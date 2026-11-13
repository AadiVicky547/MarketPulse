"""Phase 1 MarketPulse overview page and non-blocking data preparation."""
from __future__ import annotations

import atexit
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
import math
import logging
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping

import pandas as pd
import streamlit as st

from analytics.alert_service import AlertService
from analytics.anomaly_service import AnomalyService
from analytics.liquidity_service import LiquidityService
from analytics.regime_service import RegimeService
from analytics.volatility_service import VolatilityService
from config import INSTRUMENTS, REFRESH
from data.market_manager import MarketDataManager
from dashboard.theme import (
    COLORS,
    alert_container_key,
    alert_severity_badge_html,
    render_page_header,
)


ANALYTICS_CACHE_SECONDS = 60.0
WATCHLIST_COLUMNS = 3
logger = logging.getLogger(__name__)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def format_value(value: Any, *, decimals: int = 8) -> str:
    """Format a genuine numeric value while preserving zero and missingness."""
    number = _numeric(value)
    if number is None:
        return "—"
    rendered = f"{number:,.{decimals}f}".rstrip("0").rstrip(".")
    return rendered if rendered not in {"", "-0"} else "0"


def _display_metric(result: Any, metric_name: str) -> str:
    metrics = _field(result, "metrics", {})
    metric = _field(metrics, metric_name)
    if metric is None:
        return "UNAVAILABLE"
    status = str(_field(metric, "status", "UNAVAILABLE")).upper()
    value = _numeric(_field(metric, "value"))
    if status == "AVAILABLE" and value is not None:
        unit = _field(metric, "unit", "")
        return f"{format_value(value)}{f' {unit}' if unit else ''}"
    warning = _field(metric, "warning")
    return f"{status}{f' — {warning}' if warning else ''}"


def _volatility_regime_label(result: Any) -> str:
    """Display the regime service's high/low-volatility label when evidenced."""
    status = str(_field(result, "status", "UNAVAILABLE")).upper()
    if status != "AVAILABLE":
        return status
    current = _field(result, "current_regime")
    if current in {"HIGH_VOLATILITY", "LOW_VOLATILITY"}:
        return str(current)
    features = _field(result, "features", {})
    percentiles = []
    for name in ("atr_percentile", "realized_volatility_percentile"):
        feature = _field(features, name)
        if str(_field(feature, "status", "UNAVAILABLE")).upper() == "AVAILABLE":
            value = _numeric(_field(feature, "value"))
            if value is not None:
                percentiles.append(value)
    if not percentiles:
        return "UNAVAILABLE"
    # The regime service gives volatility classes precedence over directional
    # labels. This says only that no high/low class was selected, not that risk
    # is absent or low.
    return "No high/low label"


def _watchlist_price(state: Any) -> tuple[Any, str]:
    """Select the same available price used by the Overview display."""
    midpoint = _field(state, "mid_price")
    if _numeric(midpoint) is not None:
        return midpoint, "Midpoint"
    last_trade = _field(state, "last_trade_price")
    if _numeric(last_trade) is not None:
        return last_trade, "Last trade"
    return None, "Price unavailable"


def _price_direction(current: Any, previous: Any) -> str:
    """Return a presentation direction only for two valid positive prices."""
    current_number = _numeric(current)
    previous_number = _numeric(previous)
    if (current_number is None or previous_number is None
            or current_number <= 0 or previous_number <= 0):
        return "UNCHANGED"
    if current_number > previous_number:
        return "UP"
    if current_number < previous_number:
        return "DOWN"
    return "UNCHANGED"


def advance_watchlist_price_history(
    snapshot: Any,
    prior_records: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float | None], dict[str, dict[str, float | None]]]:
    """Compare snapshot prices with the previous valid displayed sample.

    This stores only six display references. It neither fetches data nor changes
    the market manager's snapshots or cadence.
    """
    previous_prices, _previous_quotes, next_records = advance_watchlist_quote_history(
        snapshot, prior_records)
    return previous_prices, next_records


def advance_watchlist_quote_history(
    snapshot: Any,
    prior_records: Mapping[str, Any] | None = None,
) -> tuple[
    dict[str, float | None],
    dict[str, dict[str, float | None]],
    dict[str, dict[str, float | None]],
]:
    """Compare midpoint, bid, and ask with prior valid values from snapshots."""
    states = _field(snapshot, "instruments", {})
    prior_records = prior_records or {}
    next_records: dict[str, dict[str, float | None]] = {}
    previous_prices: dict[str, float | None] = {}
    previous_quotes: dict[str, dict[str, float | None]] = {}
    for instrument in INSTRUMENTS:
        symbol = instrument.symbol.upper()
        state = _field(states, symbol)
        current, _ = _watchlist_price(state) if state is not None else (None, "Price unavailable")
        current_number = _numeric(current)
        old_record = _field(prior_records, symbol, {})
        old_last = _numeric(_field(old_record, "last"))
        old_bid = _numeric(_field(old_record, "bid"))
        old_ask = _numeric(_field(old_record, "ask"))
        current_bid = _numeric(_field(state, "bid")) if state is not None else None
        current_ask = _numeric(_field(state, "ask")) if state is not None else None
        previous_quotes[symbol] = {"bid": old_bid, "ask": old_ask}
        next_record = {"last": old_last, "bid": old_bid, "ask": old_ask}
        if current_number is not None and current_number > 0:
            next_record["last"] = current_number
            previous_prices[symbol] = old_last
        if current_bid is not None and current_bid > 0:
            next_record["bid"] = current_bid
        if current_ask is not None and current_ask > 0:
            next_record["ask"] = current_ask
        if any(value is not None for value in next_record.values()):
            next_records[symbol] = next_record
    return previous_prices, previous_quotes, next_records


def prepare_watchlist(snapshot: Any, *,
                      previous_prices: Mapping[str, Any] | None = None,
                      previous_quotes: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Build display rows from a manager snapshot and optional prior UI prices."""
    states = _field(snapshot, "instruments", {})
    previous_prices = previous_prices or {}
    previous_quotes = previous_quotes or {}
    rows: list[dict[str, Any]] = []
    for instrument in INSTRUMENTS:
        symbol = instrument.symbol.upper()
        state = _field(states, symbol)
        if state is None:
            rows.append({
                "instrument": symbol, "provider": instrument.source,
                "actual_symbol": "unavailable", "price_label": "Price",
                "price": "—", "bid": "—", "ask": "—", "spread": "—",
                "spread_bps": "—", "connection": "UNAVAILABLE",
                "freshness": "UNAVAILABLE", "timestamp_trust": "not reported",
                "timestamp_source": "not reported", "price_direction": "UNCHANGED",
                "bid_direction": "UNCHANGED", "ask_direction": "UNCHANGED",
            })
            continue
        price_value, price_label = _watchlist_price(state)
        price = format_value(price_value)
        old_quotes = _field(previous_quotes, symbol, {})
        bid_value = _field(state, "bid")
        ask_value = _field(state, "ask")
        rows.append({
            "instrument": symbol,
            "provider": "MT5 broker" if instrument.source == "MT5" else "Binance public",
            "actual_symbol": _field(state, "actual_symbol") or "unavailable",
            "price_label": price_label,
            "price": price,
            "price_direction": _price_direction(
                price_value, _field(previous_prices, symbol)),
            "bid": format_value(bid_value),
            "ask": format_value(ask_value),
            "bid_direction": _price_direction(bid_value, _field(old_quotes, "bid")),
            "ask_direction": _price_direction(ask_value, _field(old_quotes, "ask")),
            "spread": format_value(_field(state, "spread")),
            "spread_bps": format_value(_field(state, "spread_bps")),
            "connection": str(_field(state, "connection_state", "UNAVAILABLE")).upper(),
            "freshness": str(_field(state, "freshness", "UNAVAILABLE")).upper(),
            "timestamp_trust": _field(state, "tick_timestamp_trust") or "not reported",
            "timestamp_source": _field(state, "tick_timestamp_source") or "not reported",
        })
    return rows


def prepare_analytics_summary(volatility: Any, liquidity: Any,
                              regimes: Any, anomalies: Any) -> list[dict[str, str]]:
    """Join existing analytics results for display without recalculating them."""
    rows: list[dict[str, str]] = []
    for instrument in INSTRUMENTS:
        symbol = instrument.symbol.upper()
        vol_result = _field(volatility, symbol)
        liq_result = _field(liquidity, symbol)
        regime_result = _field(regimes, symbol)
        anomaly_result = _field(anomalies, symbol)

        regime_status = str(_field(regime_result, "status", "UNAVAILABLE")).upper()
        regime_label = _field(regime_result, "current_regime")
        regime_display = (str(regime_label) if regime_status == "AVAILABLE" and regime_label
                          else regime_status)

        anomaly_status = str(_field(anomaly_result, "status", "UNAVAILABLE")).upper()
        anomaly_value = _field(anomaly_result, "anomaly_detected")
        if anomaly_status == "AVAILABLE" and isinstance(anomaly_value, bool):
            anomaly_display = "ANOMALY" if anomaly_value else "Not flagged"
        else:
            anomaly_display = anomaly_status

        if instrument.source == "MT5":
            flow_value = (f"Tick volume {_display_metric(liq_result, 'tick_volume')} · "
                          f"activity {_display_metric(liq_result, 'tick_volume_activity_ratio')}")
            flow_note = "MT5 tick volume is broker-observed activity, not centralized volume."
        else:
            flow_value = (f"OBI {_display_metric(liq_result, 'order_book_imbalance')} · "
                          f"buy {_display_metric(liq_result, 'buy_aggressor_volume')} · "
                          f"sell {_display_metric(liq_result, 'sell_aggressor_volume')}")
            flow_note = "Displayed Binance depth is cancellable and not guaranteed executable."

        rows.append({
            "Instrument": symbol,
            "Provider": instrument.source,
            "Realized volatility (1h)": _display_metric(vol_result, "realized_volatility"),
            "Volatility percentile": _display_metric(vol_result, "volatility_percentile"),
            "ATR": _display_metric(vol_result, "atr"),
            "Volatility regime": _volatility_regime_label(regime_result),
            "Market regime": regime_display,
            "Liquidity / Flow": flow_value,
            "Anomaly status": anomaly_display,
            "Timestamp trust": str(_field(vol_result, "timestamp_trust") or
                                    _field(regime_result, "timestamp_trust") or
                                    _field(anomaly_result, "timestamp_trust") or "not reported"),
            "Data note": flow_note,
        })
    return rows


def overall_connection_status(snapshot: Any) -> str:
    """Summarize the two manager source statuses without masking partial failure."""
    sources = _field(snapshot, "sources", {})
    values = [str(_field(_field(sources, source), "connection_state", "DISCONNECTED")).upper()
              for source in ("MT5", "Binance")]
    connected = sum(value == "CONNECTED" for value in values)
    if connected == len(values):
        return "LIVE"
    if connected:
        return "PARTIAL"
    if any(value in {"CONNECTING", "RECONNECTING"} for value in values):
        return "CONNECTING"
    return "DISCONNECTED"


class _CycleLiquidityService:
    """Share one LiquidityService result between the overview and anomaly pass."""

    def __init__(self, service: Any) -> None:
        self._service = service
        self._result: Any = None
        self._attempted = False

    def begin_cycle(self) -> None:
        self._result = None
        self._attempted = False

    def analyze(self) -> Any:
        if not self._attempted:
            self._attempted = True
            self._result = self._service.analyze()
        return self._result


class _TimedAnalyticsCache:
    """Refresh candle analytics off the UI thread at a bounded cadence."""

    def __init__(self, service: Any, method_name: str, ttl_seconds: float) -> None:
        self.service = service
        self.method_name = method_name
        self.ttl_seconds = ttl_seconds
        self._result: Mapping[str, Any] = MappingProxyType({})
        self._last_attempt = 0.0
        self.updated_at: datetime | None = None
        self.error: str | None = None

    def get(self, *args: Any) -> Mapping[str, Any]:
        now = time.monotonic()
        if now - self._last_attempt >= self.ttl_seconds:
            self._last_attempt = now
            try:
                result = getattr(self.service, self.method_name)(*args)
                if isinstance(result, Mapping):
                    self._result = result
                    self.updated_at = datetime.now(timezone.utc)
                    self.error = None
                else:
                    self.error = "Analytics service returned an unexpected result structure"
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
        return self._result


class _CachedVolatilityProxy:
    """Expose the VolatilityService API while using the current cached result."""

    def __init__(self, cache: _TimedAnalyticsCache) -> None:
        self.cache = cache

    def analyze_all(self, timeframe: str, count: int) -> Mapping[str, Any]:
        return self.cache.get(timeframe, count)


@dataclass(frozen=True)
class OverviewData:
    snapshot: Any
    volatility: Mapping[str, Any] = field(default_factory=dict)
    liquidity: Mapping[str, Any] = field(default_factory=dict)
    regimes: Mapping[str, Any] = field(default_factory=dict)
    anomalies: Mapping[str, Any] = field(default_factory=dict)
    alerts: tuple[Any, ...] = ()
    analytics_pending: bool = True
    errors: tuple[str, ...] = ()
    volatility_updated_at: datetime | None = None
    regime_updated_at: datetime | None = None


class OverviewDataService:
    """Prepare snapshots and analytics asynchronously for the Streamlit page."""

    def __init__(self, market_manager: Any, *,
                 volatility_service: Any | None = None,
                 liquidity_service: Any | None = None,
                 regime_service: Any | None = None,
                 anomaly_service: Any | None = None,
                 alert_service: Any | None = None,
                 analytics_cache_seconds: float = ANALYTICS_CACHE_SECONDS,
                 executor: Any | None = None) -> None:
        self.market_manager = market_manager
        self.volatility_service = volatility_service or VolatilityService()
        self._liquidity = _CycleLiquidityService(liquidity_service or LiquidityService(market_manager))
        self.regime_service = regime_service or RegimeService(market_manager=market_manager)
        self._volatility_cache = _TimedAnalyticsCache(
            self.volatility_service, "analyze_all", analytics_cache_seconds)
        self._volatility_proxy = _CachedVolatilityProxy(self._volatility_cache)
        self.anomaly_service = anomaly_service or AnomalyService(
            market_manager, liquidity_service=self._liquidity,
            volatility_service=self._volatility_proxy,
        )
        self.alert_service = alert_service or AlertService(
            market_manager, volatility_service=self._volatility_proxy,
            liquidity_service=self._liquidity, regime_service=self.regime_service,
            anomaly_service=self.anomaly_service,
        )
        self._regime_cache = _TimedAnalyticsCache(
            self.regime_service, "analyze_all", analytics_cache_seconds)
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="marketpulse-overview")
        self._lock = threading.RLock()
        self._future: Future[Any] | None = None
        self._latest: dict[str, Any] = {
            "volatility": {}, "liquidity": {}, "regimes": {}, "anomalies": {},
            "alerts": (), "errors": (),
        }

    def refresh(self) -> OverviewData:
        """Return the latest safe snapshot immediately; analytics run in a worker."""
        snapshot = self.market_manager.get_market_snapshot()
        with self._lock:
            if self._future is not None and self._future.done():
                try:
                    self._latest = self._future.result()
                except Exception as exc:
                    self._latest["errors"] = (f"Overview analytics worker failed: {type(exc).__name__}: {exc}",)
                self._future = None
            if self._future is None:
                self._future = self._executor.submit(self._calculate_analytics)
            return OverviewData(
                snapshot=snapshot,
                volatility=self._latest["volatility"],
                liquidity=self._latest["liquidity"],
                regimes=self._latest["regimes"],
                anomalies=self._latest["anomalies"],
                alerts=self._latest["alerts"],
                analytics_pending=self._future is not None,
                errors=tuple(self._latest["errors"]),
                volatility_updated_at=self._volatility_cache.updated_at,
                regime_updated_at=self._regime_cache.updated_at,
            )

    def _calculate_analytics(self) -> dict[str, Any]:
        errors: list[str] = []
        self._liquidity.begin_cycle()
        try:
            liquidity = self._liquidity.analyze()
        except Exception as exc:
            liquidity = {}
            errors.append(f"Liquidity analysis: {type(exc).__name__}: {exc}")

        volatility = self._volatility_cache.get("1h", 200)
        if self._volatility_cache.error:
            errors.append(f"Volatility analysis: {self._volatility_cache.error}")

        regimes = self._regime_cache.get("1h", 600)
        if self._regime_cache.error:
            errors.append(f"Regime analysis: {self._regime_cache.error}")

        try:
            anomalies = self.anomaly_service.analyze()
        except Exception as exc:
            anomalies = {}
            errors.append(f"Anomaly analysis: {type(exc).__name__}: {exc}")

        # Evaluate once per completed shared cycle, reusing its results. This
        # runs on the existing analytics executor, never in a Streamlit render
        # function, and AlertService enforces freshness plus suppression rules.
        try:
            snapshot = self.market_manager.get_market_snapshot()
            self.alert_service.evaluate(
                snapshot=snapshot,
                inputs={"volatility": volatility, "liquidity": liquidity,
                        "regime": regimes, "anomaly": anomalies},
            )
        except Exception as exc:
            logger.exception("Shared-cycle alert evaluation failed")
            errors.append(f"Alert evaluation: {type(exc).__name__}: {exc}")

        try:
            alerts = tuple(self.alert_service.get_recent_alerts(limit=8))
        except Exception as exc:
            alerts = ()
            errors.append(f"Alert history: {type(exc).__name__}: {exc}")
        return {
            "volatility": volatility,
            "liquidity": liquidity,
            "regimes": regimes,
            "anomalies": anomalies,
            "alerts": alerts,
            "errors": tuple(errors),
        }

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an existing alert without creating or evaluating alerts."""
        return bool(self.alert_service.acknowledge_alert(alert_id))

    def close(self) -> None:
        """Finish queued analytics work before the manager is stopped at exit."""
        self._executor.shutdown(wait=True, cancel_futures=True)


@dataclass
class DashboardRuntime:
    manager: MarketDataManager
    data_service: OverviewDataService
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.data_service.close()
        self.manager.stop()


@st.cache_resource(show_spinner=False)
def _get_runtime() -> DashboardRuntime:
    """Create one app-process manager; Streamlit reruns reuse this resource."""
    manager = MarketDataManager()
    try:
        manager.start()
        runtime = DashboardRuntime(manager, OverviewDataService(manager))
    except Exception:
        manager.stop()
        raise
    atexit.register(runtime.close)
    return runtime


def _request_shared_analytics_cycle() -> None:
    """Request one shared analytics cycle without waiting for completion."""
    try:
        _get_runtime().data_service.refresh()
    except Exception:
        logger.exception("Shared analytics scheduler request failed")


@st.fragment(run_every=REFRESH.dashboard_seconds)
def render_shared_analytics_scheduler() -> None:
    """Keep shared analytics sampling active while any dashboard page is open.

    This lightweight Streamlit fragment runs independently of the selected
    page. ``OverviewDataService.refresh`` is single-flight: it schedules work
    on the one existing analytics executor and returns immediately when that
    cycle is already running.
    """
    _request_shared_analytics_cycle()


def _status_color(status: str) -> str:
    normalized = status.upper()
    if normalized in {"LIVE", "CONNECTED", "AVAILABLE", "SYNCHRONIZED"}:
        return COLORS["green"]
    if normalized in {"PARTIAL", "CONNECTING", "STALE", "WARMING_UP", "UNVERIFIED_TIME_BASIS"}:
        return COLORS["yellow"]
    return COLORS["red"]


def _render_status(label: str, status: str) -> None:
    color = _status_color(status)
    st.markdown(
        f"<span class='status-pill' style='--status-color:{color}'>{escape(label)}: {escape(status)}</span>",
        unsafe_allow_html=True,
    )


def _utc_text(value: Any) -> str:
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return "not available"


def _render_header(snapshot: Any) -> None:
    render_page_header("Cross-Asset Overview", "LIVE MARKET MONITOR")
    clock_col, market_col, mt5_col, binance_col = st.columns(
        [1.15, 1, 1, 1], vertical_alignment="center")
    with clock_col:
        st.metric("SYSTEM TIME · UTC", datetime.now(timezone.utc).strftime("%H:%M:%S"))
    with market_col:
        _render_status("MARKET DATA", overall_connection_status(snapshot))
    source_map = _field(snapshot, "sources", {})
    mt5_state = _field(_field(source_map, "MT5"), "connection_state", "DISCONNECTED")
    binance_state = _field(_field(source_map, "Binance"), "connection_state", "DISCONNECTED")
    with mt5_col:
        _render_status("MT5", str(mt5_state).upper())
    with binance_col:
        _render_status("BINANCE", str(binance_state).upper())


def _render_watchlist(snapshot: Any) -> None:
    st.markdown("### Cross-Asset Watchlist")
    st.caption("MT5 values are broker-provided quotes. Binance values are public Spot market data.")
    previous_prices, previous_quotes, next_records = advance_watchlist_quote_history(
        snapshot,
        st.session_state.get("marketpulse_watchlist_price_history", {}),
    )
    st.session_state["marketpulse_watchlist_price_history"] = next_records
    rows = prepare_watchlist(
        snapshot, previous_prices=previous_prices, previous_quotes=previous_quotes)
    columns = st.columns(WATCHLIST_COLUMNS, gap="medium")
    price_colors = {"UP": COLORS["green"], "DOWN": COLORS["red"],
                    "UNCHANGED": "#FFFFFF"}
    for index, row in enumerate(rows):
        with columns[index % WATCHLIST_COLUMNS]:
            with st.container(border=True):
                title_col, state_col = st.columns([1.2, 1])
                with title_col:
                    st.markdown(f"**{row['instrument']}**")
                with state_col:
                    _render_status("DATA", row["freshness"])
                st.markdown(
                    f"<div class='mp-asset-description'>{escape(row['provider'])} · "
                    f"{escape(str(row['actual_symbol']))} · Event time: "
                    f"{escape(str(row['timestamp_trust']))} · "
                    f"{escape(str(row['timestamp_source']))}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='mp-quote-metric'>"
                    f"<div class='mp-quote-label'>{escape(row['price_label'])}</div>"
                    f"<div class='mp-quote-price' style='color:{price_colors[row['price_direction']]}' "
                    f"data-direction='{row['price_direction']}'>{escape(row['price'])}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                bid_col, ask_col = st.columns(2)
                bid_col.markdown(
                    _quote_detail_html("Bid", row["bid"], price_colors[row["bid_direction"]]),
                    unsafe_allow_html=True,
                )
                ask_col.markdown(
                    _quote_detail_html("Ask", row["ask"], price_colors[row["ask_direction"]]),
                    unsafe_allow_html=True,
                )
                spread_col, bps_col = st.columns(2)
                spread_col.markdown(_quote_detail_html("Spread", row["spread"]), unsafe_allow_html=True)
                bps_col.markdown(_quote_detail_html("Spread bps", row["spread_bps"]), unsafe_allow_html=True)
                connection_col, freshness_col = st.columns(2)
                connection_col.markdown(_quote_detail_html("Connection", row["connection"]), unsafe_allow_html=True)
                freshness_col.markdown(_quote_detail_html("Freshness", row["freshness"]), unsafe_allow_html=True)


def _quote_detail_html(label: str, value: str, value_color: str = "#FFFFFF") -> str:
    """Return escaped quote details with subdued labels and opaque values."""
    return (f"<div class='mp-quote-detail'><span>{escape(label)}</span> "
            f"<strong style='--mp-value-color:{value_color}'>{escape(value)}</strong></div>")


def _render_analytics(data: OverviewData) -> None:
    st.markdown("### Analytics Summary")
    if data.analytics_pending and not data.volatility and not data.liquidity:
        st.info("Analytics are warming up. Market-data workers continue independently while history is prepared.")
    if data.errors:
        with st.expander("Analytics status details", expanded=False):
            for error in data.errors:
                st.caption(error)
    rows = prepare_analytics_summary(data.volatility, data.liquidity,
                                     data.regimes, data.anomalies)
    frame = pd.DataFrame(rows)
    st.dataframe(frame, use_container_width=True, hide_index=True, height=290)
    times = []
    if data.volatility_updated_at:
        times.append(f"Volatility data as of {_utc_text(data.volatility_updated_at)}")
    if data.regime_updated_at:
        times.append(f"Regime data as of {_utc_text(data.regime_updated_at)}")
    if times:
        st.caption(" · ".join(times) + f" · Candle analytics refresh every {ANALYTICS_CACHE_SECONDS:g}s")
    st.caption("Anomaly labels describe statistical unusualness, not harm or a trading recommendation.")


def _render_alerts(data: OverviewData, runtime: DashboardRuntime) -> None:
    st.markdown("### Recent Persisted Alerts")
    if not data.alerts:
        st.caption("No persisted alerts are available. The Overview reads alert history but does not generate alerts.")
        return
    for index, alert in enumerate(data.alerts):
        severity = str(alert.severity).upper()
        with st.container(
            border=True,
            key=alert_container_key(severity, str(index)),
        ):
            top = st.columns([1.2, 1, 2.4, 1])
            top[0].markdown(f"**{alert.instrument}** · {alert.provider}")
            top[1].markdown(
                f"{alert_severity_badge_html(severity)} · "
                f"{escape(alert.category.replace('_', ' ').title())}",
                unsafe_allow_html=True,
            )
            top[2].caption(_utc_text(alert.detected_at_utc))
            if alert.acknowledged:
                top[3].caption("Acknowledged")
            elif top[3].button("Acknowledge", key=f"ack-{alert.alert_id}",
                               use_container_width=True):
                try:
                    runtime.data_service.acknowledge_alert(alert.alert_id)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not acknowledge alert: {type(exc).__name__}: {exc}")
            st.write(alert.explanation)
            st.caption(f"Detection time provenance: {alert.timestamp_provenance}")


@st.fragment(run_every=REFRESH.dashboard_seconds)
def _overview_fragment() -> None:
    """Refresh the view periodically without blocking on analytics requests."""
    try:
        runtime = _get_runtime()
        data = runtime.data_service.refresh()
    except Exception as exc:
        st.error(f"Overview initialization failed: {type(exc).__name__}: {exc}")
        st.info("The page will retry on its next refresh.")
        return
    _render_header(data.snapshot)
    st.divider()
    _render_watchlist(data.snapshot)
    st.divider()
    _render_analytics(data)
    st.divider()
    _render_alerts(data, runtime)


def render_overview() -> None:
    """Render the initial Overview page."""
    _overview_fragment()
