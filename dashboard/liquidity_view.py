"""Liquidity and order-flow analytics page backed by shared market snapshots.

All calculations come from :class:`LiquidityService`; this module prepares and
renders its structured results. The page never owns a provider connector.
"""
from __future__ import annotations

import atexit
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import threading
from types import MappingProxyType
from typing import Any, Mapping

import plotly.graph_objects as go
import streamlit as st

from analytics.liquidity_service import (
    AVAILABLE,
    LiquidityService,
    LiquiditySettings,
    UNAVAILABLE,
    WARMING_UP,
)
from config import INSTRUMENTS, REFRESH
from dashboard.overview import _field, _get_runtime, format_value
from dashboard.theme import COLORS, render_page_header, style_figure

HISTORY_SIZE = 120
DISPLAY_BOOK_LEVELS = LiquiditySettings().book_levels
METRIC_LABELS = {
    "spread_price": "Spread",
    "spread_bps": "Spread (bps)",
    "spread_expansion_ratio": "Spread expansion",
    "spread_anomaly_zscore": "Spread anomaly z-score",
    "tick_volume": "Broker-observed tick volume",
    "tick_volume_change": "Tick-volume change",
    "tick_volume_activity_ratio": "Tick-volume activity ratio",
    "tick_volume_anomaly_zscore": "Tick-volume anomaly z-score",
    "price_move_per_tick_volume": "Price movement per tick-volume unit",
    "bid_depth": "Top-N bid depth",
    "ask_depth": "Top-N ask depth",
    "order_book_imbalance": "Order Book Imbalance (OBI)",
    "bid_top_level_concentration": "Best-level bid concentration",
    "ask_top_level_concentration": "Best-level ask concentration",
    "buy_aggressor_volume": "Buy-aggressor volume",
    "sell_aggressor_volume": "Sell-aggressor volume",
    "buy_sell_volume_imbalance": "Buy/sell volume imbalance",
    "trade_count": "Trade count",
    "trade_activity": "Executed trade activity",
    "bid_depth_change": "Bid depth change",
    "ask_depth_change": "Ask depth change",
    "bid_depth_change_pct": "Bid depth change (%)",
    "ask_depth_change_pct": "Ask depth change (%)",
    "bid_depth_anomaly_zscore": "Bid depth anomaly z-score",
    "ask_depth_anomaly_zscore": "Ask depth anomaly z-score",
}


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def metric_display(metric: Any) -> str:
    """Format an existing liquidity metric without turning missing into zero."""
    if metric is None:
        return "UNAVAILABLE"
    status = str(_field(metric, "status", UNAVAILABLE)).upper()
    value = _number(_field(metric, "value"))
    if status == AVAILABLE and value is not None:
        unit = str(_field(metric, "unit", ""))
        return f"{format_value(value)}{f' {unit}' if unit else ''}"
    warning = _field(metric, "warning")
    return f"{status}{f' — {warning}' if warning else ''}"


def prepare_instrument_view(symbol: str, result: Any, snapshot: Any,
                            history: tuple[Mapping[str, Any], ...] = ()) -> dict[str, Any]:
    """Prepare one honest display model from service results and safe snapshot."""
    configured = next((item for item in INSTRUMENTS if item.symbol == symbol), None)
    if configured is None:
        raise ValueError(f"Unknown instrument: {symbol}")
    states = _field(snapshot, "instruments", {})
    state = _field(states, symbol)
    metrics = _field(result, "metrics", {}) if result is not None else {}
    provider = configured.source
    connection = str(_field(result, "connection_state", _field(state, "connection_state", "UNAVAILABLE"))).upper()
    freshness = str(_field(result, "freshness", _field(state, "freshness", "UNAVAILABLE"))).upper()
    book_map = _field(snapshot, "binance_order_books", {})
    book = _field(book_map, symbol, {}) if provider == "Binance" else {}
    book_status = _field(book, "status", {})
    synchronized = bool(_field(book_status, "synchronized", False))
    book_stale = bool(_field(book_status, "stale", True))
    depth_metrics_available = all(
        str(_field(_field(metrics, name), "status", "")).upper() == AVAILABLE
        for name in ("bid_depth", "ask_depth", "order_book_imbalance")
    )
    allow_book = bool(
        provider == "Binance" and connection == "CONNECTED" and freshness == "LIVE"
        and synchronized and not book_stale and bool(_field(book, "available", False))
        and depth_metrics_available
    )
    levels = {
        "bids": tuple(_field(book, "bids", ()) or ())[:DISPLAY_BOOK_LEVELS] if allow_book else (),
        "asks": tuple(_field(book, "asks", ()) or ())[:DISPLAY_BOOK_LEVELS] if allow_book else (),
    }
    return {
        "instrument": symbol,
        "provider": provider,
        "status": str(_field(result, "status", UNAVAILABLE)).upper(),
        "connection": connection,
        "freshness": freshness,
        "timestamp_trust": _field(result, "timestamp_trust", _field(state, "tick_timestamp_trust", "not reported")),
        "event_timestamp": _field(result, "timestamp", _field(state, "tick_timestamp")),
        "received_at": _field(result, "received_at", _field(state, "last_update_received_at")),
        "actual_symbol": _field(state, "actual_symbol", symbol if provider == "Binance" else None),
        "bid": _field(state, "bid") if connection == "CONNECTED" and freshness == "LIVE" else None,
        "ask": _field(state, "ask") if connection == "CONNECTED" and freshness == "LIVE" else None,
        "mid_price": _field(state, "mid_price"),
        "error": _field(state, "error_message"),
        "metrics": metrics,
        "warnings": tuple(_field(result, "warnings", ()) or ()),
        "book_sync": "SYNCHRONIZED" if synchronized and not book_stale else (
            "STALE" if synchronized and book_stale else "UNSYNCHRONIZED"),
        "book_available": allow_book,
        "book_levels": levels,
        "history": tuple(history),
        "volume_semantics": _field(result, "volume_semantics"),
    }


@dataclass(frozen=True)
class LiquidityPageData:
    """Last completed analytics sample and its market snapshot."""

    results: Mapping[str, Any]
    snapshot: Any | None
    history: Mapping[str, tuple[Mapping[str, Any], ...]]
    updated_at: datetime | None
    status: str
    error: str | None = None


class LiquidityPageController:
    """Run existing liquidity analytics off the Streamlit render path."""

    def __init__(self, manager: Any, *, service: Any | None = None,
                 executor: Any | None = None, history_size: int = HISTORY_SIZE) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        self.manager = manager
        self.service = service or LiquidityService(manager)
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="marketpulse-liquidity-view")
        self._lock = threading.RLock()
        self._future: Future[Any] | None = None
        self._data = LiquidityPageData(MappingProxyType({}), None, MappingProxyType({}), None, "WARMING_UP")
        self._history: dict[str, list[Mapping[str, Any]]] = {
            item.symbol: [] for item in INSTRUMENTS
        }
        self._history_size = history_size
        self._closed = False
        self._refresh_latched = False

    def request(self, *, refresh: bool = False) -> LiquidityPageData:
        """Return current data immediately and schedule at most one refresh."""
        with self._lock:
            if self._future is not None and self._future.done():
                self._collect()
            if refresh and not self._refresh_latched:
                self._data = LiquidityPageData(
                    self._data.results, self._data.snapshot, self._data.history,
                    self._data.updated_at, "WARMING_UP")
                self._refresh_latched = True
            elif not refresh:
                self._refresh_latched = False
            if self._future is None:
                self._future = self._executor.submit(self._collect_live_data)
            return self._data

    def _collect_live_data(self) -> LiquidityPageData:
        results = self.service.analyze()
        snapshot = self.manager.get_market_snapshot()
        sampled_at = datetime.now(timezone.utc)
        for item in INSTRUMENTS:
            symbol = item.symbol
            result = _field(results, symbol)
            metrics = _field(result, "metrics", {})
            sample: dict[str, Any] = {"sampled_at": sampled_at}
            for name in (
                "spread_price", "spread_bps", "order_book_imbalance", "bid_depth", "ask_depth",
                "buy_aggressor_volume", "sell_aggressor_volume", "trade_count", "trade_activity",
            ):
                metric = _field(metrics, name)
                if str(_field(metric, "status", "")).upper() == AVAILABLE:
                    sample[name] = _number(_field(metric, "value"))
                else:
                    sample[name] = None
            with self._lock:
                history = self._history[symbol]
                history.append(MappingProxyType(sample))
                del history[:-self._history_size]
        with self._lock:
            history_copy = MappingProxyType({
                symbol: tuple(items) for symbol, items in self._history.items()
            })
        return LiquidityPageData(
            MappingProxyType(dict(results)), snapshot, history_copy, sampled_at, "AVAILABLE")

    def _collect(self) -> None:
        future = self._future
        if future is None:
            return
        try:
            self._data = future.result()
        except Exception as exc:
            self._data = LiquidityPageData(
                self._data.results, self._data.snapshot, self._data.history,
                datetime.now(timezone.utc), "ERROR", f"{type(exc).__name__}: {exc}")
        self._future = None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)


@st.cache_resource(show_spinner=False)
def _get_controller() -> LiquidityPageController:
    runtime = _get_runtime()  # The same cached manager used by Overview/Volatility.
    controller = LiquidityPageController(runtime.manager)
    atexit.register(controller.close)
    return controller


def _dark_figure(title: str, y_title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=title, yaxis_title=y_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return style_figure(figure)


def _render_book_charts(view: Mapping[str, Any]) -> None:
    metrics = view["metrics"]
    levels = view["book_levels"]
    bids, asks = levels["bids"], levels["asks"]
    if not view["book_available"] or not bids or not asks:
        st.info(f"Depth charts unavailable: book status is {view['book_sync']} or the book is not fresh and synchronized.")
        return
    bid_col, ask_col = st.columns(2)
    with bid_col:
        figure = _dark_figure("Displayed bid depth · top levels", "Price")
        figure.add_trace(go.Bar(
            x=[float(_field(level, "quantity")) for level in bids],
            y=[float(_field(level, "price")) for level in bids],
            orientation="h", name="Bids", marker_color=COLORS["green"],
            customdata=[[str(_field(level, "quantity"))] for level in bids],
            hovertemplate="Price %{y}<br>Quantity %{customdata[0]}<extra></extra>"))
        figure.update_xaxes(title="Displayed quantity (base asset)")
        st.plotly_chart(figure, use_container_width=True)
    with ask_col:
        figure = _dark_figure("Displayed ask depth · top levels", "Price")
        figure.add_trace(go.Bar(
            x=[float(_field(level, "quantity")) for level in asks],
            y=[float(_field(level, "price")) for level in asks],
            orientation="h", name="Asks", marker_color=COLORS["red"],
            customdata=[[str(_field(level, "quantity"))] for level in asks],
            hovertemplate="Price %{y}<br>Quantity %{customdata[0]}<extra></extra>"))
        figure.update_xaxes(title="Displayed quantity (base asset)")
        st.plotly_chart(figure, use_container_width=True)
    obi_metric = _field(metrics, "order_book_imbalance")
    obi = _number(_field(obi_metric, "value")) if str(_field(obi_metric, "status", "")) == AVAILABLE else None
    if obi is None:
        st.info("OBI indicator is unavailable until synchronized fresh depth is available.")
    else:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=obi,
            title={"text": "Order Book Imbalance · top-N quantity"},
            gauge={"axis": {"range": [-1, 1]},
                   "bar": {"color": COLORS["accent"]},
                   "steps": [{"range": [-1, -0.2], "color": "#38242b"},
                             {"range": [-0.2, 0.2], "color": COLORS["surface_raised"]},
                             {"range": [0.2, 1], "color": "#20362f"}]},
        ))
        style_figure(gauge, height=270)
        st.plotly_chart(gauge, use_container_width=True)


def _render_history_charts(view: Mapping[str, Any]) -> None:
    history = view["history"]
    if not history:
        st.info("A short history is collected while this page is open; charts appear as observations arrive.")
        return
    rows = list(history)
    times = [row["sampled_at"] for row in rows]
    spread_values = [row.get("spread_price") for row in rows]
    spread = _dark_figure("Observed spread history", "Spread (price units)")
    spread.add_trace(go.Scatter(x=times, y=spread_values, name="Spread",
                                connectgaps=False, line=dict(color=COLORS["yellow"], width=2)))
    spread.update_xaxes(title="Local observation receive time · UTC")
    st.plotly_chart(spread, use_container_width=True)

    if view["provider"] == "Binance":
        left, right = st.columns(2)
        with left:
            volume = _dark_figure("Aggressor-side executed volume", "Base-asset quantity per buffer sample")
            volume.add_trace(go.Bar(x=["Current trade buffer"], y=[_metric_number(view, "buy_aggressor_volume")],
                                    name="Buy aggressor", marker_color=COLORS["green"]))
            volume.add_trace(go.Bar(x=["Current trade buffer"], y=[_metric_number(view, "sell_aggressor_volume")],
                                    name="Sell aggressor", marker_color=COLORS["red"]))
            volume.update_layout(barmode="group")
            st.plotly_chart(volume, use_container_width=True)
        with right:
            activity = _dark_figure("Sampled Binance trade activity", "Trades in bounded buffer")
            activity.add_trace(go.Scatter(x=times, y=[row.get("trade_count") for row in rows],
                                          name="Trade count", connectgaps=False,
                                          line=dict(color=COLORS["accent"], width=2)))
            activity.update_xaxes(title="Local observation receive time · UTC")
            st.plotly_chart(activity, use_container_width=True)
        concentration = _dark_figure("Top-level liquidity concentration", "Fraction of top-N side depth")
        for name, color in (("bid_top_level_concentration", COLORS["green"]),
                            ("ask_top_level_concentration", COLORS["red"])):
            value = _metric_number(view, name)
            if value is not None:
                concentration.add_trace(go.Bar(x=[name.replace("_", " ").title()], y=[value],
                                                name=name.replace("_", " ").title(), marker_color=color))
        concentration.update_yaxes(range=[0, 1])
        st.plotly_chart(concentration, use_container_width=True)


def _metric_number(view: Mapping[str, Any], name: str) -> float | None:
    metric = _field(view["metrics"], name)
    if str(_field(metric, "status", "")).upper() != AVAILABLE:
        return None
    return _number(_field(metric, "value"))


def _render_metrics(view: Mapping[str, Any]) -> None:
    metrics = view["metrics"]
    if view["provider"] == "MT5":
        names = ("spread_price", "spread_bps", "spread_expansion_ratio", "spread_anomaly_zscore",
                 "tick_volume", "tick_volume_change", "tick_volume_activity_ratio",
                 "tick_volume_anomaly_zscore", "price_move_per_tick_volume")
    else:
        names = ("spread_price", "spread_bps", "spread_expansion_ratio", "spread_anomaly_zscore",
                 "bid_depth", "ask_depth", "order_book_imbalance", "bid_top_level_concentration",
                 "ask_top_level_concentration", "buy_aggressor_volume", "sell_aggressor_volume",
                 "buy_sell_volume_imbalance", "trade_count", "trade_activity", "bid_depth_change",
                 "ask_depth_change", "bid_depth_change_pct", "ask_depth_change_pct",
                 "bid_depth_anomaly_zscore", "ask_depth_anomaly_zscore")
    st.markdown("### Liquidity and activity metrics")
    columns = st.columns(3)
    for index, name in enumerate(names):
        metric = _field(metrics, name)
        with columns[index % 3]:
            st.metric(METRIC_LABELS[name], metric_display(metric))


@st.fragment(run_every=REFRESH.dashboard_seconds)
def _liquidity_fragment(symbol: str, refresh: bool) -> None:
    try:
        # Trigger the same single-flight background analytics cycle used by
        # Overview so persisted alerts continue to be evaluated on this page.
        _get_runtime().data_service.refresh()
        controller = _get_controller()
        data = controller.request(refresh=refresh)
        if data.status == "WARMING_UP" and not data.results:
            st.info("Liquidity analytics are loading in the background.")
            return
        if data.error:
            st.error(f"Liquidity analytics error: {data.error}")
        snapshot = data.snapshot
        result = _field(data.results, symbol)
        if snapshot is None or result is None:
            st.warning("No completed liquidity sample is available yet.")
            return
        view = prepare_instrument_view(symbol, result, snapshot, data.history.get(symbol, ()))
        st.markdown(f"## {symbol} · {view['provider']}")
        summary = (
            f"Status: {view['status']} · Connection: {view['connection']} · Freshness: {view['freshness']} · "
            f"Timestamp trust: {view['timestamp_trust']} · Broker symbol: {view['actual_symbol'] or 'unavailable'}"
        )
        if view["provider"] == "Binance":
            summary += f" · Order-book synchronization: {view['book_sync']}"
        st.caption(summary)
        if data.updated_at:
            st.caption(f"Last analytics sample: {data.updated_at.astimezone(timezone.utc):%Y-%m-%d %H:%M:%S UTC}")
        if view["error"]:
            st.warning(str(view["error"]))
        quote_columns = st.columns(4)
        for column, label, value in zip(quote_columns, ("Bid", "Ask", "Spread", "Spread bps"),
                                       (view["bid"], view["ask"],
                                        _metric_number(view, "spread_price"),
                                        _metric_number(view, "spread_bps"))):
            if label == "Spread bps":
                unit = "bps"
            elif label == "Spread":
                unit = "price units"
            else:
                unit = ""
            column.metric(label, f"{format_value(value)}{f' {unit}' if unit else ''}")
        _render_metrics(view)
        if view["provider"] == "Binance":
            _render_book_charts(view)
        _render_history_charts(view)
    except Exception as exc:
        st.error(f"Liquidity page could not prepare this view: {type(exc).__name__}: {exc}")


def render_liquidity_view() -> None:
    """Render the Phase 3 Liquidity & Order Flow page."""
    render_page_header("Liquidity & Order Flow", "LIQUIDITY · ORDER FLOW")
    symbols = [item.symbol for item in INSTRUMENTS]
    left, right, action = st.columns([1.2, 1, 0.8])
    symbol = left.selectbox("Instrument", symbols, key="liquidity_instrument")
    provider = next(item.source for item in INSTRUMENTS if item.symbol == symbol)
    right.caption(f"Provider: {provider}")
    # The button column has no field label above it, so add matching vertical
    # space to align its control with the instrument selectbox itself.
    action.markdown('<div aria-hidden="true" style="height: 1.9rem"></div>',
                    unsafe_allow_html=True)
    refresh = action.button("Refresh", key="liquidity_refresh", use_container_width=True)
    _liquidity_fragment(symbol, refresh)
