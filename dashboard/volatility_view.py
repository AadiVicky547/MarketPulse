"""Volatility analytics page and non-blocking historical data preparation.

The page calls the existing CandleService and VolatilityService from a bounded
worker. It never talks to MT5/Binance directly and does not start market-data
workers of its own. MT5 time axes remain explicitly labelled unverified.
"""
from __future__ import annotations

import atexit
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
import math
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.volatility import (
    average_true_range,
    ewma_volatility,
    realized_volatility,
    rolling_volatility,
    volatility_percentile,
    volatility_zscore,
)
from analytics.volatility_service import (
    STATUS_AVAILABLE,
    STATUS_TIMESTAMP_UNVERIFIED,
    STATUS_UNAVAILABLE,
    STATUS_WARMING_UP,
    VolatilityService,
)
from analytics.regime_service import RegimeSettings
from config import INSTRUMENTS, REFRESH
from data.candle_service import CandleService, _INTERVALS
from dashboard.overview import _get_runtime, format_value
from dashboard.theme import COLORS, render_page_header, style_figure

DEFAULT_CANDLE_COUNT = 300
CACHE_SECONDS = 60.0
MAX_CACHE_ENTRIES = 12
METRIC_LABELS = {
    "atr": "ATR",
    "realized_volatility": "Realized volatility",
    "ewma_volatility": "EWMA volatility",
    "rolling_volatility": "Rolling volatility",
    "volatility_zscore": "Volatility z-score",
    "volatility_percentile": "Historical volatility percentile",
    "average_daily_range": "Average Daily Range (ADR)",
    "intraday_range_expansion": "Intraday range expansion",
}


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def format_metric(metric: Any) -> str:
    """Display a metric without converting missing/unready values to zero."""
    if metric is None:
        return "UNAVAILABLE"
    status = str(_field(metric, "status", STATUS_UNAVAILABLE)).upper()
    value = _number(_field(metric, "value"))
    if status == STATUS_AVAILABLE and value is not None:
        unit = _field(metric, "unit", "")
        return f"{format_value(value)}{f' {unit}' if unit else ''}"
    warning = _field(metric, "warning")
    return f"{status}{f' — {warning}' if warning else ''}"


def metric_card_value(metric: Any) -> str:
    """Keep unavailable session values concise without implying a value."""
    if str(_field(metric, "status", "")).upper() == STATUS_TIMESTAMP_UNVERIFIED:
        return "—"
    return format_metric(metric)


def metric_card_status(metric: Any) -> str | None:
    """Show only status text needed to explain non-timestamp-based omissions."""
    status = str(_field(metric, "status", STATUS_UNAVAILABLE)).upper()
    if status in {STATUS_AVAILABLE, STATUS_TIMESTAMP_UNVERIFIED}:
        return None
    return status


def prepare_chart_data(
    candles: pd.DataFrame,
    timeframe: str,
    timestamp_trust: str,
    settings: Any,
) -> pd.DataFrame:
    """Prepare display series with the same metric formulas and gap policy.

    Trusted Binance returns use the established pure volatility functions.
    MT5 return observations are restricted to adjacent candles at the exact
    requested cadence, matching VolatilityService; gaps are excluded and never
    bridged. MT5 metric x-values are provider candle timestamps with an
    unverified basis, not claimed UTC instants.
    """
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(candles.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {', '.join(sorted(missing))}")
    if candles.empty:
        return pd.DataFrame()
    frame = candles.loc[:, ["timestamp", "open", "high", "low", "close"]].copy(deep=True)
    frame = frame.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    parsed_timestamps = []
    for value in frame.pop("timestamp"):
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            raise ValueError("Candle timestamps must be timezone-aware")
        parsed_timestamps.append(stamp.tz_convert("UTC"))
    timestamps = pd.DatetimeIndex(parsed_timestamps)
    if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError("Candle timestamps must be strictly increasing and unique")
    frame.insert(0, "timestamp", timestamps)
    # prepare_ohlc applies the engine's OHLC and finite-value validation.
    clean = frame
    atr = average_true_range(clean, settings.atr_period)
    output = pd.DataFrame(index=atr.index)
    output["atr"] = atr
    if timestamp_trust == "TRUSTED_UTC":
        output["realized_volatility"] = realized_volatility(
            clean, settings.realized_window, periods_per_year=settings.periods_per_year)
        output["ewma_volatility"] = ewma_volatility(
            clean, settings.ewma_decay, periods_per_year=settings.periods_per_year)
        output["rolling_volatility"] = rolling_volatility(
            clean, settings.rolling_window, periods_per_year=settings.periods_per_year)
    elif timestamp_trust == "UNVERIFIED_TIME_BASIS":
        expected_seconds = _INTERVALS[timeframe][1]
        close = pd.to_numeric(clean["Close"], errors="raise").astype(float).reset_index(drop=True)
        raw_returns = np.log(close / close.shift(1))
        stamps = pd.DatetimeIndex(timestamps)
        interval_valid = np.zeros(len(clean), dtype=bool)
        if len(clean) > 1:
            interval_valid[1:] = np.asarray(
                (stamps[1:] - stamps[:-1]) == pd.Timedelta(seconds=expected_seconds), dtype=bool)
        valid_returns = raw_returns.iloc[np.flatnonzero(interval_valid)].reset_index(drop=True)
        annual_factor = (math.sqrt(settings.periods_per_year)
                         if settings.periods_per_year is not None else 1.0)
        compact_realized = valid_returns.rolling(settings.realized_window).std(ddof=1) * annual_factor
        compact_ewma = np.sqrt(
            valid_returns.pow(2).ewm(alpha=1.0 - settings.ewma_decay, adjust=False).mean()
        ) * annual_factor
        compact_rolling = valid_returns.rolling(settings.rolling_window).std(ddof=1) * annual_factor
        x = timestamps.take(np.flatnonzero(interval_valid))
        output["realized_volatility"] = np.nan
        output["ewma_volatility"] = np.nan
        output["rolling_volatility"] = np.nan
        output.loc[x, "realized_volatility"] = compact_realized.to_numpy()
        output.loc[x, "ewma_volatility"] = compact_ewma.to_numpy()
        output.loc[x, "rolling_volatility"] = compact_rolling.to_numpy()
    else:
        raise ValueError(f"Unsupported timestamp trust: {timestamp_trust}")
    realized = output["realized_volatility"]
    output["volatility_zscore"] = volatility_zscore(
        realized, settings.zscore_baseline_window)
    output["volatility_percentile"] = volatility_percentile(
        realized, settings.percentile_lookback,
        min_history=settings.percentile_min_history)
    output.index.name = "timestamp"
    return output


class _CapturingCandleService:
    """Retain the requested frame while allowing existing service requests through."""

    def __init__(self, service: Any, symbol: str, timeframe: str, count: int) -> None:
        self.service = service
        self.primary = (symbol, timeframe, count)
        self.candles: pd.DataFrame | None = None

    def get_candles(self, instrument: str, timeframe: str, count: int,
                    *, include_incomplete: bool = False) -> pd.DataFrame:
        frame = self.service.get_candles(instrument, timeframe, count,
                                         include_incomplete=include_incomplete)
        if (instrument, timeframe, count) == self.primary:
            self.candles = frame.copy(deep=True) if isinstance(frame, pd.DataFrame) else None
        return frame

    def close(self) -> None:
        """The underlying candle service is owned by the worker."""


@dataclass(frozen=True)
class VolatilityViewPayload:
    instrument: str
    timeframe: str
    candle_count_requested: int
    candles: pd.DataFrame | None
    result: Any | None
    updated_at: datetime | None
    status: str
    error: str | None = None


class VolatilityViewController:
    """Bounded asynchronous historical analytics cache for the page."""

    def __init__(self, *, candle_service_factory: Callable[[], Any] = CandleService,
                 volatility_service_factory: Callable[..., Any] = VolatilityService,
                 ttl_seconds: float = CACHE_SECONDS, executor: Any | None = None) -> None:
        self._candle_service_factory = candle_service_factory
        self._volatility_service_factory = volatility_service_factory
        self._ttl = ttl_seconds
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="marketpulse-volatility-view")
        self._lock = threading.RLock()
        self._cache: dict[tuple[str, str, int], VolatilityViewPayload] = {}
        self._cache_times: dict[tuple[str, str, int], float] = {}
        self._future: Future[Any] | None = None
        self._future_key: tuple[str, str, int] | None = None
        self._closed = False
        self._refresh_latched = False

    def request(self, instrument: str, timeframe: str, count: int = DEFAULT_CANDLE_COUNT,
                *, refresh: bool = False) -> VolatilityViewPayload:
        key = (instrument.upper(), timeframe, count)
        now = time.monotonic()
        with self._lock:
            if refresh and not self._refresh_latched:
                self._cache.pop(key, None)
                self._cache_times.pop(key, None)
                self._refresh_latched = True
            elif not refresh:
                self._refresh_latched = False
            if key in self._cache and now - self._cache_times[key] < self._ttl:
                return self._cache[key]
            self._collect_completed()
            if key in self._cache and now - self._cache_times[key] < self._ttl:
                return self._cache[key]
            if self._future is None:
                self._future_key = key
                self._future = self._executor.submit(self._load, *key)
            elif self._future_key == key and self._future.done():
                self._collect_completed()
                if key in self._cache:
                    return self._cache[key]
            return VolatilityViewPayload(key[0], timeframe, count, None, None, None, "WARMING_UP")

    def _collect_completed(self) -> None:
        if self._future is None or not self._future.done():
            return
        key = self._future_key
        try:
            payload = self._future.result()
        except Exception as exc:
            payload = VolatilityViewPayload(
                key[0] if key else "", key[1] if key else "", key[2] if key else 0,
                None, None, datetime.now(timezone.utc), "UNAVAILABLE",
                f"{type(exc).__name__}: {exc}")
        if key is not None:
            self._cache[key] = payload
            self._cache_times[key] = time.monotonic()
            while len(self._cache) > MAX_CACHE_ENTRIES:
                oldest = min(self._cache_times, key=self._cache_times.get)
                self._cache.pop(oldest, None)
                self._cache_times.pop(oldest, None)
        self._future = None
        self._future_key = None

    def _load(self, instrument: str, timeframe: str, count: int) -> VolatilityViewPayload:
        service = self._candle_service_factory()
        captured = _CapturingCandleService(service, instrument, timeframe, count)
        try:
            analytics = self._volatility_service_factory(candle_service_factory=lambda: captured)
            result = analytics.analyze_instrument(instrument, timeframe, count)
            candles = captured.candles
            if (candles is not None and not candles.empty
                    and int(_field(result, "candle_count", 0) or 0) > 0):
                expected_trust = "TRUSTED_UTC" if result.source == "Binance" else "UNVERIFIED_TIME_BASIS"
                # Ensure plot preparation uses the same contract and OHLC gate
                # as the analysis service before exposing historical series.
                try:
                    VolatilityService._validate_candle_frame(
                        candles, instrument, result.source, timeframe, expected_trust)
                except (TypeError, ValueError):
                    # The service may return UNAVAILABLE for metric readiness
                    # while the price candles themselves remain valid. Its
                    # explicit candle count plus this validation distinguish
                    # that case from rejected/retrieval-failed input.
                    candles = None
            else:
                candles = None
            return VolatilityViewPayload(
                instrument, timeframe, count, candles, result,
                datetime.now(timezone.utc), result.status)
        finally:
            close = getattr(service, "close", None)
            if callable(close):
                close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)


@st.cache_resource(show_spinner=False)
def _get_controller() -> VolatilityViewController:
    controller = VolatilityViewController()
    atexit.register(controller.close)
    return controller


def _metric_regime(result: Any) -> str:
    metric = _field(_field(result, "metrics", {}), "volatility_percentile")
    if str(_field(metric, "status", "")).upper() != STATUS_AVAILABLE:
        return "Not enough valid history to interpret volatility level."
    value = _number(_field(metric, "value"))
    if value is None:
        return "Volatility interpretation unavailable."
    thresholds = RegimeSettings()
    if value >= thresholds.high_volatility_percentile:
        return "High relative volatility: current value is in the upper historical quintile."
    if value <= thresholds.low_volatility_percentile:
        return "Low relative volatility: current value is in the lower historical quintile."
    return "Mid-range relative volatility versus the prior-history baseline."


def _dark_figure(title: str, y_title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=title,
        xaxis_title="Provider candle timestamp", yaxis_title=y_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return style_figure(figure)


def _render_charts(payload: VolatilityViewPayload, chart_data: pd.DataFrame | None) -> None:
    candles = payload.candles
    if candles is None or candles.empty or chart_data is None:
        st.info("Historical chart data is unavailable until valid candles are returned and checked.")
        return
    timestamps = pd.DatetimeIndex(pd.to_datetime(candles["timestamp"], utc=True))
    trust = str(_field(payload.result, "timestamp_trust", "unknown"))
    time_axis = "Provider candle time · basis unverified" if trust == "UNVERIFIED_TIME_BASIS" else "Candle open time · UTC"
    left, right = st.columns([1.35, 1])
    with left:
        price = _dark_figure(f"{payload.instrument} · {payload.timeframe} price", "Price")
        price.add_trace(go.Candlestick(
            x=timestamps, open=candles["open"], high=candles["high"],
            low=candles["low"], close=candles["close"], name="OHLC"))
        price.update_xaxes(title=time_axis, rangeslider_visible=False)
        st.plotly_chart(price, use_container_width=True)
    with right:
        atr = _dark_figure("Average True Range", "Price units")
        atr.add_trace(go.Scatter(x=chart_data.index, y=chart_data["atr"], name="ATR", line=dict(color=COLORS["yellow"])))
        atr.update_xaxes(title=time_axis)
        st.plotly_chart(atr, use_container_width=True)

    left, right = st.columns(2)
    with left:
        returns = _dark_figure("Realized and EWMA volatility", "Return volatility per candle")
        for key, label, color in (("realized_volatility", "Realized", COLORS["accent"]),
                                  ("ewma_volatility", "EWMA", COLORS["green"])):
            returns.add_trace(go.Scatter(x=chart_data.index, y=chart_data[key], name=label,
                                         line=dict(color=color, width=1.8)))
        returns.update_xaxes(title=time_axis)
        st.plotly_chart(returns, use_container_width=True)
    with right:
        rolling = _dark_figure("Rolling volatility", "Return volatility per candle")
        rolling.add_trace(go.Scatter(x=chart_data.index, y=chart_data["rolling_volatility"],
                                     name="Rolling", line=dict(color="#A99BEA", width=1.8)))
        rolling.update_xaxes(title=time_axis)
        st.plotly_chart(rolling, use_container_width=True)


def _render_metric_panel(result: Any) -> None:
    metrics = _field(result, "metrics", {})
    names = ("atr", "realized_volatility", "ewma_volatility", "rolling_volatility",
             "volatility_zscore", "volatility_percentile", "average_daily_range",
             "intraday_range_expansion")
    st.markdown("### Current volatility metrics")
    columns = st.columns(4)
    for index, name in enumerate(names):
        metric = _field(metrics, name)
        with columns[index % 4]:
            st.metric(METRIC_LABELS[name], metric_card_value(metric))
            if metric is not None:
                metric_status = str(_field(metric, "status", STATUS_UNAVAILABLE)).upper()
                visible_status = metric_card_status(metric)
                if visible_status:
                    st.markdown(
                        f"<div class='mp-metric-status'>{escape(visible_status)}</div>",
                        unsafe_allow_html=True,
                    )
                warning = _field(metric, "warning")
                if warning and metric_status != STATUS_TIMESTAMP_UNVERIFIED:
                    st.markdown(
                        f"<div class='mp-metric-warning'>{escape(str(warning))}</div>",
                        unsafe_allow_html=True,
                    )


@st.fragment(run_every=REFRESH.dashboard_seconds)
def _volatility_fragment(instrument: str, timeframe: str, count: int, refresh: bool) -> None:
    try:
        runtime = _get_runtime()  # Shared with Overview; no second manager is created.
        # Keep the shared background analytics cycle moving on this page too.
        # It schedules at most one cycle and returns without waiting for it.
        runtime.data_service.refresh()
        snapshot = runtime.manager.get_market_snapshot()
        states = _field(snapshot, "instruments", {})
        state = _field(states, instrument)
        connection = str(_field(state, "connection_state", "UNAVAILABLE")).upper()
        freshness = str(_field(state, "freshness", "UNAVAILABLE")).upper()
        provider = next(item.source for item in INSTRUMENTS if item.symbol == instrument)
        st.markdown(f"## {instrument} · {provider}")
        st.caption(f"Market-data connection: {connection} · Tick freshness: {freshness}")
        controller = _get_controller()
        payload = controller.request(instrument, timeframe, count, refresh=refresh)
        if payload.status == "WARMING_UP" and payload.result is None:
            st.info("Historical volatility analysis is loading in the background; this is not a normal-market reading.")
            return
        if payload.error:
            st.error(f"Volatility request failed: {payload.error}")
            return
        result = payload.result
        if result is None:
            st.info("No volatility result is available yet.")
            return
        status = str(_field(result, "status", STATUS_UNAVAILABLE)).upper()
        trust = str(_field(result, "timestamp_trust", "not reported"))
        if payload.updated_at:
            st.caption(f"Historical analysis updated {_format_utc(payload.updated_at)}")
        if status == STATUS_UNAVAILABLE:
            st.markdown(
                "<div class='mp-volatility-state-unavailable'>"
                "Volatility metrics are unavailable for this request.</div>",
                unsafe_allow_html=True,
            )
        elif status == STATUS_WARMING_UP:
            st.info("History is valid but one or more required metrics need more observations.")

        _render_metric_panel(result)
        st.markdown(
            "<div class='mp-volatility-regime-gap'></div>",
            unsafe_allow_html=True,
        )
        st.info(_metric_regime(result) + " This describes historical volatility only; it is not a trading signal.")
        chart_data = None
        if payload.candles is not None:
            try:
                chart_data = prepare_chart_data(payload.candles, timeframe, trust, VolatilityService().settings)
            except Exception as exc:
                st.caption(f"Price charts unavailable: {type(exc).__name__}: {exc}")
        _render_charts(payload, chart_data)
    except Exception as exc:
        st.error(f"Volatility page could not prepare this view: {type(exc).__name__}: {exc}")


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def render_volatility_view() -> None:
    """Render the interactive Phase 2 volatility page."""
    render_page_header("Volatility Analytics", "VOLATILITY")
    symbols = [item.symbol for item in INSTRUMENTS]
    supported_timeframes = tuple(_INTERVALS)  # CandleService's actual supported interval map.
    cols = st.columns([1.2, 1, 1, 0.8], vertical_alignment="bottom")
    instrument = cols[0].selectbox("Instrument", symbols, key="volatility_instrument")
    timeframe = cols[1].selectbox(
        "Timeframe", supported_timeframes,
        index=supported_timeframes.index("1h") if "1h" in supported_timeframes else 0,
        key="volatility_timeframe")
    count = int(cols[2].number_input("Candles", min_value=50, max_value=1000,
                                    value=DEFAULT_CANDLE_COUNT, step=50, key="volatility_count"))
    refresh = cols[3].button("Refresh history", key="volatility_refresh", use_container_width=True)
    _volatility_fragment(instrument, timeframe, count, refresh)
