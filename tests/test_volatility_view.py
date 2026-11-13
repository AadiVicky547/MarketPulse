"""Mocked tests for volatility-page data preparation and controller behavior."""
from concurrent.futures import Future
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

from analytics.volatility_service import VolatilitySettings
from dashboard.theme import _THEME_CSS
from dashboard.volatility_view import (
    VolatilityViewController,
    format_metric,
    metric_card_value,
    metric_card_status,
    prepare_chart_data,
)


def _candles(*, gaps: bool = False) -> pd.DataFrame:
    times = pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="h")
    if gaps:
        times = times.delete(4) + pd.to_timedelta([0, 0, 0, 0, 2, 2, 2], unit="h")
    closes = np.array([100, 101, 102, 103, 104, 105, 104, 106], dtype=float)[:len(times)]
    return pd.DataFrame({
        "timestamp": times,
        "open": closes - 0.2,
        "high": closes + 0.8,
        "low": closes - 0.8,
        "close": closes,
    })


def test_chart_data_calculates_trusted_binance_metrics_and_preserves_zeroes():
    settings = VolatilitySettings(atr_period=2, realized_window=2, rolling_window=2,
                                  zscore_baseline_window=2, percentile_lookback=4,
                                  percentile_min_history=2)
    result = prepare_chart_data(_candles(), "1h", "TRUSTED_UTC", settings)
    assert result.index.tz is not None
    assert result["atr"].notna().any()
    assert result["realized_volatility"].notna().any()
    assert result["ewma_volatility"].notna().any()
    assert result["rolling_volatility"].equals(result["realized_volatility"])
    assert "volatility_percentile" in result


def test_mt5_gap_return_is_excluded_without_bridging_and_atr_remains_sequence_based():
    settings = VolatilitySettings(atr_period=2, realized_window=2, rolling_window=2,
                                  zscore_baseline_window=2, percentile_lookback=4,
                                  percentile_min_history=2)
    frame = _candles(gaps=True)
    original = frame.copy(deep=True)
    result = prepare_chart_data(frame, "1h", "UNVERIFIED_TIME_BASIS", settings)
    # The return ending at index 4 spans four hours and is excluded. The next
    # exact one-hour return is usable as a new observation, without a synthetic
    # return across the gap.
    assert pd.isna(result.iloc[4]["realized_volatility"])
    assert np.isfinite(result.iloc[5]["realized_volatility"])
    assert np.isfinite(result["atr"].iloc[-1])
    pd.testing.assert_frame_equal(frame, original)


def test_chart_prep_rejects_duplicate_or_out_of_order_timestamps():
    settings = VolatilitySettings(atr_period=2)
    frame = _candles()
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    try:
        prepare_chart_data(frame, "1h", "TRUSTED_UTC", settings)
    except ValueError as exc:
        assert "increasing and unique" in str(exc)
    else:
        raise AssertionError("duplicate timestamps must be rejected")


def test_chart_prep_rejects_naive_timestamps_instead_of_assuming_utc():
    settings = VolatilitySettings(atr_period=2)
    frame = _candles()
    frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
    try:
        prepare_chart_data(frame, "1h", "TRUSTED_UTC", settings)
    except ValueError as exc:
        assert "timezone-aware" in str(exc)
    else:
        raise AssertionError("naive timestamps must not be interpreted as UTC")


def test_metric_display_distinguishes_warming_unavailable_and_available_zero():
    assert format_metric(SimpleNamespace(status="AVAILABLE", value=0.0, unit="x")) == "0 x"
    assert format_metric(SimpleNamespace(status="WARMING_UP", value=None, warning="need more bars")) == (
        "WARMING_UP — need more bars")
    assert format_metric(SimpleNamespace(status="UNAVAILABLE", value=None)) == "UNAVAILABLE"
    assert format_metric(None) == "UNAVAILABLE"


def test_timestamp_unverified_metric_is_concise_but_not_fabricated():
    metric = SimpleNamespace(
        status="TIMESTAMP_UNVERIFIED", value=None,
        warning="Absolute session time is not verified.",
    )
    assert metric_card_value(metric) == "—"
    assert metric_card_status(metric) is None
    assert metric_card_status(SimpleNamespace(status="WARMING_UP")) == "WARMING_UP"


def test_volatility_interpretation_box_has_two_line_spacing():
    assert ".mp-volatility-regime-gap" in _THEME_CSS
    assert "height: 1.75rem" in _THEME_CSS


class DeferredExecutor:
    def __init__(self):
        self.fn = None
        self.future = Future()
        self.shutdown_called = False

    def submit(self, fn, *args):
        self.fn = lambda: fn(*args)
        return self.future

    def shutdown(self, *, wait, cancel_futures):
        self.shutdown_called = True
        assert wait and cancel_futures


def test_controller_returns_promptly_then_caches_mocked_result():
    now = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="h")
    candles = pd.DataFrame({
        "instrument": ["BTCUSDT"] * 4, "source": ["Binance"] * 4,
        "timeframe": ["1h"] * 4, "timestamp": now,
        "timestamp_trust": ["TRUSTED_UTC"] * 4,
        "open": [10., 11., 12., 13.], "high": [12., 13., 14., 15.],
        "low": [9., 10., 11., 12.], "close": [11., 12., 13., 14.],
    })

    class CandleMock:
        def get_candles(self, instrument, timeframe, count, *, include_incomplete=False):
            return candles

        def close(self):
            pass

    result = SimpleNamespace(
        instrument="BTCUSDT", source="Binance", timeframe="1h", status="WARMING_UP",
        timestamp_trust="TRUSTED_UTC", candle_count=4, gap_count=0,
        excluded_return_count=0, metrics={}, data_quality_warnings=(),
    )

    class VolatilityMock:
        def __init__(self, *, candle_service_factory):
            self.candles = candle_service_factory()

        def analyze_instrument(self, instrument, timeframe, count):
            self.candles.get_candles(instrument, timeframe, count)
            return result

    executor = DeferredExecutor()
    controller = VolatilityViewController(
        candle_service_factory=CandleMock,
        volatility_service_factory=VolatilityMock,
        executor=executor,
    )
    first = controller.request("BTCUSDT", "1h", 4)
    assert first.status == "WARMING_UP" and first.result is None
    payload = executor.fn()
    executor.future.set_result(payload)
    second = controller.request("BTCUSDT", "1h", 4)
    assert second.result is result
    assert len(second.candles) == 4
    controller.close()
    assert executor.shutdown_called
