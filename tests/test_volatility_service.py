"""Mocked tests for the safe candle-to-volatility integration."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from analytics.volatility_service import (
    STATUS_AVAILABLE,
    STATUS_TIMESTAMP_UNVERIFIED,
    STATUS_UNAVAILABLE,
    STATUS_WARMING_UP,
    VolatilityService,
    VolatilitySettings,
)


def candles(symbol, source, timeframe, count, *, start="2025-01-01T00:00:00Z", step="1h"):
    timestamps = pd.date_range(start, periods=count, freq=step, tz="UTC")
    close = 100 + np.arange(count, dtype=float) * 0.2
    return pd.DataFrame({
        "instrument": symbol, "source": source, "actual_symbol": symbol,
        "timeframe": timeframe, "timestamp": timestamps,
        "timestamp_trust": "TRUSTED_UTC" if source == "Binance" else "UNVERIFIED_TIME_BASIS",
        "open": close - 0.1, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": 1.0, "completed": True,
    })


class FakeCandleService:
    def __init__(self, data, *, override=None):
        self.data = data
        self.override = override or {}
        self.calls = []
        self.closed = False

    def get_candles(self, symbol, timeframe, count, *, include_incomplete=False):
        self.calls.append((symbol, timeframe, count, include_incomplete))
        key = (symbol, timeframe)
        if key in self.override:
            result = self.override[key]
            if isinstance(result, Exception):
                raise result
            return result.copy()
        return self.data[(symbol, timeframe)].copy()

    def close(self):
        self.closed = True


@pytest.fixture
def settings():
    return VolatilitySettings(
        atr_period=2, realized_window=2, ewma_decay=0.5, rolling_window=2,
        zscore_baseline_window=2, percentile_lookback=3, percentile_min_history=2,
        adr_lookback=2, range_lookback_sessions=1,
    )


def test_valid_mt5_candles_allow_sequence_metrics_only(settings):
    frame = candles("XAUUSD", "MT5", "1h", 40)
    fake = FakeCandleService({("XAUUSD", "1h"): frame})
    result = VolatilityService(candle_service_factory=lambda: fake, settings=settings).analyze_instrument(
        "XAUUSD", "1h", 40)

    assert result.status == STATUS_AVAILABLE
    assert result.timestamp_trust == "UNVERIFIED_TIME_BASIS"
    assert result.candle_count == 40
    for name in ("atr", "realized_volatility", "ewma_volatility", "rolling_volatility"):
        assert result.metrics[name].status == STATUS_AVAILABLE
        assert result.metrics[name].value is not None
    assert result.metrics["average_daily_range"].status == STATUS_TIMESTAMP_UNVERIFIED
    assert result.metrics["intraday_range_expansion"].status == STATUS_TIMESTAMP_UNVERIFIED
    assert any("sequence-based" in warning for warning in result.data_quality_warnings)
    assert fake.closed


@pytest.mark.parametrize("mutation, message", [
    ("duplicate", "duplicate"),
    ("unsorted", "chronological"),
    ("bad_ohlc", "OHLC"),
])
def test_invalid_mt5_candles_are_rejected(settings, mutation, message):
    frame = candles("XAUUSD", "MT5", "1h", 8)
    if mutation == "duplicate":
        frame.loc[3, "timestamp"] = frame.loc[2, "timestamp"]
    elif mutation == "unsorted":
        frame.loc[[2, 3], "timestamp"] = frame.loc[[3, 2], "timestamp"].to_numpy()
    else:
        frame.loc[3, "high"] = frame.loc[3, "low"] - 1
    fake = FakeCandleService({("XAUUSD", "1h"): frame})
    result = VolatilityService(candle_service_factory=lambda: fake, settings=settings).analyze_instrument(
        "XAUUSD", "1h", 8)
    assert result.status == STATUS_UNAVAILABLE
    assert message.lower() in result.data_quality_warnings[0].lower()
    assert result.metrics["atr"].value is None


def test_uninterrupted_mt5_hourly_series_has_no_excluded_returns(settings):
    frame = candles("XAUUSD", "MT5", "1h", 40)
    fake = FakeCandleService({("XAUUSD", "1h"): frame})
    result = VolatilityService(candle_service_factory=lambda: fake, settings=settings).analyze_instrument(
        "XAUUSD", "1h", 40)
    assert result.gap_count == 0
    assert result.excluded_return_count == 0
    assert not any("cadence gaps" in warning for warning in result.data_quality_warnings)


def test_mt5_weekend_sized_gap_warns_but_uses_only_hourly_returns(settings):
    timestamps = list(pd.date_range("2025-01-03T20:00:00Z", periods=3, freq="1h"))
    timestamps += list(pd.date_range("2025-01-06T00:00:00Z", periods=4, freq="1h"))
    close = np.array([100.0, 101.0, 103.0, 110.0, 111.0, 113.0, 112.0])
    frame = pd.DataFrame({
        "instrument": "XAUUSD", "source": "MT5", "actual_symbol": "XAUUSD",
        "timeframe": "1h", "timestamp": timestamps,
        "timestamp_trust": "UNVERIFIED_TIME_BASIS", "open": close - 0.2,
        "high": close + 0.5, "low": close - 0.5, "close": close,
        "volume": 1.0, "completed": None,
    })
    fake = FakeCandleService({("XAUUSD", "1h"): frame})
    result = VolatilityService(candle_service_factory=lambda: fake, settings=settings).analyze_instrument(
        "XAUUSD", "1h", len(frame))

    expected_returns = np.log(np.array([101.0 / 100.0, 103.0 / 101.0, 111.0 / 110.0,
                                        113.0 / 111.0, 112.0 / 113.0]))
    expected_rv = pd.Series(expected_returns[-2:]).std(ddof=1)
    expected_ewma_variance = expected_returns[0] ** 2
    for value in expected_returns[1:]:
        expected_ewma_variance = 0.5 * expected_ewma_variance + 0.5 * value ** 2

    assert result.status == STATUS_AVAILABLE
    assert result.gap_count == 1
    assert result.excluded_return_count == 1
    assert result.timestamp_trust == "UNVERIFIED_TIME_BASIS"
    assert result.metrics["realized_volatility"].value == pytest.approx(expected_rv)
    assert result.metrics["ewma_volatility"].value == pytest.approx(np.sqrt(expected_ewma_variance))
    assert any("1 interval(s) longer" in warning and "1 adjacent return(s) excluded" in warning
               for warning in result.data_quality_warnings)
    assert result.metrics["atr"].status == STATUS_AVAILABLE
    assert result.metrics["average_daily_range"].status == STATUS_TIMESTAMP_UNVERIFIED


def test_mt5_insufficient_consecutive_hourly_returns_warm_up_metrics(settings):
    timestamps = pd.to_datetime([
        "2025-01-01T00:00:00Z", "2025-01-01T03:00:00Z",
        "2025-01-01T06:00:00Z", "2025-01-01T09:00:00Z",
    ])
    close = np.array([100.0, 101.0, 102.0, 103.0])
    frame = pd.DataFrame({
        "instrument": "EURUSD", "source": "MT5", "actual_symbol": "EURUSD",
        "timeframe": "1h", "timestamp": timestamps,
        "timestamp_trust": "UNVERIFIED_TIME_BASIS", "open": close - 0.1,
        "high": close + 0.2, "low": close - 0.2, "close": close,
        "volume": 1.0, "completed": None,
    })
    fake = FakeCandleService({("EURUSD", "1h"): frame})
    result = VolatilityService(candle_service_factory=lambda: fake, settings=settings).analyze_instrument(
        "EURUSD", "1h", len(frame))
    assert result.gap_count == 3
    assert result.excluded_return_count == 3
    assert result.metrics["realized_volatility"].status == STATUS_WARMING_UP
    assert result.metrics["ewma_volatility"].status == STATUS_WARMING_UP
    assert result.metrics["rolling_volatility"].status == STATUS_WARMING_UP
    assert result.metrics["volatility_zscore"].status == STATUS_WARMING_UP
    assert result.metrics["volatility_percentile"].status == STATUS_WARMING_UP


def test_binance_utc_candles_allow_absolute_time_metrics(settings):
    hourly = candles("BTCUSDT", "Binance", "1h", 120)
    daily = candles("BTCUSDT", "Binance", "1d", 5, start="2025-01-01T00:00:00Z", step="1D")
    fake = FakeCandleService({("BTCUSDT", "1h"): hourly, ("BTCUSDT", "1d"): daily})
    result = VolatilityService(candle_service_factory=lambda: fake, settings=settings).analyze_instrument(
        "BTCUSDT", "1h", 120)

    assert result.timestamp_trust == "TRUSTED_UTC"
    assert result.metrics["atr"].status == STATUS_AVAILABLE
    assert result.metrics["average_daily_range"].status == STATUS_AVAILABLE
    assert result.metrics["intraday_range_expansion"].status != STATUS_TIMESTAMP_UNVERIFIED
    assert result.metrics["intraday_range_expansion"].value is not None
    assert fake.closed


def test_insufficient_history_is_warming_up(settings):
    frame = candles("EURUSD", "MT5", "1h", 2)
    fake = FakeCandleService({("EURUSD", "1h"): frame})
    result = VolatilityService(candle_service_factory=lambda: fake, settings=replace(settings, atr_period=3)).analyze_instrument(
        "EURUSD", "1h", 2)
    # EWMA is defined after one return, while the configured rolling windows
    # still need more observations. Availability is reported per metric.
    assert result.status == STATUS_AVAILABLE
    assert result.metrics["atr"].status == STATUS_WARMING_UP
    assert result.metrics["realized_volatility"].status == STATUS_WARMING_UP


def test_source_failure_isolated_and_all_six_instruments_are_registered(settings):
    class PerInstrumentService(FakeCandleService):
        def get_candles(self, symbol, timeframe, count, *, include_incomplete=False):
            self.calls.append((symbol, timeframe, count, include_incomplete))
            if symbol == "XAUUSD":
                raise RuntimeError("mock provider failure")
            return candles(symbol, "MT5" if symbol in {"EURUSD", "GBPUSD", "USDJPY"} else "Binance",
                           timeframe, 40)

    fake = PerInstrumentService({})
    results = VolatilityService(candle_service_factory=lambda: fake, settings=settings).analyze_all("1h", 40)
    assert set(results) == {"XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "BTCUSDT", "ETHUSDT"}
    assert results["XAUUSD"].status == STATUS_UNAVAILABLE
    assert results["EURUSD"].instrument == "EURUSD"
    assert fake.closed

