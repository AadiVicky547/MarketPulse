"""Deterministic mocked tests for MarketPulse regime classification."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from analytics.regime_service import (
    AVAILABLE,
    TIMESTAMP_UNVERIFIED,
    UNAVAILABLE,
    WARMING_UP,
    InstrumentRegimeResult,
    RegimeFeature,
    RegimeService,
    RegimeSettings,
)
from config import INSTRUMENTS


def candles(symbol, source, close, *, timeframe="1h", start="2025-01-01T00:00:00Z",
            timestamps=None, widths=None):
    close = np.asarray(close, dtype=float)
    ts = (pd.date_range(start, periods=len(close), freq="1h") if timestamps is None
          else pd.DatetimeIndex(timestamps))
    widths = np.full(len(close), 0.2) if widths is None else np.asarray(widths, dtype=float)
    return pd.DataFrame({
        "instrument": symbol, "source": source, "actual_symbol": symbol,
        "timeframe": timeframe, "timestamp": ts,
        "timestamp_trust": "TRUSTED_UTC" if source == "Binance" else "UNVERIFIED_TIME_BASIS",
        "open": close - widths * 0.1, "high": close + widths,
        "low": close - widths, "close": close,
        "volume": 1.0, "completed": True,
    })


class FakeCandleService:
    def __init__(self, data):
        self.data = data
        self.closed = False
        self.calls = []

    def get_candles(self, symbol, timeframe, count, *, include_incomplete=False):
        self.calls.append((symbol, timeframe, count, include_incomplete))
        frame = self.data[symbol]
        if isinstance(frame, Exception):
            raise frame
        return frame.copy()

    def close(self):
        self.closed = True


@pytest.fixture
def settings():
    return RegimeSettings(adx_period=3, ema_period=4, ema_slope_lookback=2,
        momentum_lookback=3, atr_period=3, atr_percentile_lookback=10,
        volatility_percentile_min_history=3, realized_volatility_window=3,
        range_lookback_sessions=2)


def feature(value, status=AVAILABLE):
    return RegimeFeature(status, value, "test", "mocked feature", 100)


def test_trending_classification_requires_directional_confirmation(settings):
    service = RegimeService(candle_service_factory=lambda: None, settings=settings)
    features = {
        "atr_percentile": feature(50), "realized_volatility_percentile": feature(55),
        "session_range_expansion_ratio": feature(None, TIMESTAMP_UNVERIFIED),
        "adx": feature(32), "plus_di": feature(40), "minus_di": feature(10),
        "ema_slope_bps": feature(8), "price_momentum_pct": feature(0.8),
    }
    regime, strength, explanation = service._classify(features)
    assert (regime, strength) == ("TRENDING", "STRONG")
    assert "align" in explanation


def test_ranging_classification_uses_low_directional_strength(settings):
    service = RegimeService(candle_service_factory=lambda: None, settings=settings)
    features = {
        "atr_percentile": feature(50), "realized_volatility_percentile": feature(45),
        "session_range_expansion_ratio": feature(None, TIMESTAMP_UNVERIFIED),
        "adx": feature(14), "plus_di": feature(20), "minus_di": feature(19),
        "ema_slope_bps": feature(0.5), "price_momentum_pct": feature(0.03),
    }
    regime, strength, _ = service._classify(features)
    assert (regime, strength) == ("RANGING", "STRONG")


@pytest.mark.parametrize("percentile, expected", [(95, "HIGH_VOLATILITY"), (5, "LOW_VOLATILITY")])
def test_volatility_regimes_take_precedence_over_trend(settings, percentile, expected):
    service = RegimeService(candle_service_factory=lambda: None, settings=settings)
    features = {
        "atr_percentile": feature(percentile), "realized_volatility_percentile": feature(50),
        "session_range_expansion_ratio": feature(None, TIMESTAMP_UNVERIFIED),
        "adx": feature(40), "plus_di": feature(50), "minus_di": feature(5),
        "ema_slope_bps": feature(20), "price_momentum_pct": feature(1.0),
    }
    regime, _, _ = service._classify(features)
    assert regime == expected


def test_high_volatility_wins_conflicting_high_and_low_evidence(settings):
    service = RegimeService(candle_service_factory=lambda: None, settings=settings)
    features = {
        "atr_percentile": feature(95), "realized_volatility_percentile": feature(5),
        "session_range_expansion_ratio": feature(None, TIMESTAMP_UNVERIFIED),
        "adx": feature(10), "plus_di": feature(10), "minus_di": feature(10),
        "ema_slope_bps": feature(0), "price_momentum_pct": feature(0),
    }
    assert service._classify(features)[0] == "HIGH_VOLATILITY"


def test_insufficient_history_returns_warming_up(settings):
    frame = candles("XAUUSD", "MT5", np.linspace(100, 101, 3))
    fake = FakeCandleService({"XAUUSD": frame})
    result = RegimeService(candle_service_factory=lambda: fake, settings=settings).analyze_instrument(
        "XAUUSD", "1h", 3)
    assert result.status == WARMING_UP
    assert result.current_regime is None
    assert result.features["adx"].status == WARMING_UP
    assert fake.closed


def test_mt5_candle_gaps_exclude_returns_and_session_features(settings):
    timestamps = list(pd.date_range("2025-01-01T00:00:00Z", periods=35, freq="1h"))
    timestamps[20:] = [stamp + pd.Timedelta(hours=8) for stamp in timestamps[20:]]
    frame = candles("EURUSD", "MT5", np.linspace(1.0, 1.1, 35), timestamps=timestamps)
    fake = FakeCandleService({"EURUSD": frame})
    result = RegimeService(candle_service_factory=lambda: fake, settings=settings).analyze_instrument(
        "EURUSD", "1h", len(frame))
    assert result.timestamp_trust == "UNVERIFIED_TIME_BASIS"
    assert result.features["session_range_expansion_ratio"].status == TIMESTAMP_UNVERIFIED
    assert result.features["rolling_realized_volatility"].status == AVAILABLE
    assert any("excluded" in warning for warning in result.data_quality_warnings)


def test_recent_gap_resets_sequence_features_instead_of_bridging(settings):
    timestamps = list(pd.date_range("2025-01-01T00:00:00Z", periods=30, freq="1h"))
    timestamps += list(pd.date_range("2025-01-03T00:00:00Z", periods=2, freq="1h"))
    frame = candles("GBPUSD", "MT5", np.linspace(1.0, 1.1, len(timestamps)), timestamps=timestamps)
    fake = FakeCandleService({"GBPUSD": frame})
    result = RegimeService(candle_service_factory=lambda: fake, settings=settings).analyze_instrument(
        "GBPUSD", "1h", len(frame))
    assert result.features["price_momentum_pct"].status == WARMING_UP
    assert result.features["ema_slope_bps"].status == WARMING_UP
    assert result.features["adx"].status == WARMING_UP


def test_binance_utc_session_feature_is_not_withheld(settings):
    frame = candles("BTCUSDT", "Binance", np.linspace(100, 110, 600))
    fake = FakeCandleService({"BTCUSDT": frame})
    result = RegimeService(candle_service_factory=lambda: fake, settings=settings).analyze_instrument(
        "BTCUSDT", "1h", len(frame))
    assert result.timestamp_trust == "TRUSTED_UTC"
    assert result.features["session_range_expansion_ratio"].status != TIMESTAMP_UNVERIFIED
    assert result.features["rolling_realized_volatility"].status == AVAILABLE


@pytest.mark.parametrize("mutation, reason", [
    ("duplicate", "duplicate"),
    ("out_of_order", "strictly increasing"),
    ("bad_ohlc", "OHLC"),
])
def test_invalid_candles_are_unavailable(settings, mutation, reason):
    frame = candles("XAUUSD", "MT5", np.linspace(100, 105, 40))
    if mutation == "duplicate":
        frame.loc[5, "timestamp"] = frame.loc[4, "timestamp"]
    elif mutation == "out_of_order":
        frame.loc[5, "timestamp"] = frame.loc[3, "timestamp"] + pd.Timedelta(minutes=30)
    else:
        frame.loc[5, "high"] = frame.loc[5, "low"] - 1
    fake = FakeCandleService({"XAUUSD": frame})
    result = RegimeService(candle_service_factory=lambda: fake, settings=settings).analyze_instrument(
        "XAUUSD", "1h", len(frame))
    assert result.status == UNAVAILABLE
    assert reason.lower() in result.data_quality_warnings[0].lower()


def test_all_six_instruments_are_returned_and_source_failure_is_isolated(settings):
    frames = {item.symbol: candles(item.symbol, item.source,
             np.linspace(100, 110, 80)) for item in INSTRUMENTS}
    frames["XAUUSD"] = RuntimeError("mock historical provider failure")
    fake = FakeCandleService(frames)
    result = RegimeService(candle_service_factory=lambda: fake, settings=settings).analyze_all("1h", 80)
    assert tuple(result) == tuple(item.symbol for item in INSTRUMENTS)
    assert all(isinstance(value, InstrumentRegimeResult) for value in result.values())
    assert all(value.source == item.source for value, item in zip(result.values(), INSTRUMENTS))
    assert result["XAUUSD"].status == UNAVAILABLE
    assert result["EURUSD"].status in {AVAILABLE, WARMING_UP}
    assert fake.closed

