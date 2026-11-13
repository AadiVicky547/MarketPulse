"""Deterministic, manually verifiable tests for volatility analytics."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from analytics.volatility import (
    average_daily_range,
    average_true_range,
    ewma_volatility,
    intraday_range_expansion,
    logarithmic_returns,
    prepare_ohlc,
    realized_volatility,
    rolling_volatility,
    true_range,
    volatility_percentile,
    volatility_zscore,
)


def candles(high, low, close, *, opens=None, timestamps=None):
    count = len(close)
    if timestamps is None:
        timestamps = pd.date_range("2024-01-01", periods=count, freq="D", tz="UTC")
    if opens is None:
        opens = close
    return pd.DataFrame({"timestamp": timestamps, "Open": opens, "High": high,
                         "Low": low, "Close": close})


def test_true_range_and_wilder_atr_seed_and_recursion():
    frame = candles([12, 13, 14, 16], [9, 10, 11, 12], [11, 12, 13, 15],
                    opens=[10, 11, 12, 13])
    tr = true_range(frame)
    np.testing.assert_allclose(tr.to_numpy(), [3, 3, 3, 4])
    atr = average_true_range(frame, period=3)
    assert atr.iloc[:2].isna().all()
    assert atr.iloc[2] == pytest.approx(3.0)  # mean of first three TR values
    assert atr.iloc[3] == pytest.approx(10 / 3)  # Wilder's recursive update


def test_log_returns_and_rolling_realized_volatility_are_sample_std():
    frame = candles([101, 111, 101], [99, 109, 98], [100, 110, 99])
    returns = logarithmic_returns(frame)
    expected = np.array([np.log(1.1), np.log(0.9)])
    assert np.isnan(returns.iloc[0])
    np.testing.assert_allclose(returns.iloc[1:].to_numpy(), expected)
    realized = realized_volatility(frame, window=2)
    assert realized.iloc[:2].isna().all()
    assert realized.iloc[2] == pytest.approx(np.std(expected, ddof=1))
    assert rolling_volatility(frame, 2).iloc[2] == pytest.approx(realized.iloc[2])


def test_realized_volatility_annualization_requires_explicit_periods():
    frame = candles([101, 111, 122], [99, 109, 120], [100, 110, 121])
    raw = realized_volatility(frame, window=2).iloc[-1]
    annualized = realized_volatility(frame, window=2, periods_per_year=252).iloc[-1]
    assert annualized == pytest.approx(raw * np.sqrt(252))
    with pytest.raises(ValueError, match="periods_per_year"):
        realized_volatility(frame, periods_per_year=0)


def test_ewma_uses_causal_recursion_and_first_return_initialization():
    frame = candles([101, 111, 122, 135], [99, 109, 120, 132], [100, 110, 121, 133.1])
    value = np.log(1.1)
    result = ewma_volatility(frame, decay=0.5)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(value)
    # Second equal log return: .5*v + .5*r^2 remains r^2.
    assert result.iloc[2] == pytest.approx(value)
    changed_future = frame.copy()
    changed_future.loc[3, "Close"] = 200
    changed_future.loc[3, "High"] = 201
    assert ewma_volatility(changed_future, decay=0.5).iloc[2] == pytest.approx(result.iloc[2])


def test_volatility_zscore_uses_prior_window_only():
    values = pd.Series([1.0, 2.0, 3.0, 4.0], name="vol")
    result = volatility_zscore(values, baseline_window=2)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx((3 - 1.5) / np.std([1, 2], ddof=1))
    assert result.iloc[3] == pytest.approx((4 - 2.5) / np.std([2, 3], ddof=1))
    changed = values.copy()
    changed.iloc[3] = 400
    assert volatility_zscore(changed, 2).iloc[3] != result.iloc[3]
    # Earlier outputs cannot be influenced by a later observation.
    changed.iloc[3] = -900
    pd.testing.assert_series_equal(volatility_zscore(changed, 2).iloc[:3], result.iloc[:3])


def test_zero_standard_deviation_zscore_is_unavailable_not_zero_or_infinity():
    values = pd.Series([2.0, 2.0, 2.0, 3.0])
    result = volatility_zscore(values, baseline_window=3)
    assert result.iloc[3] is np.nan or np.isnan(result.iloc[3])


def test_percentile_uses_prior_values_and_minimum_history():
    values = pd.Series([1.0, 3.0, 2.0, 4.0, 0.0])
    result = volatility_percentile(values, lookback=3, min_history=2)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(50.0)
    assert result.iloc[3] == pytest.approx(100.0)
    # The fourth observation's percentile baseline contains only [1, 3, 2].
    changed = values.copy()
    changed.iloc[4] = 1000
    pd.testing.assert_series_equal(volatility_percentile(changed, 3, min_history=2).iloc[:4],
                                   result.iloc[:4])


def test_adr_uses_prior_completed_sessions_and_ignores_current_incomplete_range():
    dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    frame = candles([11, 12, 13, 15, 200], [9, 8, 7, 5, 100],
                    [10, 10, 10, 10, 150], opens=[10, 10, 10, 10, 150], timestamps=dates)
    adr = average_daily_range(frame, lookback=3, min_periods=1,
                              session_timezone="UTC", session_start="00:00",
                              as_of=datetime(2024, 1, 5, 12, tzinfo=timezone.utc))
    assert np.isnan(adr.iloc[0])
    assert adr.iloc[1] == pytest.approx(2)
    assert adr.iloc[2] == pytest.approx(3)
    assert adr.iloc[3] == pytest.approx(4)
    # Current day is partial, but ADR uses only completed prior sessions (4, 6, 10).
    assert adr.iloc[4] == pytest.approx(20 / 3)


def test_intraday_range_expansion_and_contraction_use_prior_complete_sessions():
    starts = pd.date_range("2024-01-01", periods=12, freq="h", tz="UTC")
    # Three four-hour sessions with completed ranges 4, 8, and current cumulative 2.
    session_ranges = [4] * 4 + [8] * 4 + [2] * 4
    lows = [100.0] * 12
    highs = [100 + value for value in session_ranges]
    closes = [100.0] * 12
    frame = candles(highs, lows, closes, timestamps=starts)
    result = intraday_range_expansion(
        frame, candle_interval="1h", session_timezone="UTC", session_start="00:00",
        session_duration="4h", lookback_sessions=2, min_history=1,
        as_of=datetime(2024, 1, 1, 11, tzinfo=timezone.utc))
    # During session 2, previous completed session baseline is 4 price units.
    assert result.loc[starts[4], "baseline_range_price_units"] == pytest.approx(4)
    assert result.loc[starts[4], "range_expansion_ratio"] == pytest.approx(2)
    # Current third session is partial, has range 2, and compares with mean(4, 8)=6.
    assert result.loc[starts[8], "baseline_range_price_units"] == pytest.approx(6)
    assert result.loc[starts[8], "range_expansion_ratio"] == pytest.approx(2 / 6)
    assert result.loc[starts[8], "session_range_price_units"] == pytest.approx(2)


def test_insufficient_history_returns_nan_and_validates_windows():
    short = candles([2, 3], [1, 2], [1.5, 2.5])
    assert average_true_range(short, 3).isna().all()
    assert realized_volatility(short, 5).isna().all()
    assert volatility_percentile(pd.Series([1.0, 2.0]), 4, min_history=3).isna().all()
    with pytest.raises(ValueError):
        average_true_range(short, 0)
    with pytest.raises(ValueError):
        volatility_percentile(pd.Series([1.0]), lookback=2, min_history=3)


def test_prepare_ohlc_sorts_copy_and_does_not_mutate_input():
    frame = candles([12, 13], [9, 10], [11, 12])
    shuffled = frame.iloc[::-1].reset_index(drop=True)
    original = shuffled.copy(deep=True)
    prepared = prepare_ohlc(shuffled)
    assert prepared.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(shuffled, original)
    true_range(shuffled)
    average_true_range(shuffled, 1)
    pd.testing.assert_frame_equal(shuffled, original)


@pytest.mark.parametrize("change, message", [
    (lambda f: f.drop(columns="High"), "Missing required"),
    (lambda f: f.assign(Close=[10.0, np.nan]), "missing or nonfinite"),
    (lambda f: f.assign(Close=[10.0, 0.0]), "positive prices"),
    (lambda f: f.assign(High=[9.0, 12.0]), "inconsistent"),
])
def test_invalid_ohlc_inputs_are_rejected(change, message):
    frame = candles([11, 13], [9, 10], [10, 12])
    with pytest.raises(ValueError, match=message):
        prepare_ohlc(change(frame))


def test_duplicate_and_timezone_naive_timestamps_are_rejected():
    frame = candles([11, 12], [9, 10], [10, 11])
    duplicate = frame.copy()
    duplicate.loc[1, "timestamp"] = duplicate.loc[0, "timestamp"]
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_ohlc(duplicate)
    naive = frame.copy()
    naive["timestamp"] = pd.date_range("2024-01-01", periods=2, freq="D")
    with pytest.raises(ValueError, match="timezone-aware"):
        prepare_ohlc(naive)


def test_unverified_broker_candle_timestamps_are_rejected_by_time_sensitive_analytics():
    frame = candles([102, 103], [99, 100], [101, 102])
    frame["timestamp_trust"] = "UNVERIFIED_TIME_BASIS"
    with pytest.raises(ValueError, match="require trusted UTC candle timestamps"):
        prepare_ohlc(frame)
    frame["timestamp_trust"] = "TRUSTED_UTC"
    assert len(prepare_ohlc(frame)) == 2


def test_nonpositive_close_rejected_before_log_returns():
    frame = candles([11, 12], [9, 10], [10, 11])
    frame.loc[1, "Close"] = -1
    with pytest.raises(ValueError, match="Close.*positive"):
        logarithmic_returns(frame)


def test_range_expansion_excludes_incomplete_current_candle():
    starts = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    frame = candles([102, 103, 110, 500], [100, 100, 100, 1],
                    [101, 102, 105, 200], timestamps=starts)
    result = intraday_range_expansion(frame, candle_interval="1h", session_timezone="UTC",
                                      session_start="00:00", session_duration="4h",
                                      lookback_sessions=1, min_history=1,
                                      as_of=datetime(2024, 1, 1, 3, 30, tzinfo=timezone.utc))
    assert starts[3] not in result.index

