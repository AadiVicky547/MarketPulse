"""Pure volatility calculations for standardized historical OHLC candles.

All returned time series are indexed by UTC-aware candle timestamps. Price
ranges and ATR are in price units; volatility of log returns is dimensionless
per candle unless annualized with an explicitly supplied periods-per-year.
Functions copy and sort input chronologically and never mutate the caller's
DataFrame.
"""
from __future__ import annotations

from datetime import datetime, time as day_time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

OHLC_COLUMNS = ("Open", "High", "Low", "Close")


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _periods_per_year(value: float | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError("periods_per_year must be a finite positive number")
    return result


def prepare_ohlc(data: pd.DataFrame) -> pd.DataFrame:
    """Validate OHLC data and return a sorted, UTC-indexed copy.

    Required columns are ``timestamp``, ``Open``, ``High``, ``Low``, and
    ``Close``. Timestamps must include a timezone. Duplicate timestamps,
    missing/nonfinite values, nonpositive prices, and inconsistent candles are
    rejected instead of silently dropped or repaired.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    required = {"timestamp", *OHLC_COLUMNS}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns: {', '.join(sorted(missing))}")
    if data.empty:
        raise ValueError("OHLC data must contain at least one row")
    if "timestamp_trust" in data.columns:
        trust = data["timestamp_trust"]
        untrusted = trust.isna() | trust.ne("TRUSTED_UTC")
        if untrusted.any():
            statuses = sorted(str(value) for value in trust[untrusted].dropna().unique())
            detail = ", ".join(statuses) if statuses else "missing trust status"
            raise ValueError(f"Time-sensitive analytics require trusted UTC candle timestamps; found {detail}")

    frame = data.loc[:, ["timestamp", *OHLC_COLUMNS]].copy(deep=True)
    parsed: list[pd.Timestamp] = []
    for value in frame["timestamp"]:
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid timestamp: {value!r}") from exc
        if stamp.tzinfo is None:
            raise ValueError("All timestamps must be timezone-aware")
        parsed.append(stamp.tz_convert("UTC"))
    frame["timestamp"] = pd.DatetimeIndex(parsed)
    if frame["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps are not allowed")

    for column in OHLC_COLUMNS:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} contains a non-numeric value") from exc
        if not np.isfinite(frame[column].to_numpy()).all():
            raise ValueError(f"{column} contains a missing or nonfinite value")
        if (frame[column] <= 0).any():
            raise ValueError(f"{column} must contain positive prices")

    invalid = ((frame["High"] < frame[["Open", "Close", "Low"]].max(axis=1)) |
               (frame["Low"] > frame[["Open", "Close", "High"]].min(axis=1)))
    if invalid.any():
        raise ValueError("OHLC candle is inconsistent: High/Low do not contain Open and Close")

    frame = frame.sort_values("timestamp", kind="stable").set_index("timestamp")
    frame.index.name = "timestamp"
    return frame


def true_range(data: pd.DataFrame) -> pd.Series:
    """Return true range in price units; first candle uses High minus Low."""
    frame = prepare_ohlc(data)
    previous_close = frame["Close"].shift(1)
    values = pd.concat((frame["High"] - frame["Low"],
                        (frame["High"] - previous_close).abs(),
                        (frame["Low"] - previous_close).abs()), axis=1).max(axis=1)
    values.name = "true_range_price_units"
    return values


def average_true_range(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR with Wilder smoothing and a simple-average seed.

    True range's first candle is High-Low. ATR is unavailable (NaN) until
    ``period`` true-range observations exist. The first ATR is their arithmetic
    mean; later values use ``((period-1)*previous_ATR + current_TR)/period``.
    """
    period = _positive_integer(period, "period")
    ranges = true_range(data)
    output = pd.Series(np.nan, index=ranges.index, name=f"atr_{period}_price_units", dtype=float)
    if len(ranges) < period:
        return output
    values = ranges.to_numpy(dtype=float)
    output.iloc[period - 1] = float(np.mean(values[:period]))
    for position in range(period, len(values)):
        output.iloc[position] = ((period - 1) * output.iloc[position - 1] + values[position]) / period
    return output


def logarithmic_returns(data: pd.DataFrame) -> pd.Series:
    """Return consecutive close log returns; first observation is unavailable."""
    frame = prepare_ohlc(data)
    result = np.log(frame["Close"] / frame["Close"].shift(1))
    result.name = "log_return_per_candle"
    return result


def _annualize(series: pd.Series, periods_per_year: float | None) -> pd.Series:
    if periods_per_year is not None:
        series = series * np.sqrt(periods_per_year)
    return series


def realized_volatility(data: pd.DataFrame, window: int = 20, *,
                        periods_per_year: float | None = None,
                        min_periods: int | None = None) -> pd.Series:
    """Rolling sample standard deviation of log returns.

    By default it is nonannualized and expressed per candle. Annualization is
    ``rolling_std * sqrt(periods_per_year)`` and is performed only when the
    caller explicitly supplies a calendar-appropriate period count.
    """
    window = _positive_integer(window, "window")
    minimum = window if min_periods is None else _positive_integer(min_periods, "min_periods")
    if minimum > window:
        raise ValueError("min_periods cannot exceed window")
    annual = _periods_per_year(periods_per_year)
    result = logarithmic_returns(data).rolling(window, min_periods=minimum).std(ddof=1)
    result = _annualize(result, annual)
    result.name = "realized_volatility_annualized" if annual is not None else "realized_volatility_per_candle"
    return result


def rolling_volatility(data: pd.DataFrame, window: int = 20, *,
                       periods_per_year: float | None = None,
                       min_periods: int | None = None) -> pd.Series:
    """Rolling sample standard deviation of log returns (same units as realized volatility)."""
    return realized_volatility(data, window, periods_per_year=periods_per_year,
                               min_periods=min_periods).rename(
                                   "rolling_volatility_annualized" if periods_per_year is not None
                                   else "rolling_volatility_per_candle")


def ewma_volatility(data: pd.DataFrame, decay: float = 0.94, *,
                    periods_per_year: float | None = None) -> pd.Series:
    """Causal exponentially weighted volatility of log returns.

    Variance follows ``v[t] = decay*v[t-1] + (1-decay)*r[t]**2``. The first
    available return initializes variance as ``r**2``; the reported volatility
    is its square root. Thus each result depends only on returns through that
    timestamp. Optional annualization multiplies by ``sqrt(periods_per_year)``.
    """
    decay = float(decay)
    if not np.isfinite(decay) or not 0 < decay < 1:
        raise ValueError("decay must be strictly between zero and one")
    annual = _periods_per_year(periods_per_year)
    returns = logarithmic_returns(data)
    result = pd.Series(np.nan, index=returns.index, name="ewma_volatility_per_candle", dtype=float)
    squared = returns.dropna().pow(2)
    if not squared.empty:
        variance = squared.ewm(alpha=1.0 - decay, adjust=False).mean()
        result.loc[variance.index] = np.sqrt(variance)
    result = _annualize(result, annual)
    result.name = "ewma_volatility_annualized" if annual is not None else "ewma_volatility_per_candle"
    return result


def volatility_zscore(volatility: pd.Series, baseline_window: int = 50, *,
                      min_history: int | None = None) -> pd.Series:
    """Compare each value with the prior rolling baseline, excluding itself.

    The baseline is the prior ``baseline_window`` values. A zero/undefined
    baseline standard deviation yields NaN rather than infinity or a fabricated
    zero score.
    """
    baseline_window = _positive_integer(baseline_window, "baseline_window")
    minimum = baseline_window if min_history is None else _positive_integer(min_history, "min_history")
    if minimum > baseline_window:
        raise ValueError("min_history cannot exceed baseline_window")
    values = pd.to_numeric(volatility, errors="raise").astype(float)
    if not np.isfinite(values.dropna().to_numpy()).all():
        raise ValueError("volatility contains nonfinite values")
    baseline = values.shift(1).rolling(baseline_window, min_periods=minimum)
    mean = baseline.mean()
    std = baseline.std(ddof=1)
    result = (values - mean) / std
    result = result.where(std > 0)
    result.name = "volatility_zscore_vs_prior_history"
    return result


def volatility_percentile(volatility: pd.Series, lookback: int = 100, *,
                          min_history: int = 20) -> pd.Series:
    """Empirical percentile (0-100) versus prior values only.

    For each observation, the score is 100 times the fraction of available
    values in the preceding lookback that are less than or equal to it. The
    current value is never added to its comparison baseline. Results stay NaN
    until at least ``min_history`` valid prior observations are available.
    """
    lookback = _positive_integer(lookback, "lookback")
    min_history = _positive_integer(min_history, "min_history")
    if min_history > lookback:
        raise ValueError("min_history cannot exceed lookback")
    values = pd.to_numeric(volatility, errors="raise").astype(float)
    if not np.isfinite(values.dropna().to_numpy()).all():
        raise ValueError("volatility contains nonfinite values")
    output = np.full(len(values), np.nan, dtype=float)
    array = values.to_numpy()
    for index, current in enumerate(array):
        start = max(0, index - lookback)
        prior = array[start:index]
        prior = prior[np.isfinite(prior)]
        if np.isfinite(current) and len(prior) >= min_history:
            output[index] = 100.0 * np.count_nonzero(prior <= current) / len(prior)
    result = pd.Series(output, index=values.index, name="volatility_percentile_vs_prior_history")
    return result


def _as_utc(value: datetime | pd.Timestamp, name: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return stamp.tz_convert("UTC")


def _session_start(stamp: pd.Timestamp, zone: ZoneInfo, session_open: day_time) -> pd.Timestamp:
    local = stamp.tz_convert(zone)
    local_open = pd.Timestamp(datetime.combine(local.date(), session_open)).tz_localize(
        zone, ambiguous="raise", nonexistent="shift_forward")
    if local < local_open:
        previous_date = local.date() - timedelta(days=1)
        local_open = pd.Timestamp(datetime.combine(previous_date, session_open)).tz_localize(
            zone, ambiguous="raise", nonexistent="shift_forward")
    return local_open.tz_convert("UTC")


def _session_bounds(stamp: pd.Timestamp, zone: ZoneInfo, session_open: day_time,
                    duration: pd.Timedelta) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Map a timestamp to explicit session start/end bounds."""
    local = stamp.tz_convert(zone)
    local_open = pd.Timestamp(datetime.combine(local.date(), session_open)).tz_localize(
        zone, ambiguous="raise", nonexistent="shift_forward")
    if local < local_open:
        previous_date = local.date() - timedelta(days=1)
        local_open = pd.Timestamp(datetime.combine(previous_date, session_open)).tz_localize(
            zone, ambiguous="raise", nonexistent="shift_forward")
    if duration == pd.Timedelta(days=1):
        # A daily session follows local wall-clock boundaries; this preserves
        # 23/25-hour sessions across daylight-saving transitions.
        next_date = local_open.date() + timedelta(days=1)
        end = pd.Timestamp(datetime.combine(next_date, session_open)).tz_localize(
            zone, ambiguous="raise", nonexistent="shift_forward")
        return local_open.tz_convert("UTC"), end.tz_convert("UTC")
    elapsed = stamp - local_open.tz_convert("UTC")
    offset = int(elapsed // duration)
    start = local_open.tz_convert("UTC") + offset * duration
    return start, start + duration


def average_daily_range(data: pd.DataFrame, lookback: int = 14, *,
                        session_timezone: str = "UTC", session_start: str = "00:00",
                        as_of: datetime | pd.Timestamp | None = None,
                        min_periods: int | None = None) -> pd.Series:
    """Mean High-Low range of prior completed daily sessions, in price units.

    Input must contain one daily candle timestamped at each session opening.
    Session timezone and wall-clock opening are explicit. A candle is complete
    only when the next local session opening is at or before ``as_of`` (UTC now
    when omitted). The ADR shown for a session uses only preceding completed
    sessions; the current/incomplete session is never used in its own baseline.
    """
    lookback = _positive_integer(lookback, "lookback")
    minimum = lookback if min_periods is None else _positive_integer(min_periods, "min_periods")
    if minimum > lookback:
        raise ValueError("min_periods cannot exceed lookback")
    try:
        zone = ZoneInfo(session_timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown session timezone: {session_timezone}") from exc
    try:
        hour, minute = (int(part) for part in session_start.split(":"))
        open_time = day_time(hour, minute)
    except (ValueError, TypeError) as exc:
        raise ValueError("session_start must use HH:MM (24-hour) format") from exc
    frame = prepare_ohlc(data)
    cutoff = _as_utc(as_of, "as_of") if as_of is not None else pd.Timestamp.now(tz="UTC")
    starts = [_session_start(stamp, zone, open_time) for stamp in frame.index]
    if any(stamp != start for stamp, start in zip(frame.index, starts)):
        raise ValueError("Daily candle timestamps must align exactly with configured session openings")
    ranges = (frame["High"] - frame["Low"]).where(
        [(_session_start(stamp, zone, open_time) + pd.DateOffset(days=1)).tz_convert("UTC") <= cutoff
         for stamp in frame.index])
    result = ranges.shift(1).rolling(lookback, min_periods=minimum).mean()
    result.name = f"adr_{lookback}_price_units"
    return result


def intraday_range_expansion(
    data: pd.DataFrame, *, candle_interval: str,
    session_timezone: str, session_start: str, session_duration: str,
    lookback_sessions: int = 20, min_history: int | None = None,
    as_of: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compare cumulative session range with prior complete session ranges.

    Candle timestamps identify candle open times. The interval, timezone,
    session opening, and session duration are required explicitly. Incomplete
    candles after ``as_of`` are excluded. Historical sessions qualify as
    complete only when every expected candle timestamp is present. The baseline
    is the mean of the preceding complete sessions (never the current session).
    Output includes current cumulative range and its ratio to that baseline.
    """
    lookback_sessions = _positive_integer(lookback_sessions, "lookback_sessions")
    minimum = lookback_sessions if min_history is None else _positive_integer(min_history, "min_history")
    if minimum > lookback_sessions:
        raise ValueError("min_history cannot exceed lookback_sessions")
    try:
        interval = pd.Timedelta(candle_interval)
        duration = pd.Timedelta(session_duration)
    except (TypeError, ValueError) as exc:
        raise ValueError("candle_interval and session_duration must be valid durations") from exc
    if interval <= pd.Timedelta(0) or duration <= pd.Timedelta(0) or duration % interval != pd.Timedelta(0):
        raise ValueError("session_duration must be a positive integer multiple of candle_interval")
    try:
        zone = ZoneInfo(session_timezone)
        hour, minute = (int(part) for part in session_start.split(":"))
        open_time = day_time(hour, minute)
    except (ZoneInfoNotFoundError, ValueError, TypeError) as exc:
        raise ValueError("Invalid session_timezone or HH:MM session_start") from exc

    frame = prepare_ohlc(data)
    cutoff = _as_utc(as_of, "as_of") if as_of is not None else pd.Timestamp.now(tz="UTC")
    bounds = [_session_bounds(stamp, zone, open_time, duration) for stamp in frame.index]
    # Durations are explicit, and daily sessions honor local calendar boundaries
    # across daylight-saving changes; shorter custom sessions are elapsed-time based.
    session_starts = pd.DatetimeIndex([start for start, _ in bounds])
    session_ends = pd.DatetimeIndex([end for _, end in bounds])
    completed_candles = frame.index + interval <= cutoff
    frame = frame.loc[completed_candles].copy()
    session_starts = session_starts[completed_candles]
    if frame.empty:
        empty = pd.DataFrame(index=frame.index, columns=["session_start", "session_range_price_units",
                                                          "baseline_range_price_units", "range_expansion_ratio"])
        empty.index.name = "timestamp"
        return empty

    frame["_session_start"] = session_starts
    group = frame.groupby("_session_start", sort=True)
    cumulative_high = group["High"].cummax()
    cumulative_low = group["Low"].cummin()
    cumulative_range = cumulative_high - cumulative_low

    expected_count = int(duration / interval)
    complete_ranges: dict[pd.Timestamp, float] = {}
    for start, session in group:
        first_position = int(np.flatnonzero(session_starts == start)[0])
        end = session_ends[first_position]
        if duration == pd.Timedelta(days=1):
            expected = pd.date_range(start=start, end=end - interval, freq=interval, tz="UTC")
        else:
            expected = pd.date_range(start=start, periods=expected_count, freq=interval, tz="UTC")
        complete = session.index.equals(expected) and end <= cutoff
        if complete:
            complete_ranges[start] = float(session["High"].max() - session["Low"].min())

    range_series = pd.Series(complete_ranges, dtype=float,
                             index=pd.DatetimeIndex(list(complete_ranges), tz="UTC")).sort_index()
    baseline_by_session: dict[pd.Timestamp, float] = {}
    for start in frame["_session_start"].drop_duplicates().sort_values():
        prior_ranges = range_series.loc[range_series.index < start].tail(lookback_sessions)
        if len(prior_ranges) >= minimum:
            baseline_by_session[start] = float(prior_ranges.mean())
    baseline = frame["_session_start"].map(baseline_by_session)
    ratio = cumulative_range / baseline
    result = pd.DataFrame({"session_start": frame["_session_start"],
                           "session_range_price_units": cumulative_range,
                           "baseline_range_price_units": baseline,
                           "range_expansion_ratio": ratio}, index=frame.index)
    result.index.name = "timestamp"
    return result

