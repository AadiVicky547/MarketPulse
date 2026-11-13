"""Transparent feature-based market regime classification.

The service is analytical only: regime labels summarize observed history and
are not price predictions or trading recommendations. MT5 event timestamps stay
unverified; their values are used only to validate order/cadence, never for
session attribution. Binance session features use explicit trusted UTC time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from analytics.volatility import (
    average_true_range,
    intraday_range_expansion,
    prepare_ohlc,
)
from config import INSTRUMENTS
from data.candle_service import CandleService

logger = logging.getLogger(__name__)

AVAILABLE = "AVAILABLE"
WARMING_UP = "WARMING_UP"
UNAVAILABLE = "UNAVAILABLE"
TIMESTAMP_UNVERIFIED = "TIMESTAMP_UNVERIFIED"

_TIMEFRAME_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
    "8h": 28800, "12h": 43200, "1d": 86400, "3d": 259200,
    "1w": 604800,
}


@dataclass(frozen=True)
class RegimeSettings:
    """Feature windows and deterministic classification thresholds.

    Default rules: ATR/RV percentile >= 80 or <= 20 takes precedence over
    directional classification. Otherwise ADX >= 25, a >= 3 bp EMA slope,
    and momentum aligned with the dominant DI direction confirm TRENDING;
    remaining classifiable observations are RANGING. EMA is span 20, slope
    spans five bars, and momentum spans twelve bars.
    """

    adx_period: int = 14
    ema_period: int = 20
    ema_slope_lookback: int = 5
    momentum_lookback: int = 12
    atr_period: int = 14
    atr_percentile_lookback: int = 100
    volatility_percentile_min_history: int = 20
    realized_volatility_window: int = 20
    range_lookback_sessions: int = 10
    high_volatility_percentile: float = 80.0
    low_volatility_percentile: float = 20.0
    trending_adx_threshold: float = 25.0
    minimum_ema_slope_bps: float = 3.0
    ranging_momentum_abs_pct: float = 0.2

    def __post_init__(self) -> None:
        positive = (self.adx_period, self.ema_period, self.ema_slope_lookback,
                    self.momentum_lookback, self.atr_period,
                    self.atr_percentile_lookback, self.volatility_percentile_min_history,
                    self.realized_volatility_window, self.range_lookback_sessions)
        if any(isinstance(value, bool) or value <= 0 for value in positive):
            raise ValueError("regime feature windows must be positive integers")
        if self.volatility_percentile_min_history > self.atr_percentile_lookback:
            raise ValueError("percentile minimum history cannot exceed lookback")
        if not 0 <= self.low_volatility_percentile < self.high_volatility_percentile <= 100:
            raise ValueError("volatility percentile thresholds must be ordered within 0-100")


@dataclass(frozen=True)
class RegimeFeature:
    """One feature observation and its calculation provenance."""

    status: str
    value: float | None
    unit: str
    methodology: str
    sample_count: int
    warning: str | None = None


@dataclass(frozen=True)
class InstrumentRegimeResult:
    """Regime classification and evidence for one configured instrument."""

    instrument: str
    source: str
    timeframe: str
    status: str
    current_regime: str | None
    evidence_strength: str | None
    timestamp_trust: str
    last_observation_freshness: str
    last_update_received_at: datetime | None
    features: Mapping[str, RegimeFeature] = field(default_factory=dict)
    explanation: str = ""
    data_quality_warnings: tuple[str, ...] = ()
    candle_count: int = 0


def _feature(status: str, value: Any, unit: str, method: str, samples: int,
             warning: str | None = None) -> RegimeFeature:
    if value is None:
        return RegimeFeature(status, None, unit, method, samples, warning)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return RegimeFeature(UNAVAILABLE, None, unit, method, samples, "Feature value is not numeric")
    if not math.isfinite(numeric):
        return RegimeFeature(UNAVAILABLE, None, unit, method, samples, "Feature value is not finite")
    return RegimeFeature(status, numeric, unit, method, samples, warning)


class RegimeService:
    """Retrieve candles and produce deterministic per-instrument regimes.

    Candle retrieval uses the existing CandleService. Volatility features reuse
    the established ATR and UTC session-range primitives from analytics.volatility;
    cadence-filtered rolling returns use the same sample-standard-deviation
    convention. This avoids a second provider request through the retrieval-
    coupled VolatilityService. An optional running market manager is read only
    for receive-time freshness; this service never starts it.
    """

    def __init__(self, *, candle_service_factory: Callable[[], Any] = CandleService,
                 market_manager: Any | None = None,
                 settings: RegimeSettings | None = None) -> None:
        self._candle_service_factory = candle_service_factory
        self.market_manager = market_manager
        self.settings = settings or RegimeSettings()
        self._sources = {item.symbol.upper(): item.source for item in INSTRUMENTS}

    def analyze_all(self, timeframe: str = "1h", count: int = 600) -> Mapping[str, InstrumentRegimeResult]:
        """Return independent results for all six configured instruments."""
        self._validate_request(timeframe, count)
        service = self._candle_service_factory()
        freshness = self._freshness_snapshot()
        results: dict[str, InstrumentRegimeResult] = {}
        try:
            for configured in INSTRUMENTS:
                symbol = configured.symbol.upper()
                try:
                    results[symbol] = self._analyze_one(
                        service, symbol, configured.source, timeframe, count, freshness)
                except Exception as exc:
                    logger.exception("Regime analysis failed for %s", symbol)
                    trust = "TRUSTED_UTC" if configured.source == "Binance" else "UNVERIFIED_TIME_BASIS"
                    results[symbol] = self._failure(symbol, configured.source, timeframe, count,
                        f"Regime analysis failed: {type(exc).__name__}: {exc}", trust,
                        freshness.get(symbol, ("UNKNOWN", None)))
        finally:
            close = getattr(service, "close", None)
            if callable(close):
                close()
        return MappingProxyType(results)

    def analyze_instrument(self, instrument: str, timeframe: str = "1h",
                           count: int = 600) -> InstrumentRegimeResult:
        """Analyze one configured instrument."""
        self._validate_request(timeframe, count)
        symbol = instrument.upper()
        if symbol not in self._sources:
            raise ValueError(f"Unknown instrument: {instrument!r}")
        service = self._candle_service_factory()
        freshness = self._freshness_snapshot()
        try:
            return self._analyze_one(service, symbol, self._sources[symbol], timeframe, count, freshness)
        finally:
            close = getattr(service, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _validate_request(timeframe: str, count: int) -> None:
        if timeframe not in _TIMEFRAME_SECONDS:
            raise ValueError(f"Unsupported timeframe: {timeframe!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")

    def _freshness_snapshot(self) -> dict[str, tuple[str, datetime | None]]:
        if self.market_manager is None:
            return {}
        try:
            snapshot = self.market_manager.get_market_snapshot()
            return {
                symbol: (str(getattr(state, "freshness", "UNKNOWN")),
                         getattr(state, "last_update_received_at", None))
                for symbol, state in snapshot.instruments.items()
            }
        except Exception as exc:
            logger.warning("Could not read market-manager freshness: %s", exc)
            return {}

    def _analyze_one(self, service: Any, symbol: str, source: str, timeframe: str,
                     count: int, freshness: Mapping[str, tuple[str, datetime | None]]) -> InstrumentRegimeResult:
        trust = "TRUSTED_UTC" if source == "Binance" else "UNVERIFIED_TIME_BASIS"
        current_freshness = freshness.get(symbol)
        latest_freshness, received_at = current_freshness or ("UNKNOWN", None)
        try:
            candles = service.get_candles(symbol, timeframe, count, include_incomplete=False)
        except Exception as exc:
            return self._failure(symbol, source, timeframe, count,
                f"Candle retrieval failed: {type(exc).__name__}: {exc}", trust,
                (latest_freshness, received_at))
        if not isinstance(candles, pd.DataFrame) or candles.empty:
            return self._failure(symbol, source, timeframe, count, "No historical candles returned", trust,
                                 (latest_freshness, received_at))
        warnings: list[str] = []
        try:
            clean, timestamps, gap_count, excluded_returns = self._validate_candles(
                candles, symbol, source, timeframe, trust)
        except (TypeError, ValueError) as exc:
            return self._failure(symbol, source, timeframe, count,
                f"Candle validation failed: {exc}", trust, (latest_freshness, received_at), len(candles))
        if current_freshness is None:
            latest_freshness, received_at = self._candle_freshness(source, timestamps[-1], timeframe)
        elif source == "Binance":
            candle_freshness, _ = self._candle_freshness(source, timestamps[-1], timeframe)
            if candle_freshness == "SUSPICIOUS_FUTURE_TIMESTAMP":
                latest_freshness = candle_freshness
        if excluded_returns:
            warnings.append(
                f"{gap_count} candle interval(s) exceed {timeframe}; excluded {excluded_returns} "
                "nonmatching adjacent return(s) from timeframe-specific returns. Gaps were not filled."
            )
        if source == "MT5":
            warnings.append("MT5 candle event-time basis is UNVERIFIED_TIME_BASIS; no timezone correction applied.")
            if current_freshness is None:
                warnings.append("MT5 receive-time freshness is unavailable because no MarketDataManager was supplied.")
        if latest_freshness == "SUSPICIOUS_FUTURE_TIMESTAMP":
            warnings.append("Latest trusted Binance candle timestamp is materially in the future; freshness is suspicious.")

        features = self._calculate_features(clean, timestamps, timeframe, source, warnings)
        regime, strength, explanation = self._classify(features)
        status = AVAILABLE if regime is not None else WARMING_UP
        return InstrumentRegimeResult(symbol, source, timeframe, status, regime, strength, trust,
            latest_freshness, received_at, MappingProxyType(features), explanation,
            tuple(warnings), len(candles))

    @staticmethod
    def _validate_candles(frame: pd.DataFrame, symbol: str, source: str, timeframe: str,
                          expected_trust: str) -> tuple[pd.DataFrame, pd.DatetimeIndex, int, int]:
        required = {"instrument", "source", "timeframe", "timestamp", "timestamp_trust",
                    "open", "high", "low", "close"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
        if not frame["instrument"].astype(str).str.upper().eq(symbol).all():
            raise ValueError("instrument identity does not match request")
        if not frame["source"].astype(str).str.casefold().eq(source.casefold()).all():
            raise ValueError("source does not match configured instrument")
        if not frame["timeframe"].astype(str).eq(timeframe).all():
            raise ValueError("timeframe does not match request")
        if not frame["timestamp_trust"].astype(str).eq(expected_trust).all():
            raise ValueError(f"timestamp trust must be {expected_trust}")
        timestamps = pd.DatetimeIndex([pd.Timestamp(value) for value in frame["timestamp"]])
        if any(value.tzinfo is None for value in timestamps):
            raise ValueError("timestamps must be timezone-aware")
        if timestamps.has_duplicates:
            raise ValueError("duplicate candle timestamps")
        if not timestamps.is_monotonic_increasing:
            raise ValueError("candle timestamps are not strictly increasing")
        differences = timestamps[1:] - timestamps[:-1]
        expected = pd.Timedelta(seconds=_TIMEFRAME_SECONDS[timeframe])
        excluded = int(np.count_nonzero(differences != expected))
        gaps = int(np.count_nonzero(differences > expected))
        ohlc = frame.loc[:, ["timestamp", "open", "high", "low", "close"]].rename(
            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
        clean = prepare_ohlc(ohlc)
        return clean, timestamps, gaps, excluded

    def _calculate_features(self, clean: pd.DataFrame, timestamps: pd.DatetimeIndex,
                            timeframe: str, source: str,
                            warnings: list[str]) -> dict[str, RegimeFeature]:
        settings = self.settings
        interval = pd.Timedelta(seconds=_TIMEFRAME_SECONDS[timeframe])
        highs, lows, closes = clean["High"], clean["Low"], clean["Close"]
        # ATR and its percentile reuse the established Wilder implementation.
        atr_series = average_true_range(clean.reset_index(), settings.atr_period)
        atr = self._last_feature(atr_series, "price units",
            f"Wilder ATR with simple-average seed; period={settings.atr_period}", settings.atr_period)
        atr_percentile_series = self._percentile_series(
            atr_series, settings.atr_percentile_lookback,
            settings.volatility_percentile_min_history)
        atr_pct = self._last_feature(atr_percentile_series, "percentile (0-100)",
            f"Current ATR mid-rank against prior ATR observations (ties split evenly); "
            f"lookback={settings.atr_percentile_lookback}",
            settings.volatility_percentile_min_history + 1)

        # Only exact-interval adjacent close pairs enter timeframe return metrics.
        differences = timestamps[1:] - timestamps[:-1]
        valid_pair = np.asarray(differences == interval, dtype=bool)
        close_values = closes.to_numpy(dtype=float)
        return_values = np.log(close_values[1:] / close_values[:-1])
        valid_returns = pd.Series(return_values[valid_pair], dtype=float)
        rv_series = valid_returns.rolling(settings.realized_volatility_window).std(ddof=1)
        rv = self._last_feature(rv_series, "return per candle",
            f"Sample standard deviation of exact-{timeframe} log returns; "
            f"window={settings.realized_volatility_window}", settings.realized_volatility_window)
        rv_pct_series = self._percentile_series(rv_series, settings.atr_percentile_lookback,
            settings.volatility_percentile_min_history)
        rv_pct = self._last_feature(rv_pct_series, "percentile (0-100)",
            "Current rolling realized volatility rank against prior values", 
            settings.volatility_percentile_min_history + 1)
        if int(np.count_nonzero(~valid_pair)):
            warnings.append(f"Excluded {int(np.count_nonzero(~valid_pair))} non-{timeframe} return(s); "
                            "rolling realized volatility uses only exact-interval returns.")

        continuous = self._latest_contiguous_run(clean, timestamps, interval)
        adx, plus_di, minus_di = self._adx(continuous, settings.adx_period)
        features: dict[str, RegimeFeature] = {
            "atr": atr,
            "atr_percentile": atr_pct,
            "rolling_realized_volatility": rv,
            "realized_volatility_percentile": rv_pct,
            "adx": _feature(AVAILABLE if adx is not None else WARMING_UP, adx, "index (0-100)",
                f"Wilder ADX; period={settings.adx_period}; calculated on latest gap-free run",
                len(continuous), None if adx is not None else "Insufficient consecutive candles for ADX"),
            "plus_di": _feature(AVAILABLE if plus_di is not None else WARMING_UP, plus_di,
                "index (0-100)", "Wilder +DI from positive directional movement", len(continuous),
                None if plus_di is not None else "Insufficient consecutive candles for +DI"),
            "minus_di": _feature(AVAILABLE if minus_di is not None else WARMING_UP, minus_di,
                "index (0-100)", "Wilder -DI from negative directional movement", len(continuous),
                None if minus_di is not None else "Insufficient consecutive candles for -DI"),
        }

        ema_slope = momentum = None
        if len(continuous) >= settings.ema_period + settings.ema_slope_lookback:
            ema = continuous["Close"].ewm(span=settings.ema_period, adjust=False).mean()
            now_ema = float(ema.iloc[-1])
            prior_ema = float(ema.iloc[-1 - settings.ema_slope_lookback])
            ema_slope = (now_ema / prior_ema - 1.0) * 10000.0
        features["ema_slope_bps"] = _feature(AVAILABLE if ema_slope is not None else WARMING_UP,
            ema_slope, "basis points", f"EMA({settings.ema_period}) change over "
            f"{settings.ema_slope_lookback} consecutive bars", len(continuous),
            None if ema_slope is not None else "Insufficient consecutive candles for EMA slope")
        if len(continuous) > settings.momentum_lookback:
            momentum = (float(continuous["Close"].iloc[-1]) /
                        float(continuous["Close"].iloc[-1 - settings.momentum_lookback]) - 1.0) * 100.0
        features["price_momentum_pct"] = _feature(AVAILABLE if momentum is not None else WARMING_UP,
            momentum, "percent", f"Close-to-close percentage momentum over "
            f"{settings.momentum_lookback} consecutive bars", len(continuous),
            None if momentum is not None else "Insufficient consecutive candles for momentum")

        if source != "Binance":
            features["session_range_expansion_ratio"] = _feature(TIMESTAMP_UNVERIFIED, None, "ratio",
                "Disabled because MT5 event-time basis is unverified", 0,
                "No session feature is calculated from MT5 timestamps")
        else:
            try:
                expansion = intraday_range_expansion(
                    clean.reset_index(), candle_interval=timeframe, session_timezone="UTC",
                    session_start="00:00", session_duration="24h",
                    lookback_sessions=settings.range_lookback_sessions)
                ratio_series = expansion["range_expansion_ratio"]
                ratio = float(ratio_series.dropna().iloc[-1]) if ratio_series.notna().any() else None
                features["session_range_expansion_ratio"] = _feature(
                    AVAILABLE if ratio is not None else WARMING_UP, ratio, "ratio",
                    f"Current cumulative UTC session range / prior complete-session mean; "
                    f"lookback={settings.range_lookback_sessions}",
                    int(ratio_series.notna().sum()), None if ratio is not None else
                    "Insufficient complete UTC sessions")
            except (TypeError, ValueError) as exc:
                features["session_range_expansion_ratio"] = _feature(UNAVAILABLE, None, "ratio",
                    "UTC session range expansion", 0, str(exc))
                warnings.append(f"Binance session range feature unavailable: {exc}")
        return features

    @staticmethod
    def _percentile_series(values: pd.Series, lookback: int,
                           minimum_history: int) -> pd.Series:
        """Causal empirical mid-rank; equal observations share their rank."""
        array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        result = np.full(array.size, np.nan, dtype=float)
        for index, current in enumerate(array):
            if not np.isfinite(current):
                continue
            prior = array[max(0, index - lookback):index]
            prior = prior[np.isfinite(prior)]
            if len(prior) < minimum_history:
                continue
            less = np.count_nonzero(prior < current)
            equal = np.count_nonzero(prior == current)
            result[index] = 100.0 * (less + 0.5 * equal) / len(prior)
        return pd.Series(result, index=values.index, name="historical_midrank_percentile")

    @staticmethod
    def _latest_contiguous_run(clean: pd.DataFrame, timestamps: pd.DatetimeIndex,
                               interval: pd.Timedelta) -> pd.DataFrame:
        if len(timestamps) <= 1:
            return clean.iloc[-1:]
        differences = timestamps[1:] - timestamps[:-1]
        breaks = np.flatnonzero(differences != interval)
        start = int(breaks[-1] + 1) if len(breaks) else 0
        return clean.iloc[start:]

    @staticmethod
    def _adx(frame: pd.DataFrame, period: int) -> tuple[float | None, float | None, float | None]:
        """Wilder ADX with mean seeds; requires a continuous OHLC run."""
        if len(frame) < 2 * period:
            return None, None, None
        high, low, close = frame["High"].to_numpy(), frame["Low"].to_numpy(), frame["Close"].to_numpy()
        tr = np.maximum.reduce((high[1:] - low[1:], np.abs(high[1:] - close[:-1]),
                                np.abs(low[1:] - close[:-1])))
        up, down = np.diff(high), -np.diff(low)
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)

        def wilder(values: np.ndarray) -> np.ndarray:
            out = np.full(values.size, np.nan, dtype=float)
            if values.size < period:
                return out
            out[period - 1] = float(np.mean(values[:period]))
            for i in range(period, values.size):
                out[i] = ((period - 1) * out[i - 1] + values[i]) / period
            return out

        atr = wilder(tr)
        plus = wilder(plus_dm)
        minus = wilder(minus_dm)
        plus_di = np.divide(100.0 * plus, atr, out=np.full_like(plus, np.nan), where=atr > 0)
        minus_di = np.divide(100.0 * minus, atr, out=np.full_like(minus, np.nan), where=atr > 0)
        denominator = plus_di + minus_di
        dx = np.divide(100.0 * np.abs(plus_di - minus_di), denominator,
                       out=np.full_like(denominator, np.nan), where=denominator > 0)
        valid = np.flatnonzero(np.isfinite(dx))
        if len(valid) < period:
            return None, (float(plus_di[-1]) if np.isfinite(plus_di[-1]) else None), \
                   (float(minus_di[-1]) if np.isfinite(minus_di[-1]) else None)
        seed_positions = valid[:period]
        adx_values = np.full(dx.size, np.nan, dtype=float)
        seed = int(seed_positions[-1])
        adx_values[seed] = float(np.mean(dx[seed_positions]))
        for i in range(seed + 1, dx.size):
            if np.isfinite(dx[i]):
                adx_values[i] = ((period - 1) * adx_values[i - 1] + dx[i]) / period
        last_adx = float(adx_values[-1]) if np.isfinite(adx_values[-1]) else None
        last_plus = float(plus_di[-1]) if np.isfinite(plus_di[-1]) else None
        last_minus = float(minus_di[-1]) if np.isfinite(minus_di[-1]) else None
        return last_adx, last_plus, last_minus

    def _classify(self, features: Mapping[str, RegimeFeature]) -> tuple[str | None, str | None, str]:
        s = self.settings
        atr_rank = features["atr_percentile"].value
        rv_rank = features["realized_volatility_percentile"].value
        expansion = features["session_range_expansion_ratio"].value
        ranks = [value for value in (atr_rank, rv_rank) if value is not None]
        high_flags = [value >= s.high_volatility_percentile for value in ranks]
        low_flags = [value <= s.low_volatility_percentile for value in ranks]
        if expansion is not None:
            high_flags.append(expansion >= 1.5)
            low_flags.append(expansion <= 0.67)
        # Volatility regimes take precedence over direction; HIGH wins a direct
        # high/low feature conflict to keep the policy deterministic.
        if any(high_flags):
            confirmations = sum(high_flags)
            return "HIGH_VOLATILITY", "STRONG" if confirmations >= 2 else "MODERATE", \
                f"Volatility expansion takes precedence: {confirmations} high-volatility feature(s) crossed threshold."
        if any(low_flags):
            confirmations = sum(low_flags)
            return "LOW_VOLATILITY", "STRONG" if confirmations >= 2 else "MODERATE", \
                f"Low-volatility thresholds crossed by {confirmations} feature(s); volatility takes precedence over direction."

        adx = features["adx"].value
        plus_di, minus_di = features["plus_di"].value, features["minus_di"].value
        slope, momentum = features["ema_slope_bps"].value, features["price_momentum_pct"].value
        if adx is None:
            return None, None, "Insufficient valid feature history to classify a regime."
        if adx is not None and plus_di is not None and minus_di is not None and slope is not None and momentum is not None:
            direction = 1 if plus_di > minus_di else -1 if minus_di > plus_di else 0
            aligned = (direction != 0 and slope * direction > 0 and momentum * direction > 0
                       and abs(slope) >= s.minimum_ema_slope_bps)
            if adx >= s.trending_adx_threshold and aligned:
                return "TRENDING", "STRONG", (
                    f"ADX {adx:.2f} >= {s.trending_adx_threshold:g}; EMA slope and momentum align "
                    "with dominant directional movement.")
            range_evidence = int(adx < s.trending_adx_threshold)
            range_evidence += int(abs(slope) < s.minimum_ema_slope_bps)
            range_evidence += int(abs(momentum) < s.ranging_momentum_abs_pct)
            strength = "STRONG" if range_evidence == 3 else "MODERATE" if range_evidence == 2 else "WEAK"
            return "RANGING", strength, (
                f"No fully confirmed trend: ADX={adx:.2f}, EMA slope={slope:.3f} bp, "
                f"momentum={momentum:.4f}%; directional confirmation was incomplete.")
        return "RANGING", "WEAK", "Direction features are incomplete; no trend is confirmed."

    @staticmethod
    def _last_feature(series: pd.Series, unit: str, method: str,
                      minimum_samples: int) -> RegimeFeature:
        valid = series.dropna()
        value = float(series.iloc[-1]) if len(series) and pd.notna(series.iloc[-1]) else None
        status = AVAILABLE if value is not None else WARMING_UP if len(valid) < minimum_samples else UNAVAILABLE
        warning = None if value is not None else (
            f"Insufficient history: {len(valid)} valid sample(s), need {minimum_samples}"
            if status == WARMING_UP else "Feature undefined for available history")
        return _feature(status, value, unit, method, min(len(valid), minimum_samples), warning)

    def _candle_freshness(self, source: str, last_timestamp: Any,
                          timeframe: str) -> tuple[str, datetime | None]:
        if source == "MT5":
            return "UNAVAILABLE", None
        if last_timestamp is None:
            return "UNAVAILABLE", None
        age = (datetime.now(timezone.utc) - pd.Timestamp(last_timestamp).to_pydatetime()).total_seconds()
        if age < -60:
            return "SUSPICIOUS_FUTURE_TIMESTAMP", None
        return ("LIVE" if age <= 2 * _TIMEFRAME_SECONDS[timeframe] else "STALE", None)

    def _failure(self, symbol: str, source: str, timeframe: str, count: int, warning: str,
                 trust: str, freshness: tuple[str, datetime | None], candle_count: int = 0
                 ) -> InstrumentRegimeResult:
        return InstrumentRegimeResult(symbol, source, timeframe, UNAVAILABLE, None, None, trust,
            freshness[0], freshness[1], MappingProxyType({}), "Regime unavailable.",
            (warning,), candle_count)
