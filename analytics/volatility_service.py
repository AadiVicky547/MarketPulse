"""Safe integration of historical candles with the volatility calculations.

MT5 event timestamps remain unverified. ATR may use validated ordered candles;
return metrics use only adjacent candles at the requested exact interval.
Session/day metrics are withheld for MT5.
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
    average_daily_range,
    average_true_range,
    ewma_volatility,
    intraday_range_expansion,
    prepare_ohlc,
    realized_volatility,
    rolling_volatility,
    volatility_percentile,
    volatility_zscore,
)
from config import INSTRUMENTS
from data.candle_service import CandleService

logger = logging.getLogger(__name__)

STATUS_AVAILABLE = "AVAILABLE"
STATUS_WARMING_UP = "WARMING_UP"
STATUS_UNAVAILABLE = "UNAVAILABLE"
STATUS_TIMESTAMP_UNVERIFIED = "TIMESTAMP_UNVERIFIED"

_TIMEFRAME_SECONDS = {
    "1s": 1, "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
    "6h": 21600, "8h": 28800, "12h": 43200, "1d": 86400,
    "3d": 259200, "1w": 604800,
}


@dataclass(frozen=True)
class VolatilitySettings:
    """Calculation windows; periods-per-year remains unset unless explicitly given."""

    atr_period: int = 14
    realized_window: int = 20
    ewma_decay: float = 0.94
    rolling_window: int = 20
    zscore_baseline_window: int = 50
    percentile_lookback: int = 100
    percentile_min_history: int = 20
    adr_lookback: int = 14
    range_lookback_sessions: int = 20
    periods_per_year: float | None = None


@dataclass(frozen=True)
class VolatilityMetric:
    """One metric value and its availability/provenance."""

    status: str
    value: float | None
    unit: str
    methodology: str
    sample_count: int
    warning: str | None = None


@dataclass(frozen=True)
class InstrumentVolatilityResult:
    """Volatility result for one configured instrument and timeframe."""

    instrument: str
    source: str
    timeframe: str
    requested_candle_count: int
    candle_count: int
    timestamp_trust: str
    status: str
    metrics: Mapping[str, VolatilityMetric] = field(default_factory=dict)
    data_quality_warnings: tuple[str, ...] = ()
    gap_count: int = 0
    excluded_return_count: int = 0


class VolatilityService:
    """Fetch candles and produce timestamp-aware volatility results.

    The injected candle service factory makes this class testable without live
    MT5 or Binance access. Binance-only absolute-time metrics use explicit UTC
    day sessions because those candle timestamps have trusted UTC semantics.
    """

    def __init__(self, *, candle_service_factory: Callable[[], Any] = CandleService,
                 settings: VolatilitySettings | None = None) -> None:
        self._candle_service_factory = candle_service_factory
        self.settings = settings or VolatilitySettings()
        self._sources = {item.symbol.upper(): item.source for item in INSTRUMENTS}

    def analyze_instrument(self, instrument: str, timeframe: str = "1h",
                           count: int = 200) -> InstrumentVolatilityResult:
        """Analyze one configured instrument, reporting provider and quality failures."""
        symbol = instrument.upper()
        if symbol not in self._sources:
            raise ValueError(f"Unknown instrument: {instrument!r}")
        if timeframe not in _TIMEFRAME_SECONDS:
            raise ValueError(f"Unsupported timeframe: {timeframe!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        service = self._candle_service_factory()
        try:
            return self._analyze_with_service(service, symbol, timeframe, count)
        finally:
            close = getattr(service, "close", None)
            if callable(close):
                close()

    def analyze_all(self, timeframe: str = "1h", count: int = 200) -> Mapping[str, InstrumentVolatilityResult]:
        """Return an independent result for each configured instrument."""
        if timeframe not in _TIMEFRAME_SECONDS:
            raise ValueError(f"Unsupported timeframe: {timeframe!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        service = self._candle_service_factory()
        results: dict[str, InstrumentVolatilityResult] = {}
        try:
            for item in INSTRUMENTS:
                symbol = item.symbol.upper()
                try:
                    results[symbol] = self._analyze_with_service(service, symbol, timeframe, count)
                except Exception as exc:
                    logger.exception("Volatility integration failed for %s", symbol)
                    results[symbol] = self._failed_result(symbol, item.source, timeframe, count,
                                                          f"{type(exc).__name__}: {exc}")
        finally:
            close = getattr(service, "close", None)
            if callable(close):
                close()
        return MappingProxyType(results)

    def _analyze_with_service(self, service: Any, symbol: str, timeframe: str,
                              count: int) -> InstrumentVolatilityResult:
        source = self._sources[symbol]
        warnings: list[str] = []
        try:
            candles = service.get_candles(symbol, timeframe, count, include_incomplete=False)
        except Exception as exc:
            return self._failed_result(symbol, source, timeframe, count,
                                       f"Candle retrieval failed: {type(exc).__name__}: {exc}")
        if not isinstance(candles, pd.DataFrame) or candles.empty:
            return self._failed_result(symbol, source, timeframe, count, "No historical candles returned")

        expected_trust = "TRUSTED_UTC" if source == "Binance" else "UNVERIFIED_TIME_BASIS"
        try:
            self._validate_candle_frame(candles, symbol, source, timeframe, expected_trust)
        except (TypeError, ValueError) as exc:
            return self._failed_result(symbol, source, timeframe, count,
                                       f"Candle validation failed: {exc}", len(candles), expected_trust)

        gap_count = 0
        excluded_return_count = 0
        if source == "MT5":
            warnings.append("MT5 event timestamp basis is unverified; metrics are sequence-based only.")
            expected_delta = pd.Timedelta(seconds=_TIMEFRAME_SECONDS[timeframe])
            valid_returns, gap_count, excluded_return_count, interval_warning = self._mt5_return_intervals(
                candles, expected_delta)
            if interval_warning:
                warnings.append(interval_warning)
            metrics = self._sequence_metrics(candles, valid_return_mask=valid_returns)
        else:
            metrics = self._sequence_metrics(candles)
        if source == "MT5":
            metrics["average_daily_range"] = self._withheld_metric(
                "MT5 timestamp basis is unverified; daily-session attribution is disabled.", "price units")
            metrics["intraday_range_expansion"] = self._withheld_metric(
                "MT5 timestamp basis is unverified; session boundaries cannot be trusted.", "ratio")
        else:
            self._add_binance_time_metrics(service, symbol, timeframe, metrics, warnings)

        usable_sequence = any(
            metrics[name].status == STATUS_AVAILABLE
            for name in ("atr", "realized_volatility", "ewma_volatility", "rolling_volatility")
        )
        data_failure = any(metric.status == STATUS_UNAVAILABLE for metric in metrics.values())
        status = STATUS_AVAILABLE if usable_sequence else (
            STATUS_UNAVAILABLE if data_failure else STATUS_WARMING_UP)
        return InstrumentVolatilityResult(
            instrument=symbol, source=source, timeframe=timeframe,
            requested_candle_count=count, candle_count=len(candles),
            timestamp_trust=expected_trust, status=status,
            metrics=MappingProxyType(metrics), data_quality_warnings=tuple(warnings),
            gap_count=gap_count, excluded_return_count=excluded_return_count,
        )

    @staticmethod
    def _validate_candle_frame(frame: pd.DataFrame, symbol: str, source: str,
                               timeframe: str, expected_trust: str) -> pd.DataFrame:
        required = {"instrument", "source", "timeframe", "timestamp", "open", "high", "low", "close",
                    "timestamp_trust"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
        if not frame["instrument"].astype(str).str.upper().eq(symbol).all():
            raise ValueError("instrument identity does not match request")
        if not frame["source"].astype(str).str.casefold().eq(source.casefold()).all():
            raise ValueError("data source does not match configured instrument")
        if not frame["timeframe"].astype(str).eq(timeframe).all():
            raise ValueError("candle timeframe does not match request")
        if not frame["timestamp_trust"].astype(str).eq(expected_trust).all():
            raise ValueError(f"timestamp trust must be {expected_trust}")
        timestamps: list[pd.Timestamp] = []
        for value in frame["timestamp"]:
            stamp = pd.Timestamp(value)
            if stamp.tzinfo is None:
                raise ValueError("timestamps must be timezone-aware")
            timestamps.append(stamp)
        index = pd.DatetimeIndex(timestamps)
        if index.has_duplicates:
            raise ValueError("duplicate candle timestamps")
        if not index.is_monotonic_increasing:
            raise ValueError("candle timestamps are not strictly chronological")
        expected_delta = pd.Timedelta(seconds=_TIMEFRAME_SECONDS[timeframe])
        differences = index[1:] - index[:-1]
        if source == "Binance" and not bool((differences == expected_delta).all()):
            bad = sorted({str(delta) for delta in differences if delta != expected_delta})
            raise ValueError(f"candle interval mismatch; expected {expected_delta}, observed {', '.join(bad)}")
        # prepare_ohlc validates OHLC and missing values. Pass only these fields
        # after explicitly checking the trust policy above; no metric formulas
        # or timestamp corrections are performed here.
        return prepare_ohlc(frame.loc[:, ["timestamp", "open", "high", "low", "close"]].rename(
            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}))

    @staticmethod
    def _mt5_return_intervals(candles: pd.DataFrame, expected_delta: pd.Timedelta
                              ) -> tuple[np.ndarray, int, int, str | None]:
        """Mark only adjacent MT5 candles with the expected interval as returns."""
        timestamps = pd.DatetimeIndex([pd.Timestamp(value) for value in candles["timestamp"]])
        differences = timestamps[1:] - timestamps[:-1]
        valid = np.zeros(len(candles), dtype=bool)
        if len(differences):
            valid[1:] = np.asarray(differences == expected_delta, dtype=bool)
        gap_count = int(np.count_nonzero(differences > expected_delta))
        excluded_count = int(np.count_nonzero(differences != expected_delta))
        if not excluded_count:
            return valid, 0, 0, None
        frequencies = pd.Series(differences[differences != expected_delta]).value_counts(sort=False)
        detail = ", ".join(f"{interval}: {int(count)}" for interval, count in frequencies.items())
        warning = (
            f"MT5 cadence gaps: {gap_count} interval(s) longer than expected {expected_delta}; "
            f"{excluded_count} adjacent return(s) excluded from return-based metrics. "
            f"Nonmatching intervals: {detail}. Timestamp basis remains unverified."
        )
        return valid, gap_count, excluded_count, warning

    def _sequence_metrics(self, candles: pd.DataFrame,
                          valid_return_mask: np.ndarray | None = None) -> dict[str, VolatilityMetric]:
        source = candles.loc[:, ["timestamp", "open", "high", "low", "close"]].rename(
            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
        clean = prepare_ohlc(source)
        indexed = clean.reset_index()
        n = len(indexed)
        settings = self.settings
        results: dict[str, VolatilityMetric] = {}

        atr = average_true_range(indexed, settings.atr_period)
        results["atr"] = self._last_metric(
            atr, n, "price units", f"True range with Wilder smoothing; period={settings.atr_period}",
            settings.atr_period)

        if valid_return_mask is None:
            return_metrics = self._existing_return_metrics(indexed, n, settings)
        else:
            return_metrics = self._mt5_return_metrics(indexed, valid_return_mask, settings)
        results["realized_volatility"] = return_metrics["realized_volatility"]
        results["ewma_volatility"] = return_metrics["ewma_volatility"]
        results["rolling_volatility"] = return_metrics["rolling_volatility"]
        results["volatility_zscore"] = return_metrics["volatility_zscore"]
        results["volatility_percentile"] = return_metrics["volatility_percentile"]
        return results

    def _existing_return_metrics(self, indexed: pd.DataFrame, n: int,
                                 settings: VolatilitySettings) -> dict[str, Any]:
        """Preserve the existing provider path and formulas for trusted candles."""
        results: dict[str, VolatilityMetric] = {}
        realized = realized_volatility(indexed, settings.realized_window,
                                       periods_per_year=settings.periods_per_year)
        results["realized_volatility"] = self._last_metric(
            realized, min(max(0, n - 1), settings.realized_window), "return per candle" if
            settings.periods_per_year is None else "annualized return volatility",
            f"Rolling sample standard deviation of log returns; window={settings.realized_window}; "
            f"periods_per_year={settings.periods_per_year!r}", settings.realized_window + 1)

        ewma = ewma_volatility(indexed, settings.ewma_decay,
                               periods_per_year=settings.periods_per_year)
        results["ewma_volatility"] = self._last_metric(
            ewma, max(0, n - 1), "return per candle" if settings.periods_per_year is None else
            "annualized return volatility",
            f"EWMA variance recursion; decay={settings.ewma_decay}; "
            f"periods_per_year={settings.periods_per_year!r}", 2)

        rolling = rolling_volatility(indexed, settings.rolling_window,
                                     periods_per_year=settings.periods_per_year)
        results["rolling_volatility"] = self._last_metric(
            rolling, min(max(0, n - 1), settings.rolling_window), "return per candle" if
            settings.periods_per_year is None else "annualized return volatility",
            f"Rolling sample standard deviation of log returns; window={settings.rolling_window}; "
            f"periods_per_year={settings.periods_per_year!r}", settings.rolling_window + 1)

        zscore = volatility_zscore(realized, settings.zscore_baseline_window)
        results["volatility_zscore"] = self._last_metric(
            zscore, min(settings.zscore_baseline_window, int(realized.notna().sum())),
            "standard deviations", f"Current realized volatility versus prior rolling baseline; "
            f"window={settings.zscore_baseline_window}",
            settings.realized_window + settings.zscore_baseline_window + 1)

        percentile = volatility_percentile(realized, settings.percentile_lookback,
                                           min_history=settings.percentile_min_history)
        results["volatility_percentile"] = self._last_metric(
            percentile, min(settings.percentile_lookback, int(realized.notna().sum())),
            "percentile (0-100)", f"Empirical percentile against prior realized volatility; "
            f"lookback={settings.percentile_lookback}",
            settings.realized_window + settings.percentile_min_history + 1)
        return {"realized_series": realized, **results}

    def _mt5_return_metrics(self, indexed: pd.DataFrame, valid_return_mask: np.ndarray,
                            settings: VolatilitySettings) -> dict[str, Any]:
        """Apply established return formulas only to exact-cadence MT5 returns.

        The sample sequence is the real set of adjacent one-timeframe log returns;
        gaps are removed, never bridged with a synthetic return.
        """
        closes = indexed["Close"].astype(float)
        all_returns = np.log(closes / closes.shift(1))
        valid_returns = all_returns.iloc[np.flatnonzero(valid_return_mask)].reset_index(drop=True)
        return_count = len(valid_returns)
        realized = valid_returns.rolling(settings.realized_window).std(ddof=1)
        annual_factor = (math.sqrt(settings.periods_per_year)
                         if settings.periods_per_year is not None else 1.0)
        realized = realized * annual_factor
        unit = "return per candle" if settings.periods_per_year is None else "annualized return volatility"
        rv_method = (f"Rolling sample standard deviation of exact-cadence log returns; "
                     f"window={settings.realized_window}; annualization factor={annual_factor:g}")
        realized_metric = self._last_metric(
            realized, min(return_count, settings.realized_window), unit, rv_method,
            settings.realized_window)

        variance = valid_returns.pow(2).ewm(alpha=1.0 - settings.ewma_decay, adjust=False).mean()
        ewma = np.sqrt(variance) * annual_factor
        ewma_metric = self._last_metric(
            ewma, return_count, unit,
            f"EWMA variance recursion over exact-cadence returns; decay={settings.ewma_decay}; "
            f"initial variance=first return squared; annualization factor={annual_factor:g}", 1)

        rolling_metric = self._last_metric(
            realized, min(return_count, settings.rolling_window), unit,
            f"Rolling sample standard deviation of exact-cadence log returns; "
            f"window={settings.rolling_window}; annualization factor={annual_factor:g}",
            settings.rolling_window)
        zscore = volatility_zscore(realized, settings.zscore_baseline_window)
        zscore_metric = self._last_metric(
            zscore, min(settings.zscore_baseline_window, int(realized.notna().sum())),
            "standard deviations", f"Current realized volatility versus prior rolling baseline; "
            f"window={settings.zscore_baseline_window}",
            settings.realized_window + settings.zscore_baseline_window)
        percentile = volatility_percentile(realized, settings.percentile_lookback,
                                           min_history=settings.percentile_min_history)
        percentile_metric = self._last_metric(
            percentile, min(settings.percentile_lookback, int(realized.notna().sum())),
            "percentile (0-100)", f"Empirical percentile against prior realized volatility; "
            f"lookback={settings.percentile_lookback}",
            settings.realized_window + settings.percentile_min_history)
        return {
            "realized_series": realized,
            "realized_volatility": realized_metric,
            "ewma_volatility": ewma_metric,
            "rolling_volatility": rolling_metric,
            "volatility_zscore": zscore_metric,
            "volatility_percentile": percentile_metric,
        }

    def _add_binance_time_metrics(self, service: Any, symbol: str, timeframe: str,
                                 metrics: dict[str, VolatilityMetric], warnings: list[str]) -> None:
        settings = self.settings
        try:
            daily = service.get_candles(symbol, "1d", settings.adr_lookback + 2,
                                        include_incomplete=True)
            self._validate_candle_frame(daily, symbol, "Binance", "1d", "TRUSTED_UTC")
            daily_ohlc = self._ohlc_frame(daily)
            adr_series = average_daily_range(daily_ohlc, settings.adr_lookback,
                                             session_timezone="UTC", session_start="00:00",
                                             as_of=datetime.now(timezone.utc), min_periods=settings.adr_lookback)
            metrics["average_daily_range"] = self._last_metric(
                adr_series, min(settings.adr_lookback, len(daily)), "price units",
                f"Mean prior completed UTC daily high-low ranges; lookback={settings.adr_lookback}",
                settings.adr_lookback + 1)
        except Exception as exc:
            warnings.append(f"Binance daily candles unavailable or invalid for ADR: {exc}")
            metrics["average_daily_range"] = self._unavailable_metric(
                f"Daily candle request/validation failed: {type(exc).__name__}: {exc}", "price units")

        if timeframe not in {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h"}:
            metrics["intraday_range_expansion"] = self._unavailable_metric(
                "Range expansion requires an intraday candle interval.", "ratio")
            return
        try:
            # Binance Spot crypto trades continuously; define the session
            # explicitly as midnight-to-midnight UTC.
            candles = service.get_candles(symbol, timeframe, 500, include_incomplete=False)
            self._validate_candle_frame(candles, symbol, "Binance", timeframe, "TRUSTED_UTC")
            frame = self._ohlc_frame(candles)
            ranges = intraday_range_expansion(
                frame, candle_interval=timeframe, session_timezone="UTC", session_start="00:00",
                session_duration="24h", lookback_sessions=settings.range_lookback_sessions,
                as_of=datetime.now(timezone.utc),
            )
            metrics["intraday_range_expansion"] = self._last_metric(
                ranges["range_expansion_ratio"], len(candles), "ratio",
                f"Current cumulative UTC session range / mean of prior complete sessions; "
                f"session=00:00-24:00 UTC; lookback={settings.range_lookback_sessions}",
                settings.range_lookback_sessions * int(86400 / _TIMEFRAME_SECONDS[timeframe]))
        except Exception as exc:
            warnings.append(f"Binance intraday range history unavailable or invalid: {exc}")
            metrics["intraday_range_expansion"] = self._unavailable_metric(
                f"Intraday candle request/validation failed: {type(exc).__name__}: {exc}", "ratio")

    @staticmethod
    def _ohlc_frame(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[:, ["timestamp", "open", "high", "low", "close"]].rename(
            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})

    @staticmethod
    def _last_metric(series: pd.Series, sample_count: int, unit: str,
                     methodology: str, minimum_rows: int) -> VolatilityMetric:
        if len(series) == 0 or not math.isfinite(float(series.iloc[-1])):
            status = STATUS_WARMING_UP if sample_count < minimum_rows else STATUS_UNAVAILABLE
            warning = "Insufficient valid history" if status == STATUS_WARMING_UP else "Metric is undefined for available history"
            return VolatilityMetric(status, None, unit, methodology, sample_count, warning)
        return VolatilityMetric(STATUS_AVAILABLE, float(series.iloc[-1]), unit, methodology, sample_count)

    @staticmethod
    def _withheld_metric(warning: str, unit: str) -> VolatilityMetric:
        return VolatilityMetric(STATUS_TIMESTAMP_UNVERIFIED, None, unit,
                                "Requires trusted absolute candle time", 0, warning)

    @staticmethod
    def _unavailable_metric(warning: str, unit: str) -> VolatilityMetric:
        return VolatilityMetric(STATUS_UNAVAILABLE, None, unit, "Not calculated", 0, warning)

    def _failed_result(self, symbol: str, source: str, timeframe: str, count: int,
                       warning: str, candle_count: int = 0,
                       timestamp_trust: str | None = None) -> InstrumentVolatilityResult:
        trust = timestamp_trust or ("TRUSTED_UTC" if source == "Binance" else "UNVERIFIED_TIME_BASIS")
        names = ("atr", "realized_volatility", "ewma_volatility", "rolling_volatility",
                 "volatility_zscore", "volatility_percentile", "average_daily_range",
                 "intraday_range_expansion")
        metrics = {name: self._unavailable_metric(warning, "unavailable") for name in names}
        if source == "MT5" and trust != "TRUSTED_UTC":
            metrics["average_daily_range"] = self._withheld_metric(
                "MT5 timestamp basis is unverified; daily-session attribution is disabled.", "price units")
            metrics["intraday_range_expansion"] = self._withheld_metric(
                "MT5 timestamp basis is unverified; session boundaries cannot be trusted.", "ratio")
        return InstrumentVolatilityResult(symbol, source, timeframe, count, candle_count, trust,
                                          STATUS_UNAVAILABLE, MappingProxyType(metrics), (warning,))
