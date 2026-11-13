"""Historical OHLC retrieval and normalization for MT5 and Binance Spot.

The returned DataFrame uses UTC-aware candle-open timestamps and consistent
lowercase OHLC columns. ``volume`` keeps provider semantics: MT5 ``tick_volume``
is broker-observed tick activity; Binance ``volume`` is executed base-asset
quantity. Binance quote-asset volume and MT5 tick/real volume remain separate.
The service performs synchronous, on-demand retrieval and does not persist data.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Any, Callable, Mapping

import pandas as pd
import requests

from config import INSTRUMENTS
from data.mt5_connector import MT5Connector

BINANCE_REST_BASE = "https://api.binance.com"
BINANCE_KLINES_ENDPOINT = f"{BINANCE_REST_BASE}/api/v3/klines"
BINANCE_MAX_LIMIT = 1000

# Durations are explicit so completion can be determined from candle-open time.
_INTERVALS: dict[str, tuple[str, int]] = {
    "1s": ("S1", 1), "1m": ("M1", 60), "3m": ("M3", 180),
    "5m": ("M5", 300), "15m": ("M15", 900), "30m": ("M30", 1800),
    "1h": ("H1", 3600), "2h": ("H2", 7200), "4h": ("H4", 14400),
    "6h": ("H6", 21600), "8h": ("H8", 28800), "12h": ("H12", 43200),
    "1d": ("D1", 86400), "3d": ("D3", 259200), "1w": ("W1", 604800),
}
STANDARD_CANDLE_COLUMNS = (
    "instrument", "source", "actual_symbol", "timeframe", "timestamp",
    "raw_timestamp_epoch", "timestamp_source", "timestamp_status", "timestamp_trust",
    "open", "high", "low", "close", "volume", "volume_type",
    "tick_volume", "real_volume", "quote_volume", "spread_points", "completed",
    "completion_status",
)


class CandleDataError(RuntimeError):
    """Base error for unavailable or invalid historical candle data."""


class CandleProviderError(CandleDataError):
    """A market-data provider did not return usable historical candles."""


def _utc_now(clock: Callable[[], datetime] | None) -> datetime:
    value = clock() if clock else datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _number(value: Any, label: str, *, allow_zero: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CandleDataError(f"Invalid {label}: {value!r}") from exc
    if not math.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        raise CandleDataError(f"Invalid {label}: {value!r}")
    return result


def _validate_ohlc(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    opened = _number(row.get("open"), "open")
    high = _number(row.get("high"), "high")
    low = _number(row.get("low"), "low")
    close = _number(row.get("close"), "close")
    if high < max(opened, close, low) or low > min(opened, close, high):
        raise CandleDataError("OHLC candle is inconsistent: high/low do not contain open and close")
    return opened, high, low, close


class CandleService:
    """Fetch historical candles from local MT5 and public Binance Spot REST.

    Timeframes use Binance interval notation (for example ``1m``, ``1h``,
    ``1d``). MT5 equivalents are resolved through the official package's
    ``TIMEFRAME_*`` constants. Pass ``now`` in tests to make completion
    decisions deterministic. Incomplete candles are omitted by default.
    """

    def __init__(
        self,
        *,
        mt5_connector_factory: Callable[[], Any] = MT5Connector,
        rest_get: Callable[..., Any] | None = None,
        now: Callable[[], datetime] | None = None,
        timeout: tuple[float, float] = (3.05, 10.0),
        mt5_timeframes: Mapping[str, int] | None = None,
    ) -> None:
        self._instrument_sources = {item.symbol.upper(): item.source for item in INSTRUMENTS}
        self._mt5_connector_factory = mt5_connector_factory
        self._rest_get = rest_get or requests.get
        self._now = now
        self._timeout = timeout
        self._mt5_timeframes = dict(mt5_timeframes) if mt5_timeframes is not None else None
        self._mt5: Any = None
        self._mt5_initialized = False

    @staticmethod
    def empty_frame() -> pd.DataFrame:
        """Return an empty DataFrame with the standard output schema."""
        return pd.DataFrame(columns=STANDARD_CANDLE_COLUMNS)

    def _timeframe(self, interval: str) -> tuple[str, str, int]:
        key = str(interval).strip().lower()
        try:
            mt5_code, duration = _INTERVALS[key]
            return key, mt5_code, duration
        except KeyError as exc:
            raise ValueError(f"Unsupported timeframe {interval!r}; supported: {', '.join(_INTERVALS)}") from exc

    def _mt5_timeframe_value(self, code: str) -> int:
        if self._mt5_timeframes is not None:
            if code not in self._mt5_timeframes:
                raise ValueError(f"MT5 timeframe {code} was not configured")
            return int(self._mt5_timeframes[code])
        try:
            import MetaTrader5 as mt5
        except ImportError as exc:
            raise CandleProviderError("MetaTrader5 package is unavailable") from exc
        name = f"TIMEFRAME_{code}"
        if not hasattr(mt5, name):
            raise ValueError(f"Installed MetaTrader5 package does not expose {name}")
        return int(getattr(mt5, name))

    def _get_mt5_connector(self) -> Any:
        if self._mt5 is None:
            self._mt5 = self._mt5_connector_factory()
        if not self._mt5_initialized:
            status = self._mt5.initialize()
            if not getattr(status, "connected", False):
                message = getattr(status, "message", "MT5 initialization failed")
                error = getattr(status, "error", None)
                raise CandleProviderError(f"{message}{': ' + str(error) if error else ''}")
            self._mt5_initialized = True
        return self._mt5

    def _mt5_rows(self, instrument: str, timeframe: int, count: int,
                  include_incomplete: bool) -> list[dict[str, Any]]:
        connector = self._get_mt5_connector()
        matches = connector.discover_symbols((instrument,))
        match = matches[instrument]
        if not match.actual_symbol:
            raise CandleProviderError(f"MT5 symbol unavailable for {instrument}: {match.message}")
        try:
            rates = connector.get_rates(instrument, timeframe, count=count,
                                        include_current=include_incomplete)
        except Exception as exc:
            raise CandleProviderError(f"MT5 history unavailable for {instrument}: {exc}") from exc
        # Some terminals/brokers can return an empty position-0 result when no
        # current bar is available, even though completed history exists. A
        # request for history must not fail just because that optional bar is
        # absent; retry from position 1 (the latest completed bar).
        if include_incomplete and rates is not None and len(rates) == 0:
            try:
                rates = connector.get_rates(instrument, timeframe, count=count,
                                            include_current=False)
            except Exception as exc:
                raise CandleProviderError(
                    f"MT5 completed history unavailable for {instrument}: {exc}") from exc
        if rates is None or len(rates) == 0:
            try:
                error = connector.last_error()
            except Exception as exc:
                error = f"last_error() failed: {type(exc).__name__}: {exc}"
            detail = f"; MT5 last_error={error!r}" if error is not None else "; MT5 supplied no error details"
            raise CandleProviderError(
                f"MT5 returned no historical bars for {instrument} ({match.actual_symbol}){detail}")
        return rates

    def _binance_rows(self, instrument: str, interval: str, count: int) -> list[Any]:
        if count > BINANCE_MAX_LIMIT:
            raise ValueError(f"Binance Spot klines limit is at most {BINANCE_MAX_LIMIT}; got {count}")
        try:
            response = self._rest_get(BINANCE_KLINES_ENDPOINT,
                                      params={"symbol": instrument, "interval": interval, "limit": count},
                                      timeout=self._timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            detail = f"HTTP {status_code}" if status_code is not None else str(exc)
            raise CandleProviderError(f"Binance klines request failed for {instrument}: {detail}") from exc
        except (ValueError, TypeError) as exc:
            raise CandleProviderError(f"Malformed Binance klines response for {instrument}: {exc}") from exc
        if not isinstance(payload, list):
            raise CandleProviderError(f"Malformed Binance klines response for {instrument}: expected a list")
        return payload

    def get_candles(self, instrument: str, timeframe: str = "1h", count: int = 500,
                    *, include_incomplete: bool = False) -> pd.DataFrame:
        """Retrieve normalized candles; shorter provider history stays shorter.

        Binance limits are validated rather than silently truncated. Provider
        errors (including rate limiting) are raised as ``CandleProviderError``;
        this service does not retry a rate-limited request. Timestamps identify
        candle opens, are UTC-aware, and duplicate candle opens are rejected.
        """
        symbol = str(instrument).upper()
        if symbol not in self._instrument_sources:
            raise ValueError(f"Unknown instrument: {instrument!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        interval, mt5_code, duration_seconds = self._timeframe(timeframe)
        source = self._instrument_sources[symbol]
        if source == "Binance":
            raw = self._binance_rows(symbol, interval, count)
            rows = self._normalize_binance(symbol, interval, raw)
        else:
            timeframe_value = self._mt5_timeframe_value(mt5_code)
            raw = self._mt5_rows(symbol, timeframe_value, count, include_incomplete)
            rows = self._normalize_mt5(symbol, interval, raw)

        now = _utc_now(self._now)
        for row in rows:
            provider_completed = row.pop("_provider_completed")
            if row["timestamp_trust"] != "TRUSTED_UTC":
                # Preserve raw bars for inspection, but do not claim completion
                # from an unverified broker event-time basis.
                row["completed"] = None
                row["completion_status"] = "UNKNOWN_UNVERIFIED_TIMESTAMP"
            else:
                row["completed"] = provider_completed and (
                    row["timestamp"] + timedelta(seconds=duration_seconds) <= now)
                row["completion_status"] = "COMPLETED" if row["completed"] else "INCOMPLETE"
        frame = pd.DataFrame(rows, columns=STANDARD_CANDLE_COLUMNS)
        if frame.empty:
            return self.empty_frame()
        frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        if frame["timestamp"].duplicated().any():
            raise CandleDataError(f"Duplicate candle timestamps returned for {symbol}")
        if not include_incomplete:
            # Keep MT5 rows whose completion is unknown; filtering them as if
            # they were incomplete would hide data with unverified time basis.
            frame = frame.loc[frame["completed"].isna() | frame["completed"]].reset_index(drop=True)
        return frame

    def _normalize_mt5(self, instrument: str, timeframe: str,
                       raw: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for item in raw:
            stamp = pd.Timestamp(item.get("timestamp"))
            if stamp.tzinfo is None:
                raise CandleDataError("MT5 connector returned a timezone-naive candle timestamp")
            stamp = stamp.tz_convert("UTC")
            opened, high, low, close = _validate_ohlc(item)
            tick_volume = _optional_nonnegative(item.get("tick_volume"), "MT5 tick volume")
            real_volume = _optional_nonnegative(item.get("real_volume"), "MT5 real volume")
            rows.append({"instrument": instrument, "source": "MT5",
                         "actual_symbol": item.get("actual_symbol"), "timeframe": timeframe,
                         "timestamp": stamp.to_pydatetime(),
                         "raw_timestamp_epoch": item.get("raw_time"),
                         "timestamp_source": item.get("timestamp_source", "time"),
                         "timestamp_status": item.get("timestamp_status", "VALID"),
                         "timestamp_trust": item.get("timestamp_trust", "UNVERIFIED_TIME_BASIS"),
                         "open": opened, "high": high,
                         "low": low, "close": close, "volume": tick_volume,
                         "volume_type": "broker_tick_volume", "tick_volume": tick_volume,
                         "real_volume": real_volume, "quote_volume": None,
                         "spread_points": _optional_nonnegative(item.get("spread"), "MT5 candle spread points"),
                         "_provider_completed": True})
        return rows

    def _normalize_binance(self, instrument: str, timeframe: str,
                           raw: list[Any]) -> list[dict[str, Any]]:
        rows = []
        for index, item in enumerate(raw):
            if not isinstance(item, (list, tuple)) or len(item) < 12:
                raise CandleDataError(f"Malformed Binance kline row {index}: expected 12 fields")
            try:
                open_time = int(item[0])
                close_time = int(item[6])
                stamp = datetime.fromtimestamp(open_time / 1000, timezone.utc)
                opened, high, low, close = _validate_ohlc(
                    {"open": item[1], "high": item[2], "low": item[3], "close": item[4]})
                volume = _number(item[5], "Binance base-asset volume", allow_zero=True)
                quote_volume = _number(item[7], "Binance quote-asset volume", allow_zero=True)
            except (TypeError, ValueError, OverflowError, OSError) as exc:
                if isinstance(exc, CandleDataError):
                    raise
                raise CandleDataError(f"Malformed Binance kline row {index}: {exc}") from exc
            if open_time < 0 or close_time < open_time:
                raise CandleDataError(f"Malformed Binance kline row {index}: invalid candle times")
            rows.append({"instrument": instrument, "source": "Binance", "actual_symbol": instrument,
                         "timeframe": timeframe, "timestamp": stamp, "raw_timestamp_epoch": None,
                         "timestamp_source": None, "timestamp_status": "VALID",
                         "timestamp_trust": "TRUSTED_UTC",
                         "open": opened, "high": high,
                         "low": low, "close": close, "volume": volume,
                         "volume_type": "base_asset_trade_volume", "tick_volume": None,
                         "real_volume": None, "quote_volume": quote_volume, "spread_points": None,
                         "_provider_completed": close_time < int(_utc_now(self._now).timestamp() * 1000)})
        return rows

    def close(self) -> None:
        """Release the MT5 terminal connection owned by this service."""
        if self._mt5 is not None and self._mt5_initialized:
            try:
                self._mt5.close()
            finally:
                self._mt5_initialized = False
                self._mt5 = None

    def __enter__(self) -> "CandleService":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _optional_nonnegative(value: Any, label: str) -> float | int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CandleDataError(f"Invalid {label}: {value!r}") from exc
    if not math.isfinite(number) or number < 0:
        raise CandleDataError(f"Invalid {label}: {value!r}")
    return int(number) if number.is_integer() else number
