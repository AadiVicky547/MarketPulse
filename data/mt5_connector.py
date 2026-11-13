"""Reusable synchronous access to a locally running MetaTrader 5 terminal.

MT5 epoch fields are converted using their documented units and retained with
provenance. Their UTC time-basis is explicitly unverified; no broker offset is
applied. Historical bars exclude the currently forming candle by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import math
import re
import time
from typing import Any, Mapping, Sequence

try:
    import MetaTrader5 as _mt5
except ImportError:
    _mt5 = None

logger = logging.getLogger(__name__)
DEFAULT_INSTRUMENTS = ("XAUUSD", "EURUSD", "GBPUSD", "USDJPY")


@dataclass(frozen=True)
class ConnectionStatus:
    connected: bool
    message: str
    error: str | None = None


@dataclass(frozen=True)
class SymbolMatch:
    requested_instrument: str
    actual_symbol: str | None
    matched: bool
    match_type: str | None
    message: str


class MT5ConnectorError(RuntimeError):
    """An MT5 operation could not be completed safely."""


def _field(record: Any, name: str, default: Any = None) -> Any:
    """Read a named value from mappings, NumPy records, or ordinary objects.

    MT5 historical rates are NumPy structured arrays. Their scalar rows expose
    named fields through ``record[name]`` rather than attribute access.
    """
    if isinstance(record, Mapping):
        return record.get(name, default)

    dtype = getattr(record, "dtype", None)
    field_names = getattr(dtype, "names", None)
    if field_names is not None and name in field_names:
        try:
            return record[name]
        except (IndexError, KeyError, TypeError):
            return default

    return getattr(record, name, default)


def _positive(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _utc(value: Any) -> datetime | None:
    """Interpret a numeric MT5 epoch value as seconds since Unix epoch, in UTC."""
    try:
        number = float(value)
        return datetime.fromtimestamp(number, timezone.utc) if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _utc_milliseconds(value: Any) -> datetime | None:
    """Interpret a numeric MT5 epoch value as milliseconds since Unix epoch, in UTC."""
    try:
        number = float(value)
        return (datetime.fromtimestamp(number / 1000.0, timezone.utc)
                if math.isfinite(number) and number > 0 else None)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _timestamp_problem(value: Any, unit: str) -> str | None:
    """Describe why an MT5 timestamp field cannot be interpreted in its documented unit."""
    if value is None:
        return "missing"
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "not numeric"
    if not math.isfinite(number):
        return "not finite"
    if number <= 0:
        return "zero or negative"
    converted = _utc_milliseconds(value) if unit == "milliseconds" else _utc(value)
    return None if converted is not None else "outside datetime range"


class MT5Connector:
    """Reusable, synchronous connector for the installed MT5 terminal."""

    def __init__(self, instruments: Sequence[str] = DEFAULT_INSTRUMENTS,
                 api: Any = None, stale_after_seconds: float = 300.0) -> None:
        self.instruments = tuple(item.upper() for item in instruments)
        self._api = api if api is not None else _mt5
        self.stale_after_seconds = stale_after_seconds
        self._connected = False
        self._symbols: dict[str, SymbolMatch] = {}
        self.status = ConnectionStatus(False, "Not connected")

    @property
    def connected(self) -> bool:
        return self._connected

    def _error_text(self) -> str:
        try:
            error = self._api.last_error()
            return f"MT5 error {error[0]}: {error[1]}" if error else "MT5 did not provide error details"
        except Exception:
            return "MT5 did not provide error details"

    def last_error(self) -> Any:
        """Return the exact latest error tuple from the official MT5 API.

        This is useful when an API call returns a valid but empty result array,
        because that case is distinct from ``None`` and otherwise carries no
        error context through normalized connector methods.
        """
        if self._api is None:
            return None
        return self._api.last_error()

    def initialize(self, **kwargs: Any) -> ConnectionStatus:
        """Initialize once; optional keyword arguments are passed to MT5.initialize."""
        if self._connected:
            return self.status
        if self._api is None:
            self.status = ConnectionStatus(False, "MetaTrader5 package is unavailable",
                                           "Install MetaTrader5 in this environment")
            return self.status
        try:
            initialized = self._api.initialize(**kwargs)
        except Exception as exc:
            self.status = ConnectionStatus(False, "MT5 initialization raised an exception", str(exc))
            logger.exception("MT5 initialization failed")
            return self.status
        if not initialized:
            error = self._error_text()
            self.status = ConnectionStatus(False, "Could not initialize MT5 terminal", error)
            logger.error("%s: %s", self.status.message, error)
            return self.status
        self._connected = True
        self.status = ConnectionStatus(True, "Connected to MT5 terminal")
        logger.info("Connected to MT5 terminal")
        return self.status

    def _require_connection(self) -> None:
        if not self._connected:
            raise MT5ConnectorError(f"MT5 terminal is not connected: {self.status.message}; {self.status.error or ''}")
        try:
            if self._api.terminal_info() is None:
                self._connected = False
                self.status = ConnectionStatus(False, "MT5 terminal is disconnected", self._error_text())
                raise MT5ConnectorError(f"MT5 terminal is disconnected: {self.status.error}")
        except MT5ConnectorError:
            raise
        except Exception as exc:
            self._connected = False
            self.status = ConnectionStatus(False, "Could not verify MT5 terminal connection", str(exc))
            raise MT5ConnectorError(self.status.message) from exc

    @staticmethod
    def _key(name: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", name.upper())

    @classmethod
    def _match_symbol(cls, requested: str, available: Sequence[Any]) -> SymbolMatch:
        """Prefer exact names, else choose the closest name containing the full base."""
        requested = requested.upper()
        names = [str(_field(item, "name", "")) for item in available]
        exact = next((name for name in names if name.upper() == requested), None)
        if exact:
            return SymbolMatch(requested, exact, True, "exact", "Exact symbol match")
        key = cls._key(requested)
        candidates = [name for name in names if key and key in cls._key(name)]
        if not candidates:
            return SymbolMatch(requested, None, False, None, "No related MT5 symbol found")
        def rank(name: str) -> tuple[int, int, str]:
            candidate = cls._key(name)
            relation = 0 if candidate.startswith(key) else 1 if candidate.endswith(key) else 2
            return relation, len(candidate) - len(key), name.upper()
        best = min(candidates, key=rank)
        return SymbolMatch(requested, best, True, "broker_variant",
                           f"Selected broker variant {best!r} from {len(candidates)} related candidate(s)")

    def discover_symbols(self, instruments: Sequence[str] | None = None) -> dict[str, SymbolMatch]:
        """Find requested symbols and enable them in Market Watch."""
        self._require_connection()
        requested = tuple(item.upper() for item in (instruments or self.instruments))
        try:
            available = self._api.symbols_get()
        except Exception as exc:
            raise MT5ConnectorError(f"MT5 symbol discovery failed: {exc}") from exc
        if available is None:
            raise MT5ConnectorError(f"MT5 symbol discovery failed: {self._error_text()}")
        result = {}
        for instrument in requested:
            match = self._match_symbol(instrument, available)
            if match.actual_symbol:
                try:
                    if not self._api.symbol_select(match.actual_symbol, True):
                        match = SymbolMatch(match.requested_instrument, match.actual_symbol, True,
                                            match.match_type, match.message +
                                            f"; Market Watch enable failed: {self._error_text()}")
                except Exception as exc:
                    match = SymbolMatch(match.requested_instrument, match.actual_symbol, True,
                                        match.match_type, match.message + f"; selection failed: {exc}")
            result[instrument] = match
        self._symbols.update(result)
        return result

    def _resolve(self, instrument: str) -> SymbolMatch:
        key = instrument.upper()
        if key not in self._symbols:
            self.discover_symbols((key,))
        match = self._symbols[key]
        if not match.actual_symbol:
            raise MT5ConnectorError(f"No MT5 symbol found for requested instrument {key}")
        return match

    def get_tick(self, instrument: str) -> dict[str, Any]:
        """Get latest tick; invalid numeric fields become None and stale ticks are flagged."""
        self._require_connection()
        match = self._resolve(instrument)
        try:
            tick = self._api.symbol_info_tick(match.actual_symbol)
        except Exception as exc:
            raise MT5ConnectorError(f"Tick request failed for {match.actual_symbol}: {exc}") from exc
        received_at = datetime.now(timezone.utc)
        result = {"requested_instrument": match.requested_instrument, "actual_symbol": match.actual_symbol,
                  "connection_status": "connected", "available": False, "tick_received": False, "stale": None,
                  "bid": None, "ask": None, "last": None, "spread_price": None,
                  "spread_points": None, "tick_volume": None, "timestamp": None, "error": None,
                  "raw_time": None, "raw_time_msc": None, "timestamp_source": None,
                  "timestamp_status": "INVALID", "timestamp_trust": "UNAVAILABLE",
                  "timestamp_issues": (), "received_at": received_at}
        if tick is None:
            result["error"] = self._error_text()
            result["timestamp_issues"] = ("tick unavailable",)
            return result
        result["tick_received"] = True
        bid, ask, last = _positive(_field(tick, "bid")), _positive(_field(tick, "ask")), _positive(_field(tick, "last"))
        raw_time = _field(tick, "time")
        raw_time_msc = _field(tick, "time_msc")
        seconds_timestamp = _utc(raw_time)
        milliseconds_timestamp = _utc_milliseconds(raw_time_msc)
        seconds_problem = _timestamp_problem(raw_time, "seconds")
        milliseconds_problem = _timestamp_problem(raw_time_msc, "milliseconds")
        timestamp_issues = []
        if seconds_problem:
            timestamp_issues.append(f"time ({seconds_problem})")
        if milliseconds_problem:
            timestamp_issues.append(f"time_msc ({milliseconds_problem})")
        consistent = True
        if seconds_timestamp is not None and milliseconds_timestamp is not None:
            try:
                seconds_value = float(raw_time)
                milliseconds_value = float(raw_time_msc)
                consistent = (seconds_value.is_integer() and
                              math.floor(milliseconds_value / 1000.0) == int(seconds_value))
            except (TypeError, ValueError, OverflowError):
                consistent = False
            if not consistent:
                timestamp_issues.append("time and time_msc disagree")

        # Prefer MT5's millisecond event time; use whole seconds only when the
        # millisecond field is unavailable or invalid. This is unit handling,
        # not a broker timezone correction.
        timestamp = milliseconds_timestamp or seconds_timestamp
        timestamp_source = "time_msc" if milliseconds_timestamp is not None else (
            "time" if seconds_timestamp is not None else None)
        if timestamp is None:
            timestamp_status = "INVALID"
        elif not consistent:
            timestamp_status = "INCONSISTENT"
        elif timestamp_source == "time":
            timestamp_status = "FALLBACK_SECONDS"
        elif seconds_problem:
            timestamp_status = "MILLISECONDS_ONLY"
        else:
            timestamp_status = "VALID"
        spread = ask - bid if bid is not None and ask is not None else None
        if spread is not None and spread < 0:
            spread = None
        info = self._api.symbol_info(match.actual_symbol)
        point = _positive(_field(info, "point")) if info is not None else None
        volume = _field(tick, "volume")
        try:
            volume = int(volume) if volume is not None and int(volume) >= 0 else None
        except (TypeError, ValueError, OverflowError):
            volume = None
        result.update({"available": timestamp is not None,
                       "stale": timestamp is None or time.time() - timestamp.timestamp() > self.stale_after_seconds,
                       "bid": bid, "ask": ask, "last": last, "spread_price": spread,
                       "spread_points": spread / point if spread is not None and point else None,
                       "tick_volume": volume, "timestamp": timestamp,
                       "raw_time": raw_time, "raw_time_msc": raw_time_msc,
                       "timestamp_source": timestamp_source, "timestamp_status": timestamp_status,
                       "timestamp_trust": "UNVERIFIED_TIME_BASIS" if timestamp is not None else "UNAVAILABLE",
                       "timestamp_issues": tuple(timestamp_issues),
                       "timestamp_warning": "; ".join(timestamp_issues) if timestamp_issues else None})
        if timestamp is None:
            result["error"] = "Tick timestamp is missing or invalid"
        return result

    def get_rates(self, instrument: str, timeframe: int, count: int = 100,
                  include_current: bool = False) -> list[dict[str, Any]]:
        """Get recent OHLC bars with UTC timestamps; excludes the forming bar by default."""
        self._require_connection()
        if count <= 0:
            raise ValueError("count must be greater than zero")
        match = self._resolve(instrument)
        try:
            rows = self._api.copy_rates_from_pos(match.actual_symbol, timeframe, 0 if include_current else 1, count)
        except Exception as exc:
            raise MT5ConnectorError(f"Historical rates request failed for {match.actual_symbol}: {exc}") from exc
        if rows is None:
            raise MT5ConnectorError(f"Historical rates unavailable for {match.actual_symbol}: {self._error_text()}")
        bars = []
        for index, row in enumerate(rows):
            raw_time = _field(row, "time")
            timestamp = _utc(raw_time)
            if timestamp is None:
                problem = _timestamp_problem(raw_time, "seconds")
                logger.warning("Skipping MT5 rate row %d for %s with invalid raw time=%r (%s)",
                               index, match.actual_symbol, raw_time, problem)
                continue
            bars.append({"timestamp": timestamp, "raw_time": raw_time,
                         "timestamp_source": "time", "timestamp_status": "VALID",
                         "timestamp_trust": "UNVERIFIED_TIME_BASIS",
                         "timestamp_issues": (), "open": _positive(_field(row, "open")),
                         "high": _positive(_field(row, "high")), "low": _positive(_field(row, "low")),
                         "close": _positive(_field(row, "close")), "tick_volume": _field(row, "tick_volume"),
                         "spread": _field(row, "spread"), "real_volume": _field(row, "real_volume"),
                         "actual_symbol": match.actual_symbol, "requested_instrument": match.requested_instrument})
        return bars

    def get_market_depth(self, instrument: str) -> dict[str, Any]:
        """Return broker-provided depth when supported, never implying consolidated Forex liquidity."""
        self._require_connection()
        match = self._resolve(instrument)
        try:
            if not self._api.market_book_add(match.actual_symbol):
                return {"requested_instrument": match.requested_instrument, "actual_symbol": match.actual_symbol,
                        "status": "unavailable", "levels": [], "message": self._error_text()}
            book = self._api.market_book_get(match.actual_symbol)
        except Exception as exc:
            return {"requested_instrument": match.requested_instrument, "actual_symbol": match.actual_symbol,
                    "status": "unavailable", "levels": [], "message": str(exc)}
        finally:
            # This method performs a one-shot read, so release the terminal subscription.
            try:
                if self._api.market_book_release(match.actual_symbol) is False:
                    logger.warning("MT5 could not release market book for %s", match.actual_symbol)
            except Exception:
                logger.exception("Could not release MT5 market book for %s", match.actual_symbol)
        if book is None:
            return {"requested_instrument": match.requested_instrument, "actual_symbol": match.actual_symbol,
                    "status": "unavailable", "levels": [], "message": self._error_text()}
        levels = [dict(item) if isinstance(item, Mapping) else item._asdict() if hasattr(item, "_asdict")
                  else {key: _field(item, key) for key in ("type", "price", "volume", "volume_dbl")}
                  for item in book]
        return {"requested_instrument": match.requested_instrument, "actual_symbol": match.actual_symbol,
                "status": "available", "levels": levels,
                "message": "Broker-provided depth only; not consolidated Forex market liquidity"}

    def close(self) -> None:
        """Shut down the initialized MT5 connection."""
        if self._connected and self._api is not None:
            try:
                self._api.shutdown()
            finally:
                self._connected = False
                self.status = ConnectionStatus(False, "Disconnected")

    def __enter__(self) -> "MT5Connector":
        status = self.initialize()
        if not status.connected:
            raise MT5ConnectorError(f"{status.message}: {status.error or ''}")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
