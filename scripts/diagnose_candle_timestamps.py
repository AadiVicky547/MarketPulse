"""Compare five XAUUSD H1 MT5 epochs with Python clocks and completion rules.

Run this from the project root while the intended MT5 terminal is already
available to the existing connector. It requests five bars only, prints raw
timestamps for diagnosis, and never changes or offsets candle timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import traceback
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.mt5_connector import MT5Connector


@dataclass(frozen=True)
class CandleTimeAssessment:
    """Diagnostic classification without changing CandleService behavior."""

    expected_close_utc: datetime
    status: str
    completed: bool
    future_dated: bool


def assess_candle_time(candle_open: datetime, now_utc: datetime,
                       interval_seconds: int = 3600) -> CandleTimeAssessment:
    """Flag future opens separately from ordinarily incomplete candles."""
    if candle_open.tzinfo is None or now_utc.tzinfo is None:
        raise ValueError("candle_open and now_utc must be timezone-aware")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    opened = candle_open.astimezone(timezone.utc)
    now = now_utc.astimezone(timezone.utc)
    expected_close = opened + timedelta(seconds=interval_seconds)
    future = opened > now
    completed = not future and expected_close <= now
    status = "FUTURE_DATED" if future else "COMPLETED" if completed else "INCOMPLETE"
    return CandleTimeAssessment(expected_close, status, completed, future)


def _epoch_datetime(raw_epoch: Any) -> datetime:
    """Convert epoch seconds explicitly as UTC, matching the connector."""
    return datetime.fromtimestamp(int(raw_epoch), timezone.utc)


class CapturingMT5API:
    """Delegate to MetaTrader5 while retaining raw rate records for comparison."""

    def __init__(self, api: Any) -> None:
        self._api = api
        self.raw_rates: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._api, name)

    def copy_rates_from_pos(self, symbol: str, timeframe: int,
                            start_pos: int, count: int) -> Any:
        rates = self._api.copy_rates_from_pos(symbol, timeframe, start_pos, count)
        self.raw_rates = rates
        return rates


def _print_terminal_clock(terminal: Any) -> None:
    """Print terminal metadata and any explicit clock field, if exposed."""
    print(f"MT5 terminal connected: {getattr(terminal, 'connected', None)!r}", flush=True)
    print(f"MT5 terminal name: {getattr(terminal, 'name', None)!r}", flush=True)
    print(f"MT5 terminal path: {getattr(terminal, 'path', None)!r}", flush=True)
    values = terminal._asdict() if hasattr(terminal, "_asdict") else {}
    time_fields = {key: value for key, value in values.items()
                   if "time" in key.lower() or "server" in key.lower()}
    if time_fields:
        print(f"MT5 terminal info time/server fields (source: terminal_info): {time_fields!r}",
              flush=True)
    else:
        print("MT5 terminal_info exposes no explicit server-clock field.", flush=True)


def _print_tick_clock(api: Any, actual_symbol: str, now_utc: datetime) -> None:
    """Show the latest broker tick timestamp as an event clock, not server wall time."""
    try:
        tick = api.symbol_info_tick(actual_symbol)
        if tick is None:
            print(f"Latest broker tick: unavailable; MT5 last_error={api.last_error()!r}", flush=True)
            return
        raw_seconds = getattr(tick, "time", None)
        raw_milliseconds = getattr(tick, "time_msc", None)
        event_epoch = raw_milliseconds / 1000 if raw_milliseconds else raw_seconds
        tick_utc = _epoch_datetime(event_epoch) if event_epoch else None
        print(f"Latest broker tick raw time: time={raw_seconds!r}, time_msc={raw_milliseconds!r}",
              flush=True)
        print(f"Latest broker tick event UTC: {tick_utc.isoformat() if tick_utc else None}; "
              "source: symbol_info_tick event timestamp", flush=True)
        if tick_utc:
            print(f"Latest tick age relative to Python UTC: {now_utc - tick_utc}", flush=True)
    except Exception:
        print("Latest broker tick inspection failed; full traceback:", flush=True)
        traceback.print_exc()


def main() -> int:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 package is not installed in this Python environment.", flush=True)
        return 2

    if not hasattr(mt5, "TIMEFRAME_H1"):
        print("MetaTrader5.TIMEFRAME_H1 is unavailable.", flush=True)
        return 2

    now_utc_before = datetime.now(timezone.utc)
    local_before = datetime.now().astimezone()
    print(f"Python UTC before request: {now_utc_before.isoformat()}", flush=True)
    print(f"Python local system time before request: {local_before.isoformat()}", flush=True)
    print(f"Local clock converted to UTC: {local_before.astimezone(timezone.utc).isoformat()}",
          flush=True)

    capture = CapturingMT5API(mt5)
    connector = MT5Connector(instruments=("XAUUSD",), api=capture)
    try:
        status = connector.initialize()
        print(f"MT5 connector status: connected={status.connected}; message={status.message}; "
              f"error={status.error!r}", flush=True)
        if not status.connected:
            print(f"MT5 last_error: {connector.last_error()!r}", flush=True)
            return 1

        terminal = mt5.terminal_info()
        _print_terminal_clock(terminal)
        matches = connector.discover_symbols(("XAUUSD",))
        match = matches["XAUUSD"]
        print(f"XAUUSD actual symbol: {match.actual_symbol!r}; "
              f"match={match.match_type!r}; discovery={match.message}", flush=True)
        if not match.actual_symbol:
            print(f"MT5 last_error: {connector.last_error()!r}", flush=True)
            return 1

        bars = connector.get_rates("XAUUSD", int(mt5.TIMEFRAME_H1), count=5,
                                   include_current=False)
        now_utc = datetime.now(timezone.utc)
        local_now = datetime.now().astimezone()
        print(f"\nPython UTC after request: {now_utc.isoformat()}", flush=True)
        print(f"Python local system time after request: {local_now.isoformat()}", flush=True)
        print(f"Local clock converted to UTC: {local_now.astimezone(timezone.utc).isoformat()}",
              flush=True)
        print(f"MT5 last_error after rates request: {connector.last_error()!r}", flush=True)

        raw_rates = capture.raw_rates
        if raw_rates is None:
            print("MT5 raw rates result is None.", flush=True)
            return 1
        print(f"Raw MT5 rates type: {type(raw_rates).__module__}.{type(raw_rates).__qualname__}; "
              f"count={len(raw_rates)}", flush=True)
        print(f"Connector-converted bars count: {len(bars)}", flush=True)

        for index, raw in enumerate(raw_rates, start=1):
            raw_epoch = int(raw["time"])
            raw_utc = _epoch_datetime(raw_epoch)
            converted = bars[index - 1] if index <= len(bars) else None
            connector_utc = converted["timestamp"] if converted is not None else None
            local_rendering = datetime.fromtimestamp(raw_epoch).astimezone()
            assessment = assess_candle_time(raw_utc, now_utc, 3600)
            service_completed = raw_utc + timedelta(hours=1) <= now_utc
            print(f"\nH1 candle {index}:", flush=True)
            print(f"  raw MT5 epoch seconds: {raw_epoch}", flush=True)
            print(f"  Python fromtimestamp(epoch, UTC): {raw_utc.isoformat()}", flush=True)
            print(f"  Connector converted UTC: "
                  f"{connector_utc.isoformat() if connector_utc else '<record omitted by connector>'}",
                  flush=True)
            print(f"  Python local rendering of same epoch: {local_rendering.isoformat()}", flush=True)
            print(f"  Candle open UTC: {raw_utc.isoformat()}", flush=True)
            print(f"  Expected H1 close UTC: {assessment.expected_close_utc.isoformat()}", flush=True)
            print(f"  CandleService completion rule: {service_completed}", flush=True)
            print(f"  Diagnostic time status: {assessment.status}; "
                  f"future_dated={assessment.future_dated}", flush=True)
            print(f"  Open-time difference from Python UTC: {raw_utc - now_utc}", flush=True)

        if bars:
            latest = bars[-1]["timestamp"]
            print(f"\nLatest candle open minus Python UTC now: {latest - now_utc}", flush=True)
        _print_tick_clock(mt5, match.actual_symbol, now_utc)
        print("Raw candle epochs and displayed UTC values above are unadjusted; "
              "no broker offset was applied.", flush=True)
        return 0
    except Exception:
        print("Timestamp diagnostic failed; full traceback follows:", flush=True)
        traceback.print_exc()
        try:
            print(f"MT5 last_error: {connector.last_error()!r}", flush=True)
        except Exception:
            traceback.print_exc()
        return 1
    finally:
        connector.close()
        print("MT5 connector shut down.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
