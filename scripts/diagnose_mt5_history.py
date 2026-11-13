"""Diagnose MT5 terminal access and recent broker-provided candle history."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.mt5_connector import DEFAULT_INSTRUMENTS, MT5Connector


def _format_error(mt5: Any) -> str:
    """Return the exact most recent MT5 API last_error tuple without secrets."""
    try:
        return repr(mt5.last_error())
    except Exception as exc:
        return f"last_error() raised {type(exc).__name__}: {exc}"


def _print_rates(mt5: Any, symbol: str, timeframe: int, timeframe_name: str,
                 count: int = 5) -> bool:
    print(f"  Request: copy_rates_from_pos({symbol!r}, {timeframe_name}, 1, {count})", flush=True)
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, count)
    except Exception as exc:
        print(f"  Request raised {type(exc).__name__}: {exc}", flush=True)
        print(f"  MT5 last_error: {_format_error(mt5)}", flush=True)
        return False
    if rates is None:
        print("  Rates result: None", flush=True)
        print(f"  MT5 last_error: {_format_error(mt5)}", flush=True)
        return False
    try:
        number = len(rates)
    except TypeError:
        print(f"  Rates result has unexpected type {type(rates).__name__}", flush=True)
        print(f"  MT5 last_error: {_format_error(mt5)}", flush=True)
        return False
    print(f"  Bars returned: {number}", flush=True)
    if number == 0:
        print(f"  MT5 last_error: {_format_error(mt5)}", flush=True)
        return False
    try:
        first = datetime.fromtimestamp(int(rates[0]["time"]), timezone.utc)
        last = datetime.fromtimestamp(int(rates[-1]["time"]), timezone.utc)
        print(f"  First candle open (UTC): {first.isoformat()}", flush=True)
        print(f"  Last candle open (UTC):  {last.isoformat()}", flush=True)
        print(f"  MT5 last_error after successful request: {_format_error(mt5)}", flush=True)
        return True
    except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
        print(f"  Returned rates could not be read: {type(exc).__name__}: {exc}", flush=True)
        print(f"  MT5 last_error: {_format_error(mt5)}", flush=True)
        return False


def main() -> int:
    print("Loading the installed MetaTrader5 package...", flush=True)
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        print(f"MetaTrader5 package import failed: {exc}", flush=True)
        return 2

    connector = MT5Connector()
    try:
        print("Initializing the actual MT5 terminal (timeout: 15 seconds)...", flush=True)
        status = connector.initialize(timeout=15_000)
        print(f"Terminal connection: {'connected' if status.connected else 'failed'}", flush=True)
        print(f"Initialization message: {status.message}", flush=True)
        if status.error:
            print(f"Initialization error: {status.error}", flush=True)
        if not status.connected:
            print(f"MT5 last_error: {_format_error(mt5)}", flush=True)
            return 1

        terminal = mt5.terminal_info()
        print(f"Terminal info available: {terminal is not None}", flush=True)
        if terminal is None:
            print(f"MT5 last_error: {_format_error(mt5)}", flush=True)

        matches = connector.discover_symbols(DEFAULT_INSTRUMENTS)
        h1 = getattr(mt5, "TIMEFRAME_H1", None)
        if h1 is None:
            print("Installed MetaTrader5 package has no TIMEFRAME_H1 constant", flush=True)
            return 2
        print(f"H1 timeframe constant: {h1} (MetaTrader5.TIMEFRAME_H1)", flush=True)

        failed_h1: list[tuple[str, str]] = []
        for requested in DEFAULT_INSTRUMENTS:
            match = matches[requested]
            print(f"\n{requested}: actual symbol={match.actual_symbol!r}; "
                  f"match={match.match_type}; discovery={match.message}", flush=True)
            if not match.actual_symbol:
                print(f"  Symbol unavailable; MT5 last_error: {_format_error(mt5)}", flush=True)
                failed_h1.append((requested, "symbol unavailable"))
                continue

            actual = match.actual_symbol
            selection_ok = mt5.symbol_select(actual, True)
            symbol_info = mt5.symbol_info(actual)
            visible = getattr(symbol_info, "visible", None) if symbol_info is not None else None
            print(f"  symbol_select({actual!r}, True): {selection_ok}", flush=True)
            print(f"  symbol_info available: {symbol_info is not None}; visible={visible!r}", flush=True)
            if not selection_ok or symbol_info is None:
                print(f"  Symbol selection/info error: {_format_error(mt5)}", flush=True)

            if not _print_rates(mt5, actual, h1, "TIMEFRAME_H1"):
                failed_h1.append((requested, actual))

        # Only probe M5 if all four H1 history calls failed; this distinguishes
        # missing H1 bars from a symbol-wide lack of downloaded history.
        if len(failed_h1) == len(DEFAULT_INSTRUMENTS):
            m5 = getattr(mt5, "TIMEFRAME_M5", None)
            if m5 is not None:
                print("\nAll H1 requests returned no usable bars; probing M5 once per discovered symbol.",
                      flush=True)
                for requested, actual in failed_h1:
                    if actual == "symbol unavailable":
                        continue
                    print(f"\n{requested} fallback timeframe:", flush=True)
                    _print_rates(mt5, actual, m5, "TIMEFRAME_M5")
        return 0
    except Exception as exc:
        print(f"Diagnostic failed: {type(exc).__name__}: {exc}", flush=True)
        try:
            import MetaTrader5 as mt5
            print(f"MT5 last_error: {_format_error(mt5)}", flush=True)
        except ImportError:
            pass
        return 1
    finally:
        connector.close()
        print("MT5 connection shut down.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
