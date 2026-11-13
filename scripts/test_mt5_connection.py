"""Manually verify the local MetaTrader 5 terminal and genuine latest ticks."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.mt5_connector import DEFAULT_INSTRUMENTS, MT5Connector


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    connector = MT5Connector()
    try:
        status = connector.initialize()
        if not status.connected:
            print(f"MT5 unavailable: {status.message}. {status.error or ''}".strip())
            return 1
        print("MT5 terminal connected.")
        try:
            matches = connector.discover_symbols(DEFAULT_INSTRUMENTS)
        except Exception as exc:
            print(f"Instrument discovery failed: {exc}")
            return 1
        failed = False
        for requested, match in matches.items():
            if not match.actual_symbol:
                failed = True
                print(f"{requested}: UNAVAILABLE — {match.message}")
                continue
            print(f"{requested}: broker symbol {match.actual_symbol} ({match.match_type})")
            try:
                tick = connector.get_tick(requested)
            except Exception as exc:
                failed = True
                print(f"  Tick unavailable: {exc}")
                continue
            if not tick["available"]:
                failed = True
                print(f"  Tick unavailable: {tick['error']}")
                continue
            if tick["stale"]:
                failed = True
                print(f"  STALE tick at {tick['timestamp'].isoformat()} UTC; values shown:")
            print(f"  Bid: {tick['bid']} | Ask: {tick['ask']} | Spread: {tick['spread_price']} price units")
            print(f"  Tick volume: {tick['tick_volume']} | Timestamp: {tick['timestamp'].isoformat()} UTC")
        return 1 if failed else 0
    finally:
        connector.close()
        print("MT5 connection shut down.")


if __name__ == "__main__":
    raise SystemExit(main())
