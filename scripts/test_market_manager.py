"""Bounded integration check of all six instruments through real connectors."""
from __future__ import annotations

import argparse
from decimal import Decimal
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.market_manager import MarketDataManager


def show(value: Decimal | None) -> str:
    return "unavailable" if value is None else str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=12, help="Bounded collection interval (1-30 seconds)")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 30:
        parser.error("--seconds must be between 1 and 30")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    market_manager = MarketDataManager()
    print("Starting independent MT5 and Binance workers...", flush=True)
    market_manager.start()
    try:
        print(f"Collecting unified MT5 and Binance state for at most {args.seconds} seconds...", flush=True)
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline and market_manager.is_running():
            time.sleep(0.2)
        snapshot = market_manager.get_market_snapshot()
        failed = False
        for symbol, state in snapshot.instruments.items():
            print(f"{symbol} [{state.data_source}] {state.freshness} / connection={state.connection_state} "
                  f"actual={state.actual_symbol or 'unavailable'}")
            print(f"  bid={show(state.bid)} ask={show(state.ask)} mid={show(state.mid_price)} "
                  f"spread={show(state.spread)} spread_bps={show(state.spread_bps)}")
            print(f"  last={show(state.last_trade_price)} trade_qty={show(state.trade_quantity)} "
                  f"tick_volume={state.tick_volume if state.tick_volume is not None else 'unavailable'} "
                  f"tick_time={state.tick_timestamp.isoformat() if state.tick_timestamp else 'unavailable'}")
            print(f"  received={state.last_update_received_at.isoformat() if state.last_update_received_at else 'unavailable'} "
                  f"book={state.order_book_sync_state or 'not applicable'} "
                  f"status={state.error_message or 'OK'}")
            if state.freshness != "LIVE":
                failed = True
        for source, status in snapshot.sources.items():
            print(f"SOURCE {source}: {status.connection_state} — {status.message}; "
                  f"error={status.last_error or 'none'}", flush=True)
        if failed:
            print("Integration check: one or more instruments were unavailable, stale, or disconnected.")
            return 1
        print("Integration check: all six instruments reported live data during this run.")
        return 0
    finally:
        market_manager.stop()
        print("Market data manager stopped.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
