"""Bounded live smoke check for public Binance Spot market data."""
from __future__ import annotations

import argparse
import asyncio
from decimal import Decimal
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.binance_connector import DEFAULT_SYMBOLS, BinanceMarketDataConnector


def show_value(value):
    return str(value) if isinstance(value, Decimal) else value


async def verify(seconds: int = 25) -> int:
    connector = BinanceMarketDataConnector()
    print("Starting public Binance streams...", flush=True)
    await connector.start()
    try:
        print(f"Collecting public BTCUSDT and ETHUSDT market data for {seconds} seconds...", flush=True)
        await asyncio.sleep(seconds)
        failed = False
        for symbol in DEFAULT_SYMBOLS:
            status = connector.connection_status(symbol)["symbols"][symbol]
            print(f"\n{symbol}: connection={'connected' if status['connected'] else 'disconnected'}, "
                  f"fresh={'no' if status['stale'] else 'yes'}", flush=True)
            trades = connector.latest_trades(symbol, 1)
            if trades:
                trade = trades[-1]
                age = asyncio.get_running_loop().time() - trade["received_monotonic"]
                print(f"  Genuine trade: id={trade['trade_id']} price={trade['price']} qty={trade['quantity']} "
                      f"quote_qty={trade['quote_quantity']} time={trade['timestamp'].isoformat()} "
                  f"buyer_is_maker={trade['buyer_is_maker']} aggressor={trade['aggressor_side']} age={age:.2f}s", flush=True)
            else:
                failed = True
                print("  Trade: unavailable (no trade event received during check)", flush=True)
            ticker = connector.latest_book_ticker(symbol)
            if ticker and not ticker["stale"]:
                print(f"  Genuine best bid={ticker['bid_price']} qty={ticker['bid_quantity']} "
                      f"ask={ticker['ask_price']} qty={ticker['ask_quantity']} mid={ticker['mid_price']} "
                      f"spread={ticker['spread']} spread_bps={ticker['spread_bps']}", flush=True)
            else:
                failed = True
                print("  Best bid/ask: unavailable or stale", flush=True)
            metrics = connector.order_book_metrics(symbol, 5)
            if metrics["available"]:
                print(f"  Order book: synchronized update_id={metrics['status'].update_id}; "
                      f"mid={show_value(metrics['mid_price'])} spread={show_value(metrics['spread'])} "
                      f"OBI={show_value(metrics['obi'])}", flush=True)
                print("  Top 5 bids:", [(show_value(row['price']), show_value(row['quantity'])) for row in metrics['bids']], flush=True)
                print("  Top 5 asks:", [(show_value(row['price']), show_value(row['quantity'])) for row in metrics['asks']], flush=True)
                print(f"  Note: {metrics['liquidity_note']}", flush=True)
            else:
                failed = True
                print(f"  Order book unavailable: synchronized={metrics['status'].synchronized}, "
                      f"stale={metrics['status'].stale}, reason={metrics['message']}", flush=True)
        return 1 if failed else 0
    finally:
        await connector.stop()
        print("\nBinance WebSocket and synchronization tasks shut down.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=25, help="Verification duration (1-30 seconds)")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 30:
        parser.error("--seconds must be between 1 and 30")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    print("Launching bounded verification loop...", flush=True)
    return asyncio.run(verify(args.seconds))


if __name__ == "__main__":
    raise SystemExit(main())
