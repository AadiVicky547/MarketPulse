"""Bounded live historical-candle check for all configured instruments."""
from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.candle_service import CandleService


def _collect_one(instrument: str, timeframe: str, count: int, output: Any) -> None:
    """Run one provider request in a killable child for bounded verification."""
    service = CandleService()
    try:
        candles = service.get_candles(instrument, timeframe, count, include_incomplete=True)
        if candles.empty:
            output.put({"error": "provider returned no candles"})
            return
        first, last = candles.iloc[0], candles.iloc[-1]
        completed = candles["completed"]
        known = completed.notna()
        output.put({
            "actual_symbol": last["actual_symbol"] or instrument,
            "count": len(candles),
            "completed_count": int(completed.fillna(False).sum()),
            "incomplete_count": int((completed.eq(False) & known).sum()),
            "completion_unknown_count": int((~known).sum()),
            "timestamp_trust": str(last["timestamp_trust"]),
            "first_timestamp": first["timestamp"].isoformat(),
            "first_completion_status": first["completion_status"],
            "last_timestamp": last["timestamp"].isoformat(),
            "last_completion_status": last["completion_status"],
        })
    except Exception as exc:
        output.put({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        service.close()


def _bounded_fetch(instrument: str, timeframe: str, count: int, timeout: float) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    output = context.Queue(maxsize=1)
    process = context.Process(target=_collect_one, args=(instrument, timeframe, count, output))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        output.close()
        output.join_thread()
        return {"error": f"provider request exceeded {timeout:g}s timeout"}
    try:
        return output.get(timeout=1)
    except Exception:
        return {"error": f"provider worker exited with code {process.exitcode} without a result"}
    finally:
        output.close()
        output.join_thread()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeframe", default="1h", help="Supported candle interval (default: 1h)")
    parser.add_argument("--count", type=int, default=100, help="Requested sample size (1-1000)")
    parser.add_argument("--timeout", type=float, default=15.0,
                        help="Maximum seconds per instrument provider request")
    args = parser.parse_args()
    if not 1 <= args.count <= 1000:
        parser.error("--count must be between 1 and 1000")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    errors = 0
    for instrument, source in (("XAUUSD", "MT5"), ("EURUSD", "MT5"),
                               ("GBPUSD", "MT5"), ("USDJPY", "MT5"),
                               ("BTCUSDT", "Binance"), ("ETHUSDT", "Binance")):
        print(f"\n{instrument} | source={source} | timeframe={args.timeframe}", flush=True)
        result = _bounded_fetch(instrument, args.timeframe, args.count, args.timeout)
        if "error" in result:
            errors += 1
            print(f"  unavailable: {result['error']}", flush=True)
            continue
        print(f"  actual symbol: {result['actual_symbol']}", flush=True)
        print(f"  candles: {result['count']} (completed={result['completed_count']}, "
              f"incomplete={result['incomplete_count']}, "
              f"completion unknown={result['completion_unknown_count']})", flush=True)
        print(f"  timestamp trust: {result['timestamp_trust']}", flush=True)
        print(f"  first: {result['first_timestamp']} "
              f"({result['first_completion_status']})", flush=True)
        print(f"  last:  {result['last_timestamp']} "
              f"({result['last_completion_status']})", flush=True)
        if result["count"] < args.count:
            print(f"  note: provider returned fewer than requested ({args.count}); no candles were fabricated",
                  flush=True)
    print(f"\nVerification completed; instruments with unavailable/error data: {errors}/6", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
