"""Bounded live verification of the MarketPulse liquidity analytics service.

Run manually from the project root. This starts the configured market manager,
reads public market data only, and never places orders.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import logging
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics.liquidity_service import (  # noqa: E402
    DISCONNECTED,
    LiquidityService,
    STALE,
    UNAVAILABLE,
)
from config import INSTRUMENTS  # noqa: E402
from data.market_manager import MarketDataManager  # noqa: E402


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _format_value(value: Any) -> str:
    if value is None:
        return "unavailable"
    try:
        return f"{float(value):.10g}"
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _print_report(snapshot: Any, results: Mapping[str, Any]) -> None:
    """Print all configured instruments and their service metrics."""
    print("\nMARKETPULSE — Liquidity and Order Flow Integration Report")
    for configured in INSTRUMENTS:
        symbol = configured.symbol.upper()
        print(f"{'=' * 78}\n{symbol} | source={configured.source}")
        try:
            _print_instrument(snapshot, results, configured, symbol)
        except Exception as exc:
            print(f"Reporting error: {type(exc).__name__}: {exc}")
            print("The remaining instruments will still be reported.")


def _print_instrument(snapshot: Any, results: Mapping[str, Any], configured: Any,
                      symbol: str) -> None:
    result = results.get(symbol)
    state = _field(_field(snapshot, "instruments", {}), symbol)
    if result is None:
        print("Result: UNAVAILABLE (liquidity result missing)")
        print("Connection: unavailable | Freshness: unavailable")
        return

    print(f"Result: {_field(result, 'status', UNAVAILABLE)} | "
          f"Connection: {_field(result, 'connection_state', 'UNAVAILABLE')} | "
          f"Freshness: {_field(result, 'freshness', 'UNAVAILABLE')}")
    print(f"Timestamp trust: {_field(result, 'timestamp_trust', 'not reported')}")
    if configured.source == "MT5":
        print("Tick-volume semantics: broker-observed activity; not centralized traded volume")
    else:
        print("Displayed order-book liquidity is cancellable and not guaranteed executable.")
        book = _field(_field(snapshot, "binance_order_books", {}), symbol, {})
        book_status = _field(book, "status", {})
        synchronized = bool(_field(book_status, "synchronized", False))
        stale = bool(_field(book_status, "stale", True))
        book_label = "STALE" if synchronized and stale else (
            "SYNCHRONIZED" if synchronized and not stale else "UNSYNCHRONIZED")
        trades = _field(_field(snapshot, "binance_trades", {}), symbol, ()) or ()
        bids = _field(book, "bids", ()) or ()
        asks = _field(book, "asks", ()) or ()
        print(f"Order book: {book_label} | trade-buffer count={len(trades)} | "
              f"available depth levels: bids={len(bids)}, asks={len(asks)}")

    metrics = _field(result, "metrics", {})
    print("Metrics:")
    if isinstance(metrics, Mapping) and metrics:
        for name, metric in metrics.items():
            status = _field(metric, "status", UNAVAILABLE)
            value = _field(metric, "value")
            unit = _field(metric, "unit", "")
            sample_count = _field(metric, "sample_count", "unknown")
            method = _field(metric, "methodology", "not reported")
            warning = _field(metric, "warning")
            value_text = _format_value(value)
            suffix = f" {unit}" if unit else ""
            print(f"  {name}: {status} | value={value_text}{suffix} | samples={sample_count}")
            if method:
                print(f"    Method: {method}")
            if warning:
                print(f"    Note: {warning}")
    else:
        print("  No metric results available.")

    warnings = _field(result, "warnings", ()) or ()
    if isinstance(warnings, str):
        warnings = (warnings,)
    print("Data-quality warnings:")
    if warnings:
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("  - none")
    error = _field(state, "error_message")
    if error:
        print(f"Market-data status detail: {error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=12.0,
                        help="Live collection warm-up in seconds (1-60; default: 12)")
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= 60:
        parser.error("--seconds must be between 1 and 60")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    manager = MarketDataManager()
    try:
        print("Starting the existing MT5 and Binance market-data workers...", flush=True)
        manager.start()
        service = LiquidityService(manager)
        deadline = time.monotonic() + args.seconds
        results = service.analyze()
        while manager.is_running() and time.monotonic() < deadline:
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
            # Repeated bounded analysis samples populate spread/depth baselines.
            results = service.analyze()
        snapshot = manager.get_market_snapshot()
        results = service.analyze()
        _print_report(snapshot, results)

        failed_statuses = {UNAVAILABLE, STALE, DISCONNECTED}
        return 1 if any(_field(result, "status") in failed_statuses
                        for result in results.values()) else 0
    except KeyboardInterrupt:
        print("Verification interrupted; shutting down market-data workers.", flush=True)
        return 130
    except Exception as exc:
        print(f"Integration verification failed: {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        try:
            manager.stop()
        except Exception as exc:
            print(f"Market manager shutdown error: {type(exc).__name__}: {exc}", flush=True)
        print("Market data manager stopped.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
