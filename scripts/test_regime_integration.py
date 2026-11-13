"""Bounded live integration check for MarketPulse regime detection.

Run manually from the project root. The script starts the existing market-data
workers, reads live historical candles, and never places orders.
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

from analytics.regime_service import RegimeService  # noqa: E402
from config import INSTRUMENTS  # noqa: E402
from data.market_manager import MarketDataManager  # noqa: E402


TIMEFRAME = "1h"
CANDLE_COUNT = 600


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _value(value: Any) -> str:
    """Format None as unavailable while preserving a genuine numeric zero."""
    if value is None:
        return "unavailable"
    try:
        return f"{float(value):.10g}"
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _print_report(snapshot: Any, results: Mapping[str, Any]) -> bool:
    """Print each configured symbol independently; return whether any failed."""
    print("\nMARKETPULSE — Market Regime Integration Report")
    print(f"Timeframe: {TIMEFRAME} | requested candles: {CANDLE_COUNT}")
    print("Evidence strength describes rule support; it is not a probability.")
    failed = False
    instrument_states = _field(snapshot, "instruments", {})
    source_states = _field(snapshot, "sources", {})

    for configured in INSTRUMENTS:
        symbol = configured.symbol.upper()
        print(f"{'=' * 78}\n{symbol} | provider={configured.source}")
        try:
            result = results.get(symbol) if isinstance(results, Mapping) else None
            state = _field(instrument_states, symbol)
            source_state = _field(source_states, configured.source)
            connection = _field(state, "connection_state",
                                _field(source_state, "connection_state", "UNAVAILABLE"))
            freshness = (_field(result, "last_observation_freshness") if result is not None else None)
            if freshness is None:
                freshness = _field(state, "freshness", "UNAVAILABLE")
            print(f"Connection: {connection} | Freshness: {freshness}")

            if result is None:
                print("Status: UNAVAILABLE (no result returned)")
                print("Selected regime: unavailable")
                print("Evidence strength: unavailable")
                print("Timestamp trust: not reported")
                print("Explanation: no regime result was returned for this instrument.")
                failed = True
                continue

            status = _field(result, "status", "UNAVAILABLE")
            regime = _field(result, "current_regime")
            strength = _field(result, "evidence_strength")
            print(f"Status: {status}")
            print(f"Selected regime: {regime if regime is not None else 'unavailable'}")
            print(f"Evidence strength: {strength if strength is not None else 'unavailable'} "
                  "(rule-support category; not a probability)")
            print(f"Timestamp trust: {_field(result, 'timestamp_trust', 'not reported')}")
            print(f"Candles: {_field(result, 'candle_count', 'unknown')}")

            features = _field(result, "features", {})
            print("Features:")
            if isinstance(features, Mapping) and features:
                for name, feature in features.items():
                    feature_status = _field(feature, "status", "UNAVAILABLE")
                    feature_value = _value(_field(feature, "value"))
                    unit = _field(feature, "unit", "")
                    sample_count = _field(feature, "sample_count", "unknown")
                    suffix = f" {unit}" if unit else ""
                    print(f"  {name}: {feature_status} | value={feature_value}{suffix} | "
                          f"samples={sample_count}")
                    methodology = _field(feature, "methodology")
                    warning = _field(feature, "warning")
                    if methodology:
                        print(f"    Method: {methodology}")
                    if warning:
                        print(f"    Note: {warning}")
            else:
                print("  unavailable (no feature values returned)")

            explanation = _field(result, "explanation")
            print(f"Explanation: {explanation or 'unavailable'}")
            warnings = _field(result, "data_quality_warnings", ()) or ()
            if isinstance(warnings, str):
                warnings = (warnings,)
            print("Data-quality warnings:")
            if warnings:
                for warning in warnings:
                    print(f"  - {warning}")
            else:
                print("  - none")
            if status in {"UNAVAILABLE", "DISCONNECTED", "STALE"}:
                failed = True
        except Exception as exc:
            print(f"Reporting error: {type(exc).__name__}: {exc}")
            print("The remaining instruments will still be reported.")
            failed = True
    return failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=12.0,
                        help="Market-manager warm-up in seconds (1-60; default: 12)")
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= 60:
        parser.error("--seconds must be between 1 and 60")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    manager = MarketDataManager()
    try:
        print("Starting the existing MT5 and Binance market-data workers...", flush=True)
        manager.start()
        deadline = time.monotonic() + args.seconds
        while manager.is_running() and time.monotonic() < deadline:
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))

        # RegimeService reads manager snapshots only for receive-time freshness;
        # candles are fetched through its existing CandleService integration.
        service = RegimeService(market_manager=manager)
        results = service.analyze_all(TIMEFRAME, CANDLE_COUNT)
        snapshot = manager.get_market_snapshot()
        failed = _print_report(snapshot, results)
        print(f"\nCompleted regime report for {len(INSTRUMENTS)} configured instruments.")
        return 1 if failed else 0
    except KeyboardInterrupt:
        print("Integration test interrupted; stopping market-data workers.", flush=True)
        return 130
    except Exception as exc:
        print(f"Integration test failed: {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        try:
            manager.stop()
        except Exception as exc:
            print(f"Market-manager shutdown error: {type(exc).__name__}: {exc}", flush=True)
        print("Market data manager stopped.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
