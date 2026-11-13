"""Bounded live integration check for MarketPulse anomaly detection.

Run manually from the project root. The script starts configured market-data
workers and uses public market data only; it never places orders. Each sample is
scored through AnomalyService. Hourly candle analytics are cached for this run
to avoid repeating unchanged historical requests at the live sampling cadence.
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

from analytics.anomaly_service import (  # noqa: E402
    AVAILABLE,
    ERROR,
    UNAVAILABLE,
    WARMING_UP,
    AnomalyService,
    AnomalySettings,
)
from analytics.volatility_service import VolatilityService  # noqa: E402
from config import INSTRUMENTS  # noqa: E402
from data.market_manager import MarketDataManager  # noqa: E402


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _text(value: Any, default: str = "unavailable") -> str:
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def _number(value: Any) -> str:
    if value is None:
        return "unavailable"
    try:
        return f"{float(value):.10g}"
    except (TypeError, ValueError, OverflowError):
        return _text(value)


class _RunCachedVolatilityService:
    """Cache historical candle-derived metrics while the live sample loop runs."""

    def __init__(self, service: Any) -> None:
        self._service = service
        self._results: Mapping[str, Any] | None = None

    def analyze_all(self, timeframe: str, count: int) -> Mapping[str, Any]:
        if self._results is None:
            self._results = self._service.analyze_all(timeframe, count)
        return self._results


def _print_report(snapshot: Any, results: Mapping[str, Any]) -> bool:
    """Print the latest result for every configured instrument defensively."""
    print("\nMARKETPULSE — Anomaly Detection Integration Report")
    print("An anomaly is an unusual observation, not necessarily harmful and not a trading signal.")
    print("WARMING_UP means no decision was made; it must not be read as normal.")
    print("Anomaly score is the negated Isolation Forest decision function; higher is more unusual.")
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
            if result is None:
                print("Status: UNAVAILABLE (no result returned)")
                print(f"Connection: {_text(_field(state, 'connection_state', _field(source_state, 'connection_state')))}")
                print(f"Freshness: {_text(_field(state, 'freshness'))}")
                print("Model status: unavailable | prior samples: 0")
                print("Anomaly: not scored")
                print("Features: unavailable")
                print("Explanation: no anomaly result was returned for this instrument.")
                print("Timestamp trust: not reported")
                print("Data-quality warnings: no instrument result returned")
                failed = True
                continue

            status = _text(_field(result, "status"), UNAVAILABLE).upper()
            connection = _field(result, "connection_state",
                                _field(state, "connection_state",
                                       _field(source_state, "connection_state", "UNAVAILABLE")))
            freshness = _field(result, "freshness", _field(state, "freshness", "UNAVAILABLE"))
            model_status = _field(result, "model_status", "not reported")
            sample_count = _field(result, "sample_count", "unknown")
            print(f"Status: {status}")
            print(f"Connection: {_text(connection)} | Freshness: {_text(freshness)}")
            print(f"Model status: {_text(model_status)} | prior complete samples: {sample_count}")

            anomaly = _field(result, "anomaly_detected")
            if status == AVAILABLE and anomaly is not None:
                anomaly_text = "YES — unusual observation" if bool(anomaly) else "NO — not flagged as unusual"
            elif status == WARMING_UP:
                anomaly_text = "not scored (warming up; not a normal classification)"
            else:
                anomaly_text = "not scored"
            print(f"Anomaly detected: {anomaly_text}")
            print(f"Anomaly score: {_number(_field(result, 'anomaly_score'))}")
            print(f"Model decision: {_text(_field(result, 'model_decision'))}")

            features = _field(result, "features_used", ()) or ()
            feature_values = _field(result, "feature_values", {})
            print("Features used:")
            if features:
                for name in features:
                    value = _field(feature_values, name)
                    print(f"  - {name}: {_number(value)}")
            else:
                print("  - none")
            print(f"Explanation: {_text(_field(result, 'explanation'))}")
            print(f"Timestamp trust: {_text(_field(result, 'timestamp_trust'), 'not reported')}")

            warnings = _field(result, "data_quality_warnings", ()) or ()
            if isinstance(warnings, str):
                warnings = (warnings,)
            print("Data-quality warnings:")
            if warnings:
                for warning in warnings:
                    print(f"  - {_text(warning, '<warning unavailable>')}")
            else:
                print("  - none")
            if status in {UNAVAILABLE, ERROR}:
                failed = True
        except Exception as exc:
            print(f"Reporting error: {type(exc).__name__}: {exc}")
            print("The remaining instruments will still be reported.")
            failed = True
    return failed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between anomaly samples (default: 1)")
    parser.add_argument("--duration", type=float, default=90.0,
                        help="Total sampling duration in seconds (default: 90; maximum: 3600)")
    args = parser.parse_args(argv)
    if not 0.1 <= args.interval <= 60:
        parser.error("--interval must be between 0.1 and 60 seconds")
    if not args.interval <= args.duration <= 3600:
        parser.error("--duration must be at least --interval and no more than 3600 seconds")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manager = MarketDataManager()
    try:
        print("Starting the existing MT5 and Binance market-data workers...", flush=True)
        manager.start()
        analyzer = AnomalyService(
            manager,
            volatility_service=_RunCachedVolatilityService(VolatilityService()),
            settings=AnomalySettings(),
        )
        print(f"Sampling every {args.interval:g}s for {args.duration:g}s. "
              "Default model warm-up requires 50 prior complete observations per instrument.",
              flush=True)

        deadline = time.monotonic() + args.duration
        results: Mapping[str, Any] = {}
        samples = 0
        while time.monotonic() < deadline:
            results = analyzer.analyze()
            samples += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(args.interval, remaining))

        snapshot = manager.get_market_snapshot()
        failed = _print_report(snapshot, results)
        print(f"\nCompleted {samples} analysis sample(s) for "
              f"{len(INSTRUMENTS)} configured instruments.")
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
