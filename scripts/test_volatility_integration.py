"""Live, bounded verification of candle-to-volatility integration.

Run from the project root with the project's Python environment. This script
uses the configured CandleService providers and does not place orders.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
from collections.abc import Mapping
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics.volatility_service import (  # noqa: E402
    STATUS_AVAILABLE,
    STATUS_TIMESTAMP_UNVERIFIED,
    STATUS_UNAVAILABLE,
    STATUS_WARMING_UP,
    VolatilityService,
)
from config import INSTRUMENTS  # noqa: E402
from data.candle_service import CandleService  # noqa: E402


TIMEFRAME = "1h"
CANDLE_COUNT = 200
EXPECTED_INTERVAL = pd.Timedelta(hours=1)


class _CapturedCandleService:
    """Replay one real candle-service response to the volatility service."""

    def __init__(self, captured: dict[str, pd.DataFrame | Exception]) -> None:
        self._captured = captured

    def get_candles(self, instrument: str, timeframe: str, count: int, *,
                    include_incomplete: bool = False) -> pd.DataFrame:
        value = self._captured.get(instrument.upper())
        if isinstance(value, Exception):
            raise value
        if value is None:
            raise RuntimeError("No captured candle response for instrument")
        return value.copy()

    def close(self) -> None:
        """No provider resources are owned by this in-memory replay adapter."""


def _print_cadence_report(symbol: str, frame: Any) -> None:
    """Print timestamp cadence without changing timezones or trust labels."""
    print(f"Timestamp cadence: {symbol}")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        print("  Candle timestamps unavailable.")
        return
    if "timestamp" not in frame.columns:
        print("  Candle timestamps unavailable: timestamp column missing.")
        return

    try:
        stamps = [pd.Timestamp(value) for value in frame["timestamp"]]
        deltas = [stamps[index] - stamps[index - 1] for index in range(1, len(stamps))]
    except Exception as exc:
        print(f"  Candle timestamps could not be compared: {type(exc).__name__}: {exc}")
        return

    trust_values = frame["timestamp_trust"] if "timestamp_trust" in frame.columns else None
    first_trust = trust_values.iloc[0] if trust_values is not None else "not reported"
    last_trust = trust_values.iloc[-1] if trust_values is not None else "not reported"
    print(f"  First candle: {stamps[0].isoformat()} | trust={first_trust}")
    print(f"  Last candle:  {stamps[-1].isoformat()} | trust={last_trust}")

    duplicates = int(frame["timestamp"].duplicated().sum())
    strictly_increasing = all(delta > pd.Timedelta(0) for delta in deltas)
    print(f"  Strictly increasing: {'yes' if strictly_increasing else 'no'}")
    print(f"  Duplicate timestamps: {'yes' if duplicates else 'no'} (extra duplicate rows={duplicates})")

    frequencies = Counter(deltas)
    if frequencies:
        print("  Distinct observed intervals (frequency):")
        for interval, frequency in sorted(frequencies.items(), key=lambda item: item[0]):
            print(f"    {interval}: {frequency}")
    else:
        print("  Distinct observed intervals (frequency): none; fewer than two candles")

    one_hour = sum(delta == EXPECTED_INTERVAL for delta in deltas)
    larger = sum(delta > EXPECTED_INTERVAL for delta in deltas)
    other = len(deltas) - one_hour - larger
    print(f"  One-hour intervals: {one_hour}")
    print(f"  Larger-than-one-hour intervals: {larger}")
    print(f"  Other non-one-hour intervals (shorter/non-positive): {other}")
    for index, delta in enumerate(deltas, start=1):
        if delta != EXPECTED_INTERVAL:
            print(f"  Non-one-hour interval ({delta}): {stamps[index - 1].isoformat()} -> "
                  f"{stamps[index].isoformat()}")


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a dataclass-like or mapping field without assuming its shape."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _text(value: Any, default: str) -> str:
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def _metric_status(metric: Any) -> str:
    if metric is None:
        return "UNAVAILABLE: metric result is missing"
    status = _text(_field(metric, "status"), "UNAVAILABLE").upper()
    value = _field(metric, "value")
    unit = _text(_field(metric, "unit"), "")
    sample_count = _field(metric, "sample_count")
    warning = _text(_field(metric, "warning"), "")
    count_text = f"n={sample_count}" if sample_count is not None else "sample count unavailable"
    if status == STATUS_AVAILABLE:
        if value is None:
            return f"{status}: value unavailable ({count_text})"
        try:
            value_text = f"{float(value):.10g}"
        except (TypeError, ValueError, OverflowError):
            value_text = _text(value, "value unavailable")
        suffix = f" {unit}" if unit else ""
        return f"{status}: {value_text}{suffix} ({count_text})"
    if status == STATUS_WARMING_UP:
        detail = warning or "insufficient history for configured window"
    elif status == STATUS_TIMESTAMP_UNVERIFIED:
        detail = warning or "requires trusted absolute timestamps"
    elif status == STATUS_UNAVAILABLE:
        detail = warning or "data or calculation unavailable"
    else:
        detail = warning or "unrecognized metric status"
    return f"{status}: {detail} ({count_text})"


def _retrieval_status(result: Any) -> str:
    candle_count = _field(result, "candle_count", 0)
    try:
        candle_count = max(0, int(candle_count or 0))
    except (TypeError, ValueError, OverflowError):
        return "UNAVAILABLE (invalid candle count in result)"
    warnings = _warnings(result)
    if candle_count == 0:
        if any("candle validation failed" in warning.casefold() for warning in warnings):
            return "REJECTED (candle validation failed; see exact reason below)"
        if any("candle retrieval failed" in warning.casefold() for warning in warnings):
            return "FAILED (candle retrieval error; see details below)"
        return "UNAVAILABLE (no candles returned)"
    validation_failed = any("candle validation failed" in warning.casefold() for warning in warnings)
    if validation_failed:
        return f"REJECTED ({candle_count} candles returned; see exact reason below)"
    requested = _field(result, "requested_candle_count")
    requested_text = str(requested) if requested is not None else "?"
    return f"RETRIEVED ({candle_count}/{requested_text} requested)"


def _warnings(result: Any) -> list[str]:
    value = _field(result, "data_quality_warnings", ())
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [_text(item, "<warning could not be rendered>") for item in value]
    except TypeError:
        return [_text(value, "<warning could not be rendered>")]


def _print_instrument_report(symbol: str, configured_source: str, result: Any) -> bool:
    """Print one result defensively; return whether its overall result failed."""
    print(f"{'=' * 72}\n{symbol} | Source: {_text(_field(result, 'source'), configured_source)}")
    status = _text(_field(result, "status"), "UNAVAILABLE").upper()
    print(f"Overall status: {status}")
    print(f"Candle retrieval: {_retrieval_status(result)}")
    print(f"Timestamp trust: {_text(_field(result, 'timestamp_trust'), 'not reported')}")

    warnings = _warnings(result)
    print("Data-quality warnings:")
    if warnings:
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("  - none")

    metrics = _field(result, "metrics", {})
    print("Volatility metrics:")
    if isinstance(metrics, Mapping):
        for name, metric in metrics.items():
            try:
                rendered = _metric_status(metric)
            except Exception as exc:
                rendered = f"UNAVAILABLE: metric could not be rendered ({type(exc).__name__}: {exc})"
            print(f"  - {_text(name, 'unnamed metric')}: {rendered}")
    elif metrics is None:
        print("  - Metric results unavailable (metrics field is None)")
    else:
        try:
            items = metrics.items()
        except (AttributeError, TypeError):
            print(f"  - Unexpected metric result structure: {type(metrics).__name__}")
        else:
            try:
                for name, metric in items:
                    print(f"  - {_text(name, 'unnamed metric')}: {_metric_status(metric)}")
            except Exception as exc:
                print(f"  - Metric rendering stopped for malformed structure: {type(exc).__name__}: {exc}")

    return status == STATUS_UNAVAILABLE


def main() -> int:
    """Fetch and report current historical volatility inputs and results."""
    print("MARKETPULSE — Volatility Integration Verification")
    print(f"Timeframe: {TIMEFRAME} | Requested candles: {CANDLE_COUNT}")
    print("MT5 event-time basis remains unverified; no timestamp correction is applied.\n")

    candle_service = CandleService()
    captured: dict[str, pd.DataFrame | Exception] = {}
    try:
        for instrument in INSTRUMENTS:
            symbol = instrument.symbol.upper()
            try:
                captured[symbol] = candle_service.get_candles(
                    symbol, TIMEFRAME, CANDLE_COUNT, include_incomplete=False)
            except Exception as exc:
                captured[symbol] = exc
            if instrument.source == "MT5":
                value = captured[symbol]
                if isinstance(value, Exception):
                    print(f"Timestamp cadence: {symbol}\n  Candle retrieval failed: "
                          f"{type(value).__name__}: {value}")
                else:
                    _print_cadence_report(symbol, value)
    finally:
        try:
            candle_service.close()
        except Exception as exc:
            print(f"Candle service close warning: {type(exc).__name__}: {exc}")

    try:
        results = VolatilityService(
            candle_service_factory=lambda: _CapturedCandleService(captured)
        ).analyze_all(TIMEFRAME, CANDLE_COUNT)
    except Exception as exc:
        print(f"Integration run failed: {type(exc).__name__}: {exc}")
        return 1

    failed = False
    reported = 0
    for instrument in INSTRUMENTS:
        symbol = instrument.symbol.upper()
        reported += 1
        try:
            result = results.get(symbol) if isinstance(results, Mapping) else None
            if result is None:
                print(f"{'=' * 72}\n{symbol} | Source: {instrument.source}")
                print("Overall status: UNAVAILABLE (no result returned)")
                print("Candle retrieval: UNAVAILABLE")
                print("Timestamp trust: not reported")
                print("Volatility metrics: unavailable")
                print("Data-quality warnings: no instrument result returned")
                failed = True
            else:
                failed = _print_instrument_report(symbol, instrument.source, result) or failed
        except Exception as exc:
            print(f"{'=' * 72}\n{symbol} | Source: {instrument.source}")
            print(f"Reporting error: {type(exc).__name__}: {exc}")
            print("Remaining instruments will still be reported.")
            failed = True

    print(f"\nCompleted report for {reported}/{len(INSTRUMENTS)} configured instruments.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
