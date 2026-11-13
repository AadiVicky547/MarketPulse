"""Bounded live audit of dashboard data through the shared service architecture.

This diagnostic starts one MarketDataManager and the same OverviewDataService
used by the dashboard. It does not instantiate provider connectors directly,
does not call AlertService.evaluate(), and never generates alerts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import INSTRUMENTS  # noqa: E402
from dashboard.overview import OverviewDataService  # noqa: E402
from data.market_manager import MarketDataManager  # noqa: E402
from storage.database import AlertDatabase  # noqa: E402


def _available_metric_count(result: Any) -> tuple[int, int]:
    metrics = getattr(result, "metrics", {}) or {}
    values = tuple(metrics.values()) if hasattr(metrics, "values") else ()
    return sum(str(getattr(item, "status", "")).upper() == "AVAILABLE" for item in values), len(values)


def _available_feature_count(result: Any) -> tuple[int, int]:
    features = getattr(result, "features", {}) or {}
    values = tuple(features.values()) if hasattr(features, "values") else ()
    return sum(str(getattr(item, "status", "")).upper() == "AVAILABLE" for item in values), len(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=20,
                        help="Maximum bounded live collection interval (1-60 seconds)")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 60:
        parser.error("--seconds must be between 1 and 60")

    manager = MarketDataManager()
    service: OverviewDataService | None = None
    data = None
    failures: list[str] = []
    try:
        manager.start()
        service = OverviewDataService(manager)
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            data = service.refresh()
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
        # ``data`` is the latest result returned by refresh() during the
        # bounded loop; avoid scheduling an additional analytics cycle here.
        snapshot = data.snapshot
        instruments = getattr(snapshot, "instruments", {})
        volatility = data.volatility
        liquidity = data.liquidity
        regimes = data.regimes
        anomalies = data.anomalies

        print(f"Dashboard data audit completed at {datetime.now(timezone.utc).isoformat()}")
        print(f"Shared architecture: MarketDataManager + OverviewDataService; "
              f"analytics_pending={data.analytics_pending}")
        print(f"Snapshot generated_at={getattr(snapshot, 'generated_at', None)}")
        for item in INSTRUMENTS:
            symbol = item.symbol
            state = instruments.get(symbol)
            vol = volatility.get(symbol)
            liq = liquidity.get(symbol)
            regime = regimes.get(symbol)
            anomaly = anomalies.get(symbol)
            vol_available, vol_total = _available_metric_count(vol)
            liq_available, liq_total = _available_metric_count(liq)
            regime_available, regime_total = _available_feature_count(regime)
            print(f"\n{symbol} [{item.source}]")
            print(f"  connection={getattr(state, 'connection_state', 'UNAVAILABLE')} "
                  f"freshness={getattr(state, 'freshness', 'UNAVAILABLE')} "
                  f"snapshot_state={'present' if state is not None else 'missing'} "
                  f"error={getattr(state, 'error_message', None)}")
            print(f"  volatility_status={getattr(vol, 'status', 'NOT_RETURNED')} "
                  f"timeframe={getattr(vol, 'timeframe', 'n/a')} "
                  f"candles={getattr(vol, 'candle_count', 0)} "
                  f"available_metrics={vol_available}/{vol_total}")
            if vol is not None:
                print(f"  volatility_warnings={getattr(vol, 'data_quality_warnings', ())}")
            print(f"  liquidity_status={getattr(liq, 'status', 'NOT_RETURNED')} "
                  f"freshness={getattr(liq, 'freshness', 'n/a')} "
                  f"available_metrics={liq_available}/{liq_total} "
                  f"book_sync={getattr(liq, 'order_book_sync_state', None)}")
            if liq is not None:
                print(f"  liquidity_warnings={getattr(liq, 'warnings', ())}")
            print(f"  regime_status={getattr(regime, 'status', 'NOT_RETURNED')} "
                  f"regime={getattr(regime, 'current_regime', None)} "
                  f"freshness={getattr(regime, 'last_observation_freshness', 'n/a')} "
                  f"available_features={regime_available}/{regime_total} "
                  f"candles={getattr(regime, 'candle_count', 0)}")
            if regime is not None:
                print(f"  regime_warnings={getattr(regime, 'data_quality_warnings', ())}")
            print(f"  anomaly_status={getattr(anomaly, 'status', 'NOT_RETURNED')} "
                  f"detected={getattr(anomaly, 'anomaly_detected', None)} "
                  f"prior_training_observations={getattr(anomaly, 'sample_count', 0)} "
                  f"current_valid_features={len(getattr(anomaly, 'available_features', ()) or getattr(anomaly, 'feature_values', {}) or ())} "
                  f"model_features={len(getattr(anomaly, 'features_used', ()) or ())} "
                  f"freshness={getattr(anomaly, 'freshness', 'n/a')}")
            if anomaly is not None:
                print(f"  anomaly_feature_values={getattr(anomaly, 'feature_values', {})}")
            if anomaly is not None:
                print(f"  anomaly_warnings={getattr(anomaly, 'data_quality_warnings', ())}")

        try:
            database = AlertDatabase()
            alerts = database.fetch_alerts(limit=1000)
            print(f"\nAlert database path: {database.path.resolve()}")
            print(f"Persisted records in newest-1000 query: {len(alerts)}; "
                  f"unacknowledged={sum(not row['acknowledged'] for row in alerts)}; "
                  f"acknowledged={sum(row['acknowledged'] for row in alerts)}")
        except Exception as exc:
            failures.append(f"Alert database read failed: {type(exc).__name__}: {exc}")

        failures.extend(str(error) for error in data.errors)
        if failures:
            print("\nErrors:")
            for failure in failures:
                print(f"  - {failure}")
            return 1
        return 0
    except Exception as exc:
        print(f"Dashboard data audit failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if service is not None:
            service.close()
        manager.stop()


if __name__ == "__main__":
    raise SystemExit(main())
