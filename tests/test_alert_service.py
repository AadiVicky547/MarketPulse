"""Mocked tests for alert rules, persistence, provenance and suppression."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from analytics.alert_service import (
    ABNORMAL_SPREAD,
    AVAILABLE,
    BINANCE_ORDER_BOOK_IMBALANCE,
    CRITICAL,
    ISOLATION_FOREST_ANOMALY,
    MT5_TICK_ACTIVITY,
    REGIME_CHANGE,
    UNAVAILABLE,
    VOLATILITY_EXPANSION,
    WARNING,
    AlertService,
    AlertSettings,
)
from config import INSTRUMENTS
from storage.database import AlertDatabase


NOW = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)


def metric(value, status=AVAILABLE, samples=25):
    return SimpleNamespace(status=status, value=value, unit="test units",
                           sample_count=samples, methodology="mock method", warning=None)


def default_metrics(source):
    liquidity = {
        "spread_expansion_ratio": metric(1.0),
        "spread_anomaly_zscore": metric(0.0),
    }
    if source == "MT5":
        liquidity.update({
            "tick_volume_activity_ratio": metric(1.0),
            "tick_volume_anomaly_zscore": metric(0.0),
        })
    else:
        liquidity["order_book_imbalance"] = metric(0.0)
    return liquidity


class FakeAnalytics:
    def __init__(self, results):
        self.results = results

    def analyze(self):
        return self.results

    def analyze_all(self, timeframe, count):
        return self.results


class Fixture:
    def __init__(self, tmp_path: Path, *, settings=None):
        self.clock = [NOW]
        states = {
            item.symbol: SimpleNamespace(
                requested_instrument=item.symbol,
                freshness="LIVE",
                connection_state="CONNECTED",
                tick_timestamp=NOW - timedelta(milliseconds=100),
                tick_timestamp_trust=("UNVERIFIED_TIME_BASIS" if item.source == "MT5"
                                      else "TRUSTED_UTC"),
                tick_timestamp_source="time_msc" if item.source == "MT5" else "trade.T",
                last_update_received_at=NOW - timedelta(milliseconds=50),
                order_book_sync_state=("SYNCHRONIZED" if item.source == "Binance" else None),
            ) for item in INSTRUMENTS
        }
        self.manager = SimpleNamespace(get_market_snapshot=lambda: SimpleNamespace(
            instruments=states, sources={}))
        self.states = states
        self.volatility_results = {}
        self.liquidity_results = {}
        self.regime_results = {}
        self.anomaly_results = {}
        for item in INSTRUMENTS:
            self.volatility_results[item.symbol] = SimpleNamespace(
                instrument=item.symbol, source=item.source, timestamp_trust=(
                    "TRUSTED_UTC" if item.source == "Binance" else "UNVERIFIED_TIME_BASIS"),
                status=AVAILABLE, metrics={
                    "volatility_percentile": metric(50.0),
                    "volatility_zscore": metric(0.0),
                }, data_quality_warnings=(),
            )
            self.liquidity_results[item.symbol] = SimpleNamespace(
                instrument=item.symbol, source=item.source,
                freshness="LIVE", connection_state="CONNECTED",
                timestamp=states[item.symbol].tick_timestamp,
                received_at=states[item.symbol].last_update_received_at,
                timestamp_trust=states[item.symbol].tick_timestamp_trust,
                volume_semantics=("Broker-observed tick activity; not centralized traded volume"
                                  if item.source == "MT5" else None),
                metrics=default_metrics(item.source), warnings=(),
            )
            self.regime_results[item.symbol] = SimpleNamespace(
                instrument=item.symbol, source=item.source, status=AVAILABLE,
                current_regime="RANGING", timestamp_trust=states[item.symbol].tick_timestamp_trust,
                last_observation_freshness="LIVE", last_update_received_at=states[item.symbol].last_update_received_at,
                explanation="mock regime", data_quality_warnings=(), features={},
            )
            self.anomaly_results[item.symbol] = SimpleNamespace(
                instrument=item.symbol, provider=item.source, status=AVAILABLE,
                anomaly_detected=False, anomaly_score=-0.1, model_decision=1,
                features_used=("spread_price",), feature_values={"spread_price": 0.1},
                sample_count=50, model_status="TRAINED_ON_PRIOR_OBSERVATIONS",
                timestamp_trust=states[item.symbol].tick_timestamp_trust,
                freshness="LIVE", connection_state="CONNECTED",
                last_update_received_at=states[item.symbol].last_update_received_at,
                data_quality_warnings=(), explanation="mock normal pattern",
            )
        self.volatility = FakeAnalytics(self.volatility_results)
        self.liquidity = FakeAnalytics(self.liquidity_results)
        self.regime = FakeAnalytics(self.regime_results)
        self.anomaly = FakeAnalytics(self.anomaly_results)
        self.database = AlertDatabase(tmp_path / "alerts.sqlite3")
        self.service = AlertService(
            self.manager, volatility_service=self.volatility,
            liquidity_service=self.liquidity, regime_service=self.regime,
            anomaly_service=self.anomaly, database=self.database,
            settings=settings or AlertSettings(), clock=lambda: self.clock[0],
        )

    def set_metric(self, symbol, group, name, value, status=AVAILABLE):
        results = getattr(self, f"{group}_results")
        result = results[symbol]
        metrics = dict(result.metrics)
        metrics[name] = metric(value, status)
        results[symbol] = SimpleNamespace(**{**vars(result), "metrics": metrics})

    def evaluate(self):
        return self.service.evaluate()


def test_all_six_alert_categories_use_existing_analytics_results(tmp_path):
    fixture = Fixture(tmp_path)
    fixture.set_metric("XAUUSD", "volatility", "volatility_percentile", 99.0)
    fixture.set_metric("EURUSD", "liquidity", "spread_anomaly_zscore", 5.5)
    fixture.set_metric("GBPUSD", "liquidity", "tick_volume_activity_ratio", 4.5)
    fixture.set_metric("BTCUSDT", "liquidity", "order_book_imbalance", -0.95)
    fixture.anomaly_results["USDJPY"] = SimpleNamespace(
        **{**vars(fixture.anomaly_results["USDJPY"]), "anomaly_detected": True,
           "anomaly_score": 0.3, "model_decision": -1})

    first = fixture.evaluate()
    fixture.regime_results["ETHUSDT"] = SimpleNamespace(
        **{**vars(fixture.regime_results["ETHUSDT"]), "current_regime": "TRENDING"})
    second = fixture.evaluate()
    all_alerts = first + second
    categories = {alert.category for alert in all_alerts}
    assert categories == {
        VOLATILITY_EXPANSION, ABNORMAL_SPREAD, MT5_TICK_ACTIVITY,
        BINANCE_ORDER_BOOK_IMBALANCE, REGIME_CHANGE, ISOLATION_FOREST_ANOMALY,
    }
    assert len({alert.alert_id for alert in all_alerts}) == len(all_alerts)
    assert all(alert.detected_at_utc.tzinfo is timezone.utc for alert in all_alerts)
    assert all(alert.provider in {"MT5", "Binance"} for alert in all_alerts)
    assert any(alert.instrument == "ETHUSDT" and alert.category == REGIME_CHANGE for alert in all_alerts)


def test_shared_cycle_inputs_are_used_and_persisted_without_recollecting(tmp_path):
    fixture = Fixture(tmp_path)
    fixture.set_metric("XAUUSD", "liquidity", "spread_expansion_ratio", 2.5)
    fixture.volatility.analyze_all = lambda *args: pytest.fail("volatility recollected")
    fixture.liquidity.analyze = lambda: pytest.fail("liquidity recollected")
    fixture.regime.analyze_all = lambda *args: pytest.fail("regime recollected")
    fixture.anomaly.analyze = lambda: pytest.fail("anomaly recollected")
    shared_inputs = {
        "volatility": fixture.volatility_results,
        "liquidity": fixture.liquidity_results,
        "regime": fixture.regime_results,
        "anomaly": fixture.anomaly_results,
    }

    first = fixture.service.evaluate(
        snapshot=fixture.manager.get_market_snapshot(), inputs=shared_inputs)
    assert any(alert.instrument == "XAUUSD" and alert.category == ABNORMAL_SPREAD
               for alert in first)
    assert len(fixture.service.get_recent_alerts()) == len(first)
    assert fixture.service.evaluate(
        snapshot=fixture.manager.get_market_snapshot(), inputs=shared_inputs) == ()


def test_shared_cycle_stale_instrument_does_not_generate_alert(tmp_path):
    fixture = Fixture(tmp_path)
    fixture.set_metric("XAUUSD", "liquidity", "spread_expansion_ratio", 2.5)
    fixture.states["XAUUSD"].freshness = "STALE"
    results = fixture.service.evaluate(
        snapshot=fixture.manager.get_market_snapshot(),
        inputs={"volatility": fixture.volatility_results,
                "liquidity": fixture.liquidity_results,
                "regime": fixture.regime_results,
                "anomaly": fixture.anomaly_results})
    assert not any(alert.instrument == "XAUUSD" for alert in results)


def test_duplicate_active_conditions_are_suppressed_and_clear_then_reenter_uses_cooldown(tmp_path):
    fixture = Fixture(tmp_path, settings=AlertSettings(spread_cooldown_seconds=60))
    fixture.set_metric("XAUUSD", "liquidity", "spread_expansion_ratio", 2.5)
    first = fixture.evaluate()
    assert sum(alert.category == ABNORMAL_SPREAD for alert in first) == 1
    assert not fixture.evaluate()

    fixture.set_metric("XAUUSD", "liquidity", "spread_expansion_ratio", 1.0)
    fixture.evaluate()
    fixture.set_metric("XAUUSD", "liquidity", "spread_expansion_ratio", 2.5)
    assert not any(alert.category == ABNORMAL_SPREAD for alert in fixture.evaluate())

    fixture.set_metric("XAUUSD", "liquidity", "spread_expansion_ratio", 1.0)
    fixture.evaluate()
    fixture.clock[0] += timedelta(seconds=61)
    fixture.set_metric("XAUUSD", "liquidity", "spread_expansion_ratio", 2.5)
    assert any(alert.category == ABNORMAL_SPREAD for alert in fixture.evaluate())


def test_regime_change_is_edge_triggered_and_persisted_across_service_restart(tmp_path):
    fixture = Fixture(tmp_path)
    assert not any(alert.category == REGIME_CHANGE for alert in fixture.evaluate())
    fixture.regime_results["EURUSD"] = SimpleNamespace(
        **{**vars(fixture.regime_results["EURUSD"]), "current_regime": "TRENDING"})
    alerts = fixture.evaluate()
    changes = [alert for alert in alerts if alert.instrument == "EURUSD" and alert.category == REGIME_CHANGE]
    assert len(changes) == 1
    assert changes[0].observed_values["previous_regime"] == "RANGING"
    assert changes[0].observed_values["current_regime"] == "TRENDING"

    restarted = AlertService(
        fixture.manager, volatility_service=fixture.volatility,
        liquidity_service=fixture.liquidity, regime_service=fixture.regime,
        anomaly_service=fixture.anomaly, database=AlertDatabase(tmp_path / "alerts.sqlite3"),
        clock=lambda: fixture.clock[0],
    )
    assert not any(alert.category == REGIME_CHANGE for alert in restarted.evaluate())


def test_unavailable_stale_and_warming_inputs_do_not_create_or_clear_alerts(tmp_path):
    fixture = Fixture(tmp_path)
    fixture.set_metric("XAUUSD", "liquidity", "spread_expansion_ratio", 3.0)
    assert any(alert.category == ABNORMAL_SPREAD for alert in fixture.evaluate())
    fixture.states["XAUUSD"].freshness = "STALE"
    fixture.set_metric("XAUUSD", "liquidity", "spread_expansion_ratio", 1.0)
    assert not fixture.evaluate()
    fixture.states["XAUUSD"].freshness = "LIVE"
    # Warming-up metric values are ignored even if a numeric value is present.
    fixture.set_metric("XAUUSD", "liquidity", "spread_expansion_ratio", 1.0, "WARMING_UP")
    fixture.set_metric("XAUUSD", "liquidity", "spread_anomaly_zscore", 0.0, "UNAVAILABLE")
    assert not fixture.evaluate()
    # The prior active condition remains set through unknown/stale data, so the
    # same persisting condition does not produce a duplicate on recovery.
    fixture.set_metric("XAUUSD", "liquidity", "spread_expansion_ratio", 3.0)
    assert not any(alert.category == ABNORMAL_SPREAD for alert in fixture.evaluate())


def test_invalid_nonfinite_metrics_never_trigger_conclusions(tmp_path):
    fixture = Fixture(tmp_path)
    fixture.set_metric("BTCUSDT", "liquidity", "order_book_imbalance", float("nan"))
    fixture.set_metric("XAUUSD", "volatility", "volatility_zscore", float("inf"))
    fixture.set_metric("EURUSD", "liquidity", "spread_anomaly_zscore", None)
    alerts = fixture.evaluate()
    assert not any(alert.category in {
        BINANCE_ORDER_BOOK_IMBALANCE, VOLATILITY_EXPANSION, ABNORMAL_SPREAD,
    } for alert in alerts)


def test_stale_anomaly_result_is_not_used(tmp_path):
    fixture = Fixture(tmp_path)
    fixture.anomaly_results["XAUUSD"] = SimpleNamespace(
        **{**vars(fixture.anomaly_results["XAUUSD"]), "anomaly_detected": True,
           "freshness": "STALE"})
    alerts = fixture.evaluate()
    assert not any(alert.instrument == "XAUUSD" and
                   alert.category == ISOLATION_FOREST_ANOMALY for alert in alerts)


def test_mt5_volume_and_binance_book_alerts_include_limitations_and_trust(tmp_path):
    fixture = Fixture(tmp_path)
    fixture.set_metric("XAUUSD", "liquidity", "tick_volume_anomaly_zscore", 3.5)
    fixture.set_metric("BTCUSDT", "liquidity", "order_book_imbalance", 0.8)
    alerts = fixture.evaluate()
    mt5_alert = next(alert for alert in alerts if alert.category == MT5_TICK_ACTIVITY)
    binance_alert = next(alert for alert in alerts if alert.category == BINANCE_ORDER_BOOK_IMBALANCE)
    assert "not centralized traded volume" in mt5_alert.explanation
    assert mt5_alert.observed_values["source_timestamp_trust"] == "UNVERIFIED_TIME_BASIS"
    assert mt5_alert.observed_values["analytics_timestamp_trust"] == "UNVERIFIED_TIME_BASIS"
    assert "cancelled" in binance_alert.explanation
    assert binance_alert.observed_values["source_timestamp_trust"] == "TRUSTED_UTC"
    assert mt5_alert.timestamp_provenance.startswith("PYTHON_SYSTEM_UTC_AT_DETECTION")
    assert mt5_alert.market_data_received_at_utc == NOW - timedelta(milliseconds=50)


def test_binance_imbalance_requires_synchronized_book(tmp_path):
    fixture = Fixture(tmp_path)
    fixture.set_metric("BTCUSDT", "liquidity", "order_book_imbalance", 0.95)
    fixture.states["BTCUSDT"].order_book_sync_state = "UNSYNCHRONIZED"
    alerts = fixture.evaluate()
    assert not any(alert.category == BINANCE_ORDER_BOOK_IMBALANCE for alert in alerts)


def test_alert_persistence_filters_and_acknowledgement(tmp_path):
    fixture = Fixture(tmp_path)
    fixture.set_metric("EURUSD", "liquidity", "spread_anomaly_zscore", 4.0)
    fixture.evaluate()
    alerts = fixture.service.get_recent_alerts(limit=20, instrument="eurusd", category=ABNORMAL_SPREAD)
    assert len(alerts) == 1
    alert = alerts[0]
    assert not alert.acknowledged
    assert fixture.service.acknowledge_alert(alert.alert_id)
    assert not fixture.service.acknowledge_alert("missing-id")
    saved = fixture.service.get_recent_alerts(instrument="EURUSD", severity=WARNING)
    assert len(saved) == 1
    assert saved[0].acknowledged
    assert saved[0].acknowledged_at_utc is not None


def test_critical_severity_escalation_is_a_distinct_condition(tmp_path):
    fixture = Fixture(tmp_path)
    fixture.set_metric("GBPUSD", "liquidity", "tick_volume_activity_ratio", 3.0)
    first = fixture.evaluate()
    warning = next(alert for alert in first if alert.category == MT5_TICK_ACTIVITY)
    assert warning.severity == WARNING
    fixture.set_metric("GBPUSD", "liquidity", "tick_volume_activity_ratio", 4.5)
    second = fixture.evaluate()
    critical = next(alert for alert in second if alert.category == MT5_TICK_ACTIVITY)
    assert critical.severity == CRITICAL


def test_invalid_settings_rejected(tmp_path):
    with pytest.raises(ValueError):
        AlertSettings(spread_zscore_threshold=0)
    with pytest.raises(ValueError):
        AlertDatabase(tmp_path / "nested" / "alerts.sqlite3").fetch_alerts(limit=0)
    with pytest.raises(ValueError):
        AlertSettings(critical_binance_imbalance=1.1)

