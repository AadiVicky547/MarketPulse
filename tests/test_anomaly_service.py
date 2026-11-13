from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from analytics.anomaly_service import (
    AVAILABLE,
    ERROR,
    UNAVAILABLE,
    WARMING_UP,
    AnomalyService,
    AnomalySettings,
)


def metric(value, status="AVAILABLE"):
    return SimpleNamespace(value=value, status=status, warning=None)


class FakeManager:
    def __init__(self):
        self.states = {}
        self.snapshot = None

    def get_market_snapshot(self):
        return self.snapshot


class FakeAnalytics:
    def __init__(self, results):
        self.results = results

    def analyze(self):
        return self.results

    def analyze_all(self, timeframe, count):
        return self.results


class Fixture:
    def __init__(self, symbol="XAUUSD", source="MT5", liquidity_metrics=None,
                 volatility_metrics=None, trust="UNVERIFIED_TIME_BASIS"):
        self.symbol = symbol
        self.source = source
        self.manager = FakeManager()
        self.liquidity = FakeAnalytics({})
        self.volatility = FakeAnalytics({})
        self.settings = AnomalySettings(
            minimum_training_samples=8,
            minimum_features=2,
            history_size=12,
            max_features=4,
            n_estimators=80,
            contamination=0.1,
            random_state=17,
        )
        self.service = AnomalyService(
            self.manager,
            liquidity_service=self.liquidity,
            volatility_service=self.volatility,
            settings=self.settings,
        )
        if liquidity_metrics is None:
            liquidity_metrics = {"spread_price": metric(0.1)}
        self.liquidity.results = {symbol: SimpleNamespace(metrics=liquidity_metrics)}
        self.volatility.results = {symbol: SimpleNamespace(
            metrics=volatility_metrics or {}, timestamp_trust=trust,
        )}
        self.index = 0
        self.publish()

    def publish(self, liquidity_metrics=None, volatility_metrics=None, **state_changes):
        state = SimpleNamespace(
            freshness="LIVE",
            connection_state="CONNECTED",
            tick_timestamp_trust=("TRUSTED_UTC" if self.source == "Binance"
                                   else "UNVERIFIED_TIME_BASIS"),
            tick_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=self.index),
            last_update_received_at=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=self.index),
            bid=100.0 + self.index,
            ask=100.1 + self.index,
            tick_volume=10 + self.index,
        )
        for name, value in state_changes.items():
            setattr(state, name, value)
        self.manager.snapshot = SimpleNamespace(
            instruments={self.symbol: state},
            sources={},
            binance_trades={self.symbol: ()},
            binance_order_books={self.symbol: {"status": {"update_id": self.index}}},
        )
        if liquidity_metrics is not None:
            self.liquidity.results = {self.symbol: SimpleNamespace(metrics=liquidity_metrics)}
        if volatility_metrics is not None:
            self.volatility.results = {self.symbol: SimpleNamespace(
                metrics=volatility_metrics,
                timestamp_trust=("TRUSTED_UTC" if self.source == "Binance"
                                 else "UNVERIFIED_TIME_BASIS"),
            )}
        self.index += 1

    def analyze(self):
        return self.service.analyze()[self.symbol]


def test_warms_up_then_scores_using_only_prior_observations():
    fixture = Fixture(volatility_metrics={"atr": metric(1.0)})
    first = fixture.analyze()
    assert first.status == WARMING_UP
    assert first.anomaly_detected is None
    assert fixture.service.history_counts["XAUUSD"] == 1

    for i in range(1, 9):
        fixture.publish(volatility_metrics={"atr": metric(1.0 + i * 0.01)})
        result = fixture.analyze()
    assert result.status == AVAILABLE
    assert result.model_status == "TRAINED_ON_PRIOR_OBSERVATIONS"
    assert result.sample_count == 8
    assert fixture.service.history_counts["XAUUSD"] == 9
    assert result.timestamp_trust == "UNVERIFIED_TIME_BASIS"


def test_default_warmup_requires_exactly_50_prior_observations():
    fixture = Fixture(volatility_metrics={"atr": metric(1.0)})
    fixture.service = AnomalyService(
        fixture.manager, liquidity_service=fixture.liquidity,
        volatility_service=fixture.volatility,
        settings=AnomalySettings(minimum_training_samples=50, minimum_features=2,
                                 history_size=80, max_features=4, random_state=17),
    )
    for index in range(50):
        fixture.publish(volatility_metrics={"atr": metric(1.0 + index * 0.001)})
        result = fixture.analyze()
    assert result.status == WARMING_UP
    assert result.sample_count == 49
    assert "49/50" in result.explanation
    assert fixture.service.history_counts["XAUUSD"] == 50

    fixture.publish(volatility_metrics={"atr": metric(1.051)})
    result = fixture.analyze()
    assert result.status == AVAILABLE
    assert result.sample_count == 50
    assert fixture.service.history_counts["XAUUSD"] == 51


@pytest.mark.parametrize(
    ("symbol", "source", "trust", "liquidity_metrics", "volatility_metrics", "expected"),
    [
        (
            "XAUUSD", "MT5", "UNVERIFIED_TIME_BASIS",
            {
                "spread_price": metric(0.25),
                "spread_bps": metric(1.2),
                "tick_volume": metric(None, "UNAVAILABLE"),
                "tick_volume_activity_ratio": metric(None, "WARMING_UP"),
            },
            {
                "atr": metric(2.5),
                "realized_volatility": metric(0.002),
                "ewma_volatility": metric(None, "WARMING_UP"),
                "average_daily_range": metric(None, "TIMESTAMP_UNVERIFIED"),
            },
            {"spread_price", "spread_bps", "atr", "realized_volatility"},
        ),
        (
            "BTCUSDT", "Binance", "TRUSTED_UTC",
            {
                "spread_price": metric(1.5),
                "spread_bps": metric(0.4),
                "bid_depth": metric(12.0),
                "ask_depth": metric(10.0),
                "order_book_imbalance": metric(0.09),
                "trade_activity": metric(None, "WARMING_UP"),
            },
            {
                "atr": metric(120.0),
                "realized_volatility": metric(0.001),
                "ewma_volatility": metric(0.0012),
                "volatility_percentile": metric(None, "WARMING_UP"),
            },
            {"spread_price", "spread_bps", "bid_depth", "ask_depth",
             "order_book_imbalance", "atr", "realized_volatility", "ewma_volatility"},
        ),
    ],
)
def test_partial_provider_features_reach_current_observation_and_explain_omissions(
        symbol, source, trust, liquidity_metrics, volatility_metrics, expected):
    fixture = Fixture(symbol=symbol, source=source, trust=trust,
                      liquidity_metrics=liquidity_metrics,
                      volatility_metrics=volatility_metrics)
    fixture.service = AnomalyService(
        fixture.manager, liquidity_service=fixture.liquidity,
        volatility_service=fixture.volatility, settings=AnomalySettings(),
    )
    result = fixture.analyze()

    assert result.status == WARMING_UP
    assert set(result.available_features) == expected
    assert set(result.feature_values) == expected
    assert result.features_used == ()  # No model schema until prior coverage is sufficient.
    assert result.sample_count == 0
    assert any("50 prior observations" in warning for warning in result.data_quality_warnings)
    if source == "MT5":
        assert any("tick_volume omitted" in warning for warning in result.data_quality_warnings)
        assert any("ewma_volatility omitted" in warning for warning in result.data_quality_warnings)
        assert any("average_daily_range excluded" in warning for warning in result.data_quality_warnings)
    else:
        assert any("trade_activity omitted" in warning for warning in result.data_quality_warnings)
        assert any("volatility_percentile omitted" in warning for warning in result.data_quality_warnings)


def test_each_configured_instrument_keeps_independent_warmup_history():
    from config import INSTRUMENTS

    fixture = Fixture(volatility_metrics={"atr": metric(1.0)})
    states, liquidity, volatility = {}, {}, {}
    for index, item in enumerate(INSTRUMENTS):
        states[item.symbol] = SimpleNamespace(
            freshness="LIVE", connection_state="CONNECTED",
            tick_timestamp_trust=("TRUSTED_UTC" if item.source == "Binance"
                                  else "UNVERIFIED_TIME_BASIS"),
            tick_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            last_update_received_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            bid=100.0 + index, ask=100.1 + index, tick_volume=10 + index,
        )
        liquidity[item.symbol] = SimpleNamespace(metrics={
            "spread_price": metric(0.1 + index * 0.01),
            "spread_bps": metric(1.0 + index * 0.1),
        })
        volatility[item.symbol] = SimpleNamespace(
            metrics={"atr": metric(1.0 + index * 0.1)},
            timestamp_trust=("TRUSTED_UTC" if item.source == "Binance"
                             else "UNVERIFIED_TIME_BASIS"),
        )
    fixture.manager.snapshot = SimpleNamespace(
        instruments=states, sources={}, binance_trades={}, binance_order_books={})
    fixture.liquidity.results = liquidity
    fixture.volatility.results = volatility

    result = fixture.service.analyze()
    assert set(result) == {item.symbol for item in INSTRUMENTS}
    assert all(result[item.symbol].status == WARMING_UP for item in INSTRUMENTS)
    assert fixture.service.history_counts == {item.symbol: 1 for item in INSTRUMENTS}

    # A stale instrument is excluded independently; the other five still grow.
    states["XAUUSD"].freshness = "STALE"
    for state in states.values():
        state.tick_timestamp += timedelta(seconds=1)
    fixture.manager.snapshot = SimpleNamespace(
        instruments=states, sources={}, binance_trades={}, binance_order_books={})
    fixture.service.analyze()
    assert fixture.service.history_counts["XAUUSD"] == 1
    assert all(fixture.service.history_counts[item.symbol] == 2 for item in INSTRUMENTS[1:])


def test_normal_observation_has_structured_score_and_explanation():
    fixture = Fixture(volatility_metrics={"atr": metric(1.0)})
    for i in range(12):
        fixture.publish(volatility_metrics={"atr": metric(1.0 + (i % 4) * 0.01)})
        result = fixture.analyze()
    assert result.status == AVAILABLE
    assert result.anomaly_detected in (True, False)
    assert result.anomaly_score is not None
    assert result.model_decision in (-1, 1)
    assert result.features_used
    assert "not" in result.explanation.lower()


def test_extreme_current_observation_is_scored_against_baseline():
    fixture = Fixture(volatility_metrics={"atr": metric(1.0)})
    for i in range(20):
        fixture.publish(
            liquidity_metrics={"spread_price": metric(0.10 + (i % 5) * 0.007)},
            volatility_metrics={"atr": metric(1.0 + ((i * 3) % 7) * 0.013)},
        )
        fixture.analyze()
    fixture.publish(
        liquidity_metrics={"spread_price": metric(1000.0)},
        volatility_metrics={"atr": metric(500.0)},
    )
    result = fixture.analyze()
    assert result.status == AVAILABLE
    assert result.anomaly_detected is True
    assert result.sample_count == 12  # bounded prior history; current row excluded
    assert "Unusual" in result.explanation


def test_unavailable_and_invalid_features_are_not_replaced_with_zero():
    fixture = Fixture(
        liquidity_metrics={
            "spread_price": metric(None),
            "spread_bps": metric(float("nan")),
            "tick_volume": metric(0.0, "UNAVAILABLE"),
        },
        volatility_metrics={"atr": metric(float("inf"))},
    )
    result = fixture.analyze()
    assert result.status == UNAVAILABLE
    assert result.feature_values == {}
    assert result.anomaly_detected is None
    assert result.data_quality_warnings


def test_too_few_features_stays_in_warmup_and_skips_missing_feature_rows():
    fixture = Fixture(liquidity_metrics={}, volatility_metrics={"atr": metric(1.0)})
    for i in range(10):
        # A single feature cannot satisfy the configured minimum feature count.
        fixture.publish(volatility_metrics={"atr": metric(1.0 + i * 0.1)})
        result = fixture.analyze()
    assert result.status == WARMING_UP
    assert result.sample_count == 9
    assert result.features_used == ("atr",)


def test_mt5_session_metrics_are_excluded_when_timestamp_basis_is_unverified():
    fixture = Fixture(volatility_metrics={
        "atr": metric(1.0),
        "average_daily_range": metric(5.0),
        "intraday_range_expansion": metric(1.5),
    })
    result = fixture.analyze()
    assert "average_daily_range" not in result.features_used
    assert "intraday_range_expansion" not in result.features_used
    assert result.timestamp_trust == "UNVERIFIED_TIME_BASIS"


def test_freshness_and_connection_failures_are_unavailable():
    fixture = Fixture(volatility_metrics={"atr": metric(1.0)})
    fixture.publish(freshness="STALE")
    assert fixture.analyze().status == UNAVAILABLE
    fixture.publish(connection_state="DISCONNECTED")
    result = fixture.analyze()
    assert result.status == UNAVAILABLE
    assert result.freshness == "DISCONNECTED"


def test_mt5_and_binance_features_are_source_specific():
    mt5 = Fixture(volatility_metrics={"atr": metric(1.0)}, trust="UNVERIFIED_TIME_BASIS")
    mt5_result = mt5.analyze()
    assert "atr" in mt5_result.feature_values
    assert "buy_aggressor_volume" not in mt5_result.feature_values

    binance = Fixture(
        symbol="BTCUSDT", source="Binance", trust="TRUSTED_UTC",
        liquidity_metrics={
            "buy_aggressor_volume": metric(2.0),
            "sell_aggressor_volume": metric(1.0),
            "order_book_imbalance": metric(0.2),
        },
        volatility_metrics={"atr": metric(100.0)},
    )
    result = binance.analyze()
    assert result.timestamp_trust == "TRUSTED_UTC"
    assert "buy_aggressor_volume" in result.feature_values
    assert "order_book_imbalance" in result.feature_values
    assert "tick_volume" not in result.feature_values


def test_histories_and_models_are_independent_per_instrument():
    fixture = Fixture(volatility_metrics={"atr": metric(1.0)})
    for i in range(6):
        fixture.publish(volatility_metrics={"atr": metric(1.0 + i * 0.01)})
        fixture.analyze()
    assert fixture.service.history_counts["XAUUSD"] == 6
    assert fixture.service.history_counts["BTCUSDT"] == 0


def test_repeated_identical_snapshot_does_not_duplicate_training_row():
    fixture = Fixture(liquidity_metrics={}, volatility_metrics={"atr": metric(1.0)})
    fixture.analyze()
    fixture.index -= 1  # reuse the same event identity for the next snapshot
    fixture.publish(volatility_metrics={"atr": metric(1.0)})
    fixture.index -= 1
    fixture.analyze()
    assert fixture.service.history_counts["XAUUSD"] == 1


def test_invalid_model_settings_are_rejected():
    with pytest.raises(ValueError):
        AnomalySettings(minimum_training_samples=5, history_size=4)


def test_input_collection_failure_is_reported_without_raising():
    fixture = Fixture(volatility_metrics={"atr": metric(1.0)})
    fixture.manager.get_market_snapshot = lambda: (_ for _ in ()).throw(RuntimeError("snapshot failure"))
    results = fixture.service.analyze()
    assert results["XAUUSD"].status == ERROR
    assert "snapshot failure" in results["XAUUSD"].explanation

