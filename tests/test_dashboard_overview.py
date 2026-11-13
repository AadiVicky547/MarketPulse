"""Mocked tests for Overview data preparation and non-blocking refresh."""
from concurrent.futures import Future
from datetime import datetime, timezone
from decimal import Decimal
from types import MappingProxyType, SimpleNamespace

from config import INSTRUMENTS
from dashboard import overview
from dashboard.overview import (
    OverviewDataService,
    overall_connection_status,
    _price_direction,
    _quote_detail_html,
    advance_watchlist_price_history,
    advance_watchlist_quote_history,
    prepare_analytics_summary,
    prepare_watchlist,
)


def test_watchlist_displays_genuine_values_and_preserves_missing_data():
    instruments = {
        "XAUUSD": SimpleNamespace(
            actual_symbol="XAUUSD.a", mid_price=Decimal("2345.50"), bid=Decimal("2345"),
            ask=Decimal("2346"), spread=Decimal("1"), spread_bps=Decimal("4.26"),
            last_trade_price=None, connection_state="CONNECTED", freshness="LIVE",
            tick_timestamp_trust="UNVERIFIED_TIME_BASIS",
        ),
        "BTCUSDT": SimpleNamespace(
            actual_symbol="BTCUSDT", mid_price=None, last_trade_price=Decimal("0"),
            bid=None, ask=None, spread=None, spread_bps=None,
            connection_state="CONNECTED", freshness="LIVE", tick_timestamp_trust="TRUSTED_UTC",
        ),
    }
    rows = prepare_watchlist(SimpleNamespace(instruments=MappingProxyType(instruments)))
    assert len(rows) == 6
    gold = next(row for row in rows if row["instrument"] == "XAUUSD")
    assert gold["provider"] == "MT5 broker"
    assert gold["actual_symbol"] == "XAUUSD.a"
    assert gold["price"] == "2,345.5"
    assert gold["timestamp_trust"] == "UNVERIFIED_TIME_BASIS"
    bitcoin = next(row for row in rows if row["instrument"] == "BTCUSDT")
    assert bitcoin["provider"] == "Binance public"
    assert bitcoin["price_label"] == "Last trade"
    assert bitcoin["price"] == "0"
    assert bitcoin["bid"] == "—"
    absent = next(row for row in rows if row["instrument"] == "EURUSD")
    assert absent["connection"] == "UNAVAILABLE"
    assert absent["price"] == "—"


def test_watchlist_price_direction_uses_prior_valid_price_for_all_instruments():
    values = {
        "XAUUSD": (Decimal("2346"), Decimal("2345")),
        "EURUSD": (Decimal("1.10"), Decimal("1.11")),
        "GBPUSD": (Decimal("1.25"), Decimal("1.25")),
        "USDJPY": (Decimal("159"), None),
        "BTCUSDT": (Decimal("68000"), Decimal("67000")),
        "ETHUSDT": (Decimal("1900"), Decimal("1950")),
    }
    states = {
        symbol: SimpleNamespace(
            actual_symbol=symbol, mid_price=current, last_trade_price=None,
            bid=None, ask=None, spread=None, spread_bps=None,
            connection_state="CONNECTED", freshness="LIVE",
        )
        for symbol, (current, _previous) in values.items()
    }
    previous = {
        symbol: old for symbol, (_current, old) in values.items() if old is not None
    }

    rows = prepare_watchlist(
        SimpleNamespace(instruments=MappingProxyType(states)),
        previous_prices=previous,
    )
    directions = {row["instrument"]: row["price_direction"] for row in rows}

    assert directions == {
        "XAUUSD": "UP",
        "EURUSD": "DOWN",
        "GBPUSD": "UNCHANGED",
        "USDJPY": "UNCHANGED",
        "BTCUSDT": "UP",
        "ETHUSDT": "DOWN",
    }


def test_watchlist_direction_rejects_missing_nonfinite_and_nonpositive_previous_prices():
    assert _price_direction(10, None) == "UNCHANGED"
    assert _price_direction(10, 0) == "UNCHANGED"
    assert _price_direction(10, float("nan")) == "UNCHANGED"
    assert _price_direction(0, 9) == "UNCHANGED"


def test_quote_detail_keeps_values_opaque_and_supports_dynamic_bid_ask_color():
    markup = _quote_detail_html("Bid", "1.2345", "#35C46A")
    assert "mp-quote-detail" in markup
    assert "--mp-value-color:#35C46A" in markup
    assert "1.2345" in markup


def test_price_history_compares_consecutive_valid_snapshots_and_preserves_gaps():
    def snapshot(price):
        state = SimpleNamespace(mid_price=Decimal(str(price)), last_trade_price=None)
        return SimpleNamespace(instruments={"XAUUSD": state})

    old = {}
    previous, old = advance_watchlist_price_history(snapshot("100"), old)
    assert previous["XAUUSD"] is None
    assert old["XAUUSD"] == {"last": 100.0, "bid": None, "ask": None}

    previous, old = advance_watchlist_price_history(snapshot("100"), old)
    assert previous["XAUUSD"] == 100.0
    assert _price_direction(100, previous["XAUUSD"]) == "UNCHANGED"
    assert old["XAUUSD"] == {"last": 100.0, "bid": None, "ask": None}

    previous, old = advance_watchlist_price_history(snapshot("101"), old)
    assert previous["XAUUSD"] == 100.0
    assert old["XAUUSD"] == {"last": 101.0, "bid": None, "ask": None}

    previous, old = advance_watchlist_price_history(snapshot("101"), old)
    assert previous["XAUUSD"] == 101.0
    assert _price_direction(101, previous["XAUUSD"]) == "UNCHANGED"

    # A missing quote does not erase the last valid comparison.
    previous, old = advance_watchlist_price_history(
        SimpleNamespace(instruments={}), old)
    assert old["XAUUSD"] == {"last": 101.0, "bid": None, "ask": None}


def test_bid_and_ask_directions_are_compared_independently():
    state = SimpleNamespace(
        mid_price=Decimal("10.5"), bid=Decimal("10.4"), ask=Decimal("10.6"),
        last_trade_price=None,
    )
    snapshot = SimpleNamespace(instruments={"XAUUSD": state})
    prior = {"XAUUSD": {"last": 10.0, "bid": 10.5, "ask": 10.5}}

    previous_prices, previous_quotes, updated = advance_watchlist_quote_history(snapshot, prior)
    row = next(item for item in prepare_watchlist(
        snapshot, previous_prices=previous_prices, previous_quotes=previous_quotes
    ) if item["instrument"] == "XAUUSD")

    assert row["price_direction"] == "UP"
    assert row["bid_direction"] == "DOWN"
    assert row["ask_direction"] == "UP"
    assert updated["XAUUSD"] == {"last": 10.5, "bid": 10.4, "ask": 10.6}

def test_overall_connection_status_distinguishes_partial_from_down():
    assert overall_connection_status(SimpleNamespace(sources={
        "MT5": SimpleNamespace(connection_state="CONNECTED"),
        "Binance": SimpleNamespace(connection_state="CONNECTING"),
    })) == "PARTIAL"
    assert overall_connection_status(SimpleNamespace(sources={
        "MT5": SimpleNamespace(connection_state="DISCONNECTED"),
        "Binance": SimpleNamespace(connection_state="CONNECTING"),
    })) == "CONNECTING"
    assert overall_connection_status(SimpleNamespace(sources={})) == "DISCONNECTED"


def test_analytics_summary_renders_metric_statuses_without_zero_substitution():
    volatility = {
        "XAUUSD": SimpleNamespace(
            timestamp_trust="UNVERIFIED_TIME_BASIS",
            metrics={
                "realized_volatility": SimpleNamespace(status="AVAILABLE", value=0.0, unit="return per candle"),
                "atr": SimpleNamespace(status="WARMING_UP", value=None, warning="needs bars"),
            },
        )
    }
    liquidity = {
        "XAUUSD": SimpleNamespace(metrics={
            "tick_volume_activity_ratio": SimpleNamespace(status="UNAVAILABLE", value=None),
        })
    }
    regimes = {"XAUUSD": SimpleNamespace(status="WARMING_UP", current_regime=None)}
    anomalies = {"XAUUSD": SimpleNamespace(status="WARMING_UP", anomaly_detected=None)}
    rows = prepare_analytics_summary(volatility, liquidity, regimes, anomalies)
    gold = rows[0]
    assert gold["Realized volatility (1h)"] == "0 return per candle"
    assert gold["Volatility percentile"] == "UNAVAILABLE"
    assert gold["ATR"] == "WARMING_UP — needs bars"
    assert gold["Volatility regime"] == "WARMING_UP"
    assert gold["Market regime"] == "WARMING_UP"
    assert gold["Anomaly status"] == "WARMING_UP"
    assert gold["Liquidity / Flow"].startswith("Tick volume UNAVAILABLE · activity UNAVAILABLE")
    assert gold["Timestamp trust"] == "UNVERIFIED_TIME_BASIS"
    assert len(rows) == len(INSTRUMENTS)


class DeferredExecutor:
    """Executor test double that records work without executing it inline."""

    def __init__(self):
        self.tasks = []
        self.futures = []
        self.shutdown_called = False

    def submit(self, function):
        future = Future()
        self.tasks.append(function)
        self.futures.append(future)
        return future

    def shutdown(self, *, wait, cancel_futures):
        self.shutdown_called = True
        assert wait is True
        assert cancel_futures is True


def test_overview_refresh_is_nonblocking_and_reuses_timed_candle_analytics():
    calls = {"snapshot": 0, "liquidity": 0, "volatility": 0, "regime": 0,
             "anomaly": 0, "alerts": 0, "evaluate": 0}
    evaluation_inputs = []
    snapshot = SimpleNamespace(instruments={}, sources={})

    class Manager:
        def get_market_snapshot(self):
            calls["snapshot"] += 1
            return snapshot

    class Liquidity:
        def analyze(self):
            calls["liquidity"] += 1
            return {"XAUUSD": "liquidity"}

    class Volatility:
        def analyze_all(self, timeframe, count):
            calls["volatility"] += 1
            assert (timeframe, count) == ("1h", 200)
            return {"XAUUSD": "volatility"}

    class Regime:
        def analyze_all(self, timeframe, count):
            calls["regime"] += 1
            assert (timeframe, count) == ("1h", 600)
            return {"XAUUSD": "regime"}

    class Anomaly:
        fail = False

        def analyze(self):
            calls["anomaly"] += 1
            if self.fail:
                raise RuntimeError("model unavailable")
            return {"XAUUSD": "anomaly"}

    class Alerts:
        fail_evaluation = False

        def evaluate(self, *, snapshot, inputs):
            calls["evaluate"] += 1
            if self.fail_evaluation:
                raise RuntimeError("database unavailable")
            evaluation_inputs.append((snapshot, inputs))
            return ()

        def get_recent_alerts(self, *, limit):
            calls["alerts"] += 1
            assert limit == 8
            return ()

        def acknowledge_alert(self, alert_id):
            return alert_id == "known"

    executor = DeferredExecutor()
    alerts = Alerts()
    anomaly_service = Anomaly()
    service = OverviewDataService(
        Manager(), volatility_service=Volatility(), liquidity_service=Liquidity(),
        regime_service=Regime(), anomaly_service=anomaly_service, alert_service=alerts,
        executor=executor,
    )
    first = service.refresh()
    assert first.analytics_pending
    assert calls == {"snapshot": 1, "liquidity": 0, "volatility": 0, "regime": 0,
                     "anomaly": 0, "alerts": 0, "evaluate": 0}

    computed = executor.tasks[0]()
    executor.futures[0].set_result(computed)
    assert calls["evaluate"] == 1
    assert evaluation_inputs[0][0] is snapshot
    assert evaluation_inputs[0][1] == {
        "volatility": {"XAUUSD": "volatility"},
        "liquidity": {"XAUUSD": "liquidity"},
        "regime": {"XAUUSD": "regime"},
        "anomaly": {"XAUUSD": "anomaly"},
    }
    second = service.refresh()
    assert second.analytics_pending
    assert second.volatility["XAUUSD"] == "volatility"
    assert second.liquidity["XAUUSD"] == "liquidity"
    assert second.regimes["XAUUSD"] == "regime"
    assert second.anomalies["XAUUSD"] == "anomaly"
    assert calls["volatility"] == 1
    assert calls["regime"] == 1
    assert calls["liquidity"] == 1
    assert calls["evaluate"] == 1
    assert service.acknowledge_alert("known")
    # A persistence/evaluation exception is recorded without failing the cycle.
    alerts.fail_evaluation = True
    anomaly_service.fail = True
    service.refresh()
    failed_cycle = executor.tasks[1]()
    executor.futures[1].set_result(failed_cycle)
    after_failure = service.refresh()
    assert any(error.startswith("Anomaly analysis: RuntimeError")
               for error in after_failure.errors)
    assert any(error.startswith("Alert evaluation: RuntimeError")
               for error in after_failure.errors)
    assert calls["evaluate"] == 2
    service.close()
    assert executor.shutdown_called


def test_shared_scheduler_requests_collection_independent_of_page(monkeypatch):
    calls = []

    class SharedService:
        def refresh(self):
            calls.append("cycle-request")

    runtime = SimpleNamespace(data_service=SharedService())
    monkeypatch.setattr(overview, "_get_runtime", lambda: runtime)

    # Streamlit runs this root-level periodic fragment for the session, rather
    # than tying observation collection to any particular selected page.
    overview._request_shared_analytics_cycle()
    overview._request_shared_analytics_cycle()
    assert calls == ["cycle-request", "cycle-request"]


def test_repeated_refresh_requests_do_not_create_overlapping_workers():
    snapshot = SimpleNamespace(instruments={}, sources={})

    class Manager:
        def get_market_snapshot(self):
            return snapshot

    executor = DeferredExecutor()
    service = OverviewDataService(Manager(), executor=executor)
    service.refresh()
    service.refresh()
    service.refresh()
    assert len(executor.tasks) == 1
    service.close()

