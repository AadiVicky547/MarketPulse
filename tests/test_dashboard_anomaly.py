"""Mocked anomaly dashboard preparation and visualization tests."""
from datetime import datetime, timezone
from types import SimpleNamespace

from config import INSTRUMENTS
from dashboard.anomaly_view import (
    anomaly_score_figure,
    anomaly_status_figure,
    prepare_anomaly_rows,
    _render_instrument_cards,
)


def anomaly(symbol, source, *, status="AVAILABLE", detected=False, score=0.0,
            freshness="LIVE", connection="CONNECTED", warnings=()):
    return SimpleNamespace(
        instrument=symbol, provider=source, status=status,
        anomaly_detected=detected if status == "AVAILABLE" else None,
        anomaly_score=score if status == "AVAILABLE" else None,
        model_decision=(-1 if detected else 1) if status == "AVAILABLE" else None,
        features_used=("spread_price", "atr") if status in {"AVAILABLE", "WARMING_UP"} else (),
        explanation=("Unusual spread evidence." if detected else "Within prior learned pattern."),
        sample_count=55 if status == "AVAILABLE" else 12,
        model_status="TRAINED_ON_PRIOR_OBSERVATIONS" if status == "AVAILABLE" else status,
        timestamp_trust="UNVERIFIED_TIME_BASIS" if source == "MT5" else "TRUSTED_UTC",
        freshness=freshness, connection_state=connection,
        last_update_received_at=datetime(2026, 11, 13, tzinfo=timezone.utc),
        feature_values={"spread_price": 0.0, "atr": 1.25},
        data_quality_warnings=warnings,
    )


def test_preparation_covers_all_instruments_and_keeps_warming_distinct_from_normal():
    results = {
        "XAUUSD": anomaly("XAUUSD", "MT5", detected=True, score=0.25, warnings=("test warning",)),
        "EURUSD": anomaly("EURUSD", "MT5", detected=False, score=-0.02),
        "GBPUSD": anomaly("GBPUSD", "MT5", status="WARMING_UP"),
        "USDJPY": anomaly("USDJPY", "MT5", status="UNAVAILABLE", freshness="UNAVAILABLE"),
        "BTCUSDT": anomaly("BTCUSDT", "Binance", status="AVAILABLE", detected=False, score=0.01),
        "ETHUSDT": anomaly("ETHUSDT", "Binance", status="ERROR"),
    }
    rows = prepare_anomaly_rows(results)
    assert [row["instrument"] for row in rows] == [item.symbol for item in INSTRUMENTS]
    assert rows[0]["determination"] == "YES"
    assert rows[0]["status"] == "AVAILABLE"
    assert rows[0]["warnings"] == ("test warning",)
    assert rows[0]["timestamp_trust"] == "UNVERIFIED_TIME_BASIS"
    assert rows[1]["determination"] == "NO"
    assert rows[2]["determination"] == "NOT YET DETERMINED"
    assert rows[2]["plot_status"] == "WARMING_UP"
    assert rows[3]["status"] == "UNAVAILABLE"
    assert rows[5]["status"] == "ERROR"


def test_stale_or_disconnected_results_withhold_score_and_decision():
    results = {
        "XAUUSD": anomaly("XAUUSD", "MT5", detected=True, score=0.9, freshness="STALE"),
        "BTCUSDT": anomaly("BTCUSDT", "Binance", detected=True, score=0.8,
                           connection="DISCONNECTED"),
    }
    rows = prepare_anomaly_rows(results)
    stale = rows[0]
    disconnected = rows[4]
    assert stale["status"] == "STALE"
    assert stale["determination"] == "NOT YET DETERMINED"
    assert stale["score"] is None and stale["model_decision"] is None
    assert disconnected["status"] == "DISCONNECTED"
    assert disconnected["score"] is None


def test_score_plot_only_contains_finite_available_decisions():
    results = {
        "XAUUSD": anomaly("XAUUSD", "MT5", detected=True, score=0.2),
        "EURUSD": anomaly("EURUSD", "MT5", detected=False, score=-0.1),
        "GBPUSD": anomaly("GBPUSD", "MT5", status="WARMING_UP"),
        "USDJPY": anomaly("USDJPY", "MT5", detected=True, score=0.8, freshness="STALE"),
    }
    rows = prepare_anomaly_rows(results)
    scores = anomaly_score_figure(rows)
    assert list(scores.data[0].x) == ["XAUUSD", "EURUSD"]
    assert list(scores.data[0].y) == [0.2, -0.1]
    statuses = anomaly_status_figure(rows)
    plotted = {symbol for trace in statuses.data for symbol in trace.y}
    assert plotted == {"XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "BTCUSDT", "ETHUSDT"}


def test_actual_feature_values_are_presented_without_fake_importance():
    rows = prepare_anomaly_rows({"XAUUSD": anomaly("XAUUSD", "MT5", detected=True)})
    gold = rows[0]
    values = {item["feature"]: item for item in gold["feature_rows"]}
    assert values["spread_price"]["value"] == 0.0
    assert values["spread_price"]["display"] == "0"
    assert values["spread_price"]["used_by_model"] is True
    assert "importance" not in values["spread_price"]
    assert gold["feature_count"] == 2
    assert gold["available_feature_count"] == 2


def test_warming_result_distinguishes_current_features_from_model_schema():
    warming = anomaly("BTCUSDT", "Binance", status="WARMING_UP")
    warming.features_used = ()
    rows = prepare_anomaly_rows({"BTCUSDT": warming})
    bitcoin = rows[4]
    assert bitcoin["available_feature_count"] == 2
    assert bitcoin["feature_count"] == 0
    assert bitcoin["sample_count"] == 12
    assert bitcoin["training_target"] == 50


def test_malformed_available_result_is_not_recast_as_normal():
    malformed = anomaly("XAUUSD", "MT5", detected=False, score=-0.2)
    malformed.anomaly_detected = None
    rows = prepare_anomaly_rows({"XAUUSD": malformed})
    gold = rows[0]
    assert gold["status"] == "UNAVAILABLE"
    assert gold["determination"] == "NOT YET DETERMINED"
    assert gold["score"] is None


def test_cards_show_live_anomaly_result_without_explanation_paragraph(monkeypatch):
    """Keep key scores visible while omitting the changing explanation paragraph."""
    from contextlib import nullcontext
    import dashboard.anomaly_view as view

    calls = []

    class FakeStreamlit:
        def columns(self, count, **kwargs):
            return [nullcontext() for _ in range(count)]

        def container(self, **kwargs):
            return nullcontext()

        def expander(self, label, **kwargs):
            calls.append(("expander", label, kwargs))
            return nullcontext()

        def __getattr__(self, name):
            def record(*args, **kwargs):
                calls.append((name, *args))
            return record

    monkeypatch.setattr(view, "st", FakeStreamlit())
    row = prepare_anomaly_rows({
        "XAUUSD": anomaly("XAUUSD", "MT5", detected=True, score=0.25,
                           warnings=("sensitive warning detail",))
    })[0]

    _render_instrument_cards([row])

    assert not any(name == "warning" for name, *_ in calls)
    assert not any("sensitive warning detail" in repr(call) for call in calls)
    assert not any(name == "expander" for name, *_ in calls)
    metric_index = next(i for i, call in enumerate(calls) if call[0] == "metric")
    assert calls[metric_index][1:3] == ("Anomaly detected", "YES")
    rendered = repr(calls)
    assert "TRAINED_ON_PRIOR_OBSERVATIONS" in rendered
    assert "Score: 0.25" in rendered
    assert "Decision: -1" in rendered
    assert "Prior valid observations: 55/50" in rendered
    assert "Current valid features: 2" in rendered
    assert "Model features: 2" in rendered
    assert "Unusual spread evidence." not in rendered
    assert not any(name == "write" for name, *_ in calls)


def test_visible_warmup_progress_updates_with_new_service_result(monkeypatch):
    """A subsequent render reflects the latest prior-observation count."""
    from contextlib import nullcontext
    import dashboard.anomaly_view as view

    calls = []

    class FakeStreamlit:
        def columns(self, count, **kwargs):
            return [nullcontext() for _ in range(count)]

        def container(self, **kwargs):
            return nullcontext()

        def __getattr__(self, name):
            def record(*args, **kwargs):
                calls.append((name, *args))
            return record

    monkeypatch.setattr(view, "st", FakeStreamlit())
    for count in (12, 13):
        result = anomaly("XAUUSD", "MT5", status="WARMING_UP")
        result.sample_count = count
        result.features_used = ()
        row = prepare_anomaly_rows({"XAUUSD": result})[0]
        _render_instrument_cards([row])

    progress = [call[1] for call in calls if call[0] == "caption"
                and "Prior valid observations:" in call[1]]
    assert progress == [
        "Prior valid observations: 12/50 · Current valid features: 2 · Model features: 0",
        "Prior valid observations: 13/50 · Current valid features: 2 · Model features: 0",
    ]
