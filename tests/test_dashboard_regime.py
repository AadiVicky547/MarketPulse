"""Mocked data preparation tests for the Market Regime dashboard page."""
from datetime import datetime, timezone
from types import SimpleNamespace

from config import INSTRUMENTS
from dashboard.regime_view import (
    _render_cards,
    feature_display,
    prepare_regime_rows,
    regime_comparison_figure,
    summarize_regimes,
)


def feature(status="AVAILABLE", value=0.0, unit="basis points", warning=None):
    return SimpleNamespace(status=status, value=value, unit=unit,
                           warning=warning, methodology="mock methodology", sample_count=12)


def regime_result(symbol, source, *, status="AVAILABLE", regime="RANGING", freshness="LIVE",
                  features=None, warnings=()):
    return SimpleNamespace(
        instrument=symbol, source=source, timeframe="1h", status=status,
        current_regime=regime, evidence_strength="MODERATE",
        timestamp_trust="TRUSTED_UTC" if source == "Binance" else "UNVERIFIED_TIME_BASIS",
        last_observation_freshness=freshness,
        last_update_received_at=datetime(2026, 11, 13, tzinfo=timezone.utc),
        features=features or {}, explanation="Mock rule explanation.",
        data_quality_warnings=warnings, candle_count=200,
    )


def test_rows_include_six_configured_instruments_and_real_classifications():
    results = {item.symbol: regime_result(item.symbol, item.source) for item in INSTRUMENTS}
    analysis_time = datetime(2026, 11, 13, 8, tzinfo=timezone.utc)
    rows = prepare_regime_rows(results, analysis_time)
    assert [row["instrument"] for row in rows] == [item.symbol for item in INSTRUMENTS]
    assert all(row["regime"] == "RANGING" for row in rows)
    assert all(row["analysis_updated_at"] == analysis_time for row in rows)
    assert rows[0]["timestamp_trust"] == "UNVERIFIED_TIME_BASIS"
    assert rows[-1]["timestamp_trust"] == "TRUSTED_UTC"


def test_unavailable_and_warming_features_are_not_evidence_against_regime():
    results = {"XAUUSD": regime_result("XAUUSD", "MT5", status="WARMING_UP", regime=None,
        features={"adx": feature("WARMING_UP", None, "index", "need more history"),
                  "ema_slope_bps": feature("AVAILABLE", 0.0)})}
    rows = prepare_regime_rows(results)
    gold = rows[0]
    assert gold["status"] == "WARMING_UP"
    assert gold["regime"] is None
    features = {item["key"]: item for item in gold["features"]}
    assert features["adx"]["display"] == "WARMING_UP — need more history"
    assert features["ema_slope_bps"]["display"] == "0 basis points"
    assert features["atr_percentile"]["display"] == "UNAVAILABLE"
    assert feature_display(feature("UNAVAILABLE", None, warning="no samples")) == "UNAVAILABLE — no samples"


def test_stale_classification_is_retained_as_historical_but_not_current():
    results = {"EURUSD": regime_result("EURUSD", "MT5", regime="TRENDING", freshness="STALE")}
    row = prepare_regime_rows(results)[1]
    assert row["status"] == "STALE"
    assert row["regime"] is None
    assert row["last_classification"] == "TRENDING"


def test_summary_and_comparison_exclude_stale_unavailable_and_warming_results():
    results = {
        "XAUUSD": regime_result("XAUUSD", "MT5", regime="TRENDING"),
        "EURUSD": regime_result("EURUSD", "MT5", regime="TRENDING", freshness="STALE"),
        "GBPUSD": regime_result("GBPUSD", "MT5", status="WARMING_UP", regime=None),
        "BTCUSDT": regime_result("BTCUSDT", "Binance", regime="HIGH_VOLATILITY"),
    }
    rows = prepare_regime_rows(results)
    summary = summarize_regimes(rows)
    assert summary["TRENDING"] == 1
    assert summary["HIGH_VOLATILITY"] == 1
    assert summary["STALE"] == 1
    assert summary["WARMING_UP"] == 1
    assert summary["UNAVAILABLE"] == 2
    figure = regime_comparison_figure(rows)
    plotted = [trace.y[0] for trace in figure.data]
    assert plotted == ["XAUUSD", "BTCUSDT"]


def test_instrument_cards_do_not_render_service_warnings_as_yellow_boxes(monkeypatch):
    from contextlib import nullcontext
    import dashboard.regime_view as regime_view

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

    monkeypatch.setattr(regime_view, "st", FakeStreamlit())
    monkeypatch.setattr(regime_view, "_render_feature_breakdown", lambda row: None)
    warning_text = "cadence gaps detected; no timezone correction applied"
    row = prepare_regime_rows({
        "XAUUSD": regime_result("XAUUSD", "MT5", warnings=(warning_text,))
    })[0]

    _render_cards([row])

    assert not any(name == "warning" for name, *_ in calls)
    assert warning_text not in repr(calls)
    assert any(call[0] == "metric" and call[1] == "Market regime" for call in calls)
