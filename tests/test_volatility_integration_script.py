"""Tests for defensive output in the live volatility verification script."""
from types import SimpleNamespace

import pandas as pd

from scripts import test_volatility_integration as report


def test_available_metric_status_is_rendered_without_name_error():
    metric = SimpleNamespace(status="AVAILABLE", value=1.25, unit="price units", sample_count=8)
    assert report._metric_status(metric) == "AVAILABLE: 1.25 price units (n=8)"


def test_unavailable_result_keeps_exact_validation_warning(capsys):
    exact_reason = "Candle validation failed: candle interval mismatch; expected 0 days 01:00"
    result = SimpleNamespace(
        status="UNAVAILABLE", source="MT5", candle_count=12, requested_candle_count=20,
        timestamp_trust="UNVERIFIED_TIME_BASIS", metrics={"atr": None},
        data_quality_warnings=(exact_reason,),
    )

    report._print_instrument_report("XAUUSD", "MT5", result)
    output = capsys.readouterr().out
    assert "Candle retrieval: REJECTED" in output
    assert exact_reason in output
    assert "atr: UNAVAILABLE: metric result is missing" in output


def test_none_fields_and_unexpected_metrics_are_reported(capsys):
    result = SimpleNamespace(
        status=None, source=None, candle_count=None, requested_candle_count=None,
        timestamp_trust=None, metrics=["unexpected"],
        data_quality_warnings="provider response was malformed",
    )

    report._print_instrument_report("ETHUSDT", "Binance", result)
    output = capsys.readouterr().out
    assert "Overall status: UNAVAILABLE" in output
    assert "Timestamp trust: not reported" in output
    assert "Unexpected metric result structure: list" in output
    assert "provider response was malformed" in output


def test_reporting_error_does_not_stop_later_instruments(monkeypatch, capsys):
    class FakeService:
        def __init__(self, *, candle_service_factory):
            self.candle_service_factory = candle_service_factory

        def analyze_all(self, timeframe, count):
            return {item.symbol: object() for item in report.INSTRUMENTS}

    monkeypatch.setattr(report, "VolatilityService", FakeService)
    class FakeCandleService:
        def get_candles(self, symbol, timeframe, count, *, include_incomplete=False):
            return pd.DataFrame({
                "timestamp": pd.date_range("2025-01-01T00:00:00Z", periods=2, freq="1h"),
                "timestamp_trust": "UNVERIFIED_TIME_BASIS",
            })

        def close(self):
            pass

    monkeypatch.setattr(report, "CandleService", FakeCandleService)
    original = report._print_instrument_report

    def sometimes_fails(symbol, source, result):
        if symbol == "XAUUSD":
            raise RuntimeError("mock rendering problem")
        return original(symbol, source, SimpleNamespace(
            status="WARMING_UP", source=source, candle_count=10, requested_candle_count=200,
            timestamp_trust="TRUSTED_UTC", metrics={}, data_quality_warnings=(),
        ))

    monkeypatch.setattr(report, "_print_instrument_report", sometimes_fails)
    exit_code = report.main()
    output = capsys.readouterr().out
    assert exit_code == 1
    assert "XAUUSD" in output and "BTCUSDT" in output and "ETHUSDT" in output
    assert "Completed report for 6/6 configured instruments" in output


def test_cadence_report_shows_intervals_gaps_and_unverified_trust(capsys):
    timestamps = pd.to_datetime([
        "2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z",
        "2025-01-01T04:00:00Z", "2025-01-03T07:00:00Z",
    ])
    frame = pd.DataFrame({
        "timestamp": timestamps,
        "timestamp_trust": "UNVERIFIED_TIME_BASIS",
    })

    report._print_cadence_report("XAUUSD", frame)
    output = capsys.readouterr().out
    assert "trust=UNVERIFIED_TIME_BASIS" in output
    assert "Strictly increasing: yes" in output
    assert "Duplicate timestamps: no" in output
    assert "One-hour intervals: 1" in output
    assert "Larger-than-one-hour intervals: 2" in output
    assert "0 days 03:00:00: 1" in output
    assert "2 days 03:00:00: 1" in output
    assert "2025-01-01T01:00:00+00:00 -> 2025-01-01T04:00:00+00:00" in output
    assert "2025-01-01T04:00:00+00:00 -> 2025-01-03T07:00:00+00:00" in output


def test_cadence_report_detects_duplicates_and_non_increasing_order(capsys):
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2025-01-01T01:00:00Z", "2025-01-01T01:00:00Z",
            "2025-01-01T00:00:00Z",
        ]),
        "timestamp_trust": "UNVERIFIED_TIME_BASIS",
    })
    report._print_cadence_report("EURUSD", frame)
    output = capsys.readouterr().out
    assert "Strictly increasing: no" in output
    assert "Duplicate timestamps: yes (extra duplicate rows=1)" in output
