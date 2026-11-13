"""Mock-only report, sampling and shutdown tests for anomaly integration."""
from types import SimpleNamespace

import pytest

from config import INSTRUMENTS
from scripts import test_anomaly_integration as script


def make_result(status, *, anomaly=None, score=None, decision=None, samples=0):
    return SimpleNamespace(
        instrument="XAUUSD", provider="MT5", status=status,
        anomaly_detected=anomaly, anomaly_score=score, model_decision=decision,
        features_used=("spread_price",), explanation="Mock explanation.",
        sample_count=samples, model_status=status, timestamp_trust="UNVERIFIED_TIME_BASIS",
        freshness="LIVE", connection_state="CONNECTED",
        feature_values={"spread_price": 0.0}, data_quality_warnings=("Mock warning",),
    )


def test_report_distinguishes_available_warming_and_unavailable(capsys):
    snapshot = SimpleNamespace(
        instruments={item.symbol: SimpleNamespace(connection_state="CONNECTED", freshness="LIVE")
                     for item in INSTRUMENTS},
        sources={},
    )
    results = {item.symbol: make_result("AVAILABLE", anomaly=False, score=0.0, decision=1, samples=50)
               for item in INSTRUMENTS}
    results["EURUSD"] = make_result("WARMING_UP", samples=49)
    results["BTCUSDT"] = make_result("UNAVAILABLE")

    failed = script._print_report(snapshot, results)
    output = capsys.readouterr().out
    assert failed
    assert all(item.symbol in output for item in INSTRUMENTS)
    assert "NO — not flagged as unusual" in output
    assert "not scored (warming up; not a normal classification)" in output
    assert "prior complete samples: 49" in output
    assert "Anomaly score: 0" in output
    assert "UNVERIFIED_TIME_BASIS" in output
    assert "Mock warning" in output


def test_report_handles_missing_instrument_result(capsys):
    snapshot = SimpleNamespace(instruments={}, sources={})
    assert script._print_report(snapshot, {})
    output = capsys.readouterr().out
    assert "XAUUSD" in output
    assert "UNAVAILABLE (no result returned)" in output
    assert "no anomaly result was returned" in output
    assert "ETHUSDT" in output


def test_main_repeats_anomaly_api_and_stops_manager(monkeypatch, capsys):
    clock = [0.0]

    class FakeManager:
        stopped = False

        def start(self):
            pass

        def get_market_snapshot(self):
            return SimpleNamespace(
                instruments={item.symbol: SimpleNamespace(
                    connection_state="CONNECTED", freshness="LIVE") for item in INSTRUMENTS
                }, sources={},
            )

        def stop(self):
            self.stopped = True

    manager = FakeManager()
    monkeypatch.setattr(script, "MarketDataManager", lambda: manager)
    monkeypatch.setattr(script, "VolatilityService", lambda: object())
    calls = []

    class FakeAnomalyService:
        def __init__(self, received_manager, *, volatility_service, settings):
            assert received_manager is manager
            assert settings.minimum_training_samples == 50
            self.cached_volatility = volatility_service

        def analyze(self):
            calls.append(1)
            return {item.symbol: make_result("WARMING_UP", samples=len(calls) - 1)
                    for item in INSTRUMENTS}

    monkeypatch.setattr(script, "AnomalyService", FakeAnomalyService)
    monkeypatch.setattr(script.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(script.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))

    assert script.main(["--interval", "1", "--duration", "2"]) == 0
    assert len(calls) == 2
    assert manager.stopped
    output = capsys.readouterr().out
    assert "Completed 2 analysis sample(s)" in output
    assert "not a normal classification" in output


def test_manager_stops_when_analysis_raises(monkeypatch, capsys):
    class FakeManager:
        stopped = False

        def start(self):
            pass

        def stop(self):
            self.stopped = True

    manager = FakeManager()
    monkeypatch.setattr(script, "MarketDataManager", lambda: manager)
    monkeypatch.setattr(script, "VolatilityService", lambda: object())

    class BrokenAnomalyService:
        def __init__(self, *args, **kwargs):
            pass

        def analyze(self):
            raise RuntimeError("mock analyzer failure")

    monkeypatch.setattr(script, "AnomalyService", BrokenAnomalyService)
    monkeypatch.setattr(script.time, "monotonic", lambda: 0.0)

    assert script.main(["--interval", "1", "--duration", "1"]) == 1
    assert manager.stopped
    assert "mock analyzer failure" in capsys.readouterr().out


def test_run_cached_volatility_service_delegates_once():
    class Volatility:
        calls = 0

        def analyze_all(self, timeframe, count):
            self.calls += 1
            return {"XAUUSD": SimpleNamespace(timeframe=timeframe, count=count)}

    wrapped = Volatility()
    cache = script._RunCachedVolatilityService(wrapped)
    first = cache.analyze_all("1h", 200)
    second = cache.analyze_all("1h", 200)
    assert first is second
    assert wrapped.calls == 1


def test_command_line_bounds_sampling_duration():
    with pytest.raises(SystemExit):
        script._parse_args(["--interval", "0", "--duration", "3"])
    with pytest.raises(SystemExit):
        script._parse_args(["--interval", "2", "--duration", "1"])
