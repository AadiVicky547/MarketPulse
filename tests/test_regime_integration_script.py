"""Mocked report and lifecycle tests for the manual regime integration script."""
from types import MappingProxyType, SimpleNamespace

from config import INSTRUMENTS
from scripts import test_regime_integration as script


def feature(status, value, unit="test", samples=3):
    return SimpleNamespace(status=status, value=value, unit=unit, sample_count=samples,
                           methodology="mock formula", warning=None)


def test_report_shows_all_instruments_and_distinguishes_zero_from_unavailable(capsys):
    instrument_states = {item.symbol: SimpleNamespace(
        connection_state="CONNECTED", freshness="LIVE") for item in INSTRUMENTS}
    snapshot = SimpleNamespace(instruments=MappingProxyType(instrument_states), sources={})
    results = {item.symbol: SimpleNamespace(
        source=item.source, status="AVAILABLE", current_regime="RANGING",
        evidence_strength="STRONG", timestamp_trust=(
            "UNVERIFIED_TIME_BASIS" if item.source == "MT5" else "TRUSTED_UTC"),
        last_observation_freshness="LIVE", candle_count=600,
        features={"ema_slope_bps": feature("AVAILABLE", 0.0, "basis points"),
                  "adx": feature("WARMING_UP", None, "index", 0)},
        explanation="Mocked classification explanation.", data_quality_warnings=(),
    ) for item in INSTRUMENTS}

    failed = script._print_report(snapshot, results)
    output = capsys.readouterr().out
    assert not failed
    for symbol in ("XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "BTCUSDT", "ETHUSDT"):
        assert symbol in output
    assert "value=0 basis points" in output
    assert "value=unavailable index" in output
    assert "Evidence strength: STRONG (rule-support category; not a probability)" in output
    assert "UNVERIFIED_TIME_BASIS" in output and "TRUSTED_UTC" in output


def test_main_stops_manager_when_regime_analysis_fails(monkeypatch, capsys):
    class FakeManager:
        stopped = False

        def start(self):
            pass

        def is_running(self):
            return False

        def get_market_snapshot(self):
            return SimpleNamespace(instruments={}, sources={})

        def stop(self):
            self.stopped = True

    manager = FakeManager()
    monkeypatch.setattr(script, "MarketDataManager", lambda: manager)

    class BrokenRegimeService:
        def __init__(self, *, market_manager):
            assert market_manager is manager

        def analyze_all(self, timeframe, count):
            raise RuntimeError("mock regime failure")

    monkeypatch.setattr(script, "RegimeService", BrokenRegimeService)
    assert script.main(["--seconds", "1"]) == 1
    assert manager.stopped
    assert "mock regime failure" in capsys.readouterr().out


def test_main_runs_with_mock_manager_and_reports_six_symbols(monkeypatch, capsys):
    class FakeManager:
        stopped = False

        def start(self):
            pass

        def is_running(self):
            return False

        def get_market_snapshot(self):
            return SimpleNamespace(instruments={
                item.symbol: SimpleNamespace(connection_state="CONNECTED", freshness="LIVE")
                for item in INSTRUMENTS
            }, sources={})

        def stop(self):
            self.stopped = True

    manager = FakeManager()
    monkeypatch.setattr(script, "MarketDataManager", lambda: manager)

    class FakeRegimeService:
        def __init__(self, *, market_manager):
            pass

        def analyze_all(self, timeframe, count):
            return {item.symbol: SimpleNamespace(
                status="WARMING_UP", current_regime=None, evidence_strength=None,
                timestamp_trust="UNVERIFIED_TIME_BASIS" if item.source == "MT5" else "TRUSTED_UTC",
                last_observation_freshness="LIVE", candle_count=3, features={},
                explanation="Need more candles", data_quality_warnings=(),
            ) for item in INSTRUMENTS}

    monkeypatch.setattr(script, "RegimeService", FakeRegimeService)
    assert script.main(["--seconds", "1"]) == 0
    assert manager.stopped
    output = capsys.readouterr().out
    assert all(item.symbol in output for item in INSTRUMENTS)
    assert "Completed regime report for 6 configured instruments" in output

