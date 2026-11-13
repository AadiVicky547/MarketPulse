"""Mock-only output and lifecycle tests for the manual liquidity check."""
from types import MappingProxyType, SimpleNamespace

from config import INSTRUMENTS
from scripts import test_liquidity_integration as script


def metric(status, value=None, unit="", samples=0, methodology="mock method", warning=None):
    return SimpleNamespace(status=status, value=value, unit=unit, sample_count=samples,
                           methodology=methodology, warning=warning)


def test_report_lists_six_symbols_statuses_metrics_and_binance_buffers(capsys):
    instruments = {item.symbol: SimpleNamespace(error_message=None) for item in INSTRUMENTS}
    trades = {"BTCUSDT": ({"trade_id": 1},), "ETHUSDT": ()}
    book_status = SimpleNamespace(synchronized=True, stale=False)
    books = {
        symbol: {"status": book_status, "bids": ({"price": 99},), "asks": ({"price": 101},)}
        for symbol in ("BTCUSDT", "ETHUSDT")
    }
    snapshot = SimpleNamespace(instruments=MappingProxyType(instruments),
                               binance_trades=MappingProxyType(trades),
                               binance_order_books=MappingProxyType(books))
    results = {
        item.symbol: SimpleNamespace(
            status="WARMING_UP", connection_state="CONNECTED", freshness="LIVE",
            timestamp_trust="UNVERIFIED_TIME_BASIS" if item.source == "MT5" else "TRUSTED_UTC",
            metrics={"spread_price": metric("AVAILABLE", 0.2, "price units", 1),
                     "activity": metric("WARMING_UP", warning="needs more samples")},
            warnings=("mock warning",),
        ) for item in INSTRUMENTS
    }

    script._print_report(snapshot, results)
    output = capsys.readouterr().out
    for symbol in ("XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "BTCUSDT", "ETHUSDT"):
        assert symbol in output
    assert "broker-observed activity" in output
    assert "Order book: SYNCHRONIZED" in output
    assert "trade-buffer count=1" in output
    assert "available depth levels: bids=1, asks=1" in output
    assert "AVAILABLE | value=0.2 price units | samples=1" in output
    assert "WARMING_UP" in output and "mock warning" in output


def test_manager_is_stopped_when_analysis_raises(monkeypatch, capsys):
    class FakeManager:
        stopped = False

        def start(self):
            pass

        def stop(self):
            self.stopped = True

        def is_running(self):
            return False

    manager = FakeManager()
    monkeypatch.setattr(script, "MarketDataManager", lambda: manager)

    class BrokenLiquidityService:
        def __init__(self, manager):
            pass

        def analyze(self):
            raise RuntimeError("mock analysis failure")

    monkeypatch.setattr(script, "LiquidityService", BrokenLiquidityService)
    assert script.main(["--seconds", "1"]) == 1
    assert manager.stopped
    assert "mock analysis failure" in capsys.readouterr().out

