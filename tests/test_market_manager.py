"""Mock-based tests for unified MarketPulse market-data management."""
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from decimal import Decimal
from threading import Event
import time
from types import SimpleNamespace

import pytest

from data.market_manager import MarketDataManager


class FakeMT5:
    def __init__(self, *, initialize_ok=True, available=True, changing=False, symbol_missing=False):
        self.initialize_ok = initialize_ok
        self.available = available
        self.changing = changing
        self.event_stale = False
        self.symbol_missing = symbol_missing
        self.connected = False
        self.initialized = 0
        self.closed = 0
        self.reads = 0

    def initialize(self):
        self.initialized += 1
        self.connected = self.initialize_ok
        return SimpleNamespace(connected=self.initialize_ok,
                               message="test connected" if self.initialize_ok else "test failure",
                               error=None if self.initialize_ok else "mock initialization error")

    def discover_symbols(self):
        if self.symbol_missing:
            return {symbol: SimpleNamespace(actual_symbol=None, message="mock symbol unavailable")
                    for symbol in ("XAUUSD", "EURUSD", "GBPUSD", "USDJPY")}
        return {symbol: SimpleNamespace(actual_symbol=symbol, message="exact")
                for symbol in ("XAUUSD", "EURUSD", "GBPUSD", "USDJPY")}

    def get_tick(self, symbol):
        self.reads += 1
        if not self.available:
            return {"available": False, "error": "mock tick unavailable"}
        offset = self.reads if self.changing else 0
        return {"available": True, "bid": 100.0 + offset, "ask": 101.0 + offset,
                "last": 100.5 + offset, "tick_volume": 0,
                "timestamp": datetime.fromtimestamp(1_800_000_000 + offset, timezone.utc),
                "received_at": datetime.now(timezone.utc),
                "raw_time": 1_800_000_000 + offset,
                "raw_time_msc": (1_800_000_000 + offset) * 1000,
                "timestamp_source": "time_msc", "timestamp_status": "VALID",
                "timestamp_trust": "UNVERIFIED_TIME_BASIS",
                "stale": self.event_stale, "error": None}

    def close(self):
        self.connected = False
        self.closed += 1


class FakeBinance:
    def __init__(self, *, available=True, old=False):
        self.available = available
        self.old = old
        self.running = False
        self.started = Event()
        self.stopped = Event()
        received = time.monotonic() - (10 if old else 0)
        self.trade = {"trade_id": 7, "price": Decimal("200"), "quantity": Decimal("0.5"),
                      "timestamp": datetime.now(timezone.utc), "received_monotonic": received,
                      "stale": old}
        self.ticker = {"update_id": 22, "bid_price": Decimal("199"),
                       "ask_price": Decimal("201"), "received_monotonic": received,
                       "stale": old}
        self.order_books = {symbol: SimpleNamespace(status=SimpleNamespace(synchronized=True, stale=False))
                            for symbol in ("BTCUSDT", "ETHUSDT")}

    async def start(self):
        self.running = True
        self.started.set()

    async def stop(self):
        self.running = False
        self.stopped.set()

    def connection_status(self):
        state = "connected" if self.running else "stopped"
        return {"status": state, "symbols": {symbol: {"connected": self.running, "stale": False}
                                               for symbol in self.order_books}}

    def latest_trades(self, symbol, count=1):
        return [self.trade] if self.available else []

    def latest_book_ticker(self, symbol):
        return self.ticker if self.available else None

    def order_book_metrics(self, symbol, levels=5):
        status = SimpleNamespace(synchronized=self.available, stale=self.old,
                                 update_id=22, message="mock book",
                                 last_update_monotonic=time.monotonic())
        return {"available": self.available and not self.old, "status": status,
                "bids": [{"price": Decimal("199"), "quantity": Decimal("2")}],
                "asks": [{"price": Decimal("201"), "quantity": Decimal("3")}],
                "total_bid_depth": Decimal("2"), "total_ask_depth": Decimal("3"),
                "mid_price": Decimal("200"), "spread": Decimal("2"), "obi": Decimal("-0.2")}


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def manager(mt5_factory=None, binance_factory=None, **kwargs):
    subject = MarketDataManager(mt5_poll_interval=kwargs.pop("mt5_poll_interval", 0.01),
                                 binance_poll_interval=kwargs.pop("binance_poll_interval", 0.01),
                                 mt5_connector_factory=mt5_factory or FakeMT5,
                                 binance_connector_factory=binance_factory or FakeBinance,
                                 **kwargs)

    # These orchestration tests do not start a real event loop or network
    # connector. The production worker's async lifecycle is covered through its
    # injected connector boundary and the synchronous normalization interface.
    def mocked_binance_worker():
        connector = None
        try:
            connector = subject._binance_factory()
            connector.running = True
            with subject._lock:
                subject._connectors["Binance"] = connector
            subject._set_source("Binance", "CONNECTED", "Mock Binance stream connected")
            while not subject._stop_event.is_set():
                for symbol in subject._binance_symbols():
                    subject._ingest_binance(symbol, connector)
                subject._stop_event.wait(subject.binance_poll_interval)
        except Exception as exc:
            subject._set_source("Binance", "DISCONNECTED", "Mock Binance source failed", str(exc))
        finally:
            if connector is not None:
                connector.running = False
                connector.stopped.set()
                with subject._lock:
                    subject._connectors.pop("Binance", None)
            if subject._stop_event.is_set():
                subject._set_source("Binance", "DISCONNECTED", "Binance manager stopped")

    subject._binance_thread_main = mocked_binance_worker
    return subject


def test_six_instruments_registered_before_start():
    subject = manager()
    assert subject.instruments == ("XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "BTCUSDT", "ETHUSDT")
    snapshot = subject.get_market_snapshot()
    assert len(snapshot.instruments) == 6
    assert snapshot.instruments["XAUUSD"].data_source == "MT5"
    assert snapshot.instruments["BTCUSDT"].data_source == "Binance"


def test_sources_start_independently_and_normalize_distinct_volume_fields():
    mt5_instance = FakeMT5()
    binance_instances = []
    def binance_factory():
        item = FakeBinance()
        binance_instances.append(item)
        return item
    subject = manager(lambda: mt5_instance, binance_factory)
    subject.start()
    try:
        assert wait_until(lambda: subject.get_instrument_state("XAUUSD").freshness == "LIVE")
        assert wait_until(lambda: subject.get_instrument_state("BTCUSDT").freshness == "LIVE")
        mt5_state = subject.get_instrument_state("XAUUSD")
        binance_state = subject.get_instrument_state("BTCUSDT")
        assert mt5_state.actual_symbol == "XAUUSD"
        assert mt5_state.tick_volume == 0
        assert mt5_state.trade_quantity is None
        assert "broker-observed" in mt5_state.volume_semantics
        assert binance_state.tick_volume is None
        assert binance_state.trade_quantity == Decimal("0.5")
        assert "exchange-reported" in binance_state.volume_semantics
        assert binance_state.order_book_sync_state == "SYNCHRONIZED"
    finally:
        subject.stop()
    assert mt5_instance.closed == 1
    assert binance_instances[0].stopped.is_set()


def test_failure_in_mt5_source_does_not_stop_binance():
    def broken_mt5():
        raise RuntimeError("mock MT5 construction failure")
    binance_instances = []
    def binance_factory():
        item = FakeBinance()
        binance_instances.append(item)
        return item
    subject = manager(broken_mt5, binance_factory)
    subject.start()
    try:
        assert wait_until(lambda: subject.get_instrument_state("BTCUSDT").freshness == "LIVE")
        assert wait_until(lambda: subject.get_source_status()["MT5"].connection_state == "DISCONNECTED")
        assert subject.get_instrument_state("XAUUSD").freshness == "DISCONNECTED"
        assert subject.get_source_status()["Binance"].connection_state == "CONNECTED"
    finally:
        subject.stop()


def test_mt5_initialization_failure_does_not_block_binance_worker():
    mt5 = FakeMT5(initialize_ok=False)
    subject = manager(lambda: mt5, FakeBinance)
    subject.start()
    try:
        assert wait_until(lambda: subject.get_source_status()["MT5"].connection_state == "DISCONNECTED")
        assert wait_until(lambda: subject.get_instrument_state("BTCUSDT").freshness == "LIVE")
        assert mt5.initialized >= 1
    finally:
        subject.stop()


def test_start_returns_promptly_while_mt5_initialization_is_blocked():
    entered = Event()
    release = Event()

    class SlowMT5(FakeMT5):
        def initialize(self):
            entered.set()
            release.wait(3.0)
            return super().initialize()

    mt5 = SlowMT5()
    subject = manager(lambda: mt5, FakeBinance, shutdown_timeout_seconds=1.0)
    started_at = time.monotonic()
    subject.start()
    elapsed = time.monotonic() - started_at
    try:
        assert elapsed < 0.5
        assert entered.wait(1.0)
        # MT5 is still initializing, but the independently launched Binance
        # source reaches live state without waiting for the MT5 call to finish.
        assert wait_until(lambda: subject.get_instrument_state("BTCUSDT").freshness == "LIVE")
        assert subject.get_source_status()["MT5"].connection_state == "CONNECTING"
    finally:
        release.set()
        subject.stop()
    assert mt5.closed == 1


def test_mt5_receive_freshness_is_independent_of_unverified_event_clock():
    mt5 = FakeMT5()
    mt5.event_stale = True  # Event-age estimate is untrusted; receive is recent.
    binance = FakeBinance(old=True)
    subject = manager(lambda: mt5, lambda: binance, mt5_freshness_seconds=0.05,
                      binance_freshness_seconds=0.05)
    subject.start()
    try:
        assert wait_until(lambda: subject.get_source_status()["Binance"].connection_state == "CONNECTED")
        assert subject.get_instrument_state("BTCUSDT").freshness == "STALE"
        assert wait_until(lambda: subject.get_instrument_state("XAUUSD").last_update_received_at is not None)
        mt5_state = subject.get_instrument_state("XAUUSD")
        assert mt5_state.freshness == "LIVE"
        assert mt5_state.tick_timestamp_trust == "UNVERIFIED_TIME_BASIS"
        assert mt5_state.tick_timestamp_source == "time_msc"
        assert mt5_state.tick_raw_time == 1_800_000_000
        assert mt5_state.tick_raw_time_msc == 1_800_000_000_000
        time.sleep(0.08)
        # Successful polls refresh receive time even when the broker event
        # itself has not changed; this tests arrival/connection health only.
        assert subject.get_instrument_state("XAUUSD").freshness == "LIVE"
    finally:
        subject.stop()


def test_mt5_stops_refreshing_receive_time_when_tick_becomes_unavailable():
    mt5 = FakeMT5()
    subject = manager(lambda: mt5, FakeBinance, mt5_freshness_seconds=0.05,
                      mt5_poll_interval=0.01)
    subject.start()
    try:
        assert wait_until(lambda: subject.get_instrument_state("XAUUSD").last_update_received_at is not None)
        mt5.available = False
        assert wait_until(lambda: subject.get_instrument_state("XAUUSD").freshness == "STALE")
    finally:
        subject.stop()


def test_received_tick_with_invalid_event_time_keeps_raw_provenance_and_receive_time():
    class InvalidTimestampMT5(FakeMT5):
        def get_tick(self, symbol):
            return {"tick_received": True, "available": False,
                    "bid": 100.0, "ask": 101.0, "last": 100.5, "tick_volume": 0,
                    "timestamp": None, "received_at": datetime.now(timezone.utc),
                    "raw_time": 0, "raw_time_msc": -1,
                    "timestamp_source": None, "timestamp_status": "INVALID",
                    "timestamp_trust": "UNAVAILABLE", "timestamp_warning": "invalid epoch",
                    "stale": None, "error": "invalid epoch"}

    mt5 = InvalidTimestampMT5()
    subject = manager(lambda: mt5, FakeBinance)
    subject.start()
    try:
        assert wait_until(lambda: subject.get_instrument_state("XAUUSD").last_update_received_at is not None)
        state = subject.get_instrument_state("XAUUSD")
        assert state.tick_timestamp is None
        assert state.tick_timestamp_trust == "UNAVAILABLE"
        assert state.tick_timestamp_status == "INVALID"
        assert state.tick_raw_time == 0 and state.tick_raw_time_msc == -1
        assert state.error_message == "invalid epoch"
    finally:
        subject.stop()


def test_snapshot_is_immutable_and_history_is_bounded():
    mt5 = FakeMT5(changing=True)
    subject = manager(lambda: mt5, FakeBinance, history_size=2)
    subject.start()
    try:
        assert wait_until(lambda: len(subject.get_history("XAUUSD")) == 2)
        assert wait_until(lambda: subject.get_market_snapshot().binance_order_books["BTCUSDT"]["available"])
        frozen_snapshot = subject.get_market_snapshot()
        old_state = frozen_snapshot.instruments["XAUUSD"]
        time.sleep(0.04)
        assert len(subject.get_history("XAUUSD")) == 2
        assert subject.get_instrument_state("XAUUSD").last_trade_price != old_state.last_trade_price
        with pytest.raises(TypeError):
            frozen_snapshot.instruments["NEW"] = old_state
        assert frozen_snapshot.binance_trades["BTCUSDT"]
        assert frozen_snapshot.binance_order_books["BTCUSDT"]["available"] is True
        with pytest.raises(TypeError):
            frozen_snapshot.binance_order_books["BTCUSDT"]["available"] = False
        with pytest.raises(FrozenInstanceError):
            old_state.bid = Decimal("0")
    finally:
        subject.stop()


def test_missing_market_data_is_unavailable_without_fabricated_prices():
    subject = manager(lambda: FakeMT5(available=False), lambda: FakeBinance(available=False))
    subject.start()
    try:
        assert wait_until(lambda: subject.get_source_status()["MT5"].connection_state == "CONNECTED")
        mt5_state = subject.get_instrument_state("EURUSD")
        binance_state = subject.get_instrument_state("ETHUSDT")
        assert mt5_state.freshness == "UNAVAILABLE"
        assert mt5_state.bid is mt5_state.ask is mt5_state.last_trade_price is None
        assert binance_state.freshness == "UNAVAILABLE"
        assert binance_state.bid is binance_state.trade_quantity is None
    finally:
        subject.stop()


def test_repeated_start_and_stop_do_not_duplicate_workers():
    mt5_instances = []
    binance_instances = []
    def mt5_factory():
        item = FakeMT5()
        mt5_instances.append(item)
        return item
    def binance_factory():
        item = FakeBinance()
        binance_instances.append(item)
        return item
    subject = manager(mt5_factory, binance_factory)
    subject.start()
    first_threads = dict(subject._threads)
    subject.start()
    assert subject._threads == first_threads
    assert wait_until(lambda: subject.get_instrument_state("BTCUSDT").freshness == "LIVE")
    subject.stop()
    subject.stop()
    assert not subject.is_running()
    if mt5_instances:
        assert mt5_instances[0].closed == 1
    assert binance_instances[0].stopped.is_set()
    subject.start()
    try:
        assert wait_until(lambda: len(binance_instances) == 2)
    finally:
        subject.stop()


def test_missing_mt5_symbol_is_unavailable_even_if_source_is_connected():
    subject = manager(lambda: FakeMT5(symbol_missing=True), FakeBinance)
    subject.start()
    try:
        assert wait_until(lambda: subject.get_source_status()["MT5"].connection_state == "CONNECTED")
        state = subject.get_instrument_state("XAUUSD")
        assert state.freshness == "UNAVAILABLE"
        assert state.actual_symbol is None
        assert "unavailable" in state.error_message
    finally:
        subject.stop()


def test_unexpected_mt5_poll_exception_recreates_connector_and_recovers_binance_continues():
    instances = []

    class FirstPollFails(FakeMT5):
        def get_tick(self, symbol):
            raise RuntimeError("mock unexpected poll failure")

    def mt5_factory():
        connector = FirstPollFails() if not instances else FakeMT5()
        instances.append(connector)
        return connector

    subject = manager(mt5_factory, FakeBinance,
                      reconnect_initial_seconds=0.05, reconnect_max_seconds=0.1)
    subject.start()
    try:
        assert wait_until(lambda: len(instances) >= 1 and instances[0].closed == 1)
        assert subject.get_source_status()["Binance"].connection_state == "CONNECTED"
        assert subject.get_instrument_state("BTCUSDT").freshness == "LIVE"
        assert wait_until(lambda: len(instances) >= 2 and
                          subject.get_instrument_state("XAUUSD").freshness == "LIVE")
        assert instances[0].initialized == 1
        assert instances[0].closed == 1
        assert instances[1].initialized == 1
        assert subject.get_source_status()["MT5"].connection_state == "CONNECTED"
    finally:
        subject.stop()


def test_start_restarts_only_dead_mt5_worker_while_binance_is_running():
    subject = manager()
    original_mt5_worker = subject._mt5_thread_main
    attempts = 0

    def exits_once_then_runs():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return
        original_mt5_worker()

    subject._mt5_thread_main = exits_once_then_runs
    subject.start()
    try:
        assert wait_until(lambda: subject.get_worker_health()["Binance"] and
                          not subject.get_worker_health()["MT5"])
        binance_thread = subject._threads["Binance"]
        first_mt5_thread = subject._threads["MT5"]
        assert not subject.is_running()

        subject.start()

        assert subject._threads["Binance"] is binance_thread
        assert subject._threads["MT5"] is not first_mt5_thread
        assert wait_until(lambda: subject.get_instrument_state("XAUUSD").freshness == "LIVE")
        assert subject.is_running()
    finally:
        subject.stop()


def test_shutdown_interrupts_mt5_reconnection_backoff_without_new_connector():
    instances = []

    class AlwaysFailsPolling(FakeMT5):
        def get_tick(self, symbol):
            raise RuntimeError("mock poll failure before shutdown")

    def mt5_factory():
        connector = AlwaysFailsPolling()
        instances.append(connector)
        return connector

    subject = manager(mt5_factory, FakeBinance,
                      reconnect_initial_seconds=5.0, reconnect_max_seconds=10.0)
    subject.start()
    assert wait_until(lambda: instances and instances[0].closed == 1)
    started = time.monotonic()
    subject.stop()
    assert time.monotonic() - started < 1.0
    assert len(instances) == 1
    assert instances[0].closed == 1
    assert not subject.get_worker_health()["MT5"]
