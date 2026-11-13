"""Mocked tests for Liquidity & Order Flow page preparation and sampling."""
from concurrent.futures import Future
from datetime import datetime, timezone
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch

from config import INSTRUMENTS
from dashboard.liquidity_view import (
    LiquidityPageController,
    _render_metrics,
    metric_display,
    prepare_instrument_view,
)
import dashboard.liquidity_view as liquidity_view


def metric(status, value=None, unit="", samples=0, warning=None):
    return SimpleNamespace(status=status, value=value, unit=unit,
                           sample_count=samples, warning=warning,
                           methodology="mock methodology")


def result(symbol, source, *, status="AVAILABLE", freshness="LIVE", metrics=None,
           connection="CONNECTED", warnings=()):
    return SimpleNamespace(
        instrument=symbol, source=source, status=status, freshness=freshness,
        connection_state=connection, timestamp_trust=("UNVERIFIED_TIME_BASIS" if source == "MT5" else "TRUSTED_UTC"),
        timestamp=datetime.now(timezone.utc), received_at=datetime.now(timezone.utc),
        order_book_sync_state="SYNCHRONIZED" if source == "Binance" else None,
        volume_semantics="Broker-observed tick volume; not centralized traded volume" if source == "MT5" else "Binance executed quantity",
        metrics=metrics or {}, warnings=warnings,
    )


def test_metric_display_keeps_zero_and_unavailable_distinct():
    assert metric_display(metric("AVAILABLE", 0, "ratio")) == "0 ratio"
    assert metric_display(metric("WARMING_UP", None, "ratio", warning="more samples needed")) == (
        "WARMING_UP — more samples needed")
    assert metric_display(metric("STALE", None)) == "STALE"
    assert metric_display(None) == "UNAVAILABLE"


def test_metric_cards_do_not_render_repeated_status_methodology_or_warning_captions():
    calls = []

    class Column:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def metric(self, label, value):
            calls.append((label, value))

    metrics = {"spread_price": metric("AVAILABLE", 0.1, "price units", 1)}
    with patch("dashboard.liquidity_view.st.columns", return_value=[Column(), Column(), Column()]), \
         patch("dashboard.liquidity_view.st.markdown"), \
         patch("dashboard.liquidity_view.st.metric", side_effect=lambda label, value: calls.append((label, value))), \
         patch("dashboard.liquidity_view.st.caption") as caption:
        _render_metrics({"metrics": metrics, "provider": "MT5"})

    caption.assert_not_called()
    assert calls[0] == ("Spread", "0.1 price units")
    assert len(calls) == 9


def test_mt5_view_exposes_broker_quotes_and_no_fake_book():
    state = SimpleNamespace(
        actual_symbol="EURUSDm", connection_state="CONNECTED", freshness="LIVE",
        bid=1.1, ask=1.1002, mid_price=1.1001,
        tick_timestamp_trust="UNVERIFIED_TIME_BASIS",
        error_message=None, last_update_received_at=datetime.now(timezone.utc),
    )
    metrics = {
        "spread_price": metric("AVAILABLE", 0.0002, "price units", 1),
        "tick_volume": metric("AVAILABLE", 0, "broker tick-volume units", 1),
        "price_move_per_tick_volume": metric("UNAVAILABLE", None,
                                              "price units per broker tick-volume unit", warning="zero volume"),
    }
    prepared = prepare_instrument_view(
        "EURUSD", result("EURUSD", "MT5", metrics=metrics),
        SimpleNamespace(instruments={"EURUSD": state}),
    )
    assert prepared["provider"] == "MT5"
    assert prepared["actual_symbol"] == "EURUSDm"
    assert prepared["bid"] == 1.1 and prepared["ask"] == 1.1002
    assert prepared["timestamp_trust"] == "UNVERIFIED_TIME_BASIS"
    assert prepared["book_available"] is False
    assert prepared["book_levels"] == {"bids": (), "asks": ()}
    assert metric_display(prepared["metrics"]["tick_volume"]) == "0 broker tick-volume units"
    assert metric_display(prepared["metrics"]["price_move_per_tick_volume"]).startswith("UNAVAILABLE")


def test_binance_depth_is_only_exposed_when_synchronized_fresh_and_metrics_available():
    state = SimpleNamespace(connection_state="CONNECTED", freshness="LIVE", bid=100,
                            ask=101, mid_price=100.5, tick_timestamp_trust="TRUSTED_UTC")
    book_status = SimpleNamespace(synchronized=True, stale=False)
    book = {"available": True, "status": book_status,
            "bids": (MappingProxyType({"price": 100, "quantity": 2}),),
            "asks": (MappingProxyType({"price": 101, "quantity": 3}),)}
    metrics = {name: metric("AVAILABLE", value, "units", 1) for name, value in (
        ("bid_depth", 2), ("ask_depth", 3), ("order_book_imbalance", -0.2))}
    snapshot = SimpleNamespace(instruments={"BTCUSDT": state}, binance_order_books={"BTCUSDT": book})
    prepared = prepare_instrument_view("BTCUSDT", result("BTCUSDT", "Binance", metrics=metrics), snapshot)
    assert prepared["book_available"]
    assert prepared["book_sync"] == "SYNCHRONIZED"
    assert len(prepared["book_levels"]["bids"]) == 1

    book_status.stale = True
    stale = prepare_instrument_view("BTCUSDT", result("BTCUSDT", "Binance", metrics=metrics), snapshot)
    assert not stale["book_available"]
    assert stale["book_levels"]["bids"] == ()
    assert stale["book_sync"] == "STALE"


def test_binance_unsynchronized_or_unavailable_metrics_withhold_depth():
    state = SimpleNamespace(connection_state="CONNECTED", freshness="LIVE", bid=100,
                            ask=101, mid_price=100.5, tick_timestamp_trust="TRUSTED_UTC")
    book = {"available": True, "status": SimpleNamespace(synchronized=False, stale=False),
            "bids": ({"price": 100, "quantity": 2},), "asks": ({"price": 101, "quantity": 3},)}
    snapshot = SimpleNamespace(instruments={"BTCUSDT": state}, binance_order_books={"BTCUSDT": book})
    metrics = {name: metric("UNAVAILABLE", None) for name in
               ("bid_depth", "ask_depth", "order_book_imbalance")}
    prepared = prepare_instrument_view("BTCUSDT", result("BTCUSDT", "Binance", metrics=metrics), snapshot)
    assert prepared["book_sync"] == "UNSYNCHRONIZED"
    assert not prepared["book_available"]
    assert prepared["book_levels"]["asks"] == ()


def test_refresh_button_is_spaced_to_align_with_instrument_dropdown(monkeypatch):
    calls = []

    class Column:
        def selectbox(self, label, options, **kwargs):
            calls.append(("selectbox", label))
            return "XAUUSD"

        def caption(self, text):
            calls.append(("caption", text))

        def markdown(self, text, **kwargs):
            calls.append(("spacer", text))

        def button(self, label, **kwargs):
            calls.append(("button", label))
            return False

    columns = [Column(), Column(), Column()]
    monkeypatch.setattr(liquidity_view, "render_page_header", lambda *args: None)
    monkeypatch.setattr(liquidity_view.st, "columns", lambda *_args, **_kwargs: columns)
    monkeypatch.setattr(liquidity_view, "_liquidity_fragment", lambda *args: None)

    liquidity_view.render_liquidity_view()

    spacer_index = next(i for i, item in enumerate(calls) if item[0] == "spacer")
    button_index = next(i for i, item in enumerate(calls) if item[0] == "button")
    assert calls[spacer_index][1].find("height: 1.9rem") >= 0
    assert spacer_index < button_index


class DeferredExecutor:
    def __init__(self):
        self.fn = None
        self.future = None
        self.shutdown_called = False

    def submit(self, fn):
        self.fn = fn
        self.future = Future()
        return self.future

    def run(self):
        self.future.set_result(self.fn())

    def shutdown(self, *, wait, cancel_futures):
        self.shutdown_called = True
        assert wait and cancel_futures


def test_controller_runs_mocked_analysis_off_render_path_and_bounds_sample_history():
    states = {item.symbol: SimpleNamespace(connection_state="CONNECTED", freshness="LIVE")
              for item in INSTRUMENTS}
    snapshot = SimpleNamespace(instruments=MappingProxyType(states), binance_order_books={})

    class Manager:
        def get_market_snapshot(self):
            return snapshot

    metrics = {"spread_price": metric("AVAILABLE", 0.01, "price units", 1)}

    class Service:
        def analyze(self):
            return {item.symbol: result(item.symbol, item.source, metrics=metrics) for item in INSTRUMENTS}

    executor = DeferredExecutor()
    controller = LiquidityPageController(Manager(), service=Service(), executor=executor, history_size=2)
    assert controller.request().status == "WARMING_UP"
    executor.run()
    first = controller.request()
    assert first.status == "AVAILABLE"
    assert len(first.history["BTCUSDT"]) == 1
    executor.run()
    second = controller.request()
    assert len(second.history["BTCUSDT"]) == 2
    executor.run()
    bounded = controller.request()
    assert len(bounded.history["BTCUSDT"]) == 2
    controller.close()
    assert executor.shutdown_called
