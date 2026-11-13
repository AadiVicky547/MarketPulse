"""Mocked tests for MT5 activity and Binance order-flow analysis."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import MappingProxyType, SimpleNamespace
import time

import pytest

from analytics.liquidity_service import (
    AVAILABLE,
    DISCONNECTED,
    STALE,
    UNAVAILABLE,
    WARMING_UP,
    LiquidityService,
    LiquiditySettings,
)
from config import INSTRUMENTS
from data.market_manager import InstrumentMarketState, MarketSnapshot, SourceStatus


NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


def state(symbol, source, *, freshness="LIVE", connection="CONNECTED", spread="1",
          bid="100", ask="101", mid="100.5", tick_volume=10, timestamp=NOW):
    return InstrumentMarketState(
        requested_instrument=symbol, actual_symbol=symbol, data_source=source,
        bid=Decimal(bid) if bid is not None else None,
        ask=Decimal(ask) if ask is not None else None,
        mid_price=Decimal(mid) if mid is not None else None,
        spread=Decimal(spread) if spread is not None else None,
        spread_bps=(Decimal(spread) / Decimal(mid) * Decimal(10000)
                    if spread is not None and mid is not None else None),
        last_trade_price=Decimal(mid) if mid is not None else None,
        trade_quantity=Decimal("0.5") if source == "Binance" else None,
        tick_volume=tick_volume, volume_semantics=(
            "broker-observed tick activity; not centralized traded volume" if source == "MT5"
            else "Binance exchange-reported executed trade quantity"),
        tick_timestamp=timestamp, last_update_received_at=NOW,
        tick_timestamp_trust="UNVERIFIED_TIME_BASIS" if source == "MT5" else "TRUSTED_UTC",
        freshness=freshness, connection_state=connection,
        order_book_sync_state="SYNCHRONIZED" if source == "Binance" else None,
    )


def trade(trade_id, quantity, buyer_is_maker, *, stale=False):
    return MappingProxyType({
        "trade_id": trade_id, "price": Decimal("100"), "quantity": Decimal(str(quantity)),
        "quote_quantity": Decimal(str(quantity)) * 100,
        "timestamp": NOW, "buyer_is_maker": buyer_is_maker,
        "aggressor_side": "SELL" if buyer_is_maker else "BUY", "stale": stale,
    })


def book(*, synchronized=True, stale=False, update_id=10, available=True,
         bids=None, asks=None):
    return MappingProxyType({
        "available": available, "message": "mock order book",
        "status": MappingProxyType({"synchronized": synchronized, "stale": stale,
            "update_id": update_id, "message": "mock order book",
            "last_update_monotonic": time.monotonic()}),
        "bids": tuple(bids if bids is not None else [
            MappingProxyType({"price": Decimal("99"), "quantity": Decimal("3")}),
            MappingProxyType({"price": Decimal("98"), "quantity": Decimal("1")}),
        ]),
        "asks": tuple(asks if asks is not None else [
            MappingProxyType({"price": Decimal("101"), "quantity": Decimal("1")}),
            MappingProxyType({"price": Decimal("102"), "quantity": Decimal("1")}),
        ]),
    })


class FakeManager:
    def __init__(self, states=None, histories=None, trades=None, books=None,
                 *, source_state="CONNECTED"):
        self.states = states or {}
        self.histories = histories or {}
        self.trades = trades or {}
        self.books = books or {}
        self.source_state = source_state

    def get_market_snapshot(self):
        sources = {
            "MT5": SourceStatus("MT5", "CONNECTED", "mock"),
            "Binance": SourceStatus("Binance", self.source_state, "mock"),
        }
        return MarketSnapshot(NOW, MappingProxyType(self.states), MappingProxyType(sources),
                              MappingProxyType(self.trades), MappingProxyType(self.books))

    def get_history(self, symbol):
        return self.histories.get(symbol, ())


def manager_for(symbol, source, current, *, history=(), trades=None, order_book=None,
                source_state="CONNECTED"):
    states = {item.symbol: state(item.symbol, item.source,
             freshness="UNAVAILABLE", connection="DISCONNECTED", timestamp=None)
             for item in INSTRUMENTS}
    states[symbol] = current
    return FakeManager(states, {symbol: history}, {symbol: trades or ()},
                       {symbol: order_book} if order_book is not None else {}, source_state=source_state)


def test_mt5_spread_and_broker_activity_are_normalized():
    history = [state("EURUSD", "MT5", spread=str(v), tick_volume=v, mid="100.5")
               for v in (1.0, 1.2, 1.4)]
    current = state("EURUSD", "MT5", spread="1.6", tick_volume=16, mid="100.5")
    result = LiquidityService(manager_for("EURUSD", "MT5", current, history=history),
                              settings=LiquiditySettings(baseline_window=5, minimum_baseline_samples=2)).analyze()["EURUSD"]
    assert result.metrics["spread_price"].value == pytest.approx(1.6)
    assert result.metrics["spread_bps"].value == pytest.approx(1.6 / 100.5 * 10000)
    assert result.metrics["spread_expansion_ratio"].value == pytest.approx(1.6 / 1.2)
    assert result.metrics["tick_volume"].value == 16
    assert result.metrics["tick_volume_activity_ratio"].value == pytest.approx(16 / 1.2)
    assert "not centralized" in result.volume_semantics
    assert result.timestamp_trust == "UNVERIFIED_TIME_BASIS"


def test_mt5_spread_expansion_and_tick_activity_anomaly_use_prior_baseline():
    history = [state("XAUUSD", "MT5", spread=str(v), tick_volume=v, mid="100.5")
               for v in (1.0, 1.2, 1.4)]
    current = state("XAUUSD", "MT5", spread="2.4", tick_volume=24, mid="100.5")
    result = LiquidityService(manager_for("XAUUSD", "MT5", current, history=history),
                              settings=LiquiditySettings(baseline_window=5, minimum_baseline_samples=3)).analyze()["XAUUSD"]
    assert result.metrics["spread_expansion_ratio"].status == AVAILABLE
    assert result.metrics["spread_expansion_ratio"].value == pytest.approx(2.0)
    assert result.metrics["spread_anomaly_zscore"].status == AVAILABLE
    assert result.metrics["tick_volume_anomaly_zscore"].status == AVAILABLE


@pytest.mark.parametrize("tick_volume", [0, None])
def test_mt5_zero_or_missing_tick_volume_is_not_fabricated(tick_volume):
    current = state("GBPUSD", "MT5", tick_volume=tick_volume)
    result = LiquidityService(manager_for("GBPUSD", "MT5", current)).analyze()["GBPUSD"]
    if tick_volume == 0:
        assert result.metrics["tick_volume"].status == AVAILABLE
        assert result.metrics["tick_volume"].value == 0
        assert result.metrics["price_move_per_tick_volume"].status == UNAVAILABLE
    else:
        assert result.metrics["tick_volume"].status == WARMING_UP
        assert result.metrics["tick_volume"].value is None


def test_mt5_price_move_per_activity_requires_nonzero_volume_and_history():
    prior = state("USDJPY", "MT5", tick_volume=5, mid="100.0", timestamp=NOW - timedelta(seconds=1))
    current = state("USDJPY", "MT5", tick_volume=10, mid="101.0")
    result = LiquidityService(manager_for("USDJPY", "MT5", current, history=(prior,))).analyze()["USDJPY"]
    assert result.metrics["price_move_per_tick_volume"].value == pytest.approx(0.1)


def test_binance_book_imbalance_and_aggressor_side_are_correct():
    current = state("BTCUSDT", "Binance")
    trades = [trade(1, 2, False), trade(2, 1, True), trade(3, 1, False)]
    manager = manager_for("BTCUSDT", "Binance", current, trades=trades, order_book=book())
    result = LiquidityService(manager).analyze()["BTCUSDT"]
    assert result.metrics["bid_depth"].value == pytest.approx(4)
    assert result.metrics["ask_depth"].value == pytest.approx(2)
    assert result.metrics["order_book_imbalance"].value == pytest.approx(1 / 3)
    assert result.metrics["buy_aggressor_volume"].value == pytest.approx(3)
    assert result.metrics["sell_aggressor_volume"].value == pytest.approx(1)
    assert result.metrics["buy_sell_volume_imbalance"].value == pytest.approx(0.5)
    assert result.metrics["trade_count"].value == 3
    assert result.metrics["spread_price"].value == pytest.approx(1)
    assert "cancellable" in result.warnings[0]


@pytest.mark.parametrize("snapshot_book, expected", [
    (book(synchronized=True, stale=True, available=False), STALE),
    (book(synchronized=False, stale=True, available=False), UNAVAILABLE),
    (book(synchronized=False, stale=False, available=False), UNAVAILABLE),
])
def test_stale_or_unsynchronized_binance_book_does_not_expose_depth(snapshot_book, expected):
    manager = manager_for("BTCUSDT", "Binance", state("BTCUSDT", "Binance"),
                          trades=[trade(1, 1, False)], order_book=snapshot_book)
    result = LiquidityService(manager).analyze()["BTCUSDT"]
    assert result.metrics["order_book_imbalance"].status == expected
    assert result.metrics["order_book_imbalance"].value is None


def test_empty_trades_warm_up_and_disconnected_source_is_explicit():
    current = state("ETHUSDT", "Binance")
    live = LiquidityService(manager_for("ETHUSDT", "Binance", current,
                            trades=[], order_book=book())).analyze()["ETHUSDT"]
    assert live.metrics["trade_count"].status == WARMING_UP
    disconnected_manager = manager_for("ETHUSDT", "Binance",
        state("ETHUSDT", "Binance", connection="DISCONNECTED", freshness="DISCONNECTED"),
        trades=[], order_book=book(), source_state="DISCONNECTED")
    disconnected = LiquidityService(disconnected_manager).analyze()["ETHUSDT"]
    assert disconnected.status == DISCONNECTED
    assert disconnected.metrics["order_book_imbalance"].status == DISCONNECTED


def test_invalid_book_level_and_missing_quotes_are_reported():
    invalid = book(bids=[{"price": "bad", "quantity": "2"}],
                   asks=[{"price": "101", "quantity": "1"}])
    result = LiquidityService(manager_for("BTCUSDT", "Binance",
        state("BTCUSDT", "Binance", bid=None, ask=None, mid=None, spread=None),
        trades=[], order_book=invalid)).analyze()["BTCUSDT"]
    assert result.metrics["spread_price"].status == UNAVAILABLE
    assert result.metrics["bid_depth"].status == UNAVAILABLE
    assert any("Invalid bid book level" in warning for warning in result.warnings)


def test_depth_history_is_bounded_and_reports_changes():
    current_book = book(update_id=1)
    manager = manager_for("BTCUSDT", "Binance", state("BTCUSDT", "Binance"),
                          trades=[trade(1, 1, False)], order_book=current_book)
    service = LiquidityService(manager, settings=LiquiditySettings(
        baseline_window=2, minimum_baseline_samples=2, book_history_size=3))
    service.analyze()
    for update_id, bid_qty in ((2, 4), (3, 5), (4, 6)):
        manager.books["BTCUSDT"] = book(update_id=update_id, bids=[
            {"price": Decimal("99"), "quantity": Decimal(str(bid_qty))}])
        result = service.analyze()["BTCUSDT"]
    assert len(service._book_history["BTCUSDT"]) == 3
    assert result.metrics["bid_depth_change"].value == pytest.approx(1)
    assert result.metrics["bid_depth_anomaly_zscore"].status == AVAILABLE


def test_all_six_configured_instruments_return_structured_results():
    manager = FakeManager()
    results = LiquidityService(manager).analyze()
    assert tuple(results) == tuple(item.symbol for item in INSTRUMENTS)
    assert all(result.status == UNAVAILABLE for result in results.values())

