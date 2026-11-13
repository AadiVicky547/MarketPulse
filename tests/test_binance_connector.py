"""Tests for public Binance Spot payload normalization and task lifecycle."""
from datetime import datetime, timezone
from decimal import Decimal
import asyncio
from unittest.mock import Mock

import pytest

from data.binance_connector import BinanceMarketDataConnector, normalize_book_ticker, normalize_trade


def test_trade_normalization_and_aggressor_interpretation():
    trade = normalize_trade({"e": "trade", "s": "BTCUSDT", "t": 42, "p": "10.25", "q": "2",
                             "T": 1700000000123, "m": True})
    assert trade["trade_id"] == 42
    assert trade["quote_quantity"] == Decimal("20.50")
    assert trade["buyer_is_maker"] is True
    assert trade["aggressor_side"] == "SELL"
    assert trade["timestamp"].tzinfo == timezone.utc
    taker_buy = normalize_trade({"s": "ETHUSDT", "t": 43, "p": "1", "q": "1", "T": 1700000000123, "m": False})
    assert taker_buy["aggressor_side"] == "BUY"


def test_best_bid_ask_mid_and_spread_basis_points():
    ticker = normalize_book_ticker({"s": "BTCUSDT", "u": 8, "b": "99", "B": "2",
                                    "a": "101", "A": "3"})
    assert ticker["bid_price"] == Decimal("99")
    assert ticker["bid_quantity"] == Decimal("2")
    assert ticker["ask_price"] == Decimal("101")
    assert ticker["ask_quantity"] == Decimal("3")
    assert ticker["mid_price"] == Decimal("100")
    assert ticker["spread"] == Decimal("2")
    assert ticker["spread_bps"] == Decimal("200")


@pytest.mark.parametrize("payload", [
    {"s": "BTCUSDT", "t": 1, "p": "0", "q": "1", "T": 1, "m": False},
    {"s": "BTCUSDT", "t": 1, "p": "NaN", "q": "1", "T": 1, "m": False},
    {"s": "BTCUSDT", "t": 1, "p": "1", "q": "1", "T": 1, "m": "false"},
])
def test_invalid_trade_payload_rejected(payload):
    with pytest.raises(ValueError):
        normalize_trade(payload)


def test_invalid_crossed_or_missing_ticker_rejected():
    with pytest.raises(ValueError, match="below best bid"):
        normalize_book_ticker({"s": "BTCUSDT", "u": 1, "b": "2", "B": "1", "a": "1", "A": "1"})
    with pytest.raises(ValueError, match="payload"):
        normalize_book_ticker({"s": "BTCUSDT"})


def test_order_book_metrics_unavailable_until_synchronized():
    connector = BinanceMarketDataConnector()
    result = connector.order_book_metrics("BTCUSDT")
    assert result["available"] is False
    assert result["bids"] == []


def test_rest_snapshot_uses_public_depth_endpoint_and_validates_response():
    response = Mock()
    response.json.return_value = {"lastUpdateId": 123, "bids": [["10", "2"]], "asks": [["11", "3"]]}
    get = Mock(return_value=response)
    connector = BinanceMarketDataConnector(rest_get=get)
    snapshot = connector._get_snapshot("BTCUSDT")
    assert snapshot["lastUpdateId"] == 123
    get.assert_called_once_with("https://api.binance.com/api/v3/depth",
                                params={"symbol": "BTCUSDT", "limit": 1000}, timeout=(3, 5))


def test_order_book_metrics_zero_depth_has_no_obi():
    from data.orderbook import LocalOrderBook
    item = LocalOrderBook("BTCUSDT")
    item.load_snapshot({"lastUpdateId": 5, "bids": [], "asks": []})
    assert item.synchronize_buffer([{"U": 6, "u": 6, "b": [], "a": []}])
    assert item.top_levels()["obi"] is None


def test_stream_url_uses_official_public_stream_names():
    url = BinanceMarketDataConnector().websocket_url
    assert "btcusdt@trade" in url
    assert "btcusdt@bookTicker" in url
    assert "btcusdt@depth@100ms" in url
    assert "ethusdt@trade" in url


def _drive_immediate(coroutine):
    """Run a coroutine whose awaitables are mocked to complete without a loop."""
    with pytest.raises(StopIteration):
        coroutine.send(None)


def test_clean_shutdown_closes_socket_and_awaits_cancelled_tasks(monkeypatch):
    connector = BinanceMarketDataConnector()
    task = Mock()
    task.get_name.return_value = "reader"
    connector._tasks = [task]

    class Socket:
        closed = False
        async def close(self):
            self.closed = True

    socket = Socket()
    connector._active_websocket = socket

    async def immediate_wait(awaitable, timeout):
        return await awaitable

    async def immediate_gather(*tasks, return_exceptions=False):
        return [None for _ in tasks]

    monkeypatch.setattr(asyncio, "wait_for", immediate_wait)
    monkeypatch.setattr(asyncio, "gather", immediate_gather)
    _drive_immediate(connector.stop())

    assert socket.closed
    task.cancel.assert_called_once()
    assert connector._tasks == []
    assert connector.connection_status()["status"] == "stopped"


def test_shutdown_propagates_cancellation_after_cleaning_tasks(monkeypatch):
    connector = BinanceMarketDataConnector()
    task = Mock()
    task.get_name.return_value = "reader"
    connector._tasks = [task]

    class Socket:
        async def close(self):
            raise asyncio.CancelledError

    connector._active_websocket = Socket()

    async def immediate_wait(awaitable, timeout):
        return await awaitable

    async def immediate_gather(*tasks, return_exceptions=False):
        return [None for _ in tasks]

    monkeypatch.setattr(asyncio, "wait_for", immediate_wait)
    monkeypatch.setattr(asyncio, "gather", immediate_gather)
    coroutine = connector.stop()
    with pytest.raises(asyncio.CancelledError):
        coroutine.send(None)
    task.cancel.assert_called_once()
    assert connector._tasks == []
    assert connector.connection_status()["status"] == "stopped"
