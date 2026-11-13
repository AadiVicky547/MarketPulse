"""Async Binance Spot public market-data connector (no account or trading API)."""
from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import logging
import time
from typing import Any, Callable, Mapping, Sequence

import requests
import websockets

from data.orderbook import LocalOrderBook, OrderBookError

logger = logging.getLogger(__name__)
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT")
REST_BASE = "https://api.binance.com"
WS_BASE = "wss://stream.binance.com:9443/stream"


def _decimal(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid decimal value: {value!r}") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"Expected positive finite value, got {value!r}")
    return result


def normalize_trade(payload: Mapping[str, Any], received_at: float | None = None) -> dict[str, Any]:
    """Normalize official Spot <symbol>@trade payload and derive taker side."""
    try:
        price, quantity = _decimal(payload["p"]), _decimal(payload["q"])
        timestamp = int(payload["T"])
        buyer_is_maker = payload["m"]
        if not isinstance(buyer_is_maker, bool):
            raise ValueError("Trade buyer-maker flag must be boolean")
        trade_id = int(payload["t"])
        symbol = str(payload["s"]).upper()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Binance trade payload: {exc}") from exc
    if timestamp <= 0 or trade_id < 0 or not symbol:
        raise ValueError("Trade timestamp, ID, or symbol is invalid")
    return {"symbol": symbol, "trade_id": trade_id, "price": price, "quantity": quantity,
            "quote_quantity": price * quantity,
            "timestamp": datetime.fromtimestamp(timestamp / 1000, timezone.utc),
            "buyer_is_maker": buyer_is_maker,
            "aggressor_side": "SELL" if buyer_is_maker else "BUY",
            "connection_status": "connected", "received_monotonic":
            received_at if received_at is not None else time.monotonic()}


def normalize_book_ticker(payload: Mapping[str, Any], received_at: float | None = None) -> dict[str, Any]:
    """Normalize official Spot <symbol>@bookTicker payload."""
    try:
        bid, bid_qty = _decimal(payload["b"]), _decimal(payload["B"])
        ask, ask_qty = _decimal(payload["a"]), _decimal(payload["A"])
        symbol = str(payload["s"]).upper()
        update_id = int(payload["u"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Binance bookTicker payload: {exc}") from exc
    if ask < bid:
        raise ValueError("Best ask is below best bid")
    mid = (bid + ask) / 2
    return {"symbol": symbol, "update_id": update_id, "bid_price": bid, "bid_quantity": bid_qty,
            "ask_price": ask, "ask_quantity": ask_qty, "mid_price": mid,
            "spread": ask - bid, "spread_bps": ((ask - bid) / mid) * Decimal(10000) if mid else None,
            "connection_status": "connected", "received_monotonic":
            received_at if received_at is not None else time.monotonic()}


class BinanceMarketDataConnector:
    """One multiplexed public WebSocket for trades, bookTicker, and depth diff streams."""

    def __init__(self, symbols: Sequence[str] = DEFAULT_SYMBOLS, depth_limit: int = 1000,
                 trade_buffer_size: int = 500, queue_size: int = 10000,
                 stale_after_seconds: float = 5.0,
                 rest_get: Callable[..., Any] | None = None,
                 websocket_connect: Callable[..., Any] | None = None) -> None:
        self.symbols = tuple(symbol.upper() for symbol in symbols)
        self.depth_limit = depth_limit
        self.stale_after_seconds = stale_after_seconds
        self._rest_get = rest_get or requests.get
        self._websocket_connect = websocket_connect or websockets.connect
        self.trades: dict[str, deque[dict[str, Any]]] = {
            symbol: deque(maxlen=trade_buffer_size) for symbol in self.symbols}
        self.best_bid_ask: dict[str, dict[str, Any] | None] = {symbol: None for symbol in self.symbols}
        self.order_books = {symbol: LocalOrderBook(symbol, stale_after_seconds) for symbol in self.symbols}
        self._depth_queues = {symbol: asyncio.Queue(maxsize=queue_size) for symbol in self.symbols}
        self._tasks: list[asyncio.Task[Any]] = []
        self._stop = asyncio.Event()
        self._state = "stopped"
        self._active_websocket: Any = None
        self._last_message: dict[str, float | None] = {symbol: None for symbol in self.symbols}
        self._queue_overflow: set[str] = set()

    @property
    def websocket_url(self) -> str:
        streams = []
        for symbol in self.symbols:
            normalized = symbol.lower()
            streams.extend((f"{normalized}@trade", f"{normalized}@bookTicker", f"{normalized}@depth@100ms"))
        return f"{WS_BASE}?streams={'/'.join(streams)}"

    def connection_status(self, symbol: str | None = None) -> dict[str, Any]:
        now = time.monotonic()
        symbols = (symbol.upper(),) if symbol else self.symbols
        per_symbol = {}
        for item in symbols:
            last = self._last_message.get(item)
            per_symbol[item] = {"connected": self._state == "connected",
                                "stale": last is None or now - last > self.stale_after_seconds,
                                "last_message_monotonic": last}
        return {"status": self._state, "symbols": per_symbol}

    def latest_trades(self, symbol: str, count: int = 20) -> list[dict[str, Any]]:
        if count <= 0:
            raise ValueError("count must be positive")
        now = time.monotonic()
        result = []
        for trade in list(self.trades[symbol.upper()])[-count:]:
            item = dict(trade)
            item["stale"] = now - item["received_monotonic"] > self.stale_after_seconds
            item["connection_status"] = self._state
            result.append(item)
        return result

    def latest_book_ticker(self, symbol: str) -> dict[str, Any] | None:
        item = self.best_bid_ask[symbol.upper()]
        if item is None:
            return None
        result = dict(item)
        result["stale"] = time.monotonic() - result["received_monotonic"] > self.stale_after_seconds
        result["connection_status"] = self._state
        return result

    def order_book_metrics(self, symbol: str, levels: int = 5) -> dict[str, Any]:
        return self.order_books[symbol.upper()].top_levels(levels)

    def _get_snapshot(self, symbol: str) -> Mapping[str, Any]:
        response = self._rest_get(f"{REST_BASE}/api/v3/depth",
                                  params={"symbol": symbol, "limit": self.depth_limit}, timeout=(3, 5))
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping) or "lastUpdateId" not in payload:
            raise ValueError("Depth snapshot response is missing lastUpdateId")
        return payload

    async def _depth_worker(self, symbol: str) -> None:
        queue = self._depth_queues[symbol]
        book = self.order_books[symbol]
        buffered: list[Mapping[str, Any]] = []
        snapshot_loaded = False
        retry_delay = 0.5
        next_snapshot_at = 0.0
        while not self._stop.is_set():
            if symbol in self._queue_overflow:
                self._queue_overflow.discard(symbol)
                book.invalidate("Depth event buffer overflow; resynchronization required")
                buffered.clear()
                snapshot_loaded = False

            if book.synchronized:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if not book.apply_event(event):
                    # Keep the event which exposed the gap; a new snapshot may
                    # already cover it, or it may be the first event to bridge.
                    buffered = [event]
                    snapshot_loaded = False
                    next_snapshot_at = max(next_snapshot_at, time.monotonic() + retry_delay)
                    retry_delay = min(retry_delay * 2, 8.0)
                continue

            if not buffered:
                try:
                    buffered.append(await asyncio.wait_for(queue.get(), timeout=0.5))
                except asyncio.TimeoutError:
                    continue

            if not snapshot_loaded:
                delay = next_snapshot_at - time.monotonic()
                if delay > 0:
                    try:
                        await asyncio.wait_for(self._stop.wait(), timeout=delay)
                        break
                    except asyncio.TimeoutError:
                        pass
                self._queue_overflow.discard(symbol)
                try:
                    snapshot = await asyncio.to_thread(self._get_snapshot, symbol)
                    book.load_snapshot(snapshot)
                    snapshot_loaded = True
                    # The WebSocket reader continues buffering while the REST call runs.
                    while True:
                        buffered.append(queue.get_nowait())
                except asyncio.QueueEmpty:
                    pass
                except Exception as exc:
                    book.invalidate(f"Depth snapshot failed: {exc}")
                    logger.warning("Snapshot fetch failed for %s: %s", symbol, exc)
                    snapshot_loaded = False
                    next_snapshot_at = time.monotonic() + retry_delay
                    retry_delay = min(retry_delay * 2, 8.0)
                    continue

            if len(buffered) > queue.maxsize:
                book.invalidate("Local depth buffer exceeded its configured limit; restarting snapshot sync")
                logger.warning("Depth buffer for %s exceeded limit; restarting snapshot synchronization", symbol)
                buffered.clear()
                snapshot_loaded = False
                next_snapshot_at = time.monotonic() + retry_delay
                retry_delay = min(retry_delay * 2, 8.0)
                continue

            if book.synchronize_buffer(buffered):
                buffered.clear()
                snapshot_loaded = False
                retry_delay = 0.5
                next_snapshot_at = 0.0
                logger.info("Order book %s synchronized at update ID %s", symbol, book.update_id)
                continue

            if book.awaiting_new_event:
                # All events so far are already reflected in lastUpdateId. Retain
                # that snapshot and wait; do not issue another REST request.
                snapshot_id = book.update_id
                buffered = [event for event in buffered if int(event["u"]) > snapshot_id]
                try:
                    buffered.append(await asyncio.wait_for(queue.get(), timeout=0.5))
                except asyncio.TimeoutError:
                    pass
                continue

            # A genuine bridge mismatch or sequence gap requires a fresh snapshot.
            # Exponential backoff prevents repeated REST requests while events keep
            # arriving faster than snapshot acquisition.
            logger.warning("Order book %s resynchronizing: %s", symbol, book.message)
            snapshot_loaded = False
            next_snapshot_at = time.monotonic() + retry_delay
            retry_delay = min(retry_delay * 2, 8.0)

    async def _consume_connection(self, websocket: Any) -> None:
        self._active_websocket = websocket
        self._state = "connected"
        for book in self.order_books.values():
            book.invalidate("WebSocket connected; waiting for a fresh synchronized snapshot")
        try:
            async for raw in websocket:
                received = time.monotonic()
                try:
                    envelope = __import__("json").loads(raw)
                    payload = envelope["data"]
                    stream = envelope["stream"]
                    symbol = str(payload["s"]).upper()
                    if symbol not in self.trades:
                        continue
                    self._last_message[symbol] = received
                    if stream.endswith("@trade"):
                        self.trades[symbol].append(normalize_trade(payload, received))
                    elif stream.endswith("@bookTicker"):
                        self.best_bid_ask[symbol] = normalize_book_ticker(payload, received)
                    elif "@depth" in stream:
                        queue = self._depth_queues[symbol]
                        try:
                            queue.put_nowait(payload)
                        except asyncio.QueueFull:
                            self._queue_overflow.add(symbol)
                            self.order_books[symbol].invalidate("Depth event buffer overflow; resynchronization required")
                except (KeyError, TypeError, ValueError, __import__("json").JSONDecodeError) as exc:
                    logger.warning("Ignoring invalid Binance WebSocket payload: %s", exc)
        finally:
            self._active_websocket = None

    async def _reader(self) -> None:
        delay = 1.0
        while not self._stop.is_set():
            self._state = "connecting"
            try:
                async with self._websocket_connect(self.websocket_url, ping_interval=20,
                                                   ping_timeout=20, close_timeout=5,
                                                   open_timeout=5,
                                                   max_queue=1000) as websocket:
                    delay = 1.0
                    await self._consume_connection(websocket)
                    if not self._stop.is_set():
                        raise ConnectionError("Binance WebSocket closed")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._state = "disconnected"
                for book in self.order_books.values():
                    book.invalidate(f"WebSocket disconnected: {exc}")
                if not self._stop.is_set():
                    logger.warning("Binance WebSocket disconnected: %s; retrying in %.1fs", exc, delay)
                    try:
                        await asyncio.wait_for(self._stop.wait(), timeout=delay)
                    except asyncio.TimeoutError:
                        pass
                    delay = min(delay * 2, 30.0)
        self._state = "stopped"

    async def start(self) -> None:
        """Start one websocket reader and one depth synchronizer per symbol; idempotent."""
        if self._tasks and any(not task.done() for task in self._tasks):
            return
        self._stop = asyncio.Event()
        self._state = "connecting"
        self._tasks = [asyncio.create_task(self._reader(), name="binance-websocket-reader")]
        self._tasks.extend(asyncio.create_task(self._depth_worker(symbol), name=f"depth-sync-{symbol}")
                           for symbol in self.symbols)

    async def stop(self) -> None:
        """Cancel and await all owned tasks; the websocket context closes cleanly."""
        if not self._tasks:
            self._state = "stopped"
            return
        self._stop.set()
        was_cancelled = False
        if self._active_websocket is not None:
            try:
                # Match/exceed websockets.connect(close_timeout=5); a two-second
                # deadline caused false close warnings during a normal handshake.
                await asyncio.wait_for(self._active_websocket.close(), timeout=7)
            except asyncio.CancelledError:
                was_cancelled = True
            except Exception as exc:
                logger.warning("WebSocket close did not finish cleanly: %s", exc)
        for task in self._tasks:
            task.cancel()
        results: list[Any] = []
        try:
            results = await asyncio.wait_for(asyncio.gather(*self._tasks, return_exceptions=True), timeout=10)
        except asyncio.CancelledError:
            was_cancelled = True
        except asyncio.TimeoutError:
            logger.error("Timed out waiting for Binance connector tasks to stop")
            for task in self._tasks:
                if not task.done():
                    task.cancel()
        for task, result in zip(self._tasks, results):
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                logger.error("Binance background task %s failed during shutdown: %r", task.get_name(), result)
        self._tasks.clear()
        self._state = "stopped"
        for book in self.order_books.values():
            book.invalidate("Connector stopped")
        if was_cancelled:
            raise asyncio.CancelledError

    async def run_for(self, seconds: float) -> None:
        """Run for a bounded interval and always shut down."""
        if seconds <= 0:
            raise ValueError("seconds must be positive")
        await self.start()
        try:
            await asyncio.sleep(seconds)
        finally:
            await self.stop()
