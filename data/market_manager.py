"""Thread-safe orchestration and normalization for local market-data sources.

The synchronous MT5 connector is owned by its polling thread. Binance is owned
by a separate thread with one asyncio event loop; its connector is created and
used only from a coroutine running on that loop. Public methods return frozen
records or immutable copies, so callers never receive live connector buffers.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import logging
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping

from config import INSTRUMENTS as CONFIGURED_INSTRUMENTS, REFRESH
from data.binance_connector import BinanceMarketDataConnector
from data.mt5_connector import MT5Connector
from data.normalizer import as_utc_datetime

logger = logging.getLogger(__name__)


def _number(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return result if result.is_finite() else None


@dataclass(frozen=True)
class InstrumentMarketState:
    """Latest state; tick_timestamp is event time, receive time is separate."""

    requested_instrument: str
    actual_symbol: str | None
    data_source: str
    bid: Decimal | None = None
    ask: Decimal | None = None
    mid_price: Decimal | None = None
    spread: Decimal | None = None
    spread_bps: Decimal | None = None
    last_trade_price: Decimal | None = None
    trade_quantity: Decimal | None = None
    tick_volume: int | None = None
    volume_semantics: str | None = None
    tick_timestamp: datetime | None = None
    tick_raw_time: Any = None
    tick_raw_time_msc: Any = None
    tick_timestamp_source: str | None = None
    tick_timestamp_status: str | None = None
    tick_timestamp_trust: str | None = None
    last_update_received_at: datetime | None = None
    freshness: str = "UNAVAILABLE"
    connection_state: str = "DISCONNECTED"
    order_book_sync_state: str | None = None
    order_book_liquidity_source: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class SourceStatus:
    """Connection and error state for one independent data source."""

    source: str
    connection_state: str
    message: str
    last_error: str | None = None
    last_success_at: datetime | None = None
    reconnect_attempts: int = 0


@dataclass(frozen=True)
class MarketSnapshot:
    """Immutable point-in-time view of all instruments and source statuses."""

    generated_at: datetime
    instruments: Mapping[str, InstrumentMarketState]
    sources: Mapping[str, SourceStatus]
    binance_trades: Mapping[str, tuple[Mapping[str, Any], ...]] = field(default_factory=dict)
    binance_order_books: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


class MarketDataManager:
    """Run MT5 and Binance ingestion independently and publish safe snapshots."""

    def __init__(self, *, mt5_poll_interval: float | None = None,
                 binance_poll_interval: float = 0.2,
                 mt5_freshness_seconds: float = 10.0,
                 binance_freshness_seconds: float = 3.0,
                 history_size: int = 100,
                 analytics_trade_buffer_size: int = 500,
                 analytics_book_levels: int = 100,
                 mt5_connector_factory: Callable[[], Any] = MT5Connector,
                 binance_connector_factory: Callable[[], Any] = BinanceMarketDataConnector,
                 reconnect_initial_seconds: float = 1.0,
                 reconnect_max_seconds: float = 30.0,
                 shutdown_timeout_seconds: float = 10.0) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        if analytics_trade_buffer_size <= 0 or analytics_book_levels <= 0:
            raise ValueError("analytics buffer size and book levels must be positive")
        if mt5_poll_interval is not None and mt5_poll_interval <= 0:
            raise ValueError("mt5_poll_interval must be positive")
        if binance_poll_interval <= 0 or mt5_freshness_seconds <= 0 or binance_freshness_seconds <= 0:
            raise ValueError("poll intervals and freshness thresholds must be positive")
        if reconnect_initial_seconds <= 0 or reconnect_max_seconds < reconnect_initial_seconds:
            raise ValueError("reconnect backoff values are invalid")
        if shutdown_timeout_seconds <= 0:
            raise ValueError("shutdown_timeout_seconds must be positive")
        self.mt5_poll_interval = (mt5_poll_interval if mt5_poll_interval is not None
                                  else float(REFRESH.mt5_seconds))
        self.binance_poll_interval = binance_poll_interval
        self.mt5_freshness_seconds = mt5_freshness_seconds
        self.binance_freshness_seconds = binance_freshness_seconds
        self.history_size = history_size
        self.analytics_trade_buffer_size = analytics_trade_buffer_size
        self.analytics_book_levels = analytics_book_levels
        self._mt5_factory = mt5_connector_factory
        self._binance_factory = binance_connector_factory
        self._reconnect_initial = reconnect_initial_seconds
        self._reconnect_max = reconnect_max_seconds
        self._shutdown_timeout = shutdown_timeout_seconds

        self._lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        # Source workers wait here until both worker threads have been launched.
        # This prevents slow/native startup in MT5 from racing the Binance thread's
        # creation (and vice versa) inside the synchronous start() call.
        self._startup_gate = threading.Event()
        self._running = False
        self._threads: dict[str, threading.Thread] = {}
        self._connectors: dict[str, Any] = {}
        self._mt5_matches: dict[str, Any] = {}
        self._event_keys: dict[str, Any] = {}
        self._force_stale: set[str] = set()
        self._states: dict[str, InstrumentMarketState] = {}
        self._history: dict[str, deque[InstrumentMarketState]] = {}
        self._binance_trades: dict[str, tuple[Mapping[str, Any], ...]] = {}
        self._binance_order_books: dict[str, Mapping[str, Any]] = {}
        self._sources = {
            "MT5": SourceStatus("MT5", "DISCONNECTED", "Not started"),
            "Binance": SourceStatus("Binance", "DISCONNECTED", "Not started"),
        }

        for configured in CONFIGURED_INSTRUMENTS:
            symbol, source = configured.symbol.upper(), configured.source
            self._states[symbol] = InstrumentMarketState(
                requested_instrument=symbol,
                actual_symbol=symbol if source == "Binance" else None,
                data_source=source,
                connection_state="DISCONNECTED",
                order_book_sync_state="UNSYNCHRONIZED" if source == "Binance" else None,
                order_book_liquidity_source=("Binance Spot displayed, cancellable order book"
                                             if source == "Binance"
                                             else "MT5 broker depth is separate and not loaded by this manager"),
            )
            self._history[symbol] = deque(maxlen=history_size)
            if source == "Binance":
                self._binance_trades[symbol] = ()
                self._binance_order_books[symbol] = MappingProxyType({
                    "available": False, "message": "No book snapshot received",
                    "bids": (), "asks": (), "status": MappingProxyType({
                        "synchronized": False, "stale": True, "update_id": None,
                        "message": "No book snapshot received", "last_update_monotonic": None,
                    }),
                })

    @property
    def instruments(self) -> tuple[str, ...]:
        return tuple(self._states)

    def is_running(self) -> bool:
        """Return true only when both configured provider workers are alive."""
        with self._lifecycle_lock:
            return self._running and all(
                self._threads.get(source) is not None and self._threads[source].is_alive()
                for source in ("MT5", "Binance")
            )

    def get_worker_health(self) -> Mapping[str, bool]:
        """Return independent liveness for the MT5 and Binance worker threads."""
        with self._lifecycle_lock:
            return MappingProxyType({
                source: bool(thread and thread.is_alive())
                for source in ("MT5", "Binance")
                for thread in (self._threads.get(source),)
            })

    def start(self) -> None:
        """Start each source once; returns immediately without waiting on network I/O."""
        with self._lifecycle_lock:
            alive = {source: bool(self._threads.get(source) and self._threads[source].is_alive())
                     for source in ("MT5", "Binance")}
            if self._stop_event.is_set() and any(alive.values()):
                self._startup_diagnostic(
                    "start() deferred; previous workers are still stopping")
                return
            if self._stop_event.is_set():
                self._stop_event.clear()
            self._running = True
            targets = {
                "MT5": (self._mt5_thread_main, "marketpulse-mt5", "Starting MT5 poller"),
                "Binance": (self._binance_thread_main, "marketpulse-binance",
                            "Starting Binance event loop"),
            }
            started_sources: list[str] = []
            for source, (target, name, message) in targets.items():
                if alive[source]:
                    self._startup_diagnostic(f"start() retained healthy {source} worker")
                    continue
                previous = self._threads.get(source)
                if previous is not None:
                    logger.warning("Restarting exited %s market-data worker", source)
                self._set_source(source, "CONNECTING", message)
                self._startup_diagnostic(f"creating {source} Thread object")
                thread = threading.Thread(target=target, name=name, daemon=True)
                self._threads[source] = thread
                self._startup_diagnostic(f"created {source} Thread object")
                started_sources.append(source)
            for source in started_sources:
                thread = self._threads[source]
                self._startup_diagnostic(f"{source}: calling Thread.start()")
                try:
                    thread.start()
                except Exception as exc:
                    logger.exception("[MARKET-STARTUP] %s: Thread.start() failed", source)
                    self._set_source(source, "DISCONNECTED", f"Could not start {source} worker", str(exc))
                else:
                    self._startup_diagnostic(f"{source}: Thread.start() returned")
            # The initial workers wait here until both start attempts finish.
            # During an independent restart the gate is already open and the
            # healthy peer is left untouched.
            self._startup_gate.set()
            self._startup_diagnostic("startup gate released; start() returning")

    @staticmethod
    def _startup_diagnostic(operation: str) -> None:
        """Emit temporary, timestamped startup trace points for hang diagnosis."""
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        logger.info("[MARKET-STARTUP] %s %s", timestamp, operation)

    def stop(self) -> None:
        """Request both sources to stop and wait up to the configured join timeout."""
        with self._lifecycle_lock:
            threads = tuple(self._threads.values())
            if not threads and not self._running:
                return
            self._stop_event.set()
        deadline = time.monotonic() + self._shutdown_timeout
        for thread in threads:
            if thread is threading.current_thread():
                continue
            thread.join(max(0.0, deadline - time.monotonic()))
        alive = [thread.name for thread in threads if thread.is_alive()]
        with self._lifecycle_lock:
            self._running = bool(alive)
        if alive:
            logger.warning("Market data workers did not stop before timeout: %s", ", ".join(alive))

    def _set_source(self, source: str, state: str, message: str,
                    error: str | None = None, success: bool = False,
                    reconnect_attempts: int | None = None) -> None:
        with self._lock:
            current = self._sources[source]
            self._sources[source] = SourceStatus(
                source=source,
                connection_state=state,
                message=message,
                last_error=error,
                last_success_at=datetime.now(timezone.utc) if success else current.last_success_at,
                reconnect_attempts=(current.reconnect_attempts if reconnect_attempts is None
                                    else reconnect_attempts),
            )

    def _publish(self, symbol: str, state: InstrumentMarketState, *, record: bool,
                 force_stale: bool = False) -> None:
        with self._lock:
            self._states[symbol] = state
            if force_stale:
                self._force_stale.add(symbol)
            else:
                self._force_stale.discard(symbol)
            if record:
                self._history[symbol].append(state)

    def _source_for(self, symbol: str) -> SourceStatus:
        source = self._states[symbol].data_source
        with self._lock:
            return self._sources[source]

    def _mt5_thread_main(self) -> None:
        attempts = 0
        delay = self._reconnect_initial
        connector = None
        symbols_discovered = False
        try:
            self._startup_diagnostic("MT5 worker entered; waiting for startup gate")
            self._startup_gate.wait()
            self._startup_diagnostic("MT5 worker passed startup gate")
            while not self._stop_event.is_set():
                try:
                    if connector is None:
                        self._startup_diagnostic("MT5 connector construction started")
                        connector = self._mt5_factory()
                        self._startup_diagnostic("MT5 connector construction completed")
                        with self._lock:
                            self._connectors["MT5"] = connector
                    if not getattr(connector, "connected", False):
                        attempts += 1
                        self._set_source("MT5", "CONNECTING", "Initializing MT5 terminal",
                                         reconnect_attempts=attempts)
                        logger.info("MT5 initialization attempt %d", attempts)
                        try:
                            self._startup_diagnostic("MT5 initialize() started")
                            status = connector.initialize()
                            self._startup_diagnostic("MT5 initialize() returned")
                        except Exception as exc:
                            logger.exception("MT5 initialization raised an exception")
                            status = None
                            error = str(exc)
                        else:
                            error = getattr(status, "error", None)
                        if not status or not getattr(status, "connected", False):
                            message = getattr(status, "message", "MT5 initialization failed")
                            self._set_source("MT5", "DISCONNECTED", message, error or message,
                                             reconnect_attempts=attempts)
                            logger.warning("MT5 initialization attempt %d failed: %s",
                                           attempts, error or message)
                            self._close_mt5_connector(connector)
                            connector = None
                            symbols_discovered = False
                            if self._wait_for_mt5_retry(delay):
                                break
                            delay = min(delay * 2, self._reconnect_max)
                            continue
                        self._set_source("MT5", "CONNECTED", "MT5 terminal connected",
                                         success=True, reconnect_attempts=attempts)
                        logger.info("MT5 terminal connected after %d attempt(s)", attempts)

                    if not symbols_discovered:
                        try:
                            self._startup_diagnostic("MT5 discover_symbols() started")
                            self._mt5_matches = connector.discover_symbols()
                            self._startup_diagnostic("MT5 discover_symbols() returned")
                            symbols_discovered = True
                        except Exception as exc:
                            attempts += 1
                            self._set_source("MT5", "CONNECTED", "MT5 symbol discovery failed", str(exc),
                                             reconnect_attempts=attempts)
                            logger.exception("MT5 symbol discovery failed; retrying")
                            if self._wait_for_mt5_retry(delay):
                                break
                            delay = min(delay * 2, self._reconnect_max)
                            continue

                    for symbol in self._mt5_symbols():
                        if self._stop_event.is_set():
                            break
                        match = self._mt5_matches.get(symbol)
                        actual = getattr(match, "actual_symbol", None)
                        if not actual:
                            self._publish_error(symbol, getattr(match, "message", "MT5 symbol unavailable"),
                                                actual_symbol=None)
                            continue
                        try:
                            tick = connector.get_tick(symbol)
                        except Exception as exc:
                            self._publish_error(symbol, str(exc), actual_symbol=actual)
                            logger.warning("MT5 tick polling failed for %s; rebuilding connector: %s",
                                           symbol, exc)
                            # Treat raised polling exceptions as a failed connector
                            # session even if its cached connected flag is stale.
                            raise
                        self._ingest_mt5(symbol, actual, tick)
                    # Reset exponential backoff only after a complete polling
                    # pass succeeds, not merely after initialize() succeeds.
                    if attempts:
                        self._set_source("MT5", "CONNECTED", "MT5 market data received",
                                         success=True, reconnect_attempts=0)
                    attempts = 0
                    delay = self._reconnect_initial
                    self._stop_event.wait(self.mt5_poll_interval)
                except Exception as exc:
                    attempts += 1
                    self._set_source("MT5", "DISCONNECTED", "MT5 worker cycle failed", str(exc),
                                     reconnect_attempts=attempts)
                    logger.exception("Unexpected MT5 worker exception; connector will be recreated")
                    self._close_mt5_connector(connector)
                    connector = None
                    symbols_discovered = False
                    with self._lock:
                        self._connectors.pop("MT5", None)
                    logger.warning("MT5 reconnect attempt %d scheduled in %.2f seconds",
                                   attempts, delay)
                    if self._wait_for_mt5_retry(delay):
                        break
                    delay = min(delay * 2, self._reconnect_max)
        finally:
            self._close_mt5_connector(connector)
            with self._lock:
                self._connectors.pop("MT5", None)
            if self._stop_event.is_set():
                self._set_source("MT5", "DISCONNECTED", "MT5 manager stopped")
                logger.info("MT5 worker shut down")

    def _wait_for_mt5_retry(self, seconds: float) -> bool:
        """Wait for bounded reconnect backoff; return true when shutting down."""
        return self._stop_event.wait(seconds)

    @staticmethod
    def _close_mt5_connector(connector: Any) -> None:
        if connector is None:
            return
        try:
            connector.close()
        except Exception:
            logger.exception("Could not close failed MT5 connector")

    @staticmethod
    def _mt5_symbols() -> tuple[str, ...]:
        return tuple(item.symbol.upper() for item in CONFIGURED_INSTRUMENTS if item.source == "MT5")

    @staticmethod
    def _binance_symbols() -> tuple[str, ...]:
        return tuple(item.symbol.upper() for item in CONFIGURED_INSTRUMENTS if item.source == "Binance")

    def _ingest_mt5(self, symbol: str, actual_symbol: str, tick: Mapping[str, Any]) -> None:
        tick_received = bool(tick.get("tick_received", tick.get("available")))
        with self._lock:
            previous = self._states[symbol]
            previous_key = self._event_keys.get(symbol)
        if not tick_received:
            self._publish_error(symbol, str(tick.get("error") or "No valid MT5 tick available"),
                                actual_symbol=actual_symbol)
            return
        bid, ask, last = _number(tick.get("bid")), _number(tick.get("ask")), _number(tick.get("last"))
        spread = ask - bid if bid is not None and ask is not None and ask >= bid else None
        mid = (ask + bid) / 2 if spread is not None else None
        spread_bps = spread / mid * Decimal(10000) if spread is not None and mid else None
        tick_time = as_utc_datetime(tick.get("timestamp"))
        volume_value = tick.get("tick_volume")
        try:
            tick_volume = int(volume_value) if volume_value is not None and int(volume_value) >= 0 else None
        except (ValueError, TypeError, OverflowError):
            tick_volume = None
        event_key = (tick_time, bid, ask, last, tick_volume)
        changed = event_key != previous_key
        received_at = as_utc_datetime(tick.get("received_at")) or datetime.now(timezone.utc)
        self._event_keys[symbol] = event_key
        status = self._source_for(symbol)
        state = InstrumentMarketState(
            requested_instrument=symbol, actual_symbol=actual_symbol, data_source="MT5",
            bid=bid, ask=ask, mid_price=mid, spread=spread, spread_bps=spread_bps,
            last_trade_price=last, tick_volume=tick_volume,
            volume_semantics="broker-observed tick activity; not centralized traded volume",
            tick_timestamp=tick_time, tick_raw_time=tick.get("raw_time"),
            tick_raw_time_msc=tick.get("raw_time_msc"),
            tick_timestamp_source=tick.get("timestamp_source"),
            tick_timestamp_status=tick.get("timestamp_status"),
            tick_timestamp_trust=tick.get("timestamp_trust", "UNVERIFIED_TIME_BASIS"),
            last_update_received_at=received_at,
            freshness="LIVE",
            connection_state=status.connection_state,
            order_book_sync_state=None,
            order_book_liquidity_source="MT5 broker depth is separate and not loaded by this manager",
            error_message=tick.get("error"),
        )
        self._publish(symbol, state, record=changed, force_stale=False)
        self._set_source("MT5", status.connection_state, status.message, success=True)

    def _publish_error(self, symbol: str, error: str, actual_symbol: str | None) -> None:
        with self._lock:
            old = self._states[symbol]
            source_status = self._sources[old.data_source]
        connected = source_status.connection_state == "CONNECTED"
        state = replace(old, actual_symbol=actual_symbol,
                        connection_state="CONNECTED" if connected else source_status.connection_state,
                        error_message=error)
        self._publish(symbol, state, record=False)

    def _binance_thread_main(self) -> None:
        self._startup_diagnostic("Binance worker entered; waiting for startup gate")
        self._startup_gate.wait()
        self._startup_diagnostic("Binance worker passed startup gate")
        # Use the platform default loop (Proactor on supported Windows Python),
        # scoped to this worker thread rather than changing the global policy.
        self._startup_diagnostic("Binance asyncio.new_event_loop() started")
        loop = asyncio.new_event_loop()
        self._startup_diagnostic("Binance asyncio.new_event_loop() returned")
        try:
            asyncio.set_event_loop(loop)
            # The Binance connector and its asyncio queues are constructed inside
            # _binance_main, while this dedicated loop is running.
            loop.run_until_complete(self._binance_main())
        except Exception as exc:
            logger.exception("Binance manager worker failed")
            self._set_source("Binance", "DISCONNECTED", "Binance worker failed", str(exc))
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
            except Exception:
                logger.exception("Error while closing Binance event loop")
            finally:
                loop.close()
                asyncio.set_event_loop(None)
                with self._lock:
                    self._connectors.pop("Binance", None)
                if self._stop_event.is_set():
                    self._set_source("Binance", "DISCONNECTED", "Binance manager stopped")

    async def _binance_main(self) -> None:
        attempts = 0
        delay = self._reconnect_initial
        while not self._stop_event.is_set():
            connector = None
            try:
                self._startup_diagnostic("Binance connector construction started")
                connector = self._binance_factory()
                self._startup_diagnostic("Binance connector construction completed")
                with self._lock:
                    self._connectors["Binance"] = connector
                self._startup_diagnostic("Binance connector.start() started")
                await connector.start()
                self._startup_diagnostic("Binance connector.start() returned")
                try:
                    attempts = 0
                    delay = self._reconnect_initial
                    while not self._stop_event.is_set():
                        try:
                            source_status = connector.connection_status()
                            source_state = source_status.get("status")
                            connected = source_state == "connected"
                            manager_state = ("CONNECTED" if connected else "DISCONNECTED"
                                             if source_state == "disconnected" else "CONNECTING")
                            self._set_source("Binance", manager_state,
                                             f"Binance WebSocket {source_state or 'unknown'}",
                                             reconnect_attempts=attempts)
                            for symbol in self._binance_symbols():
                                try:
                                    self._ingest_binance(symbol, connector)
                                except Exception as exc:
                                    logger.exception("Could not read Binance state for %s", symbol)
                                    self._publish_error(symbol, str(exc), actual_symbol=symbol)
                        except Exception as exc:
                            logger.exception("Binance state polling failed")
                            self._set_source("Binance", "DISCONNECTED", "Binance state polling failed", str(exc),
                                             reconnect_attempts=attempts + 1)
                        await asyncio.sleep(self.binance_poll_interval)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    attempts += 1
                    self._set_source("Binance", "DISCONNECTED", "Binance connector failed", str(exc),
                                     reconnect_attempts=attempts)
                    logger.exception("Binance connector startup or polling failed")
                finally:
                    if connector is not None:
                        try:
                            await connector.stop()
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            logger.exception("Could not stop Binance connector cleanly")
                    with self._lock:
                        self._connectors.pop("Binance", None)
            except asyncio.CancelledError:
                raise
            if not self._stop_event.is_set():
                await self._async_stop_wait(delay)
                delay = min(delay * 2, self._reconnect_max)

    async def _async_stop_wait(self, seconds: float) -> None:
        """Yield for retry backoff while checking the thread-owned stop signal."""
        deadline = time.monotonic() + seconds
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    def _ingest_binance(self, symbol: str, connector: Any) -> None:
        trades = connector.latest_trades(symbol, self.analytics_trade_buffer_size)
        trade = trades[-1] if trades else None
        ticker = connector.latest_book_ticker(symbol)
        book = connector.order_books[symbol].status
        try:
            book_metrics = connector.order_book_metrics(symbol, self.analytics_book_levels)
        except Exception as exc:
            logger.exception("Could not read Binance order-book metrics for %s", symbol)
            book_metrics = {"available": False, "message": f"Book metrics read failed: {exc}",
                            "bids": [], "asks": []}
        status = self._source_for(symbol)
        with self._lock:
            previous = self._states[symbol]
            previous_key = self._event_keys.get(symbol)
            self._binance_trades[symbol] = tuple(
                MappingProxyType(dict(item)) for item in trades
            )
            frozen_book = dict(book_metrics) if isinstance(book_metrics, Mapping) else {
                "available": False, "message": "Invalid order-book metrics response", "bids": [], "asks": []}
            book_status = frozen_book.get("status")
            if book_status is not None and not isinstance(book_status, Mapping):
                frozen_book["status"] = MappingProxyType({
                    key: getattr(book_status, key, None) for key in
                    ("synchronized", "stale", "update_id", "message", "last_update_monotonic")
                })
            for side in ("bids", "asks"):
                frozen_book[side] = tuple(
                    MappingProxyType(dict(level)) if isinstance(level, Mapping) else level
                    for level in frozen_book.get(side, ())
                )
            self._binance_order_books[symbol] = MappingProxyType(frozen_book)

        bid = _number(ticker.get("bid_price")) if ticker else None
        ask = _number(ticker.get("ask_price")) if ticker else None
        spread = ask - bid if bid is not None and ask is not None and ask >= bid else None
        mid = (bid + ask) / 2 if spread is not None else None
        spread_bps = spread / mid * Decimal(10000) if spread is not None and mid else None
        last = _number(trade.get("price")) if trade else None
        quantity = _number(trade.get("quantity")) if trade else None
        tick_time = as_utc_datetime(trade.get("timestamp")) if trade else None

        receive_times = [value for value in (
            trade.get("received_monotonic") if trade else None,
            ticker.get("received_monotonic") if ticker else None,
        ) if isinstance(value, (int, float))]
        newest_monotonic = max(receive_times) if receive_times else None
        now_monotonic = time.monotonic()
        received_at = (datetime.now(timezone.utc) - timedelta(seconds=max(0.0, now_monotonic - newest_monotonic))
                       if newest_monotonic is not None else None)
        event_key = (trade.get("trade_id") if trade else None,
                     ticker.get("update_id") if ticker else None,
                     ticker.get("received_monotonic") if ticker else None)
        changed = event_key != previous_key and newest_monotonic is not None
        self._event_keys[symbol] = event_key
        if not changed:
            received_at = previous.last_update_received_at

        if newest_monotonic is not None and now_monotonic - newest_monotonic > self.binance_freshness_seconds:
            freshness_force = True
        else:
            freshness_force = False
        connection_state = status.connection_state
        has_market_data = any(value is not None for value in (bid, ask, last, quantity))
        state = InstrumentMarketState(
            requested_instrument=symbol, actual_symbol=symbol, data_source="Binance",
            bid=bid, ask=ask, mid_price=mid, spread=spread, spread_bps=spread_bps,
            last_trade_price=last, trade_quantity=quantity,
            volume_semantics="Binance exchange-reported executed trade quantity",
            tick_timestamp=tick_time, last_update_received_at=received_at,
            tick_timestamp_source="trade.T" if trade else None,
            tick_timestamp_status="VALID" if tick_time else "UNAVAILABLE",
            tick_timestamp_trust="TRUSTED_UTC" if tick_time else "UNAVAILABLE",
            freshness=("UNAVAILABLE" if not has_market_data else
                       "DISCONNECTED" if connection_state == "DISCONNECTED" else
                       "STALE" if freshness_force else "LIVE"),
            connection_state=connection_state,
            order_book_sync_state=("SYNCHRONIZED" if book.synchronized and not book.stale
                                   else "STALE" if book.synchronized and book.stale else "UNSYNCHRONIZED"),
            order_book_liquidity_source="Binance Spot displayed, cancellable order book",
            error_message=(None if connection_state == "CONNECTED" else status.last_error or status.message),
        )
        self._publish(symbol, state, record=changed, force_stale=freshness_force)
        if changed and connection_state == "CONNECTED":
            self._set_source("Binance", "CONNECTED", "Binance market data received", success=True)

    def _freshness(self, symbol: str, state: InstrumentMarketState, now: datetime,
                   source: SourceStatus, force_stale: bool) -> InstrumentMarketState:
        if source.connection_state == "DISCONNECTED":
            freshness = "DISCONNECTED"
            connection = "DISCONNECTED"
        elif state.last_update_received_at is None:
            freshness = "UNAVAILABLE"
            connection = source.connection_state
        else:
            threshold = (self.mt5_freshness_seconds if state.data_source == "MT5"
                         else self.binance_freshness_seconds)
            age = (now - state.last_update_received_at).total_seconds()
            freshness = ("STALE" if source.connection_state != "CONNECTED" or force_stale or age > threshold
                         else "LIVE")
            connection = source.connection_state
        return replace(state, freshness=freshness, connection_state=connection,
                       error_message=state.error_message or source.last_error)

    def get_market_snapshot(self) -> MarketSnapshot:
        """Return an immutable, freshness-updated copy of all six instrument states."""
        now = datetime.now(timezone.utc)
        with self._lock:
            states = dict(self._states)
            sources = dict(self._sources)
            force_stale = set(self._force_stale)
            trades = dict(self._binance_trades)
            books = dict(self._binance_order_books)
        instrument_view = {}
        for symbol, state in states.items():
            instrument_view[symbol] = self._freshness(
                symbol, state, now, sources[state.data_source], symbol in force_stale)
        return MarketSnapshot(now, MappingProxyType(instrument_view), MappingProxyType(sources),
                              MappingProxyType(trades), MappingProxyType(books))

    def get_instrument_state(self, symbol: str) -> InstrumentMarketState:
        """Return the current frozen state for one requested symbol."""
        key = symbol.upper()
        with self._lock:
            state = self._states[key]
            source = self._sources[state.data_source]
            force_stale = key in self._force_stale
        return self._freshness(key, state, datetime.now(timezone.utc), source, force_stale)

    def get_history(self, symbol: str) -> tuple[InstrumentMarketState, ...]:
        """Return an immutable bounded history tuple for one instrument."""
        key = symbol.upper()
        with self._lock:
            return tuple(self._history[key])

    def get_source_status(self) -> Mapping[str, SourceStatus]:
        """Return an immutable copy of the per-source lifecycle status."""
        with self._lock:
            return MappingProxyType(dict(self._sources))
