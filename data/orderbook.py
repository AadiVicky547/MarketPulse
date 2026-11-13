"""Precision-safe Binance Spot local order book synchronization."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import time
from typing import Any, Mapping, Sequence


class OrderBookError(ValueError):
    """A snapshot or depth event is invalid or cannot be sequenced safely."""


def _decimal(value: Any, *, allow_zero: bool = False) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise OrderBookError(f"Invalid decimal value: {value!r}") from exc
    if not result.is_finite() or result < 0 or (result == 0 and not allow_zero):
        raise OrderBookError(f"Price/quantity must be {'nonnegative' if allow_zero else 'positive'}: {value!r}")
    return result


@dataclass(frozen=True)
class BookStatus:
    synchronized: bool
    stale: bool
    update_id: int | None
    message: str
    last_update_monotonic: float | None


class LocalOrderBook:
    """A single-symbol book; never exposes metrics unless synchronized and fresh."""

    def __init__(self, symbol: str, stale_after_seconds: float = 5.0,
                 clock=time.monotonic) -> None:
        self.symbol = symbol.upper()
        self.stale_after_seconds = stale_after_seconds
        self._clock = clock
        self.bids: dict[Decimal, Decimal] = {}
        self.asks: dict[Decimal, Decimal] = {}
        self.update_id: int | None = None
        self.synchronized = False
        self.awaiting_new_event = False
        self.message = "Waiting for depth snapshot and stream bridge"
        self.last_update_monotonic: float | None = None

    @property
    def stale(self) -> bool:
        return (not self.synchronized or self.last_update_monotonic is None or
                self._clock() - self.last_update_monotonic > self.stale_after_seconds)

    @property
    def status(self) -> BookStatus:
        return BookStatus(self.synchronized, self.stale, self.update_id,
                          self.message, self.last_update_monotonic)

    def invalidate(self, message: str) -> None:
        self.synchronized = False
        self.awaiting_new_event = False
        self.message = message

    @staticmethod
    def _levels(rows: Sequence[Sequence[Any]]) -> dict[Decimal, Decimal]:
        levels: dict[Decimal, Decimal] = {}
        for row in rows:
            if len(row) < 2:
                raise OrderBookError("A book level must contain price and quantity")
            price, quantity = _decimal(row[0]), _decimal(row[1], allow_zero=True)
            if quantity > 0:
                levels[price] = quantity
        return levels

    def load_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        """Replace local state from REST snapshot; remain unavailable until bridged."""
        try:
            update_id = int(snapshot["lastUpdateId"])
            bids = self._levels(snapshot["bids"])
            asks = self._levels(snapshot["asks"])
        except (KeyError, TypeError, ValueError) as exc:
            self.invalidate(f"Invalid depth snapshot: {exc}")
            raise OrderBookError(self.message) from exc
        if update_id < 0:
            raise OrderBookError("Snapshot lastUpdateId must be nonnegative")
        self.bids, self.asks, self.update_id = bids, asks, update_id
        self.synchronized = False
        self.awaiting_new_event = False
        self.message = "Snapshot loaded; waiting for bridging depth event"
        self.last_update_monotonic = self._clock()

    def _apply_levels(self, target: dict[Decimal, Decimal], rows: Sequence[Sequence[Any]]) -> None:
        for row in rows:
            if len(row) < 2:
                raise OrderBookError("A depth update level must contain price and quantity")
            price, quantity = _decimal(row[0]), _decimal(row[1], allow_zero=True)
            if quantity == 0:
                target.pop(price, None)
            else:
                target[price] = quantity

    def _apply(self, event: Mapping[str, Any]) -> None:
        try:
            self._apply_levels(self.bids, event["b"])
            self._apply_levels(self.asks, event["a"])
            self.update_id = int(event["u"])
        except (KeyError, TypeError, ValueError) as exc:
            self.invalidate(f"Invalid diff-depth event: {exc}")
            raise OrderBookError(self.message) from exc
        self.last_update_monotonic = self._clock()

    def synchronize_buffer(self, events: Sequence[Mapping[str, Any]]) -> bool:
        """Apply buffered events only when the snapshot is bridged per Binance update IDs."""
        if self.update_id is None:
            self.invalidate("Cannot synchronize without a snapshot")
            return False
        snapshot_id = self.update_id
        pending = []
        for event in events:
            try:
                first, final = int(event["U"]), int(event["u"])
            except (KeyError, TypeError, ValueError):
                self.invalidate("Invalid diff-depth update IDs")
                return False
            if first > final:
                self.invalidate("Invalid diff-depth event: U exceeds u")
                return False
            if final <= snapshot_id:
                continue
            pending.append(event)
        if not pending:
            # A REST snapshot can already include every event received so far.
            # Keep the snapshot and wait for a newer stream event; this is not a gap.
            self.invalidate("Waiting for a depth event newer than the snapshot")
            self.awaiting_new_event = True
            return False
        first_event = pending[0]
        first, final = int(first_event["U"]), int(first_event["u"])
        if not first <= snapshot_id + 1 <= final:
            self.invalidate("Snapshot predates the first buffered update; a fresh snapshot is required")
            return False
        self._apply(first_event)
        # The bridge has now been applied, so later buffered events can use the
        # ordinary synchronized sequence and gap checks.
        self.synchronized = True
        for event in pending[1:]:
            if not self.apply_event(event):
                return False
        self.synchronized = True
        self.awaiting_new_event = False
        self.message = "Synchronized"
        return True

    def apply_event(self, event: Mapping[str, Any]) -> bool:
        """Apply one in-sequence event; ignore duplicates and invalidate on gaps."""
        if not self.synchronized or self.update_id is None:
            self.invalidate("Book is not synchronized")
            return False
        try:
            first, final = int(event["U"]), int(event["u"])
        except (KeyError, TypeError, ValueError):
            self.invalidate("Invalid diff-depth update IDs")
            return False
        if first > final:
            self.invalidate("Invalid diff-depth event: U exceeds u")
            return False
        if final <= self.update_id:
            return True
        if first > self.update_id + 1:
            self.invalidate(f"Update ID gap: expected {self.update_id + 1}, received {first}")
            return False
        try:
            self._apply(event)
        except OrderBookError:
            return False
        self.message = "Synchronized"
        return True

    def top_levels(self, count: int = 5) -> dict[str, Any]:
        """Return sorted top levels, or explicitly unavailable when stale/unsynced."""
        if count <= 0:
            raise ValueError("count must be positive")
        if not self.synchronized or self.stale:
            return {"available": False, "status": self.status, "bids": [], "asks": [],
                    "message": self.message if not self.synchronized else "Order book data is stale"}
        bids = sorted(self.bids.items(), reverse=True)[:count]
        asks = sorted(self.asks.items())[:count]
        bid_total, ask_total = sum((qty for _, qty in bids), Decimal(0)), sum((qty for _, qty in asks), Decimal(0))
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        mid = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None
        denominator = bid_total + ask_total
        obi = (bid_total - ask_total) / denominator if denominator else None
        return {"available": True, "status": self.status,
                "bids": [{"price": price, "quantity": qty} for price, qty in bids],
                "asks": [{"price": price, "quantity": qty} for price, qty in asks],
                "total_bid_depth": bid_total, "total_ask_depth": ask_total,
                "mid_price": mid, "spread": best_ask - best_bid if mid is not None else None,
                "obi": obi, "liquidity_note": "Displayed liquidity is cancellable and not guaranteed executable."}
