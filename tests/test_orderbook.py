"""Deterministic tests of Binance snapshot/diff depth synchronization."""
from decimal import Decimal

from data.orderbook import LocalOrderBook


class Clock:
    def __init__(self):
        self.value = 10.0
    def __call__(self):
        return self.value


def event(first, final, bids=(), asks=()):
    return {"U": first, "u": final, "b": bids, "a": asks}


def book(clock=None):
    item = LocalOrderBook("BTCUSDT", stale_after_seconds=5, clock=clock or Clock())
    item.load_snapshot({"lastUpdateId": 100, "bids": [["100", "2"], ["99", "3"]],
                        "asks": [["101", "4"], ["102", "5"]]})
    return item


def test_snapshot_bridge_and_buffered_events_apply_in_order():
    item = book()
    assert item.synchronize_buffer([event(98, 100), event(101, 102, [["100.5", "1"]])])
    assert item.update_id == 102
    assert item.bids[Decimal("100.5")] == Decimal("1")
    assert item.status.synchronized


def test_zero_quantity_deletes_level():
    item = book()
    assert item.synchronize_buffer([event(101, 101)])
    assert item.apply_event(event(102, 102, [["100", "0"]]))
    assert Decimal("100") not in item.bids


def test_update_gap_invalidates_book_and_metrics():
    item = book()
    item.synchronize_buffer([event(101, 101)])
    assert not item.apply_event(event(103, 103))
    assert not item.status.synchronized
    assert item.top_levels()["available"] is False


def test_cannot_synchronize_when_snapshot_not_bridged():
    item = book()
    assert not item.synchronize_buffer([event(103, 104)])
    assert "fresh snapshot" in item.message


def test_snapshot_covering_all_buffered_events_waits_for_next_event_without_resnapshot():
    item = book()
    assert not item.synchronize_buffer([event(98, 100)])
    assert item.message == "Waiting for a depth event newer than the snapshot"
    assert item.awaiting_new_event
    assert item.update_id == 100
    # The next event bridges the same snapshot. No new snapshot is needed.
    assert item.synchronize_buffer([event(101, 102, [["100.5", "2"]])])
    assert item.update_id == 102
    assert item.synchronized
    assert not item.awaiting_new_event


def test_top_levels_sort_and_obi_metrics():
    item = book()
    item.synchronize_buffer([event(101, 101)])
    metrics = item.top_levels(2)
    assert metrics["available"]
    assert [level["price"] for level in metrics["bids"]] == [Decimal("100"), Decimal("99")]
    assert [level["price"] for level in metrics["asks"]] == [Decimal("101"), Decimal("102")]
    assert metrics["total_bid_depth"] == Decimal("5")
    assert metrics["total_ask_depth"] == Decimal("9")
    assert metrics["obi"] == Decimal( -4) / Decimal(14)
    assert "cancellable" in metrics["liquidity_note"]


def test_stale_book_is_not_available():
    clock = Clock()
    item = book(clock)
    item.synchronize_buffer([event(101, 101)])
    clock.value += 6
    assert item.status.stale
    assert item.top_levels()["available"] is False


def test_resnapshot_and_resynchronize_after_gap():
    item = book()
    item.synchronize_buffer([event(101, 101)])
    assert not item.apply_event(event(104, 104))
    item.load_snapshot({"lastUpdateId": 103, "bids": [["100", "2"]], "asks": [["101", "3"]]})
    assert item.synchronize_buffer([event(104, 105, [["99", "1"]])])
    assert item.update_id == 105
    assert item.synchronized


def test_malformed_payload_fails_closed():
    item = book()
    assert not item.synchronize_buffer([{"U": "bad", "u": 101}])
    assert not item.status.synchronized
