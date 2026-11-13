"""Mocked Alerts Center filtering, caching, and persistence tests."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from analytics.alert_service import ABNORMAL_SPREAD, CRITICAL, WARNING, Alert, AlertService
from dashboard.alerts_view import (
    ALERTS_PER_PAGE,
    ALERT_QUERY_LIMIT,
    AlertsViewController,
    empty_alert_message,
    filter_alerts,
    paginate_alerts,
)
import dashboard.alerts_view as alerts_view
from storage.database import AlertDatabase


NOW = datetime(2026, 11, 13, 12, tzinfo=timezone.utc)


def alert(alert_id, *, instrument="EURUSD", provider="MT5", category=ABNORMAL_SPREAD,
          severity=WARNING, minutes_ago=0, acknowledged=False):
    return Alert(
        alert_id=alert_id, instrument=instrument, provider=provider,
        detected_at_utc=NOW - timedelta(minutes=minutes_ago),
        timestamp_provenance="PYTHON_SYSTEM_UTC_AT_DETECTION",
        market_data_received_at_utc=NOW - timedelta(minutes=minutes_ago + 1),
        category=category, severity=severity, title=f"Alert {alert_id}",
        explanation="Observed spread crossed its configured threshold.",
        observed_values={"spread_expansion_ratio": 2.4, "threshold": 2.0},
        acknowledged=acknowledged,
        acknowledged_at_utc=NOW if acknowledged else None,
    )


def test_filter_alerts_applies_all_filters_and_returns_newest_first():
    rows = (
        alert("older", minutes_ago=15),
        alert("critical", instrument="BTCUSDT", provider="Binance", severity=CRITICAL,
              minutes_ago=4),
        alert("ack", acknowledged=True, minutes_ago=2),
        alert("newest", minutes_ago=1),
    )
    filtered = filter_alerts(
        rows, instrument="eurusd", provider="MT5", category=ABNORMAL_SPREAD,
        severity=WARNING, acknowledgement="Unacknowledged",
        start_utc=NOW - timedelta(minutes=10), end_utc=NOW,
    )
    assert [item.alert_id for item in filtered] == ["newest"]
    assert filter_alerts(rows, acknowledgement="Acknowledged")[0].alert_id == "ack"


def test_alert_pagination_bounds_rendered_cards_without_dropping_records():
    rows = tuple(alert(str(index)) for index in range(63))
    first = paginate_alerts(rows, 0)
    second = paginate_alerts(rows, 1)
    last = paginate_alerts(rows, 99)

    assert len(first) == ALERTS_PER_PAGE == 25
    assert len(second) == ALERTS_PER_PAGE
    assert len(last) == 13
    assert first[0].alert_id == "0"
    assert second[0].alert_id == "25"
    assert last[0].alert_id == "50"
    assert len(first + second + last) == len(rows)


def test_alert_refresh_button_aligns_with_filter_fields(monkeypatch):
    calls = []

    class Control:
        def selectbox(self, label, options, **kwargs):
            return "All"

        def date_input(self, label, **kwargs):
            return kwargs["value"]

        def markdown(self, text, **kwargs):
            calls.append(("spacer", text))

        def button(self, label, **kwargs):
            calls.append(("button", label))
            return False

    controls = [Control() for _ in range(7)]
    monkeypatch.setattr(alerts_view, "render_page_header", lambda *args: None)
    monkeypatch.setattr(alerts_view.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(alerts_view.st, "columns", lambda *_args, **_kwargs: controls)
    monkeypatch.setattr(alerts_view, "_alerts_fragment", lambda *args: None)

    alerts_view.render_alerts_view()

    spacer_index = next(i for i, item in enumerate(calls) if item[0] == "spacer")
    button_index = next(i for i, item in enumerate(calls) if item[0] == "button")
    assert "height: 1.75rem" in calls[spacer_index][1]
    assert spacer_index < button_index


def test_empty_state_distinguishes_empty_store_from_filter_miss():
    assert "No alerts have been persisted" in empty_alert_message((), persisted_count=0)
    assert empty_alert_message((alert("present"),), persisted_count=1) == ""
    assert empty_alert_message((), persisted_count=7) == "No persisted alerts match these filters."


def test_controller_caches_reads_refreshes_and_invalidates_after_acknowledgement():
    class FakeAlertService:
        def __init__(self):
            self.alerts = [alert("one")]
            self.reads = 0
            self.acknowledged = []

        def get_recent_alerts(self, *, limit):
            assert limit == ALERT_QUERY_LIMIT
            self.reads += 1
            return tuple(self.alerts)

        def acknowledge_alert(self, alert_id):
            self.acknowledged.append(alert_id)
            return alert_id == "one"

    now = [10.0]
    service = FakeAlertService()
    controller = AlertsViewController(service, ttl_seconds=15, clock=lambda: now[0])
    assert controller.load().alerts[0].alert_id == "one"
    assert controller.load().alerts[0].alert_id == "one"
    assert service.reads == 1
    now[0] += 2
    controller.load(force=True)
    controller.load(force=True)  # held fragment reruns do not force repeated DB reads
    assert service.reads == 2
    assert controller.acknowledge("one")
    controller.load()
    assert service.reads == 3
    assert service.acknowledged == ["one"]


def test_acknowledge_many_reports_per_record_failures():
    class FakeAlertService:
        def acknowledge_alert(self, alert_id):
            if alert_id == "explode":
                raise OSError("database unavailable")
            return alert_id == "valid"

    count, errors = AlertsViewController(FakeAlertService()).acknowledge_many(
        ("valid", "missing", "explode"))
    assert count == 1
    assert errors == ("explode: OSError: database unavailable",)


def test_alert_service_acknowledgement_persists_in_existing_sqlite_store(tmp_path):
    database = AlertDatabase(tmp_path / "alerts.sqlite3")
    stored = alert("persisted", severity=CRITICAL)
    database.insert_alert(AlertService._alert_record(stored))
    service = AlertService(object(), database=database, clock=lambda: NOW)

    assert not service.get_recent_alerts()[0].acknowledged
    assert service.acknowledge_alert("persisted")
    reloaded_service = AlertService(object(), database=AlertDatabase(tmp_path / "alerts.sqlite3"),
                                    clock=lambda: NOW)
    saved = reloaded_service.get_recent_alerts(severity=CRITICAL)
    assert len(saved) == 1
    assert saved[0].acknowledged
    assert saved[0].acknowledged_at_utc == NOW
    assert saved[0].timestamp_provenance == "PYTHON_SYSTEM_UTC_AT_DETECTION"
    assert saved[0].observed_values["threshold"] == 2.0
