"""Small local SQLite store for persisted MarketPulse alerts and alert state."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping


class AlertDatabase:
    """Persist alerts and duplicate-suppression state in one SQLite database.

    The default database is ``storage/marketpulse.sqlite3``. Tests and other
    callers can supply an explicit path. Connections are short-lived and
    created per operation, so the store is safe to use from multiple local
    threads without sharing a sqlite connection across threads.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else Path(__file__).with_name("marketpulse.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    instrument TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    detected_at_utc TEXT NOT NULL,
                    timestamp_provenance TEXT NOT NULL,
                    market_data_received_at_utc TEXT,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    observed_values_json TEXT NOT NULL,
                    acknowledged INTEGER NOT NULL DEFAULT 0,
                    acknowledged_at_utc TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_alerts_recent
                    ON alerts(detected_at_utc DESC, alert_id);
                CREATE INDEX IF NOT EXISTS idx_alerts_filters
                    ON alerts(instrument, category, severity, detected_at_utc DESC);
                CREATE TABLE IF NOT EXISTS alert_condition_state (
                    instrument TEXT NOT NULL,
                    category TEXT NOT NULL,
                    active_condition TEXT,
                    last_emitted_at_utc TEXT,
                    state_value TEXT,
                    PRIMARY KEY (instrument, category)
                );
                """
            )

    def insert_alert(self, alert: Mapping[str, Any]) -> None:
        """Insert one alert record. Alert IDs are unique and duplicate IDs fail."""
        with self._connect() as connection:
            self._insert_alert(connection, alert)

    def persist_alert_and_condition(self, alert: Mapping[str, Any], *, instrument: str,
                                    category: str, active_condition: str,
                                    state_value: str | None) -> None:
        """Atomically persist an emitted alert and its duplicate-suppression state."""
        emitted = _iso(alert["detected_at_utc"])
        with self._connect() as connection:
            self._insert_alert(connection, alert)
            connection.execute(
                """INSERT INTO alert_condition_state
                   (instrument, category, active_condition, last_emitted_at_utc, state_value)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(instrument, category) DO UPDATE SET
                     active_condition = excluded.active_condition,
                     last_emitted_at_utc = excluded.last_emitted_at_utc,
                     state_value = excluded.state_value""",
                (instrument.upper(), category.upper(), active_condition, emitted, state_value),
            )

    @staticmethod
    def _insert_alert(connection: sqlite3.Connection, alert: Mapping[str, Any]) -> None:
        connection.execute(
            """INSERT INTO alerts (
                alert_id, instrument, provider, detected_at_utc, timestamp_provenance,
                market_data_received_at_utc, category, severity, title, explanation,
                observed_values_json, acknowledged, acknowledged_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(alert["alert_id"]), str(alert["instrument"]), str(alert["provider"]),
                _iso(alert["detected_at_utc"]), str(alert["timestamp_provenance"]),
                _iso(alert.get("market_data_received_at_utc")), str(alert["category"]),
                str(alert["severity"]), str(alert["title"]), str(alert["explanation"]),
                json.dumps(alert.get("observed_values", {}), sort_keys=True, default=_json_default),
                int(bool(alert.get("acknowledged", False))),
                _iso(alert.get("acknowledged_at_utc")),
            ),
        )

    def fetch_alerts(self, *, limit: int = 100, instrument: str | None = None,
                     category: str | None = None, severity: str | None = None
                     ) -> list[dict[str, Any]]:
        """Return newest alerts matching optional exact filters."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in (("instrument", instrument), ("category", category),
                              ("severity", severity)):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value.upper() if column != "instrument" else value.upper())
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM alerts{where} ORDER BY detected_at_utc DESC, rowid DESC LIMIT ?",
                parameters,
            ).fetchall()
        results = []
        for row in rows:
            result = dict(row)
            result["observed_values"] = json.loads(result.pop("observed_values_json"))
            result["acknowledged"] = bool(result["acknowledged"])
            results.append(result)
        return results

    def acknowledge_alert(self, alert_id: str, acknowledged_at: datetime | None = None) -> bool:
        """Mark an existing alert acknowledged; return whether it was found."""
        timestamp = _iso(acknowledged_at or datetime.now(timezone.utc))
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE alerts SET acknowledged = 1, acknowledged_at_utc = ? WHERE alert_id = ?",
                (timestamp, alert_id),
            )
            return cursor.rowcount > 0

    def get_condition_state(self, instrument: str, category: str) -> dict[str, str | None]:
        """Read persisted active-condition/cooldown state for suppression logic."""
        with self._connect() as connection:
            row = connection.execute(
                """SELECT active_condition, last_emitted_at_utc, state_value
                   FROM alert_condition_state WHERE instrument = ? AND category = ?""",
                (instrument.upper(), category.upper()),
            ).fetchone()
        if row is None:
            return {"active_condition": None, "last_emitted_at_utc": None, "state_value": None}
        return dict(row)

    def set_condition_state(self, instrument: str, category: str, *,
                            active_condition: str | None,
                            last_emitted_at_utc: datetime | str | None,
                            state_value: str | None) -> None:
        """Persist one instrument/category suppression state."""
        emitted = (_iso(last_emitted_at_utc) if isinstance(last_emitted_at_utc, datetime)
                   else last_emitted_at_utc)
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO alert_condition_state
                   (instrument, category, active_condition, last_emitted_at_utc, state_value)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(instrument, category) DO UPDATE SET
                     active_condition = excluded.active_condition,
                     last_emitted_at_utc = excluded.last_emitted_at_utc,
                     state_value = excluded.state_value""",
                (instrument.upper(), category.upper(), active_condition, emitted, state_value),
            )

    def close(self) -> None:
        """Compatibility no-op; this store does not keep a connection open."""


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if not isinstance(value, datetime):
        raise TypeError(f"Expected datetime, got {type(value).__name__}")
    if value.tzinfo is None:
        raise ValueError("database timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return _iso(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)
