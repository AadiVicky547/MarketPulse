"""Alerts Center for persisted MarketPulse alert observations.

This page reads and acknowledges existing records through the shared
AlertService. It never evaluates analytics or creates alerts.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as day_time, timedelta, timezone
import hashlib
import json
import threading
import time
from typing import Any, Iterable, Mapping

import streamlit as st

from analytics.alert_service import (
    ABNORMAL_SPREAD,
    BINANCE_ORDER_BOOK_IMBALANCE,
    CRITICAL,
    INFO,
    ISOLATION_FOREST_ANOMALY,
    MT5_TICK_ACTIVITY,
    REGIME_CHANGE,
    VOLATILITY_EXPANSION,
    WARNING,
)
from config import INSTRUMENTS, REFRESH
from dashboard.overview import _get_runtime, _utc_text
from dashboard.theme import (
    alert_container_key,
    alert_severity_badge_html,
    render_page_header,
)

ALERT_CATEGORIES = (
    VOLATILITY_EXPANSION,
    ABNORMAL_SPREAD,
    MT5_TICK_ACTIVITY,
    BINANCE_ORDER_BOOK_IMBALANCE,
    REGIME_CHANGE,
    ISOLATION_FOREST_ANOMALY,
)
ALERT_SEVERITIES = (INFO, WARNING, CRITICAL)
ALERT_QUERY_LIMIT = 1000
ALERT_CACHE_SECONDS = 15.0
ALERTS_PER_PAGE = 25


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def filter_alerts(
    alerts: Iterable[Any], *,
    instrument: str | None = None,
    provider: str | None = None,
    category: str | None = None,
    severity: str | None = None,
    acknowledgement: str = "All",
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
) -> tuple[Any, ...]:
    """Filter persisted records locally; input is returned newest first."""
    rows = []
    start = start_utc.astimezone(timezone.utc) if start_utc else None
    end = end_utc.astimezone(timezone.utc) if end_utc else None
    for alert in alerts:
        if instrument and str(_field(alert, "instrument", "")).upper() != instrument.upper():
            continue
        if provider and str(_field(alert, "provider", "")).casefold() != provider.casefold():
            continue
        if category and str(_field(alert, "category", "")) != category:
            continue
        if severity and str(_field(alert, "severity", "")) != severity:
            continue
        acknowledged = bool(_field(alert, "acknowledged", False))
        if acknowledgement == "Unacknowledged" and acknowledged:
            continue
        if acknowledgement == "Acknowledged" and not acknowledged:
            continue
        detected = _field(alert, "detected_at_utc")
        if not isinstance(detected, datetime) or detected.tzinfo is None:
            continue
        detected = detected.astimezone(timezone.utc)
        if start and detected < start:
            continue
        if end and detected > end:
            continue
        rows.append((detected, alert))
    rows.sort(key=lambda pair: pair[0], reverse=True)
    return tuple(alert for _, alert in rows)


def paginate_alerts(alerts: Iterable[Any], page: int, *, page_size: int = ALERTS_PER_PAGE) -> tuple[Any, ...]:
    """Return one bounded UI page without dropping records from the alert store."""
    rows = tuple(alerts)
    if page_size < 1:
        raise ValueError("page_size must be positive")
    page_count = max(1, (len(rows) + page_size - 1) // page_size)
    safe_page = min(max(int(page), 0), page_count - 1)
    start = safe_page * page_size
    return rows[start:start + page_size]


@dataclass(frozen=True)
class AlertLoad:
    """Cached read result and its retrieval status."""

    alerts: tuple[Any, ...]
    loaded_at: datetime | None
    error: str | None = None


class AlertsViewController:
    """Cache persisted reads and delegate all mutations to AlertService."""

    def __init__(self, alert_service: Any, *, ttl_seconds: float = ALERT_CACHE_SECONDS,
                 clock: Any = time.monotonic) -> None:
        self.alert_service = alert_service
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.RLock()
        self._last_fetch_monotonic = float("-inf")
        self._cached = AlertLoad((), None)
        self._force_latched = False

    def load(self, *, force: bool = False) -> AlertLoad:
        """Read newest stored alerts only when cache expires or is refreshed."""
        with self._lock:
            now = self._clock()
            if force and not self._force_latched:
                self._force_latched = True
                force_once = True
            elif force:
                force_once = False
            elif not force:
                self._force_latched = False
                force_once = False
            if not force_once and now - self._last_fetch_monotonic < self.ttl_seconds:
                return self._cached
            self._last_fetch_monotonic = now
            try:
                records = self.alert_service.get_recent_alerts(limit=ALERT_QUERY_LIMIT)
                self._cached = AlertLoad(tuple(records), datetime.now(timezone.utc))
            except Exception as exc:
                self._cached = AlertLoad(
                    self._cached.alerts, self._cached.loaded_at,
                    f"{type(exc).__name__}: {exc}")
            return self._cached

    def acknowledge(self, alert_id: str) -> bool:
        """Persist acknowledgement through the established AlertService API."""
        with self._lock:
            changed = bool(self.alert_service.acknowledge_alert(alert_id))
            if changed:
                self._last_fetch_monotonic = float("-inf")
            return changed

    def acknowledge_many(self, alert_ids: Iterable[str]) -> tuple[int, tuple[str, ...]]:
        """Persist each visible record individually and report failures."""
        acknowledged = 0
        errors = []
        for alert_id in alert_ids:
            try:
                acknowledged += int(self.acknowledge(alert_id))
            except Exception as exc:
                errors.append(f"{alert_id}: {type(exc).__name__}: {exc}")
        return acknowledged, tuple(errors)


@st.cache_resource(show_spinner=False)
def _get_controller() -> AlertsViewController:
    runtime = _get_runtime()
    return AlertsViewController(runtime.data_service.alert_service)


def _date_boundaries(value: Any) -> tuple[datetime | None, datetime | None]:
    """Convert a Streamlit date range to inclusive UTC day boundaries."""
    if isinstance(value, date) and not isinstance(value, datetime):
        first = last = value
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        first, last = value
    else:
        return None, None
    if not isinstance(first, date) or not isinstance(last, date):
        return None, None
    start = datetime.combine(first, day_time.min, timezone.utc)
    end_exclusive = datetime.combine(last + timedelta(days=1), day_time.min, timezone.utc)
    return start, end_exclusive - timedelta(microseconds=1)


def _observed_json(alert: Any) -> str:
    values = _field(alert, "observed_values", {}) or {}
    try:
        return json.dumps(values, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(values)


def empty_alert_message(alerts: Iterable[Any], persisted_count: int | None = None) -> str:
    """Return a non-fabricating empty-state message for the current filter."""
    if tuple(alerts):
        return ""
    if persisted_count == 0:
        return "No alerts have been persisted yet. The Alerts Center only reads existing AlertService records."
    return "No persisted alerts match these filters."


def _severity_badge(severity: str) -> str:
    return alert_severity_badge_html(severity)


def _render_alert_card(alert: Any, controller: AlertsViewController) -> None:
    alert_id = str(_field(alert, "alert_id", "unknown"))
    severity = str(_field(alert, "severity", "UNKNOWN"))
    acknowledged = bool(_field(alert, "acknowledged", False))
    severity_key = hashlib.sha1(alert_id.encode("utf-8")).hexdigest()[:10]
    with st.container(
        border=True,
        key=alert_container_key(severity, severity_key),
    ):
        head = st.columns([1.2, 1.1, 2.2, 1.2])
        head[0].markdown(f"**{_field(alert, 'instrument', 'Unknown')} · {_field(alert, 'provider', 'Unknown')}**")
        head[1].markdown(_severity_badge(severity), unsafe_allow_html=True)
        head[2].caption(f"{_field(alert, 'category', 'UNKNOWN')} · {_utc_text(_field(alert, 'detected_at_utc'))}")
        if acknowledged:
            head[3].caption(f"Acknowledged · {_utc_text(_field(alert, 'acknowledged_at_utc'))}")
        elif head[3].button("Acknowledge", key=f"ack-{alert_id}", use_container_width=True):
            try:
                if controller.acknowledge(alert_id):
                    st.rerun()
                else:
                    st.warning("Alert was not found in the persisted store.")
            except Exception as exc:
                st.error(f"Acknowledgement failed: {type(exc).__name__}: {exc}")
        st.markdown(f"**{_field(alert, 'title', 'Untitled alert')}**")
        st.write(_field(alert, "explanation", "No explanation stored."))
        with st.expander("Alert details and provenance", expanded=False):
            st.caption(f"Alert ID: {alert_id}")
            st.caption(f"Detection timestamp provenance: {_field(alert, 'timestamp_provenance', 'not reported')}")
            st.caption(f"Market-data receive time: {_utc_text(_field(alert, 'market_data_received_at_utc'))}")
            st.markdown("**Observed values and thresholds**")
            st.code(_observed_json(alert), language="json")


def _render_alerts(alerts: tuple[Any, ...], controller: AlertsViewController,
                   persisted_count: int) -> None:
    if not alerts:
        st.success(empty_alert_message(alerts, persisted_count))
        st.caption("The Alerts Center never creates alerts or sample data for demonstration.")
        return
    pending_ids = [str(_field(alert, "alert_id", "")) for alert in alerts
                   if not bool(_field(alert, "acknowledged", False))]
    if pending_ids:
        confirmation_key = hashlib.sha1("|".join(pending_ids).encode("utf-8")).hexdigest()[:12]
        confirm = st.checkbox(
            f"Confirm acknowledgement of all {len(pending_ids)} visible unacknowledged alerts",
            key=f"alerts_ack_all_confirm_{confirmation_key}")
        if st.button("Acknowledge all visible", disabled=not confirm, key="alerts_ack_all"):
            acknowledged, errors = controller.acknowledge_many(pending_ids)
            if errors:
                st.error("Some acknowledgements failed: " + " | ".join(errors))
            st.success(f"Acknowledged {acknowledged} alert(s).")
            st.rerun()
    for alert in alerts:
        _render_alert_card(alert, controller)


@st.fragment(run_every=REFRESH.dashboard_seconds)
def _alerts_fragment(force_refresh: bool, instrument: str, provider: str, category: str,
                     severity: str, acknowledgement: str, date_range: Any) -> None:
    try:
        # Alert reads remain read-only; this only requests the shared
        # background analytics cycle that performs evaluation off the UI thread.
        _get_runtime().data_service.refresh()
        controller = _get_controller()
        loaded = controller.load(force=force_refresh)
        if loaded.error:
            st.warning(f"Alert store read error: {loaded.error}")
        start, end = _date_boundaries(date_range)
        filtered = filter_alerts(
            loaded.alerts,
            instrument=None if instrument == "All" else instrument,
            provider=None if provider == "All" else provider,
            category=None if category == "All" else category,
            severity=None if severity == "All" else severity,
            acknowledgement=acknowledgement,
            start_utc=start,
            end_utc=end,
        )
        page_count = max(1, (len(filtered) + ALERTS_PER_PAGE - 1) // ALERTS_PER_PAGE)
        page_index = min(max(int(st.session_state.get("alerts_page_index", 0)), 0), page_count - 1)
        st.session_state["alerts_page_index"] = page_index
        if page_count > 1:
            page_index = st.selectbox(
                "Alert page",
                options=range(page_count),
                index=page_index,
                format_func=lambda value: f"Page {value + 1} of {page_count}",
                key="alerts_page_index",
            )
        visible_alerts = paginate_alerts(filtered, page_index)
        total = len(filtered)
        acknowledged = sum(bool(_field(alert, "acknowledged", False)) for alert in filtered)
        active = total - acknowledged
        st.markdown("### Matching alert summary")
        summary_cols = st.columns(3)
        summary_cols[0].metric("Total", total)
        summary_cols[1].metric("Active / unacknowledged", active)
        summary_cols[2].metric("Acknowledged", acknowledged)
        if total:
            first = page_index * ALERTS_PER_PAGE + 1
            last = min(first + len(visible_alerts) - 1, total)
            st.caption(
                f"Showing {first}–{last} of {total} matching alerts · "
                f"Loaded newest {len(loaded.alerts)} persisted records · "
                f"Cache refreshed {_utc_text(loaded.loaded_at)}"
            )
        else:
            st.caption(f"Loaded newest {len(loaded.alerts)} persisted records · Cache refreshed {_utc_text(loaded.loaded_at)}")
        _render_alerts(visible_alerts, controller, len(loaded.alerts))
    except Exception as exc:
        st.error(f"Alerts Center could not load persisted records: {type(exc).__name__}: {exc}")


def render_alerts_view() -> None:
    """Render the Phase 6 Alerts Center."""
    render_page_header("Alerts Center", "PERSISTED MARKET OBSERVATIONS")
    st.caption("Persisted market observations only. Alerts are not trading recommendations.")
    symbols = [item.symbol for item in INSTRUMENTS]
    today = datetime.now(timezone.utc).date()
    controls = st.columns([1, 1, 1.4, 1, 1.2, 1.6, 0.7])
    instrument = controls[0].selectbox("Instrument", ["All", *symbols], key="alerts_instrument")
    provider = controls[1].selectbox("Provider", ["All", "MT5", "Binance"], key="alerts_provider")
    category = controls[2].selectbox("Category", ["All", *ALERT_CATEGORIES], key="alerts_category")
    severity = controls[3].selectbox("Severity", ["All", *ALERT_SEVERITIES], key="alerts_severity")
    acknowledgement = controls[4].selectbox(
        "Acknowledgement", ["All", "Unacknowledged", "Acknowledged"], key="alerts_ack_filter")
    date_range = controls[5].date_input(
        "Detection date range (UTC)", value=(today - timedelta(days=30), today),
        key="alerts_date_range")
    # Match the field label height so Refresh aligns with the filter inputs.
    controls[6].markdown('<div aria-hidden="true" style="height: 1.75rem"></div>',
                         unsafe_allow_html=True)
    force_refresh = controls[6].button("Refresh", key="alerts_refresh", use_container_width=True)
    _alerts_fragment(force_refresh, instrument, provider, category, severity, acknowledgement, date_range)
