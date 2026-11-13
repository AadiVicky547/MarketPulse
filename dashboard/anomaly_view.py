"""Anomaly Detection Intelligence page using shared Overview results.

The page presents the existing Isolation Forest service output. It does not
create a model, score observations, or infer feature importance.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import plotly.graph_objects as go
import streamlit as st

from analytics.anomaly_service import AVAILABLE, ERROR, UNAVAILABLE, WARMING_UP
from config import INSTRUMENTS, REFRESH
from dashboard.overview import _field, _get_runtime, format_value
from dashboard.theme import COLORS, render_page_header, style_figure

STATUSES = ("ANOMALY", "NORMAL", "WARMING_UP", "STALE", "DISCONNECTED", "UNAVAILABLE", "ERROR")
STATUS_COLORS = {
    "ANOMALY": COLORS["red"], "NORMAL": COLORS["green"], "WARMING_UP": COLORS["yellow"],
    "STALE": COLORS["yellow"], "DISCONNECTED": COLORS["red"], "UNAVAILABLE": COLORS["muted"],
    "ERROR": "#c39cff",
}


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _display_number(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else format_value(number)


def prepare_anomaly_rows(results: Any, *, minimum_training_samples: int = 50) -> list[dict[str, Any]]:
    """Normalize six service results for honest status and score presentation."""
    rows: list[dict[str, Any]] = []
    for configured in INSTRUMENTS:
        symbol = configured.symbol
        result = _field(results, symbol)
        service_status = str(_field(result, "status", UNAVAILABLE)).upper()
        freshness = str(_field(result, "freshness", "UNKNOWN")).upper()
        connection = str(_field(result, "connection_state", "UNAVAILABLE")).upper()
        status = service_status
        if result is not None:
            if connection != "CONNECTED":
                status = "DISCONNECTED" if connection == "DISCONNECTED" else UNAVAILABLE
            elif freshness == "DISCONNECTED":
                status = "DISCONNECTED"
            elif freshness == "STALE":
                status = "STALE"
            elif freshness != "LIVE":
                status = UNAVAILABLE
        raw_anomaly = _field(result, "anomaly_detected") if result is not None else None
        if status == AVAILABLE and not isinstance(raw_anomaly, bool):
            status = UNAVAILABLE
        anomaly_value = raw_anomaly if status == AVAILABLE else None
        if status == AVAILABLE and isinstance(anomaly_value, bool):
            determination = "YES" if anomaly_value else "NO"
            plot_status = "ANOMALY" if anomaly_value else "NORMAL"
        else:
            determination = "NOT YET DETERMINED"
            plot_status = status if status in STATUSES else "UNAVAILABLE"
        features_used = tuple(_field(result, "features_used", ()) or ())
        raw_values = _field(result, "feature_values", {}) or {}
        available_features = tuple(_field(result, "available_features", ()) or ())
        if not available_features and isinstance(raw_values, Mapping):
            # Backward-compatible preparation for results created before the
            # explicit current-feature provenance field was added.
            available_features = tuple(raw_values)
        feature_rows = []
        if isinstance(raw_values, Mapping):
            for name, raw_value in raw_values.items():
                value = _number(raw_value)
                feature_rows.append({
                    "feature": str(name), "value": value,
                    "display": _display_number(value),
                    "used_by_model": name in features_used,
                })
        warnings = tuple(_field(result, "data_quality_warnings", ()) or ())
        explanation = str(_field(result, "explanation", "") or "")
        if status != AVAILABLE and result is not None and freshness != "LIVE":
            explanation = f"Not evaluated as a current observation because data freshness is {freshness}."
        rows.append({
            "instrument": symbol,
            "provider": configured.source,
            "status": status,
            "service_status": service_status,
            "determination": determination,
            "plot_status": plot_status,
            "anomaly_detected": anomaly_value if isinstance(anomaly_value, bool) else None,
            "score": (_number(_field(result, "anomaly_score"))
                      if status == AVAILABLE and isinstance(anomaly_value, bool) else None),
            "model_decision": _field(result, "model_decision") if status == AVAILABLE else None,
            "sample_count": int(_field(result, "sample_count", 0) or 0),
            "training_target": minimum_training_samples,
            "feature_count": len(features_used),
            "available_feature_count": len(available_features),
            "available_features": available_features,
            "features_used": features_used,
            "feature_rows": tuple(feature_rows),
            "model_status": str(_field(result, "model_status", "NOT_FIT")),
            "timestamp_trust": _field(result, "timestamp_trust", "not reported"),
            "freshness": freshness,
            "connection": connection,
            "received_at": _field(result, "last_update_received_at"),
            "explanation": explanation or "No explanation is available.",
            "warnings": warnings,
        })
    return rows


def anomaly_status_figure(rows: list[Mapping[str, Any]]) -> go.Figure:
    """Show one classified status for each instrument; pending is not normal."""
    figure = go.Figure()
    for status in STATUSES:
        selected = [row for row in rows if row["plot_status"] == status]
        if not selected:
            continue
        figure.add_trace(go.Scatter(
            x=[status.replace("_", " ") for _ in selected],
            y=[row["instrument"] for row in selected],
            mode="markers", name=status.replace("_", " "),
            marker={"size": 17, "color": STATUS_COLORS[status],
                    "line": {"width": 1, "color": "#e6edf5"}},
            customdata=[[row["provider"], row["determination"], row["freshness"]]
                        for row in selected],
            hovertemplate=("%{y}<br>Provider: %{customdata[0]}<br>" 
                           "Evaluation: %{customdata[1]}<br>Freshness: %{customdata[2]}<extra></extra>")))
    figure.update_layout(
        title="Anomaly evaluation status by instrument",
        height=330,
        xaxis={"title": "Current evaluation state", "categoryorder": "array",
               "categoryarray": [status.replace("_", " ") for status in STATUSES]},
        yaxis={"title": "Instrument", "categoryorder": "array",
               "categoryarray": [item.symbol for item in INSTRUMENTS]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return style_figure(figure)


def anomaly_score_figure(rows: list[Mapping[str, Any]]) -> go.Figure:
    """Compare only finite anomaly scores from fresh AVAILABLE model results."""
    valid = [row for row in rows if row["status"] == AVAILABLE and _number(row["score"]) is not None]
    figure = go.Figure()
    if valid:
        figure.add_trace(go.Bar(
            x=[row["instrument"] for row in valid], y=[row["score"] for row in valid],
            marker_color=[STATUS_COLORS["ANOMALY"] if row["anomaly_detected"]
                          else STATUS_COLORS["NORMAL"] for row in valid],
            customdata=[["ANOMALY" if row["anomaly_detected"] else "NORMAL", row["model_decision"]]
                        for row in valid],
            hovertemplate="%{x}<br>Score: %{y:.6g}<br>Status: %{customdata[0]}<br>Decision: %{customdata[1]}<extra></extra>"))
    figure.update_layout(
        title="Isolation Forest score · available results only",
        height=320,
        xaxis_title="Instrument", yaxis_title="Anomaly score (higher = more unusual)",
    )
    return style_figure(figure)


def _render_status_summary(rows: list[dict[str, Any]]) -> None:
    st.plotly_chart(anomaly_status_figure(rows), use_container_width=True)
    st.plotly_chart(anomaly_score_figure(rows), use_container_width=True)
    unavailable = sum(row["status"] != AVAILABLE for row in rows)
    if unavailable:
        st.caption(f"{unavailable} instrument(s) do not have a current evaluable score. These are not counted as normal.")


def _render_instrument_cards(rows: list[dict[str, Any]]) -> None:
    st.markdown("### Instrument results")
    for start in range(0, len(rows), 2):
        columns = st.columns(2, gap="medium")
        for column, row in zip(columns, rows[start:start + 2]):
            with column:
                with st.container(border=True):
                    st.markdown(f"#### {row['instrument']} · {row['provider']}")
                    # Keep operational state easy to find, while moving the
                    # verbose model/provenance details out of the card face.
                    st.caption(
                        f"{row['status']} · {row['connection']} · {row['freshness']} · "
                        f"Timestamp: {row['timestamp_trust']}"
                    )
                    st.metric("Anomaly detected", row["determination"])
                    # Keep the useful scoring context visible below the result.
                    # Values come directly from the latest shared-service result.
                    decision = (row["model_decision"] if row["model_decision"] is not None
                                else "not available")
                    st.caption(
                        f"Model: {row['model_status']} · Score: {_display_number(row['score'])} · "
                        f"Decision: {decision}"
                    )
                    st.caption(
                        f"Prior valid observations: {row['sample_count']}/{row['training_target']} · "
                        f"Current valid features: {row['available_feature_count']} · "
                        f"Model features: {row['feature_count']}"
                    )


@st.fragment(run_every=REFRESH.dashboard_seconds)
def _anomaly_fragment() -> None:
    try:
        runtime = _get_runtime()
        # This is the same shared analytics cycle/model used by Overview; it
        # schedules work asynchronously and never constructs another detector.
        data = runtime.data_service.refresh()
        if data.analytics_pending and not data.anomalies:
            st.info("Anomaly analysis is warming up in the shared analytics worker. Warm-up is not a normal result.")
        rows = prepare_anomaly_rows(
            data.anomalies,
            minimum_training_samples=runtime.data_service.anomaly_service.settings.minimum_training_samples,
        )
        if data.errors:
            with st.expander("Analytics status details", expanded=False):
                for error in data.errors:
                    st.caption(error)
        _render_status_summary(rows)
        _render_instrument_cards(rows)
    except Exception as exc:
        st.error(f"Anomaly page could not prepare this view: {type(exc).__name__}: {exc}")


def render_anomaly_view() -> None:
    """Render the Phase 5 Anomaly Detection Intelligence page."""
    render_page_header("Anomaly Detection Intelligence", "STATISTICAL OBSERVATIONS")
    _anomaly_fragment()
