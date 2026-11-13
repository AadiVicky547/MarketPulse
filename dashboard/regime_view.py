"""Market Regime Intelligence page over the existing Overview analytics cache.

This page reuses the shared MarketDataManager runtime and its cached
RegimeService results. It adds no classifier, connector, or background worker.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Mapping

import plotly.graph_objects as go
import streamlit as st

from config import INSTRUMENTS, REFRESH
from dashboard.overview import _field, _get_runtime, format_value
from dashboard.theme import COLORS, render_page_header, style_figure

REGIMES = ("TRENDING", "RANGING", "HIGH_VOLATILITY", "LOW_VOLATILITY")
REGIME_COLORS = {
    "TRENDING": COLORS["accent"],
    "RANGING": COLORS["secondary"],
    "HIGH_VOLATILITY": COLORS["red"],
    "LOW_VOLATILITY": COLORS["green"],
}
FEATURES = (
    ("adx", "ADX"),
    ("atr_percentile", "ATR percentile"),
    ("realized_volatility_percentile", "Realized volatility percentile"),
    ("ema_slope_bps", "EMA slope"),
    ("price_momentum_pct", "Price momentum"),
    ("session_range_expansion_ratio", "Session range expansion"),
)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def feature_display(feature: Any) -> str:
    """Format feature evidence while keeping unavailable distinct from zero."""
    if feature is None:
        return "UNAVAILABLE"
    status = str(_field(feature, "status", "UNAVAILABLE")).upper()
    value = _number(_field(feature, "value"))
    if status == "AVAILABLE" and value is not None:
        unit = str(_field(feature, "unit", ""))
        return f"{format_value(value)}{f' {unit}' if unit else ''}"
    warning = _field(feature, "warning")
    return f"{status}{f' — {warning}' if warning else ''}"


def prepare_regime_rows(results: Any, analysis_updated_at: datetime | None = None) -> list[dict[str, Any]]:
    """Prepare six display rows without converting missing evidence to values."""
    rows: list[dict[str, Any]] = []
    for configured in INSTRUMENTS:
        symbol = configured.symbol
        result = _field(results, symbol)
        service_status = str(_field(result, "status", "UNAVAILABLE")).upper()
        freshness = str(_field(result, "last_observation_freshness", "UNKNOWN")).upper()
        classification = _field(result, "current_regime")
        display_status = service_status
        display_regime = (classification if service_status == "AVAILABLE" and classification in REGIMES
                          else None)
        if service_status == "AVAILABLE" and display_regime is None:
            display_status = "UNAVAILABLE"
        # Historical classifications are retained as evidence, but stale or
        # disconnected observations are never presented as the current regime.
        if result is not None and freshness != "LIVE":
            display_status = freshness if freshness not in {"", "UNKNOWN"} else "FRESHNESS_UNKNOWN"
            display_regime = None
        features = _field(result, "features", {})
        feature_rows = []
        for key, label in FEATURES:
            feature = _field(features, key)
            feature_rows.append({
                "key": key,
                "label": label,
                "status": str(_field(feature, "status", "UNAVAILABLE")).upper(),
                "value": _number(_field(feature, "value")),
                "display": feature_display(feature),
                "unit": _field(feature, "unit", ""),
                "methodology": _field(feature, "methodology", ""),
                "sample_count": _field(feature, "sample_count", 0),
                "warning": _field(feature, "warning"),
            })
        warning_values = tuple(_field(result, "data_quality_warnings", ()) or ())
        # Feature-specific warnings are shown alongside each unavailable item;
        # service-level warnings remain in the separate data-quality section.
        rows.append({
            "instrument": symbol,
            "source": configured.source,
            "timeframe": _field(result, "timeframe", "1h"),
            "status": display_status,
            "service_status": service_status,
            "freshness": freshness,
            "regime": display_regime,
            "last_classification": classification if display_regime is None else None,
            "evidence_strength": _field(result, "evidence_strength") if result is not None else None,
            "timestamp_trust": _field(result, "timestamp_trust", "not reported"),
            "analysis_updated_at": analysis_updated_at,
            "received_at": _field(result, "last_update_received_at"),
            "explanation": _field(result, "explanation", "No regime explanation is available."),
            "warnings": warning_values,
            "candle_count": _field(result, "candle_count", 0),
            "features": tuple(feature_rows),
        })
    return rows


def summarize_regimes(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    """Count only fresh available classifications; keep other states distinct."""
    summary = {name: 0 for name in REGIMES}
    summary.update({"STALE": 0, "WARMING_UP": 0, "UNAVAILABLE": 0, "OTHER": 0})
    for row in rows:
        status = str(row.get("status", "UNAVAILABLE")).upper()
        regime = row.get("regime")
        if status == "AVAILABLE" and regime in REGIMES:
            summary[regime] += 1
        elif status in summary:
            summary[status] += 1
        else:
            summary["OTHER"] += 1
    return summary


def regime_comparison_figure(rows: list[Mapping[str, Any]]) -> go.Figure:
    """Build a categorical cross-instrument plot using fresh classifications only."""
    figure = go.Figure()
    for regime in REGIMES:
        selected = [row for row in rows if row.get("status") == "AVAILABLE" and row.get("regime") == regime]
        if not selected:
            continue
        figure.add_trace(go.Scatter(
            x=[regime.replace("_", " ") for _ in selected],
            y=[row["instrument"] for row in selected],
            mode="markers+text", name=regime.replace("_", " "),
            marker={"size": 16, "color": REGIME_COLORS[regime], "line": {"width": 1, "color": COLORS["text"]}},
            text=[str(row.get("evidence_strength") or "") for row in selected],
            textposition="top center",
            hovertemplate="%{y}<br>Regime: %{x}<br>Evidence: %{text}<extra></extra>"))
    figure.update_layout(
        title="Current regime by instrument · fresh, available classifications",
        height=330,
        xaxis={"title": "Rule-based classification", "categoryorder": "array",
               "categoryarray": [regime.replace("_", " ") for regime in REGIMES]},
        yaxis={"title": "Instrument", "categoryorder": "array",
               "categoryarray": [item.symbol for item in INSTRUMENTS]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return style_figure(figure)


def _utc_text(value: Any) -> str:
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return "not reported"


def _render_summary(rows: list[dict[str, Any]]) -> None:
    summary = summarize_regimes(rows)
    st.markdown("### Cross-asset regime summary")
    columns = st.columns(4)
    for column, regime in zip(columns, REGIMES):
        column.metric(regime.replace("_", " "), summary[regime])
    status_columns = st.columns(4)
    for column, status in zip(status_columns, ("STALE", "WARMING_UP", "UNAVAILABLE", "OTHER")):
        column.caption(f"{status}: {summary[status]}")
    st.plotly_chart(regime_comparison_figure(rows), use_container_width=True)
    st.caption("Evidence strength is a qualitative rule-confirmation label, not a probability. The classifier describes observed candles; it does not predict prices or provide a trading recommendation.")


def _render_feature_breakdown(row: Mapping[str, Any]) -> None:
    feature_frame = []
    for feature in row["features"]:
        feature_frame.append({
            "Feature": feature["label"],
            "Value / status": feature["display"],
            "Samples": feature["sample_count"],
            "Method": feature["methodology"],
            "Data note": feature["warning"] or "",
        })
    st.dataframe(feature_frame, hide_index=True, use_container_width=True)


def _render_cards(rows: list[dict[str, Any]]) -> None:
    st.markdown("### Instrument detail")
    for start in range(0, len(rows), 2):
        columns = st.columns(2, gap="medium")
        for column, row in zip(columns, rows[start:start + 2]):
            with column:
                with st.container(border=True):
                    regime_label = (row["regime"] or row["status"]).replace("_", " ")
                    st.markdown(f"#### {row['instrument']} · {row['source']}")
                    st.metric("Market regime", regime_label)
                    st.caption(
                        f"Status: {row['status']} · Freshness: {row['freshness']} · "
                        f"Evidence strength: {row['evidence_strength'] or 'not available for current state'} · "
                        f"Timeframe: {row['timeframe']}"
                    )
                    st.caption(f"Analysis updated: {_utc_text(row['analysis_updated_at'])} · Candles: {row['candle_count']}")
                    st.caption(f"Timestamp trust: {row['timestamp_trust']} · Receive time: {_utc_text(row['received_at'])}")
                    if row["last_classification"]:
                        st.info(f"Last historical classification: {row['last_classification']} (not shown as current because data is {row['status'].lower()}).")
                    explanation = row["explanation"]
                    if explanation:
                        st.write(explanation)
                    _render_feature_breakdown(row)


@st.fragment(run_every=REFRESH.dashboard_seconds)
def _regime_fragment() -> None:
    try:
        runtime = _get_runtime()
        # refresh() reads an immutable manager snapshot immediately and reuses
        # the Overview's existing single analytics worker/cache.
        data = runtime.data_service.refresh()
        if data.analytics_pending and not data.regimes:
            st.info("Regime analysis is warming up in the shared background analytics worker.")
        rows = prepare_regime_rows(data.regimes, data.regime_updated_at)
        if data.errors:
            with st.expander("Analytics status details", expanded=False):
                for error in data.errors:
                    st.caption(error)
        _render_summary(rows)
        st.divider()
        _render_cards(rows)
    except Exception as exc:
        st.error(f"Market Regime page could not prepare this view: {type(exc).__name__}: {exc}")


def render_regime_view() -> None:
    """Render the Phase 4 Market Regime Intelligence page."""
    render_page_header("Market Regime Intelligence", "REGIME CLASSIFICATION")
    _regime_fragment()
