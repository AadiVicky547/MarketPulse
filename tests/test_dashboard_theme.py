"""Tests for the shared visual presentation helpers."""
from html import escape

import plotly.graph_objects as go
from unittest.mock import patch

from dashboard.theme import (
    CHART_COLORS,
    COLORS,
    alert_container_key,
    alert_severity_badge_html,
    page_header_html,
    render_top_navigation,
    style_figure,
    top_brand_html,
    _THEME_CSS,
)


def test_page_header_escapes_text_and_uses_shared_heading_structure():
    markup = page_header_html("Quotes <today>", "OVERVIEW & STATUS")

    assert 'class="mp-page-header"' in markup
    assert f"{escape('Quotes <today>')}" in markup
    assert f"{escape('OVERVIEW & STATUS')}" in markup
    assert "Multi-Asset Liquidity, Volatility &amp; Market Regime Intelligence" in markup
    assert "<h1" in markup


def test_style_figure_applies_shared_dark_theme_without_changing_data():
    figure = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4], name="sample")])
    original_x = list(figure.data[0].x)
    original_y = list(figure.data[0].y)

    styled = style_figure(figure, height=360)

    assert styled.layout.paper_bgcolor == COLORS["surface"]
    assert styled.layout.plot_bgcolor == COLORS["surface"]
    assert list(styled.data[0].x) == original_x
    assert list(styled.data[0].y) == original_y
    assert styled.layout.height == 360
    assert list(styled.layout.colorway) == list(CHART_COLORS)


def test_shared_palette_uses_restrained_terminal_colors():
    assert COLORS["background"].startswith("#")
    assert COLORS["green"] != COLORS["red"]
    assert COLORS["yellow"] != COLORS["accent"]
    assert COLORS["accent"] == "#F0A52B"


def test_top_navigation_retains_all_pages_and_uses_horizontal_radio():
    pages = (
        "Overview", "Volatility", "Liquidity & Order Flow", "Market Regime",
        "Anomaly Detection", "Alerts Center",
    )
    with patch("dashboard.theme.st.radio", return_value="Volatility") as radio:
        selected = render_top_navigation(pages)

    assert selected == "Volatility"
    radio.assert_called_once_with(
        "MarketPulse navigation",
        pages,
        horizontal=True,
        label_visibility="collapsed",
        key="marketpulse_top_navigation",
    )


def test_brand_has_visible_wordmark_and_custom_vector_mark():
    markup = top_brand_html()

    assert "MARKETPULSE" in markup
    assert '<svg class="mp-logo-mark"' in markup
    assert 'aria-label="MarketPulse market pulse mark"' in markup
    assert "Multi-Asset Liquidity" in markup
    assert '[data-testid="stHeader"]' in _THEME_CSS
    assert "display: none" in _THEME_CSS


def test_alert_severity_colors_and_panel_keys_are_visual_only():
    critical = alert_severity_badge_html("CRITICAL")
    warning = alert_severity_badge_html("WARNING")

    assert COLORS["red"] in critical
    assert COLORS["yellow"] in warning
    assert alert_container_key("CRITICAL", "item-1") == "mp-alert-critical-item-1"
    assert "st-key-mp-alert-critical" in _THEME_CSS
    assert "st-key-mp-alert-warning" in _THEME_CSS
    assert "background: rgba(224, 82, 82, .075)" in _THEME_CSS
    assert "background: rgba(229, 196, 71, .075)" in _THEME_CSS


def test_watchlist_quote_styles_dim_labels_but_keep_values_opaque():
    assert "color: #A7AFB7 !important" in _THEME_CSS
    assert "color: var(--mp-value-color, #FFFFFF) !important" in _THEME_CSS
