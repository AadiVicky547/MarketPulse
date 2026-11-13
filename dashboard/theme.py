"""Shared visual system for the MarketPulse Streamlit terminal.

This module contains presentation helpers only. It does not read market data,
start workers, or alter analytical results.
"""
from __future__ import annotations

from html import escape
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from config import DASHBOARD


COLORS = {
    "background": "#080A0D",
    "sidebar": "#0B0D10",
    "surface": "#111418",
    "surface_raised": "#1A1F24",
    "border": "#343A40",
    "grid": "#292E33",
    "text": "#E6E8EA",
    "secondary": "#B2B8BF",
    "muted": "#818B94",
    "accent": "#F0A52B",
    "green": "#35C46A",
    "red": "#E05252",
    "yellow": "#E5C447",
    "blue": "#2B7BBB",
}

FONT_STACK = "Inter, Segoe UI, Arial, sans-serif"
CHART_COLORS = (
    COLORS["accent"], COLORS["green"], COLORS["yellow"], COLORS["red"],
    COLORS["blue"], "#9A7BBD",
)

_THEME_CSS = f"""
<style>
:root {{
  --mp-bg: {COLORS['background']};
  --mp-sidebar: {COLORS['sidebar']};
  --mp-surface: {COLORS['surface']};
  --mp-surface-raised: {COLORS['surface_raised']};
  --mp-border: {COLORS['border']};
  --mp-grid: {COLORS['grid']};
  --mp-text: {COLORS['text']};
  --mp-secondary: {COLORS['secondary']};
  --mp-muted: {COLORS['muted']};
  --mp-accent: {COLORS['accent']};
  --mp-green: {COLORS['green']};
  --mp-red: {COLORS['red']};
  --mp-yellow: {COLORS['yellow']};
}}

html, body, [class*="css"] {{ font-family: {FONT_STACK}; }}
.stApp, [data-testid="stAppViewContainer"] {{
  background: var(--mp-bg);
  color: var(--mp-text);
}}
[data-testid="stHeader"] {{
  display: none;
}}
[data-testid="stSidebar"] {{
  background: var(--mp-sidebar);
  border-right: 1px solid var(--mp-border);
}}
[data-testid="stSidebarContent"] {{ padding-top: 1.1rem; }}
.block-container {{
  max-width: 1660px;
  padding-top: 1.35rem;
  padding-bottom: 1.6rem;
}}

.mp-top-brand {{
  align-items: center;
  border-bottom: 1px solid var(--mp-border);
  display: flex;
  gap: .8rem;
  margin: 0 0 .55rem;
  min-height: 3.55rem;
  padding: .15rem 0 .65rem;
}}
.mp-logo-mark {{
  display: block;
  flex: 0 0 42px;
  height: 42px;
  width: 42px;
}}
.mp-brand-name {{
  color: #FFFFFF !important;
  font-size: 1.28rem;
  font-weight: 850;
  letter-spacing: .17em;
  line-height: 1.15;
  opacity: 1 !important;
  white-space: nowrap;
}}
.mp-brand-subtitle {{
  color: #B9C0C7 !important;
  border-left: 1px solid var(--mp-border);
  font-size: .76rem;
  line-height: 1.4;
  opacity: 1 !important;
  padding-left: 1.15rem;
}}
.mp-page-header {{
  border-bottom: 1px solid var(--mp-border);
  margin: 0 0 1rem;
  padding: .15rem 0 .9rem;
}}
.mp-page-eyebrow {{
  color: var(--mp-accent);
  font-size: .68rem;
  font-weight: 700;
  letter-spacing: .13em;
  margin-bottom: .35rem;
  text-transform: uppercase;
}}
.mp-page-title {{
  color: var(--mp-text);
  font-size: 1.62rem;
  font-weight: 690;
  letter-spacing: -.025em;
  line-height: 1.2;
  margin: 0;
}}
.mp-page-subtitle {{
  color: var(--mp-secondary);
  font-size: .83rem;
  margin: .35rem 0 0;
}}

[data-testid="stMetric"] {{
  background: var(--mp-surface);
  border: 1px solid var(--mp-border);
  border-radius: 7px;
  padding: .62rem .8rem;
}}
[data-testid="stMetricLabel"] {{ color: var(--mp-secondary); font-size: .76rem; }}
[data-testid="stMetricValue"] {{
  color: var(--mp-text);
  font-size: 1.35rem;
  font-variant-numeric: tabular-nums;
}}
.mp-asset-description {{
  color: #A7AFB7 !important;
  font-size: .74rem;
  margin-top: .2rem;
  opacity: .92 !important;
}}
.mp-quote-metric {{
  background: var(--mp-surface);
  border: 1px solid var(--mp-border);
  border-radius: 5px;
  margin: .75rem 0 .55rem;
  padding: .62rem .75rem;
}}
.mp-quote-label {{
  color: #A7AFB7 !important;
  font-size: .78rem;
  opacity: 1 !important;
}}
.mp-quote-price {{
  font-size: 1.42rem;
  font-variant-numeric: tabular-nums;
  font-weight: 650;
  line-height: 1.5;
  opacity: 1 !important;
}}
.mp-quote-detail {{
  align-items: baseline;
  color: #FFFFFF !important;
  display: flex;
  font-size: .82rem;
  gap: .38rem;
  line-height: 1.6;
  opacity: 1 !important;
  padding: .2rem 0;
  white-space: nowrap;
}}
.mp-quote-detail span {{ color: #A7AFB7 !important; }}
.mp-quote-detail strong {{
  color: var(--mp-value-color, #FFFFFF) !important;
  font-variant-numeric: tabular-nums;
  font-weight: 650;
}}
[data-testid="stCaptionContainer"] p {{ color: var(--mp-muted); }}
.mp-metric-status {{
  color: #9AA3AB;
  font-size: .68rem;
  line-height: 1.3;
  margin: -.35rem 0 .35rem;
}}
.mp-metric-warning {{
  border-left: 2px solid #6F7881;
  color: #A7AFB7;
  font-size: .69rem;
  line-height: 1.35;
  margin: .1rem 0 .35rem;
  padding-left: .42rem;
}}
.mp-volatility-regime-gap {{
  height: 1.75rem;
  width: 100%;
}}
.mp-volatility-state-unavailable {{
  border-left: 2px solid #E05252;
  color: #E58C8C;
  font-size: .76rem;
  margin: .35rem 0;
  padding-left: .5rem;
}}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: var(--mp-border); }}
[data-testid="stVerticalBlockBorderWrapper"]:has([class*="st-key-mp-alert-critical"]),
[data-testid="stVerticalBlockBorderWrapper"].st-key-mp-alert-critical {{
  background: rgba(224, 82, 82, .075) !important;
  border: 1px solid #E05252 !important;
}}
[data-testid="stVerticalBlockBorderWrapper"]:has([class*="st-key-mp-alert-warning"]),
[data-testid="stVerticalBlockBorderWrapper"].st-key-mp-alert-warning {{
  background: rgba(229, 196, 71, .075) !important;
  border: 1px solid #E5C447 !important;
}}
[data-testid="stVerticalBlockBorderWrapper"]:has([class*="st-key-mp-alert-info"]),
[data-testid="stVerticalBlockBorderWrapper"].st-key-mp-alert-info {{
  background: rgba(43, 123, 187, .06) !important;
  border: 1px solid #2B7BBB !important;
}}
.mp-alert-severity {{
  border-radius: 3px;
  display: inline-block;
  font-size: .72rem;
  font-weight: 750;
  letter-spacing: .04em;
  padding: .17rem .42rem;
}}
[data-testid="stDataFrame"], [data-testid="stTable"] {{
  border: 1px solid var(--mp-border);
  border-radius: 7px;
  overflow: hidden;
}}
[data-testid="stPlotlyChart"] {{
  background: var(--mp-surface);
  border: 1px solid rgba(52, 58, 64, .8);
  border-radius: 7px;
  padding: .25rem;
}}

[data-testid="stRadio"] div[role="radiogroup"] {{
  align-items: stretch;
  background: #0E1114;
  border: 1px solid var(--mp-border);
  border-radius: 4px;
  display: flex;
  flex-wrap: nowrap;
  gap: 0;
  margin: 0 0 1rem;
  overflow-x: auto;
  padding: .22rem;
}}
[data-testid="stRadio"] div[role="radiogroup"] label {{
  align-items: center;
  border-right: 1px solid #292E33;
  border-radius: 2px;
  color: #D6DADF !important;
  display: flex;
  font-size: .8rem;
  font-weight: 650;
  justify-content: center;
  margin: 0;
  min-height: 2.35rem;
  opacity: 1 !important;
  padding: .25rem .75rem;
  white-space: nowrap;
}}
[data-testid="stRadio"] div[role="radiogroup"] label * {{
  color: inherit !important;
  opacity: 1 !important;
}}
[data-testid="stRadio"] div[role="radiogroup"] label:last-child {{
  border-right: 0;
}}
[data-testid="stRadio"] div[role="radiogroup"] label:hover {{
  background: #1A1F24;
  color: var(--mp-text);
}}
[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {{
  background: var(--mp-accent) !important;
  color: #101214;
  font-weight: 750;
}}
[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) * {{
  color: #101214 !important;
}}
[data-testid="stButton"] button {{
  background: var(--mp-surface-raised);
  border: 1px solid var(--mp-border);
  border-radius: 5px;
  color: var(--mp-text);
  font-weight: 600;
  min-height: 2.25rem;
}}
[data-testid="stButton"] button:hover {{
  background: #282D32;
  border-color: var(--mp-accent);
  color: var(--mp-text);
}}
[data-baseweb="select"] > div, [data-baseweb="input"] > div {{
  background: var(--mp-surface);
  border-color: var(--mp-border);
}}
hr {{ border-color: var(--mp-border); }}
</style>
"""


def apply_theme() -> None:
    """Apply consistent app-wide Streamlit component styling."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def page_header_html(title: str, eyebrow: str = "MARKETPULSE · MARKET INTELLIGENCE") -> str:
    """Return escaped, consistent page-heading markup for the six views."""
    return (
        '<header class="mp-page-header">'
        f'<div class="mp-page-eyebrow">{escape(eyebrow)}</div>'
        f'<h1 class="mp-page-title">{escape(title)}</h1>'
        f'<p class="mp-page-subtitle">{escape(DASHBOARD.subtitle)}</p>'
        '</header>'
    )


def render_page_header(title: str, eyebrow: str = "MARKETPULSE · MARKET INTELLIGENCE") -> None:
    """Render a shared terminal-style page heading."""
    st.markdown(page_header_html(title, eyebrow), unsafe_allow_html=True)


def top_brand_html() -> str:
    """Return high-contrast brand markup with the custom MarketPulse glyph."""
    return (
        '<div class="mp-top-brand">'
        '<svg class="mp-logo-mark" viewBox="0 0 48 48" role="img" '
        'aria-label="MarketPulse market pulse mark" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="1" y="1" width="46" height="46" rx="7" '
        'fill="#15191D" stroke="#F0A52B" stroke-width="1.5"/>'
        '<path d="M10 31h5V20h-5zm9 0h5V13h-5zm9 0h5V17h-5zm9 0h2V9h-2" '
        'fill="#F0A52B" opacity=".88"/>'
        '<path d="M8 36h7l4-7 5 4 6-11 5 5 5-9" fill="none" '
        'stroke="#F2F4F5" stroke-linecap="round" stroke-linejoin="round" '
        'stroke-width="2"/>'
        '<circle cx="40" cy="18" r="2.1" fill="#35C46A"/>'
        '</svg>'
        '<div class="mp-brand-name">MARKETPULSE</div>'
        f'<div class="mp-brand-subtitle">{escape(DASHBOARD.subtitle)}</div>'
        '</div>'
    )


def render_top_brand() -> None:
    """Render application branding above the horizontal page navigation."""
    st.markdown(top_brand_html(), unsafe_allow_html=True)


def render_top_navigation(options: tuple[str, ...]) -> str:
    """Render the existing page choices as a horizontal terminal navigation bar."""
    return st.radio(
        "MarketPulse navigation",
        options,
        horizontal=True,
        label_visibility="collapsed",
        key="marketpulse_top_navigation",
    )


def alert_severity_color(severity: str) -> str:
    """Map persisted severity labels to visual colors without reinterpreting them."""
    return {
        "CRITICAL": COLORS["red"],
        "WARNING": COLORS["yellow"],
        "INFO": COLORS["blue"],
    }.get(str(severity).upper(), COLORS["muted"])


def alert_container_key(severity: str, unique_suffix: str) -> str:
    """Create a stable visual key for a severity-tinted alert card."""
    normalized = str(severity).lower()
    if normalized not in {"critical", "warning", "info"}:
        normalized = "other"
    return f"mp-alert-{normalized}-{unique_suffix}"


def alert_severity_badge_html(severity: str) -> str:
    """Return an escaped severity badge using the established severity color."""
    label = escape(str(severity).upper())
    color = alert_severity_color(str(severity))
    return (
        f'<span class="mp-alert-severity" style="border:1px solid {color};'
        f'color:{color};background:rgba(0,0,0,.14)">{label}</span>'
    )


def style_figure(figure: go.Figure, *, height: int | None = None) -> go.Figure:
    """Apply shared Plotly colors, typography, grid, and hover styling."""
    layout: dict[str, Any] = {
        "template": "plotly_dark",
        "paper_bgcolor": COLORS["surface"],
        "plot_bgcolor": COLORS["surface"],
        "font": {"family": FONT_STACK, "size": 11, "color": COLORS["secondary"]},
        "colorway": list(CHART_COLORS),
        "hoverlabel": {
            "bgcolor": COLORS["surface_raised"],
            "bordercolor": COLORS["border"],
            "font": {"family": FONT_STACK, "color": COLORS["text"]},
        },
        "margin": {"l": 42, "r": 20, "t": 54, "b": 38},
    }
    if height is not None:
        layout["height"] = height
    figure.update_layout(**layout)
    axis_style = {
        "gridcolor": COLORS["grid"],
        "linecolor": COLORS["border"],
        "zerolinecolor": COLORS["border"],
        "tickfont": {"color": COLORS["secondary"], "size": 10},
        "title_font": {"color": COLORS["secondary"], "size": 11},
    }
    figure.update_xaxes(**axis_style)
    figure.update_yaxes(**axis_style)
    return figure
