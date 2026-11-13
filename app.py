"""Streamlit entry point for MarketPulse."""
from __future__ import annotations

import streamlit as st
import logging

from config import DASHBOARD
from dashboard.overview import render_overview, render_shared_analytics_scheduler
from dashboard.volatility_view import render_volatility_view
from dashboard.liquidity_view import render_liquidity_view
from dashboard.regime_view import render_regime_view
from dashboard.anomaly_view import render_anomaly_view
from dashboard.alerts_view import render_alerts_view
from dashboard.theme import apply_theme, render_top_brand, render_top_navigation


def _configure_marketpulse_logging() -> None:
    """Send MarketPulse market-worker lifecycle logs to the Streamlit console."""
    market_logger = logging.getLogger("data.market_manager")
    market_logger.setLevel(logging.INFO)
    if not any(getattr(handler, "_marketpulse_console", False)
               for handler in market_logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handler._marketpulse_console = True
        market_logger.addHandler(handler)
    # Avoid duplicate records through Streamlit's root logger.
    market_logger.propagate = False


def main() -> None:
    """Configure the application and render the selected dashboard page."""
    _configure_marketpulse_logging()
    st.set_page_config(
        page_title=DASHBOARD.title,
        layout=DASHBOARD.layout,
        initial_sidebar_state="collapsed",
    )
    apply_theme()
    # A session-level fragment continuously requests the one shared analytics
    # cycle regardless of which page is selected. The request never blocks the
    # page render and cannot start an overlapping cycle.
    render_shared_analytics_scheduler()
    render_top_brand()
    page = render_top_navigation((
        "Overview",
        "Volatility",
        "Liquidity & Order Flow",
        "Market Regime",
        "Anomaly Detection",
        "Alerts Center",
    ))
    if page == "Volatility":
        render_volatility_view()
    elif page == "Liquidity & Order Flow":
        render_liquidity_view()
    elif page == "Market Regime":
        render_regime_view()
    elif page == "Anomaly Detection":
        render_anomaly_view()
    elif page == "Alerts Center":
        render_alerts_view()
    else:
        render_overview()


if __name__ == "__main__":
    main()
