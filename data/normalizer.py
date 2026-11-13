"""Small shared helpers for timezone-aware timestamps in MarketPulse.

This module normalizes representation only. It does not infer a provider's
timezone or correct broker event timestamps.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def as_utc_datetime(value: Any) -> datetime | None:
    """Return an aware datetime in UTC; reject naive values instead of assuming a zone."""
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(timezone.utc)
