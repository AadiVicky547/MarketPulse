"""Mocked tests for the candle timestamp investigation helper."""
from datetime import datetime, timezone

import pytest

from scripts.diagnose_candle_timestamps import assess_candle_time


def test_h1_assessment_marks_a_finished_candle_completed():
    opened = datetime(2026, 11, 12, 15, tzinfo=timezone.utc)
    now = datetime(2026, 11, 12, 17, 28, tzinfo=timezone.utc)

    result = assess_candle_time(opened, now)

    assert result.expected_close_utc == datetime(2026, 11, 12, 16, tzinfo=timezone.utc)
    assert result.status == "COMPLETED"
    assert result.completed is True
    assert result.future_dated is False


def test_h1_assessment_marks_the_open_current_candle_incomplete():
    opened = datetime(2026, 11, 12, 17, tzinfo=timezone.utc)
    now = datetime(2026, 11, 12, 17, 28, tzinfo=timezone.utc)

    result = assess_candle_time(opened, now)

    assert result.expected_close_utc == datetime(2026, 11, 12, 18, tzinfo=timezone.utc)
    assert result.status == "INCOMPLETE"
    assert result.completed is False
    assert result.future_dated is False


def test_h1_assessment_flags_a_future_open_separately_from_incomplete():
    opened = datetime(2026, 11, 12, 19, tzinfo=timezone.utc)
    now = datetime(2026, 11, 12, 17, 28, tzinfo=timezone.utc)

    result = assess_candle_time(opened, now)

    assert result.expected_close_utc == datetime(2026, 11, 12, 20, tzinfo=timezone.utc)
    assert result.status == "FUTURE_DATED"
    assert result.completed is False
    assert result.future_dated is True


def test_timestamp_assessment_requires_aware_datetimes_and_positive_interval():
    aware = datetime(2026, 11, 12, 17, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="timezone-aware"):
        assess_candle_time(datetime(2026, 11, 12, 17), aware)
    with pytest.raises(ValueError, match="positive"):
        assess_candle_time(aware, aware, 0)
