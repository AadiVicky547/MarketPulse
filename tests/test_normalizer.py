"""Tests for strict UTC representation normalization."""
from datetime import datetime, timedelta, timezone

from data.normalizer import as_utc_datetime


def test_as_utc_datetime_converts_aware_values_without_changing_the_instant():
    source = datetime(2026, 11, 12, 19, 0, tzinfo=timezone(timedelta(hours=2)))
    assert as_utc_datetime(source) == datetime(2026, 11, 12, 17, 0, tzinfo=timezone.utc)


def test_as_utc_datetime_does_not_assume_a_zone_for_naive_or_invalid_values():
    assert as_utc_datetime(datetime(2026, 11, 12, 17, 0)) is None
    assert as_utc_datetime(None) is None
    assert as_utc_datetime("2026-11-12T17:00:00") is None
