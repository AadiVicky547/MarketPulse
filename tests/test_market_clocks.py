"""Mocked unit tests for the market-clock diagnostic helpers."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from scripts.diagnose_market_clocks import (
    account_identity_display,
    binance_milliseconds_to_utc,
    classify_mt5_offsets,
    clock_differences,
    extract_mt5_tick_times,
    fetch_binance_server_time,
    mask_account_login,
)


def test_binance_milliseconds_convert_to_aware_utc():
    assert binance_milliseconds_to_utc(1_800_000_000_123) == datetime.fromtimestamp(
        1_800_000_000.123, timezone.utc)


@pytest.mark.parametrize(
    ("binance_delta", "mt5_delta", "expected"),
    [(4, 8, (4, 8, 4)), (-4, -8, (-4, -8, -4))],
)
def test_clock_differences_preserve_positive_and_negative_signs(binance_delta, mt5_delta, expected):
    base = datetime(2026, 11, 12, tzinfo=timezone.utc)
    result = clock_differences(base, base + timedelta(seconds=binance_delta),
                               base + timedelta(seconds=mt5_delta))
    assert result["binance_minus_python_seconds"] == expected[0]
    assert result["mt5_minus_python_seconds"] == expected[1]
    assert result["mt5_minus_binance_seconds"] == expected[2]


def test_missing_mt5_tick_is_unavailable():
    assert extract_mt5_tick_times(None) is None
    base = datetime.now(timezone.utc)
    assert clock_differences(base, None, None) == {
        "binance_minus_python_seconds": None,
        "mt5_minus_python_seconds": None,
        "mt5_minus_binance_seconds": None,
    }


def test_mt5_raw_seconds_and_milliseconds_are_both_retained_and_converted():
    result = extract_mt5_tick_times(SimpleNamespace(time=1_800_000_000,
                                                    time_msc=1_800_000_000_456))
    assert result["raw_time"] == 1_800_000_000
    assert result["raw_time_msc"] == 1_800_000_000_456
    assert result["time_utc"] == datetime.fromtimestamp(1_800_000_000, timezone.utc)
    assert result["time_msc_utc"] == datetime.fromtimestamp(1_800_000_000.456, timezone.utc)
    assert result["event_utc"] == result["time_msc_utc"]


def test_binance_http_and_network_failures_are_reported_without_raising():
    http_response = Mock()
    http_response.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
    value, error = fetch_binance_server_time(get=Mock(return_value=http_response))
    assert value is None
    assert "HTTP/network error" in error
    assert "429" in error

    value, error = fetch_binance_server_time(get=Mock(side_effect=requests.Timeout("timed out")))
    assert value is None
    assert "timed out" in error


def test_binance_malformed_response_is_reported():
    response = Mock()
    response.json.return_value = {"unexpected": 1}
    value, error = fetch_binance_server_time(get=Mock(return_value=response))
    assert value is None
    assert "missing serverTime" in error


def test_stable_plus_two_hour_offset_is_classified():
    assert classify_mt5_offsets([7200.0, 7201.2, 7199.7]) == "STABLE_NEAR_PLUS_2_HOURS"


def test_variable_offsets_are_classified():
    assert classify_mt5_offsets([7200.0, 7180.0, 7225.0]) == "VARIABLE_OFFSET"


def test_other_stable_offset_and_no_data_classifications():
    assert classify_mt5_offsets([3600.0, 3602.0, 3599.0]) == "OTHER_STABLE_OFFSET"
    assert classify_mt5_offsets([]) == "UNAVAILABLE"


def test_account_identity_shows_server_and_masks_login():
    server, login = account_identity_display(
        SimpleNamespace(server="FundingPips-SIM1", login=12345678))
    assert server == "FundingPips-SIM1"
    assert login == "******78"
    assert "12345678" not in login


def test_account_identity_handles_missing_account_information():
    assert account_identity_display(None) is None
    assert account_identity_display({"server": "FundingPips-SIM1"}) == (
        "FundingPips-SIM1", "unavailable")
    assert mask_account_login(None) == "unavailable"
