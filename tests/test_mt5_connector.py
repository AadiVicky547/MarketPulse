"""Mock-based unit tests for the MT5 connector."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from data.mt5_connector import MT5Connector, MT5ConnectorError, _field


def connected_api(symbols=("EURUSDm",)):
    api = Mock()
    api.initialize.return_value = True
    api.terminal_info.return_value = object()
    api.symbols_get.return_value = [SimpleNamespace(name=name) for name in symbols]
    api.symbol_select.return_value = True
    api.symbol_info.return_value = SimpleNamespace(point=0.00001)
    api.last_error.return_value = (-1, "no data")
    return api


def test_symbol_matching_exact_then_variant_and_unrelated_rejected():
    api = connected_api(("EURUSDm", "EURUSD", "EURUSD.pro", "GBPUSD"))
    connector = MT5Connector(api=api)
    assert connector.initialize().connected
    matches = connector.discover_symbols(("EURUSD", "XAUUSD"))
    assert matches["EURUSD"].actual_symbol == "EURUSD"
    assert matches["EURUSD"].match_type == "exact"
    assert matches["XAUUSD"].actual_symbol is None


def test_broker_suffix_is_discovered_and_enabled():
    api = connected_api(("EURUSDm",))
    connector = MT5Connector(api=api)
    connector.initialize()
    result = connector.discover_symbols(("EURUSD",))["EURUSD"]
    assert result.actual_symbol == "EURUSDm"
    api.symbol_select.assert_called_once_with("EURUSDm", True)


def test_tick_normalization_and_spread():
    api = connected_api(("EURUSD",))
    api.symbol_info_tick.return_value = SimpleNamespace(bid=1.1, ask=1.1002, last=1.1001,
        time=1_800_000_000, time_msc=1_800_000_000_123, volume=7)
    connector = MT5Connector(api=api)
    connector.initialize()
    tick = connector.get_tick("EURUSD")
    assert tick["spread_price"] == pytest.approx(0.0002)
    assert tick["spread_points"] == pytest.approx(20)
    assert tick["tick_volume"] == 7
    assert tick["timestamp"].tzinfo == timezone.utc
    assert tick["timestamp"] == datetime.fromtimestamp(1_800_000_000.123, timezone.utc)
    assert tick["raw_time"] == 1_800_000_000
    assert tick["raw_time_msc"] == 1_800_000_000_123
    assert tick["timestamp_source"] == "time_msc"
    assert tick["timestamp_status"] == "VALID"
    assert tick["timestamp_trust"] == "UNVERIFIED_TIME_BASIS"
    assert tick["received_at"].tzinfo == timezone.utc
    assert tick["actual_symbol"] == "EURUSD"


def test_tick_falls_back_to_seconds_when_milliseconds_are_zero():
    api = connected_api(("EURUSD",))
    api.symbol_info_tick.return_value = SimpleNamespace(bid=1.1, ask=1.1002, last=1.1,
        time=1_800_000_000, time_msc=0, volume=0)
    connector = MT5Connector(api=api)
    connector.initialize()

    tick = connector.get_tick("EURUSD")

    assert tick["timestamp"] == datetime.fromtimestamp(1_800_000_000, timezone.utc)
    assert tick["timestamp_source"] == "time"
    assert tick["timestamp_status"] == "FALLBACK_SECONDS"
    assert tick["raw_time_msc"] == 0
    assert "time_msc (zero or negative)" in tick["timestamp_issues"]


@pytest.mark.parametrize("invalid_milliseconds", [float("nan"), -1, "not-a-time"])
def test_tick_reports_invalid_milliseconds_and_uses_valid_seconds_fallback(invalid_milliseconds):
    api = connected_api(("EURUSD",))
    api.symbol_info_tick.return_value = SimpleNamespace(bid=1.1, ask=1.1002, last=1.1,
        time=1_800_000_000, time_msc=invalid_milliseconds, volume=0)
    connector = MT5Connector(api=api)
    connector.initialize()

    tick = connector.get_tick("EURUSD")

    assert tick["timestamp"] == datetime.fromtimestamp(1_800_000_000, timezone.utc)
    assert tick["timestamp_source"] == "time"
    assert tick["timestamp_status"] == "FALLBACK_SECONDS"
    if isinstance(invalid_milliseconds, float) and np.isnan(invalid_milliseconds):
        assert np.isnan(tick["raw_time_msc"])
    else:
        assert tick["raw_time_msc"] == invalid_milliseconds
    assert tick["timestamp_warning"]


def test_tick_reports_missing_and_zero_timestamp_fields_without_fabrication():
    api = connected_api(("EURUSD",))
    api.symbol_info_tick.return_value = SimpleNamespace(bid=1.1, ask=1.1002, last=1.1,
        time=0, time_msc=0, volume=0)
    connector = MT5Connector(api=api)
    connector.initialize()

    tick = connector.get_tick("EURUSD")

    assert tick["timestamp"] is None
    assert tick["available"] is False
    assert tick["raw_time"] == 0 and tick["raw_time_msc"] == 0
    assert tick["timestamp_source"] is None
    assert tick["timestamp_status"] == "INVALID"
    assert tick["timestamp_issues"] == ("time (zero or negative)", "time_msc (zero or negative)")


def test_tick_reports_seconds_milliseconds_disagreement_and_prefers_milliseconds():
    api = connected_api(("EURUSD",))
    api.symbol_info_tick.return_value = SimpleNamespace(bid=1.1, ask=1.1002, last=1.1,
        time=1_800_000_000, time_msc=1_800_000_001_234, volume=0)
    connector = MT5Connector(api=api)
    connector.initialize()

    tick = connector.get_tick("EURUSD")

    assert tick["timestamp"] == datetime.fromtimestamp(1_800_000_001.234, timezone.utc)
    assert tick["timestamp_source"] == "time_msc"
    assert tick["timestamp_status"] == "INCONSISTENT"
    assert "time and time_msc disagree" in tick["timestamp_issues"]


def test_tick_uses_milliseconds_when_seconds_field_is_missing():
    api = connected_api(("EURUSD",))
    api.symbol_info_tick.return_value = {"bid": 1.1, "ask": 1.1002, "last": 1.1,
                                         "time_msc": 1_800_000_000_234, "volume": 0}
    connector = MT5Connector(api=api)
    connector.initialize()

    tick = connector.get_tick("EURUSD")

    assert tick["timestamp"] == datetime.fromtimestamp(1_800_000_000.234, timezone.utc)
    assert tick["timestamp_source"] == "time_msc"
    assert tick["timestamp_status"] == "MILLISECONDS_ONLY"
    assert tick["raw_time"] is None


def test_missing_tick_is_explicit():
    api = connected_api(("EURUSD",))
    api.symbol_info_tick.return_value = None
    connector = MT5Connector(api=api)
    connector.initialize()
    tick = connector.get_tick("EURUSD")
    assert tick["available"] is False
    assert tick["tick_received"] is False
    assert tick["bid"] is None
    assert tick["error"] == "MT5 error -1: no data"


def test_invalid_timestamp_tick_is_retained_as_received_but_not_event_available():
    api = connected_api(("EURUSD",))
    api.symbol_info_tick.return_value = SimpleNamespace(bid=1.1, ask=1.1002, last=1.1,
        time=0, time_msc=0, volume=0)
    connector = MT5Connector(api=api)
    connector.initialize()

    tick = connector.get_tick("EURUSD")

    assert tick["tick_received"] is True
    assert tick["available"] is False
    assert tick["received_at"].tzinfo == timezone.utc
    assert tick["raw_time"] == tick["raw_time_msc"] == 0


def test_invalid_prices_and_timestamp_are_not_fabricated():
    api = connected_api(("EURUSD",))
    api.symbol_info_tick.return_value = SimpleNamespace(bid=0, ask=float("nan"), last=-1,
        time=0, time_msc=0, volume=-4)
    connector = MT5Connector(api=api)
    connector.initialize()
    tick = connector.get_tick("EURUSD")
    assert tick["bid"] is tick["ask"] is tick["last"] is None
    assert tick["spread_price"] is None
    assert tick["tick_volume"] is None
    assert tick["timestamp"] is None
    assert tick["available"] is False


def test_initialization_error_is_returned_and_operations_rejected():
    api = connected_api()
    api.initialize.return_value = False
    api.last_error.return_value = (10004, "terminal unavailable")
    connector = MT5Connector(api=api)
    status = connector.initialize()
    assert not status.connected
    assert "10004" in status.error
    with pytest.raises(MT5ConnectorError, match="not connected"):
        connector.get_tick("EURUSD")


def test_disconnected_terminal_detected():
    api = connected_api()
    connector = MT5Connector(api=api)
    connector.initialize()
    api.terminal_info.return_value = None
    with pytest.raises(MT5ConnectorError, match="disconnected"):
        connector.discover_symbols()
    assert connector.connected is False


def mt5_rates_array():
    """Build structured rows shaped like MetaTrader5 copy_rates results."""
    dtype = np.dtype([
        ("time", np.int64),
        ("open", np.float64),
        ("high", np.float64),
        ("low", np.float64),
        ("close", np.float64),
        ("tick_volume", np.uint64),
        ("spread", np.int32),
        ("real_volume", np.uint64),
    ])
    start = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
    return np.array([
        (start + index * 3600, 1.10 + index * 0.001, 1.12 + index * 0.001,
         1.09 + index * 0.001, 1.11 + index * 0.001,
         (12, 0, 20, 30, 40)[index], (15, 0, 17, 18, 19)[index],
         (120, 0, 200, 300, 400)[index])
        for index in range(5)
    ], dtype=dtype)


def test_field_supports_numpy_structured_records_and_preserves_zero():
    rates = mt5_rates_array()
    row = rates[1]

    assert isinstance(row, np.void)
    assert _field(row, "time") == int(row["time"])
    assert _field(row, "tick_volume") == 0
    assert _field(row, "spread") == 0
    assert _field(row, "real_volume") == 0
    assert _field(row, "missing", "fallback") == "fallback"
    np.testing.assert_array_equal(_field(rates, "time"), rates["time"])


def test_field_preserves_mapping_and_attribute_record_behavior():
    mapping = {"value": 0}
    record = SimpleNamespace(value=0)

    assert _field(mapping, "value", 99) == 0
    assert _field(mapping, "missing", 99) == 99
    assert _field(record, "value", 99) == 0
    assert _field(record, "missing", 99) == 99


def test_get_rates_converts_five_mt5_structured_rows_without_dropping_them():
    api = connected_api(("EURUSD",))
    api.copy_rates_from_pos.return_value = mt5_rates_array()
    connector = MT5Connector(api=api)
    assert connector.initialize().connected

    bars = connector.get_rates("EURUSD", timeframe=16385, count=5)

    assert isinstance(bars, list)
    assert len(bars) == 5
    api.copy_rates_from_pos.assert_called_once_with("EURUSD", 16385, 1, 5)
    assert [bar["timestamp"] for bar in bars] == sorted(bar["timestamp"] for bar in bars)
    assert all(bar["timestamp"].tzinfo == timezone.utc for bar in bars)
    assert bars[0]["timestamp"] == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert bars[0]["raw_time"] == int(mt5_rates_array()[0]["time"])
    assert bars[0]["timestamp_source"] == "time"
    assert bars[0]["timestamp_status"] == "VALID"
    assert bars[0]["timestamp_trust"] == "UNVERIFIED_TIME_BASIS"
    assert bars[1]["open"] == pytest.approx(1.101)
    assert bars[1]["high"] == pytest.approx(1.121)
    assert bars[1]["low"] == pytest.approx(1.091)
    assert bars[1]["close"] == pytest.approx(1.111)
    assert bars[1]["tick_volume"] == 0
    assert bars[1]["spread"] == 0
    assert bars[1]["real_volume"] == 0
    assert bars[0]["tick_volume"] == 12
    assert bars[0]["real_volume"] == 120
