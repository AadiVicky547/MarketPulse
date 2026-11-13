"""Deterministic tests for standardized historical candle retrieval."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from data.candle_service import (
    BINANCE_KLINES_ENDPOINT,
    CandleDataError,
    CandleProviderError,
    CandleService,
)
from data.mt5_connector import MT5Connector

NOW = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
OPEN_MS = int(datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc).timestamp() * 1000)


def kline(open_ms=OPEN_MS, *, close_ms=None, close="101", volume="2.5", quote="252.5"):
    close_ms = open_ms + 3_599_999 if close_ms is None else close_ms
    return [open_ms, "100", "103", "99", close, volume, close_ms,
            quote, 10, "1", "100", "0"]


class FakeResponse:
    def __init__(self, payload, error=None):
        self.payload = payload
        self.error = error
        self.status_code = 200

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.payload


def service_for_binance(payload, *, rest_get=None):
    get = rest_get or Mock(return_value=FakeResponse(payload))
    return CandleService(rest_get=get, now=lambda: NOW), get


def test_binance_normalizes_utc_ohlcv_and_volume_semantics():
    service, get = service_for_binance([kline()])
    frame = service.get_candles("BTCUSDT", "1h", 10)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row.instrument == "BTCUSDT" and row.source == "Binance"
    assert row.timestamp == pd.Timestamp("2025-01-01T10:00:00Z")
    assert (row.open, row.high, row.low, row.close) == (100, 103, 99, 101)
    assert row.volume == 2.5 and row.quote_volume == 252.5
    assert row.volume_type == "base_asset_trade_volume"
    assert row.timestamp_trust == "TRUSTED_UTC"
    assert bool(row.completed) is True
    assert row.completion_status == "COMPLETED"
    get.assert_called_once_with(BINANCE_KLINES_ENDPOINT,
                                params={"symbol": "BTCUSDT", "interval": "1h", "limit": 10},
                                timeout=(3.05, 10.0))


def test_current_binance_candle_is_marked_and_filtered_by_default():
    current_open = int(datetime(2025, 1, 1, 11, 30, tzinfo=timezone.utc).timestamp() * 1000)
    payload = [kline(current_open, close_ms=int(NOW.timestamp() * 1000) + 30_000)]
    service, _ = service_for_binance(payload)
    assert service.get_candles("BTCUSDT", "1h").empty
    included = service.get_candles("BTCUSDT", "1h", include_incomplete=True)
    assert len(included) == 1 and bool(included.iloc[0].completed) is False


def test_binance_reorders_by_timestamp_and_rejects_duplicates():
    later = OPEN_MS + 3_600_000
    service, _ = service_for_binance([kline(later), kline(OPEN_MS)])
    frame = service.get_candles("ETHUSDT", "1h")
    assert frame.timestamp.is_monotonic_increasing
    service, _ = service_for_binance([kline(), kline()])
    with pytest.raises(CandleDataError, match="Duplicate"):
        service.get_candles("BTCUSDT", "1h")


def test_binance_rejects_invalid_ohlc_and_malformed_payload():
    bad_ohlc = kline()
    bad_ohlc[2] = "98"  # High below open/close.
    service, _ = service_for_binance([bad_ohlc])
    with pytest.raises(CandleDataError, match="inconsistent"):
        service.get_candles("BTCUSDT", "1h")
    service, _ = service_for_binance([[1, "100"]])
    with pytest.raises(CandleDataError, match="12 fields"):
        service.get_candles("BTCUSDT", "1h")


def test_binance_handles_network_rate_limit_and_short_history_explicitly():
    short, get = service_for_binance([kline()])
    assert len(short.get_candles("BTCUSDT", "1h", 50)) == 1
    assert get.call_count == 1
    rate_limit = requests.HTTPError("429 Too Many Requests")
    rate_limit.response = SimpleNamespace(status_code=429)
    service, _ = service_for_binance([], rest_get=Mock(return_value=FakeResponse([], rate_limit)))
    with pytest.raises(CandleProviderError, match="HTTP 429"):
        service.get_candles("ETHUSDT", "1m")
    service, _ = service_for_binance([], rest_get=Mock(side_effect=requests.Timeout("timeout")))
    with pytest.raises(CandleProviderError, match="Binance klines request failed"):
        service.get_candles("ETHUSDT", "1m")
    with pytest.raises(ValueError, match="at most 1000"):
        short.get_candles("BTCUSDT", "1h", 1001)


class FakeMT5:
    def __init__(self, rates=None, connected=True, last_error=(-5, "No history")):
        self.rates = rates or []
        self.connected = connected
        self.error = last_error
        self.initialize_calls = 0
        self.discover_calls = []
        self.rate_calls = []
        self.close_calls = 0

    def initialize(self):
        self.initialize_calls += 1
        return SimpleNamespace(connected=self.connected, message="disconnected", error="offline")

    def discover_symbols(self, instruments):
        self.discover_calls.append(instruments)
        return {symbol: SimpleNamespace(actual_symbol=f"{symbol}.a", message="broker variant")
                for symbol in instruments}

    def get_rates(self, instrument, timeframe, *, count, include_current):
        self.rate_calls.append((instrument, timeframe, count, include_current))
        return self.rates

    def close(self):
        self.close_calls += 1

    def last_error(self):
        return self.error


def mt5_bar(stamp, *, high=1.12, low=1.08):
    return {"requested_instrument": "EURUSD", "actual_symbol": "EURUSD.a",
            "timestamp": stamp, "open": 1.1, "high": high, "low": low, "close": 1.11,
            "tick_volume": 0, "real_volume": 0, "spread": 3,
            "raw_time": 1_735_714_800, "timestamp_source": "time", "timestamp_status": "VALID",
            "timestamp_trust": "UNVERIFIED_TIME_BASIS"}


def test_mt5_uses_existing_connector_suffix_and_preserves_zero_tick_volume():
    earlier = datetime(2025, 1, 1, 9, tzinfo=timezone.utc)
    later = datetime(2025, 1, 1, 10, tzinfo=timezone.utc)
    connector = FakeMT5([mt5_bar(later), mt5_bar(earlier)])
    service = CandleService(mt5_connector_factory=lambda: connector,
                            mt5_timeframes={"H1": 60}, now=lambda: NOW)
    frame = service.get_candles("EURUSD", "1h", 4)
    assert len(frame) == 2
    row = frame.iloc[-1]
    assert row.actual_symbol == "EURUSD.a"
    assert row.volume == row.tick_volume == 0
    assert row.real_volume == 0 and row.volume_type == "broker_tick_volume"
    assert row.timestamp.tzinfo is not None and pd.isna(row.completed)
    assert row.completion_status == "UNKNOWN_UNVERIFIED_TIMESTAMP"
    assert row.raw_timestamp_epoch == 1_735_714_800
    assert row.timestamp_source == "time" and row.timestamp_status == "VALID"
    assert frame.timestamp.is_monotonic_increasing
    assert connector.initialize_calls == 1
    assert connector.rate_calls == [("EURUSD", 60, 4, False)]
    service.get_candles("EURUSD", "1h", 4)
    assert connector.initialize_calls == 1  # Reuses its initialized connection.
    service.close()
    service.close()
    assert connector.close_calls == 1


def test_mt5_excludes_forming_bar_and_allows_explicit_include():
    current = datetime(2025, 1, 1, 11, 30, tzinfo=timezone.utc)
    connector = FakeMT5([mt5_bar(current)])
    service = CandleService(mt5_connector_factory=lambda: connector,
                            mt5_timeframes={"H1": 60}, now=lambda: NOW)
    default_frame = service.get_candles("EURUSD", "1h")
    assert len(default_frame) == 1
    assert pd.isna(default_frame.iloc[0].completed)
    assert default_frame.iloc[0].completion_status == "UNKNOWN_UNVERIFIED_TIMESTAMP"
    frame = service.get_candles("EURUSD", "1h", include_incomplete=True)
    assert len(frame) == 1 and pd.isna(frame.iloc[0].completed)
    assert connector.rate_calls[-1][-1] is True


def test_mt5_h1_completion_uses_utc_open_plus_one_hour():
    now = datetime(2026, 11, 12, 17, 28, tzinfo=timezone.utc)
    opens = [datetime(2026, 11, 12, hour, tzinfo=timezone.utc)
             for hour in (15, 16, 17, 18, 19)]
    connector = FakeMT5([mt5_bar(stamp) for stamp in opens])
    service = CandleService(mt5_connector_factory=lambda: connector,
                            mt5_timeframes={"H1": 60}, now=lambda: now)

    candles = service.get_candles("EURUSD", "1h", 5, include_incomplete=True)

    assert candles["completed"].isna().all()
    assert set(candles["completion_status"]) == {"UNKNOWN_UNVERIFIED_TIMESTAMP"}
    assert set(candles["timestamp_trust"]) == {"UNVERIFIED_TIME_BASIS"}
    assert candles.iloc[2]["timestamp"] == pd.Timestamp("2026-11-12T17:00:00Z")
    assert candles.iloc[3]["timestamp"] > now
    assert candles.iloc[4]["timestamp"] > now


def test_binance_epoch_timestamp_still_normalizes_to_utc():
    service, _ = service_for_binance([kline()])
    candles = service.get_candles("BTCUSDT", "1h", 5)

    assert candles.iloc[0]["timestamp"] == pd.Timestamp("2025-01-01T10:00:00Z")
    assert candles.iloc[0]["timestamp"].tzinfo is not None
    assert candles.iloc[0]["timestamp_trust"] == "TRUSTED_UTC"


def test_mt5_empty_current_bar_falls_back_to_completed_history():
    stamp = datetime(2025, 1, 1, 10, tzinfo=timezone.utc)

    class EmptyCurrentBarMT5(FakeMT5):
        def get_rates(self, instrument, timeframe, *, count, include_current):
            self.rate_calls.append((instrument, timeframe, count, include_current))
            return [] if include_current else self.rates

    connector = EmptyCurrentBarMT5([mt5_bar(stamp)])
    service = CandleService(mt5_connector_factory=lambda: connector,
                            mt5_timeframes={"H1": 60}, now=lambda: NOW)
    frame = service.get_candles("EURUSD", "1h", 5, include_incomplete=True)
    assert len(frame) == 1
    assert frame.iloc[0].timestamp == pd.Timestamp(stamp)
    assert pd.isna(frame.iloc[0].completed)
    assert frame.iloc[0].completion_status == "UNKNOWN_UNVERIFIED_TIMESTAMP"
    assert connector.rate_calls == [
        ("EURUSD", 60, 5, True),
        ("EURUSD", 60, 5, False),
    ]


def test_mt5_initialization_and_history_errors_are_reported():
    disconnected = FakeMT5(connected=False)
    service = CandleService(mt5_connector_factory=lambda: disconnected, mt5_timeframes={"H1": 60})
    with pytest.raises(CandleProviderError, match="offline"):
        service.get_candles("EURUSD", "1h")
    missing = FakeMT5([])
    missing.discover_symbols = lambda instruments: {
        symbol: SimpleNamespace(actual_symbol=None, message="not listed") for symbol in instruments}
    service = CandleService(mt5_connector_factory=lambda: missing, mt5_timeframes={"H1": 60})
    with pytest.raises(CandleProviderError, match="not listed"):
        service.get_candles("EURUSD", "1h")


def test_mt5_empty_history_preserves_exact_api_error():
    api = Mock()
    api.initialize.return_value = True
    api.terminal_info.return_value = object()
    api.symbols_get.return_value = [SimpleNamespace(name="EURUSD")]
    api.symbol_select.return_value = True
    api.copy_rates_from_pos.return_value = []
    api.last_error.return_value = (-5, "No history")
    connector = MT5Connector(api=api)
    service = CandleService(mt5_connector_factory=lambda: connector,
                            mt5_timeframes={"H1": 60}, now=lambda: NOW)
    with pytest.raises(CandleProviderError, match=r"MT5 last_error=\(-5, 'No history'\)"):
        service.get_candles("EURUSD", "1h", 5)
    api.copy_rates_from_pos.assert_called_once_with("EURUSD", 60, 1, 5)
    api.last_error.assert_called_once_with()
    service.close()


def test_invalid_timestamps_timeframe_and_count_are_rejected():
    service, _ = service_for_binance([kline()])
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        service.get_candles("BTCUSDT", "10m")
    with pytest.raises(ValueError, match="positive integer"):
        service.get_candles("BTCUSDT", "1m", 0)
    with pytest.raises(ValueError, match="Unknown instrument"):
        service.get_candles("DOGEUSDT")
