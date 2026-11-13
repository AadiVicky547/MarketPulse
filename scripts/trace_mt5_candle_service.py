"""Trace the MT5 history calls made by CandleService for XAUUSD.

Run manually from the project root in the same Windows environment as the
connected MT5 terminal. This script uses the existing MT5Connector lifecycle,
requests at most five H1 bars per call, and does not launch a terminal or
perform any account operation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import inspect
from pathlib import Path
import sys
import traceback
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MetaTrader5 as mt5

import data.candle_service as candle_service_module
import data.mt5_connector as mt5_connector_module
from data.candle_service import CandleService
from data.mt5_connector import MT5Connector


def _timeframe_name(api: Any, timeframe: Any) -> str:
    """Find a readable MetaTrader5 timeframe constant for its numeric value."""
    for code in ("M1", "M2", "M3", "M4", "M5", "M6", "M10", "M12", "M15",
                 "M20", "M30", "H1", "H2", "H3", "H4", "H6", "H8", "H12",
                 "D1", "W1", "MN1"):
        name = f"TIMEFRAME_{code}"
        if getattr(api, name, object()) == timeframe:
            return name
    return f"unknown timeframe constant {timeframe!r}"


def _bar_timestamp(bar: Any) -> str:
    """Format an MT5 rates row's epoch-seconds ``time`` field as UTC."""
    try:
        value = bar["time"]
        return datetime.fromtimestamp(int(value), timezone.utc).isoformat()
    except Exception:
        return f"<timestamp unavailable: {bar!r}>"


class TracingMT5API:
    """Transparent proxy that logs actual rates calls and delegates all APIs."""

    def __init__(self, api: Any) -> None:
        self._api = api
        self.rate_calls: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._api, name)

    def initialize(self, **kwargs: Any) -> Any:
        print(f"\n[TRACE] MT5Connector.initialize({kwargs!r})", flush=True)
        result = self._api.initialize(**kwargs)
        print(f"[TRACE] initialize result: {result!r}", flush=True)
        if result:
            terminal = self._api.terminal_info()
            path = getattr(terminal, "path", None) if terminal is not None else None
            connected = getattr(terminal, "connected", None) if terminal is not None else None
            print(f"[TRACE] terminal path: {path!r}", flush=True)
            print(f"[TRACE] terminal connected: {connected!r}", flush=True)
        else:
            print(f"[TRACE] MT5 last_error after initialize: {self._api.last_error()!r}", flush=True)
        return result

    def copy_rates_from_pos(self, symbol: str, timeframe: int,
                            start_pos: int, count: int) -> Any:
        readable = _timeframe_name(self._api, timeframe)
        record: dict[str, Any] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timeframe_name": readable,
            "start_pos": start_pos,
            "count": count,
        }
        print("\n[TRACE] Actual MT5 API call:", flush=True)
        print(f"  copy_rates_from_pos(symbol={symbol!r}, timeframe={timeframe!r} ({readable}), "
              f"start_pos={start_pos}, count={count})", flush=True)
        try:
            result = self._api.copy_rates_from_pos(symbol, timeframe, start_pos, count)
        except Exception:
            record.update({"raised": True, "is_none": None, "count_returned": None})
            self.rate_calls.append(record)
            print("  API raised; full traceback follows:", flush=True)
            traceback.print_exc()
            try:
                record["last_error"] = self._api.last_error()
                print(f"  MT5 last_error immediately after exception: {record['last_error']!r}", flush=True)
            except Exception:
                traceback.print_exc()
            raise

        try:
            error = self._api.last_error()
        except Exception:
            print("  last_error() raised; full traceback follows:", flush=True)
            traceback.print_exc()
            error = None
        is_none = result is None
        try:
            returned_count = None if is_none else len(result)
        except Exception:
            print("  Could not get result length; full traceback follows:", flush=True)
            traceback.print_exc()
            returned_count = None
        record.update({"raised": False, "is_none": is_none,
                       "count_returned": returned_count, "last_error": error})
        self.rate_calls.append(record)
        print(f"  returned None: {is_none!r}", flush=True)
        print(f"  raw return type: {type(result).__module__}.{type(result).__qualname__}", flush=True)
        print(f"  returned bars: {returned_count!r}", flush=True)
        print(f"  MT5 last_error immediately after call: {error!r}", flush=True)
        if returned_count:
            try:
                print(f"  first raw record: {result[0]!r}", flush=True)
                print(f"  last raw record:  {result[-1]!r}", flush=True)
                print(f"  first candle timestamp (UTC): {_bar_timestamp(result[0])}", flush=True)
                print(f"  last candle timestamp (UTC):  {_bar_timestamp(result[-1])}", flush=True)
            except Exception:
                print("  Could not inspect first/last candle; full traceback follows:", flush=True)
                traceback.print_exc()
        return result


def main() -> int:
    print("Imported module paths:", flush=True)
    print(f"  Candle Service: {inspect.getfile(CandleService)}", flush=True)
    print(f"  MT5 connector: {inspect.getfile(MT5Connector)}", flush=True)
    print(f"  MetaTrader5 API: {getattr(mt5, '__file__', '<unknown>')}", flush=True)
    if not hasattr(mt5, "TIMEFRAME_H1"):
        print("Installed MetaTrader5 package has no TIMEFRAME_H1 constant.", flush=True)
        return 2

    tracing_api = TracingMT5API(mt5)
    connector = MT5Connector(instruments=("XAUUSD",), api=tracing_api)
    resolved_symbols: dict[str, str | None] = {}
    connector_rate_results: list[dict[str, Any]] = []

    # Trace the complete connector return boundary without editing production
    # code. The service still calls the very same MT5Connector.get_rates method.
    original_get_rates = connector.get_rates

    def traced_get_rates(instrument: str, timeframe: int, count: int = 100,
                         include_current: bool = False):
        print(f"\n[TRACE] MT5Connector.get_rates entered: instrument={instrument!r}, "
              f"timeframe={timeframe!r} ({_timeframe_name(mt5, timeframe)}), "
              f"count={count}, include_current={include_current}", flush=True)
        try:
            result = original_get_rates(instrument, timeframe, count=count,
                                        include_current=include_current)
        except Exception as exc:
            connector_rate_results.append({"raised": True, "exception": exc})
            print(f"[TRACE] MT5Connector.get_rates raised {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            raise
        try:
            result_length = None if result is None else len(result)
        except Exception:
            result_length = None
            print("[TRACE] Could not determine connector result length; full traceback follows:", flush=True)
            traceback.print_exc()
        connector_rate_results.append({"raised": False, "value": result,
                                       "length": result_length})
        print(f"[TRACE] MT5Connector.get_rates return type: "
              f"{type(result).__module__}.{type(result).__qualname__}", flush=True)
        print(f"[TRACE] MT5Connector.get_rates returned None: {result is None}", flush=True)
        print(f"[TRACE] MT5Connector.get_rates length: {result_length!r}", flush=True)
        print(f"[TRACE] MT5Connector.get_rates returns list/ndarray/DataFrame: "
              f"{isinstance(result, list)}/{type(result).__module__ == 'numpy'}/"
              f"{type(result).__name__ == 'DataFrame'}", flush=True)
        if result_length:
            try:
                print(f"[TRACE] MT5Connector first record: {result[0]!r}", flush=True)
                print(f"[TRACE] MT5Connector last record:  {result[-1]!r}", flush=True)
            except Exception:
                print("[TRACE] Could not inspect connector records; full traceback follows:", flush=True)
                traceback.print_exc()
        print("[TRACE] MT5Connector.get_rates exception occurred: False", flush=True)
        return result

    connector.get_rates = traced_get_rates

    # Observe the exact helper used by get_rates to read each MT5 NumPy record.
    # This can expose field-access incompatibilities while leaving the connector
    # source and its behavior unchanged.
    original_field = mt5_connector_module._field

    def traced_field(record: Any, name: str, default: Any = None) -> Any:
        value = original_field(record, name, default)
        if name == "time":
            try:
                indexed_value = record[name]
            except Exception:
                indexed_value = "<record does not support indexed field access>"
            print(f"[TRACE] connector _field(time): record_type="
                  f"{type(record).__module__}.{type(record).__qualname__}, "
                  f"result={value!r}, indexed_record[name]={indexed_value!r}", flush=True)
        return value

    mt5_connector_module._field = traced_field

    # Interpose only on the Candle Service module's pandas reference so the
    # diagnostic can observe the DataFrame construction boundary.
    pandas_module = candle_service_module.pd
    original_sort_values = pandas_module.DataFrame.sort_values

    def traced_sort_values(frame: Any, *args: Any, **kwargs: Any) -> Any:
        print(f"[TRACE] Rows before Candle Service sort_values: {len(frame)}", flush=True)
        result = original_sort_values(frame, *args, **kwargs)
        print(f"[TRACE] Rows after Candle Service sort_values: {len(result)}", flush=True)
        return result

    pandas_module.DataFrame.sort_values = traced_sort_values

    class TracingPandas:
        def __getattr__(self, name: str) -> Any:
            return getattr(pandas_module, name)

        def DataFrame(self, data: Any = None, *args: Any, **kwargs: Any) -> Any:
            try:
                input_count = len(data)
            except Exception:
                input_count = None
            print(f"[TRACE] Rows before DataFrame construction: {input_count!r}", flush=True)
            frame = pandas_module.DataFrame(data, *args, **kwargs)
            print(f"[TRACE] Rows after DataFrame construction: {len(frame)}", flush=True)
            return frame

    candle_service_module.pd = TracingPandas()
    original_discover = connector.discover_symbols

    def traced_discover(instruments=None):
        matches = original_discover(instruments)
        for requested, match in matches.items():
            resolved_symbols[requested] = match.actual_symbol
            print(f"[TRACE] symbol discovery: requested={requested}, "
                  f"actual={match.actual_symbol!r}, matched={match.matched}, "
                  f"match_type={match.match_type!r}, message={match.message}", flush=True)
        return matches

    connector.discover_symbols = traced_discover
    service = CandleService(
        mt5_connector_factory=lambda: connector,
        mt5_timeframes={"H1": int(mt5.TIMEFRAME_H1)},
    )

    original_rows = service._mt5_rows
    stage_counts: dict[str, int | None] = {}

    def traced_rows(instrument: str, timeframe: int, count: int,
                    include_incomplete: bool):
        print(f"\n[TRACE] CandleService._mt5_rows entered: instrument={instrument}, "
              f"timeframe={timeframe} ({_timeframe_name(mt5, timeframe)}), "
              f"count={count}, include_incomplete={include_incomplete}", flush=True)
        call_start = len(tracing_api.rate_calls)
        rate_result_start = len(connector_rate_results)

        def report_fallback_attempted() -> None:
            service_calls = tracing_api.rate_calls[call_start:]
            fallback = (include_incomplete and len(service_calls) >= 2
                        and service_calls[0]["start_pos"] == 0
                        and any(call["start_pos"] == 1 for call in service_calls[1:]))
            print(f"[TRACE] Position-1 fallback attempted: {'YES' if fallback else 'NO'}", flush=True)
            print(f"[TRACE] Position values used by this service request: "
                  f"{[call['start_pos'] for call in service_calls]}", flush=True)

        try:
            result = original_rows(instrument, timeframe, count, include_incomplete)
        except Exception:
            print("[TRACE] CandleService._mt5_rows raised; full traceback follows:", flush=True)
            traceback.print_exc()
            for result in connector_rate_results[rate_result_start:]:
                if result["raised"]:
                    print("[TRACE] Service-side empty condition was not evaluated: "
                          "connector.get_rates raised first.", flush=True)
                    continue
                value = result["value"]
                try:
                    length = None if value is None else len(value)
                except Exception:
                    length = "<len raised>"
                    traceback.print_exc()
                empty_condition = value is None or (isinstance(length, int) and length == 0)
                print(f"[TRACE] Exact service empty check: rates is None ({value is None}) "
                      f"OR len(rates) == 0 ({length!r} == 0); evaluates to {empty_condition}",
                      flush=True)
                print(f"[TRACE] Service received object: {type(value).__module__}."
                      f"{type(value).__qualname__}; safe contents={value!r}", flush=True)
            report_fallback_attempted()
            raise
        stage_counts["connector_rows"] = len(result)
        print(f"[TRACE] Connector-normalized raw row count before candle DataFrame: "
              f"{len(result)}", flush=True)
        service_results = connector_rate_results[rate_result_start:]
        if service_results and not service_results[-1]["raised"]:
            last_value = service_results[-1]["value"]
            last_length = service_results[-1]["length"]
            empty_condition = last_value is None or last_length == 0
            print(f"[TRACE] Exact final service empty check: rates is None ({last_value is None}) "
                  f"OR len(rates) == 0 ({last_length!r} == 0); evaluates to {empty_condition}",
                  flush=True)
            print(f"[TRACE] Final rates object sample before normalization: {last_value!r}", flush=True)
        report_fallback_attempted()
        return result

    service._mt5_rows = traced_rows
    original_normalize = service._normalize_mt5

    def traced_normalize(instrument: str, timeframe: str, raw: list[dict[str, Any]]):
        print(f"\n[TRACE] Before MT5 candle normalization: {len(raw)} rows", flush=True)
        result = original_normalize(instrument, timeframe, raw)
        stage_counts["normalized_rows"] = len(result)
        print(f"[TRACE] After MT5 candle normalization: {len(result)} rows", flush=True)
        return result

    service._normalize_mt5 = traced_normalize

    result_code = 0
    try:
        print("\n[TRACE] Candle Service reproduction request (XAUUSD, 1h, count=5, "
              "include_incomplete=True; matches the live verification call mode).", flush=True)
        before_service = len(tracing_api.rate_calls)
        try:
            frame = service.get_candles("XAUUSD", "1h", 5, include_incomplete=True)
            print(f"[TRACE] CandleService returned DataFrame: {len(frame)} rows", flush=True)
            if not frame.empty:
                print(f"[TRACE] Completed rows: {int(frame['completed'].sum())}; "
                      f"incomplete rows: {int((~frame['completed']).sum())}", flush=True)
                print(f"[TRACE] Final timestamps: {frame['timestamp'].iloc[0].isoformat()} .. "
                      f"{frame['timestamp'].iloc[-1].isoformat()}", flush=True)
                print("[TRACE] include_incomplete=True, so the completed-candle filter was not applied.",
                      flush=True)
            else:
                print("[TRACE] Rows disappeared before the service returned its DataFrame.", flush=True)
        except Exception:
            result_code = 1
            print("[TRACE] CandleService request raised; full traceback follows:", flush=True)
            traceback.print_exc()

        print(f"\n[TRACE] MT5 rates calls made by Candle Service: "
              f"{len(tracing_api.rate_calls) - before_service}", flush=True)
        print(f"[TRACE] Stage counts: {stage_counts!r}", flush=True)
        if not stage_counts.get("normalized_rows"):
            print("[TRACE] DataFrame construction and service sorting/filtering were not reached; "
                  "the service stopped in _mt5_rows.", flush=True)

        actual_symbol = resolved_symbols.get("XAUUSD")
        if connector.connected and actual_symbol:
            print("\n[TRACE] Direct comparison on the same initialized connector and terminal:", flush=True)
            print("[TRACE] Requesting five completed H1 bars with start_pos=1.", flush=True)
            try:
                direct = tracing_api.copy_rates_from_pos(
                    actual_symbol, int(mt5.TIMEFRAME_H1), 1, 5)
                if direct is None:
                    print("[TRACE] Direct position-1 result: None", flush=True)
                else:
                    try:
                        direct_count = len(direct)
                    except Exception:
                        traceback.print_exc()
                        direct_count = None
                    print(f"[TRACE] Direct position-1 result count: {direct_count!r}", flush=True)
            except Exception:
                result_code = 1
                print("[TRACE] Direct position-1 request raised; full traceback follows:", flush=True)
                traceback.print_exc()
        else:
            print("\n[TRACE] Direct comparison skipped: connector did not remain connected "
                  "or XAUUSD actual symbol was not discovered.", flush=True)
    except Exception:
        result_code = 1
        print("[TRACE] Unexpected tracing failure; full traceback follows:", flush=True)
        traceback.print_exc()
    finally:
        service.close()
        pandas_module.DataFrame.sort_values = original_sort_values
        mt5_connector_module._field = original_field
        candle_service_module.pd = pandas_module
        print("\n[TRACE] Candle Service connector closed.", flush=True)

    return result_code


if __name__ == "__main__":
    raise SystemExit(main())
