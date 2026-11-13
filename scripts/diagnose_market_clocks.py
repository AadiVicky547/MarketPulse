"""Compare Python UTC, Binance public server time, and MT5 tick event clocks.

This is a bounded diagnostic only. It reports clock differences but never
adjusts timestamps. Binance and MT5 observations occur sequentially, so they
are not perfectly simultaneous.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import requests

# Match the project's direct-run script convention so root modules resolve
# when Python sets sys.path[0] to the scripts directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import INSTRUMENTS
from data.mt5_connector import MT5Connector

BINANCE_TIME_URL = "https://api.binance.com/api/v3/time"
HTTP_TIMEOUT_SECONDS = (3.0, 5.0)
STALE_AFTER_SECONDS = 300.0
CLASSIFICATION_TOLERANCE_SECONDS = 10.0


def binance_milliseconds_to_utc(server_time_ms: int) -> datetime:
    """Convert Binance's integer epoch milliseconds into aware UTC."""
    if isinstance(server_time_ms, bool) or not isinstance(server_time_ms, int) or server_time_ms < 0:
        raise ValueError("serverTime must be a non-negative integer in milliseconds")
    return datetime.fromtimestamp(server_time_ms / 1000, timezone.utc)


def clock_differences(
    python_utc: datetime,
    binance_utc: datetime | None,
    mt5_tick_utc: datetime | None,
) -> dict[str, float | None]:
    """Return signed clock offsets in seconds; positive means the first clock is ahead."""
    for value in (python_utc, binance_utc, mt5_tick_utc):
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("clock values must be timezone-aware")
    return {
        "binance_minus_python_seconds": (binance_utc - python_utc).total_seconds() if binance_utc else None,
        "mt5_minus_python_seconds": (mt5_tick_utc - python_utc).total_seconds() if mt5_tick_utc else None,
        "mt5_minus_binance_seconds": (mt5_tick_utc - binance_utc).total_seconds()
        if mt5_tick_utc and binance_utc else None,
    }


def extract_mt5_tick_times(tick: Any) -> dict[str, Any] | None:
    """Extract raw MT5 epoch fields and convert each independently to UTC."""
    if tick is None:
        return None

    def field(name: str) -> Any:
        if isinstance(tick, Mapping):
            return tick.get(name)
        try:
            return tick[name]
        except (IndexError, KeyError, TypeError):
            return getattr(tick, name, None)

    raw_time = field("time")
    raw_time_msc = field("time_msc")
    seconds_utc = _epoch_utc(raw_time, 1)
    milliseconds_utc = _epoch_utc(raw_time_msc, 1000)
    return {
        "raw_time": raw_time,
        "raw_time_msc": raw_time_msc,
        "time_utc": seconds_utc,
        "time_msc_utc": milliseconds_utc,
        "event_utc": milliseconds_utc or seconds_utc,
    }


def _epoch_utc(value: Any, divisor: int) -> datetime | None:
    try:
        number = int(value)
        if number <= 0:
            return None
        return datetime.fromtimestamp(number / divisor, timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def fetch_binance_server_time(
    get: Callable[..., Any] = requests.get,
    timeout: tuple[float, float] = HTTP_TIMEOUT_SECONDS,
) -> tuple[datetime | None, str | None]:
    """Fetch public Binance server time, returning errors instead of raising."""
    try:
        response = get(BINANCE_TIME_URL, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping) or "serverTime" not in payload:
            raise ValueError("Binance response is missing serverTime")
        return binance_milliseconds_to_utc(payload["serverTime"]), None
    except requests.RequestException as exc:
        return None, f"Binance HTTP/network error: {exc}"
    except (ValueError, TypeError, OverflowError) as exc:
        return None, f"Invalid Binance time response: {exc}"


def classify_mt5_offsets(
    offsets_seconds: Sequence[float], tolerance_seconds: float = CLASSIFICATION_TOLERANCE_SECONDS,
) -> str:
    """Classify observed MT5-minus-Binance offsets without correcting them."""
    values = [float(value) for value in offsets_seconds if value is not None]
    if not values:
        return "UNAVAILABLE"
    if all(abs(value - 7200.0) <= tolerance_seconds for value in values):
        return "STABLE_NEAR_PLUS_2_HOURS"
    if max(values) - min(values) <= tolerance_seconds:
        return "OTHER_STABLE_OFFSET"
    return "VARIABLE_OFFSET"


def _fmt_time(value: datetime | None) -> str:
    return value.isoformat(timespec="milliseconds") if value else "unavailable"


def _fmt_delta(value: float | None) -> str:
    return f"{value:+.3f}s" if value is not None else "unavailable"


def mask_account_login(login: Any) -> str:
    """Mask an MT5 account login, retaining at most its last two characters."""
    if login is None:
        return "unavailable"
    value = str(login)
    if not value:
        return "unavailable"
    visible = value[-2:] if len(value) > 2 else ""
    return "*" * (len(value) - len(visible)) + visible


def account_identity_display(account_info: Any) -> tuple[str, str] | None:
    """Return only the non-secret server name and masked login from account_info."""
    if account_info is None:
        return None
    if isinstance(account_info, Mapping):
        server, login = account_info.get("server"), account_info.get("login")
    else:
        server = getattr(account_info, "server", None)
        login = getattr(account_info, "login", None)
    return str(server) if server else "unavailable", mask_account_login(login)


def _mt5_instruments() -> tuple[str, ...]:
    return tuple(item.symbol for item in INSTRUMENTS if item.source.upper() == "MT5")


def run_diagnostic(samples: int = 3, interval_seconds: float = 5.0) -> None:
    """Run bounded sequential samples using the existing MT5 connector."""
    if samples < 3:
        raise ValueError("At least three samples are required")
    if interval_seconds < 0:
        raise ValueError("Sample interval cannot be negative")

    try:
        import MetaTrader5 as mt5
    except ImportError:
        mt5 = None

    connector = MT5Connector(instruments=_mt5_instruments(), api=mt5,
                             stale_after_seconds=STALE_AFTER_SECONDS)
    offset_observations: list[float] = []
    matches: dict[str, Any] = {}
    try:
        status = connector.initialize()
        print(f"MT5 connection: {'CONNECTED' if status.connected else 'UNAVAILABLE'} — {status.message}"
              + (f"; {status.error}" if status.error else ""))
        if status.connected:
            try:
                terminal = connector._api.terminal_info()
                terminal_path = getattr(terminal, "path", None) if terminal is not None else None
                print(f"MT5 terminal path: {terminal_path or 'not reported by terminal_info'}")
                terminal_connected = getattr(terminal, "connected", None) if terminal is not None else None
                terminal_state = ("CONNECTED" if terminal_connected is True else
                                  "DISCONNECTED" if terminal_connected is False else "UNAVAILABLE")
                print(f"MT5 terminal connection status: {terminal_state}")
            except Exception as exc:
                print(f"MT5 terminal path unavailable: {type(exc).__name__}: {exc}")
                print("MT5 terminal connection status: UNAVAILABLE")
            try:
                account_info = connector._api.account_info()
                identity = account_identity_display(account_info)
                if identity is None:
                    print("MT5 account information: unavailable (account_info() returned None)")
                else:
                    server, masked_login = identity
                    print(f"MT5 account server: {server}")
                    print(f"MT5 account login (masked): {masked_login}")
            except Exception as exc:
                print(f"MT5 account information: unavailable ({type(exc).__name__})")
            try:
                matches = connector.discover_symbols(_mt5_instruments())
            except Exception as exc:
                print(f"MT5 symbol discovery error: {exc}")

        print("Clock samples are sequential and are not perfectly simultaneous.")
        for sample_number in range(1, samples + 1):
            if sample_number > 1:
                time.sleep(interval_seconds)
            python_before = datetime.now(timezone.utc)
            binance_utc, binance_error = fetch_binance_server_time()
            instrument_rows: list[dict[str, Any]] = []
            for instrument in _mt5_instruments():
                match = matches.get(instrument)
                actual_symbol = match.actual_symbol if match else None
                if not connector.connected or not actual_symbol:
                    reason = match.message if match and not actual_symbol else connector.status.message
                    instrument_rows.append({"instrument": instrument, "actual": actual_symbol or "unavailable",
                                            "status": "UNAVAILABLE", "error": reason})
                    continue
                try:
                    normalized = connector.get_tick(instrument)
                    event_utc = normalized.get("timestamp")
                    received_at = normalized.get("received_at")
                    python_mid = datetime.now(timezone.utc)
                    diffs = clock_differences(python_mid, binance_utc, event_utc)
                    event_receive_delta = ((event_utc - received_at).total_seconds()
                                           if event_utc is not None and received_at is not None else None)
                    receive_age = ((python_mid - received_at).total_seconds()
                                   if received_at is not None else None)
                    event_age = ((python_mid - event_utc).total_seconds()
                                 if event_utc is not None else None)
                    if diffs["mt5_minus_binance_seconds"] is not None:
                        offset_observations.append(diffs["mt5_minus_binance_seconds"])
                    receive_state = ("UNAVAILABLE" if received_at is None else
                                     "STALE" if receive_age is not None and receive_age > STALE_AFTER_SECONDS
                                     else "FRESH")
                    instrument_rows.append({
                        "instrument": instrument, "actual": actual_symbol,
                        "raw_time": normalized.get("raw_time"),
                        "raw_time_msc": normalized.get("raw_time_msc"),
                        "selected": normalized.get("timestamp_source"),
                        "integrity": normalized.get("timestamp_status"),
                        "trust": normalized.get("timestamp_trust"),
                        "event": event_utc, "received": received_at,
                        "event_receive": event_receive_delta,
                        "mt5_python": diffs["mt5_minus_python_seconds"],
                        "mt5_binance": diffs["mt5_minus_binance_seconds"],
                        "event_age": event_age, "receive_age": receive_age,
                        "receive_state": receive_state,
                        "issues": normalized.get("timestamp_warning") or normalized.get("error") or "none",
                    })
                except Exception as exc:
                    instrument_rows.append({"instrument": instrument, "actual": actual_symbol,
                                            "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
            python_after = datetime.now(timezone.utc)
            python_mid = python_before + (python_after - python_before) / 2
            binance_diff = (binance_utc - python_mid).total_seconds() if binance_utc else None
            print(f"\nSAMPLE {sample_number}/{samples} | Python before={_fmt_time(python_before)}"
                  f" | after={_fmt_time(python_after)} | Binance={_fmt_time(binance_utc)}"
                  f" | Binance−Python(mid)={_fmt_delta(binance_diff)}")
            if binance_error:
                print(f"Binance error: {binance_error}")
            for row in instrument_rows:
                if "status" in row:
                    print(f"{row['instrument']} actual={row['actual']} status={row['status']} "
                          f"detail={row['error']}")
                    continue
                print(f"{row['instrument']} actual={row['actual']} | raw time={row['raw_time']!r} "
                      f"| raw time_msc={row['raw_time_msc']!r} | selected={row['selected']} "
                      f"| field_status={row['integrity']} | time_trust={row['trust']}")
                print(f"  event timestamp (connector): {_fmt_time(row['event'])} "
                      f"[time basis UNVERIFIED] | receive timestamp (local UTC): {_fmt_time(row['received'])}")
                print(f"  event−receive={_fmt_delta(row['event_receive'])} "
                      f"| MT5 event−Python={_fmt_delta(row['mt5_python'])} "
                      f"| MT5 event−Binance={_fmt_delta(row['mt5_binance'])}")
                print(f"  event age (unverified basis)={_fmt_delta(row['event_age'])} "
                      f"| receive age={_fmt_delta(row['receive_age'])} "
                      f"| arrival={row['receive_state']} | issues={row['issues']}")

        classification = classify_mt5_offsets(offset_observations)
        print("\nFINAL MT5−BINANCE OFFSET SUMMARY")
        print(f"Classification: {classification}")
        if offset_observations:
            print(f"Raw epoch comparisons interpreted as UTC (MT5 basis unverified): "
                  f"{len(offset_observations)}; median={statistics.median(offset_observations):+.3f}s; "
                  f"range={min(offset_observations):+.3f}s to {max(offset_observations):+.3f}s")
        else:
            print("No valid MT5 tick and Binance time pairs were available.")
        print("No timestamp correction has been applied.")
    finally:
        try:
            connector.close()
        except Exception as exc:
            print(f"MT5 shutdown error: {type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3, help="number of samples (minimum 3; default 3)")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between samples (default 5)")
    args = parser.parse_args()
    try:
        run_diagnostic(args.samples, args.interval)
    except Exception as exc:
        import traceback
        print(f"Diagnostic failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
