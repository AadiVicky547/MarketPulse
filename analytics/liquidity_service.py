"""Liquidity and order-flow summaries over immutable market-manager snapshots.

This module performs analysis only. It never owns or calls a data connector.
MT5 tick volume remains broker-observed activity, while Binance quantities are
exchange-reported executed base-asset quantities. Displayed book liquidity is
cancellable and is not guaranteed executable.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import threading
import time
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from config import INSTRUMENTS

AVAILABLE = "AVAILABLE"
WARMING_UP = "WARMING_UP"
UNAVAILABLE = "UNAVAILABLE"
STALE = "STALE"
DISCONNECTED = "DISCONNECTED"


@dataclass(frozen=True)
class LiquiditySettings:
    """Bounded windows for baselines and depth comparisons."""

    baseline_window: int = 20
    minimum_baseline_samples: int = 5
    book_levels: int = 20
    book_history_size: int = 100
    book_freshness_seconds: float = 3.0

    def __post_init__(self) -> None:
        if self.baseline_window <= 0 or self.minimum_baseline_samples < 2:
            raise ValueError("baseline_window must be positive and minimum_baseline_samples at least two")
        if self.minimum_baseline_samples > self.baseline_window:
            raise ValueError("minimum_baseline_samples cannot exceed baseline_window")
        if (self.book_levels <= 0 or self.book_history_size <= 0 or
                self.book_freshness_seconds <= 0):
            raise ValueError("book levels and book history size must be positive")


@dataclass(frozen=True)
class LiquidityMetric:
    """One measured value with status, method and sample provenance."""

    status: str
    value: float | None
    unit: str
    methodology: str
    sample_count: int
    warning: str | None = None


@dataclass(frozen=True)
class InstrumentLiquidityResult:
    """Liquidity and flow metrics for one configured instrument."""

    instrument: str
    source: str
    status: str
    timestamp: Any
    received_at: Any
    timestamp_trust: str | None
    freshness: str
    connection_state: str
    order_book_sync_state: str | None
    volume_semantics: str | None
    metrics: Mapping[str, LiquidityMetric] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _number(value: Any, *, positive: bool = False, nonnegative: bool = False) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result):
        return None
    if positive and result <= 0:
        return None
    if nonnegative and result < 0:
        return None
    return result


def _metric(status: str, value: float | None, unit: str, method: str,
            sample_count: int, warning: str | None = None) -> LiquidityMetric:
    return LiquidityMetric(status, value, unit, method, sample_count, warning)


def _unavailable_metrics(warning: str, status: str = UNAVAILABLE) -> dict[str, LiquidityMetric]:
    names = {
        "spread_price": "price units", "spread_bps": "basis points",
        "spread_expansion_ratio": "ratio", "spread_anomaly_zscore": "standard deviations",
        "tick_volume": "broker tick-volume units", "tick_volume_activity_ratio": "ratio",
        "tick_volume_anomaly_zscore": "standard deviations", "tick_volume_change": "broker tick-volume units",
        "price_move_per_tick_volume": "price units per broker tick-volume unit",
        "bid_depth": "base-asset units", "ask_depth": "base-asset units",
        "order_book_imbalance": "ratio", "buy_aggressor_volume": "base-asset units",
        "sell_aggressor_volume": "base-asset units", "buy_sell_volume_imbalance": "ratio",
        "trade_count": "trades", "trade_activity": "base-asset units per sample",
        "bid_top_level_concentration": "fraction of bid depth",
        "ask_top_level_concentration": "fraction of ask depth",
        "bid_depth_change": "base-asset units", "ask_depth_change": "base-asset units",
        "bid_depth_change_pct": "percent", "ask_depth_change_pct": "percent",
        "bid_depth_anomaly_zscore": "standard deviations",
        "ask_depth_anomaly_zscore": "standard deviations",
    }
    return {name: _metric(status, None, unit, "Not calculated", 0, warning)
            for name, unit in names.items()}


class LiquidityService:
    """Compute MT5 activity and Binance book/trade metrics from a manager.

    ``manager`` must implement the established ``get_market_snapshot`` and
    ``get_history(symbol)`` methods. Trade history comes from the manager's
    immutable bounded Binance snapshot buffers. Depth comparison history is
    held in bounded in-process deques owned by this analytics instance.
    """

    def __init__(self, manager: Any, *, settings: LiquiditySettings | None = None) -> None:
        self.manager = manager
        self.settings = settings or LiquiditySettings()
        self._lock = threading.RLock()
        self._book_history: dict[str, deque[tuple[Any, float, float, float]]] = {
            item.symbol.upper(): deque(maxlen=self.settings.book_history_size)
            for item in INSTRUMENTS if item.source == "Binance"
        }
        self._book_keys: dict[str, Any] = {}

    def analyze(self) -> Mapping[str, InstrumentLiquidityResult]:
        """Return a result for every configured instrument without blocking ingestion."""
        try:
            snapshot = self.manager.get_market_snapshot()
        except Exception as exc:
            return MappingProxyType({
                item.symbol.upper(): self._failed(item.symbol.upper(), item.source,
                    f"Market snapshot unavailable: {type(exc).__name__}: {exc}")
                for item in INSTRUMENTS
            })

        results: dict[str, InstrumentLiquidityResult] = {}
        for item in INSTRUMENTS:
            symbol, source = item.symbol.upper(), item.source
            state = None
            try:
                state = _field(snapshot.instruments, symbol)
                if state is None:
                    results[symbol] = self._failed(symbol, source, "Instrument state missing from snapshot")
                    continue
                history = self.manager.get_history(symbol) if source == "MT5" else ()
                if source == "MT5":
                    results[symbol] = self._analyze_mt5(symbol, state, history)
                else:
                    results[symbol] = self._analyze_binance(symbol, snapshot, state)
            except Exception as exc:
                results[symbol] = self._failed(symbol, source,
                    f"Liquidity analysis failed: {type(exc).__name__}: {exc}", state=state)
        return MappingProxyType(results)

    def _analyze_mt5(self, symbol: str, state: Any, history: Any) -> InstrumentLiquidityResult:
        freshness = str(_field(state, "freshness", "UNAVAILABLE")).upper()
        connection = str(_field(state, "connection_state", "DISCONNECTED")).upper()
        volume_semantics = _field(state, "volume_semantics") or (
            "Broker-observed tick volume; not centralized traded volume")
        warnings: list[str] = ["MT5 tick volume is broker-observed activity, not centralized traded volume."]
        if connection == DISCONNECTED:
            return self._failed(symbol, "MT5", "MT5 source is disconnected", state=state,
                                status=DISCONNECTED, freshness=freshness)
        if freshness == STALE:
            return self._failed(symbol, "MT5", "Latest MT5 observation is stale", state=state,
                                status=STALE, freshness=freshness)
        if freshness != "LIVE":
            return self._failed(symbol, "MT5", "No live MT5 observation is available", state=state,
                                status=UNAVAILABLE, freshness=freshness)

        current_spread = _number(_field(state, "spread"), nonnegative=True)
        bid, ask = _number(_field(state, "bid"), positive=True), _number(_field(state, "ask"), positive=True)
        mid = _number(_field(state, "mid_price"), positive=True)
        if current_spread is None and bid is not None and ask is not None and ask >= bid:
            current_spread = ask - bid
        if mid is None and bid is not None and ask is not None and ask >= bid:
            mid = (bid + ask) / 2
        spread_bps = (current_spread / mid * 10000) if current_spread is not None and mid else None

        observations = list(history or ())
        current_key = self._mt5_key(state)
        if not observations or self._mt5_key(observations[-1]) != current_key:
            observations.append(state)
        observations = observations[-self.settings.baseline_window - 1:]
        prior_spreads = [value for item in observations[:-1]
                         if (value := _number(_field(item, "spread"), nonnegative=True)) is not None]
        prior_volumes = [value for item in observations[:-1]
                         if (value := _number(_field(item, "tick_volume"), nonnegative=True)) is not None]
        metrics: dict[str, LiquidityMetric] = {
            "spread_price": _metric(AVAILABLE, current_spread, "price units",
                "Current ask minus bid from broker quote", 1) if current_spread is not None else
                _metric(UNAVAILABLE, None, "price units", "Current ask minus bid from broker quote", 0,
                        "Valid bid/ask spread unavailable"),
            "spread_bps": _metric(AVAILABLE, spread_bps, "basis points",
                "(ask - bid) / midpoint * 10,000", 1) if spread_bps is not None else
                _metric(UNAVAILABLE, None, "basis points", "(ask - bid) / midpoint * 10,000", 0,
                        "Valid positive midpoint unavailable"),
        }
        self._add_relative_metrics(metrics, "spread", current_spread, prior_spreads,
                                   "price units", "spread_expansion_ratio", "spread_anomaly_zscore")
        tick_volume = _number(_field(state, "tick_volume"), nonnegative=True)
        if tick_volume is None:
            warnings.append("MT5 tick volume is missing or invalid.")
            for name, unit in (("tick_volume", "broker tick-volume units"),
                               ("tick_volume_activity_ratio", "ratio"),
                               ("tick_volume_anomaly_zscore", "standard deviations"),
                               ("tick_volume_change", "broker tick-volume units"),
                               ("price_move_per_tick_volume", "price units per broker tick-volume unit")):
                metrics[name] = _metric(WARMING_UP, None, unit, "Broker tick-volume observations", 0,
                                        "No valid tick-volume observation")
        else:
            metrics["tick_volume"] = _metric(AVAILABLE, tick_volume, "broker tick-volume units",
                "Latest MT5 tick volume; broker-observed activity", 1)
            self._add_relative_metrics(metrics, "tick_volume", tick_volume, prior_volumes,
                                       "broker tick-volume units", "tick_volume_activity_ratio",
                                       "tick_volume_anomaly_zscore")
            previous_volume = prior_volumes[-1] if prior_volumes else None
            metrics["tick_volume_change"] = (_metric(AVAILABLE, tick_volume - previous_volume,
                "broker tick-volume units", "Current tick volume minus previous valid observation", 2)
                if previous_volume is not None else _metric(WARMING_UP, None, "broker tick-volume units",
                "Current tick volume minus previous valid observation", 1, "Need a prior observation"))
            previous_mid = next((_number(_field(item, "mid_price"), positive=True)
                                 for item in reversed(observations[:-1])
                                 if _number(_field(item, "mid_price"), positive=True) is not None), None)
            if mid is None or previous_mid is None or tick_volume == 0:
                metrics["price_move_per_tick_volume"] = _metric(UNAVAILABLE, None,
                    "price units per broker tick-volume unit", "Absolute midpoint change / current broker tick volume", 0,
                    "Requires two valid midpoints and nonzero tick volume")
            else:
                metrics["price_move_per_tick_volume"] = _metric(AVAILABLE, abs(mid - previous_mid) / tick_volume,
                    "price units per broker tick-volume unit", "Absolute midpoint change / current broker tick volume", 1)
        metric_statuses = [metric.status for metric in metrics.values()]
        result_status = (AVAILABLE if AVAILABLE in metric_statuses else
                         WARMING_UP if WARMING_UP in metric_statuses else UNAVAILABLE)
        return InstrumentLiquidityResult(symbol, "MT5", result_status, _field(state, "tick_timestamp"),
            _field(state, "last_update_received_at"), _field(state, "tick_timestamp_trust"), freshness,
            connection, None, volume_semantics, MappingProxyType(metrics), tuple(warnings))

    @staticmethod
    def _mt5_key(state: Any) -> tuple[Any, ...]:
        return (_field(state, "tick_timestamp"), _field(state, "bid"), _field(state, "ask"),
                _field(state, "tick_volume"))

    def _analyze_binance(self, symbol: str, snapshot: Any, state: Any) -> InstrumentLiquidityResult:
        freshness = str(_field(state, "freshness", "UNAVAILABLE")).upper()
        connection = str(_field(state, "connection_state", "DISCONNECTED")).upper()
        source_status = _field(snapshot.sources, "Binance")
        source_connection = str(_field(source_status, "connection_state", connection)).upper()
        warnings = ["Binance displayed order-book liquidity is cancellable and is not guaranteed executable."]
        metrics = _unavailable_metrics("No valid Binance observation available")
        timestamp = _field(state, "tick_timestamp")
        received_at = _field(state, "last_update_received_at")
        if source_connection != "CONNECTED" or connection == DISCONNECTED:
            return self._failed(symbol, "Binance", "Binance source is disconnected", state=state,
                                status=DISCONNECTED, freshness=freshness, warnings=warnings)
        freshness_status = STALE if freshness == STALE else (
            AVAILABLE if freshness == "LIVE" else UNAVAILABLE)

        trades_by_symbol = _field(snapshot, "binance_trades", {})
        trades = _field(trades_by_symbol, symbol, ()) or ()
        valid_trades: list[Any] = []
        invalid_trade_count = 0
        stale_trade_count = 0
        for trade in trades:
            if bool(_field(trade, "stale", False)):
                stale_trade_count += 1
                continue
            price = _number(_field(trade, "price"), positive=True)
            quantity = _number(_field(trade, "quantity"), positive=True)
            maker = _field(trade, "buyer_is_maker")
            if price is None or quantity is None or not isinstance(maker, bool):
                invalid_trade_count += 1
                continue
            valid_trades.append(trade)
        if stale_trade_count:
            warnings.append(f"Ignored {stale_trade_count} stale Binance trade(s).")
        if invalid_trade_count:
            warnings.append(f"Ignored {invalid_trade_count} invalid Binance trade(s).")
        if valid_trades:
            buy_volume = sum(_number(_field(trade, "quantity"), positive=True) or 0
                             for trade in valid_trades if not _field(trade, "buyer_is_maker"))
            sell_volume = sum(_number(_field(trade, "quantity"), positive=True) or 0
                              for trade in valid_trades if _field(trade, "buyer_is_maker"))
            total = buy_volume + sell_volume
            metrics["buy_aggressor_volume"] = _metric(AVAILABLE, buy_volume, "base-asset units",
                "Sum quantity where buyer_is_maker is false (buyer aggressor)", len(valid_trades))
            metrics["sell_aggressor_volume"] = _metric(AVAILABLE, sell_volume, "base-asset units",
                "Sum quantity where buyer_is_maker is true (seller aggressor)", len(valid_trades))
            metrics["buy_sell_volume_imbalance"] = _metric(AVAILABLE,
                (buy_volume - sell_volume) / total if total else None, "ratio",
                "(buy-aggressor quantity - sell-aggressor quantity) / total quantity",
                len(valid_trades), "Zero total executed quantity" if not total else None)
            metrics["trade_count"] = _metric(AVAILABLE, float(len(valid_trades)), "trades",
                "Count of fresh valid trades in the manager's bounded trade buffer", len(valid_trades))
            metrics["trade_activity"] = _metric(AVAILABLE, total, "base-asset units per sample",
                "Sum of fresh executed base-asset quantities in the bounded trade buffer", len(valid_trades))
        else:
            no_trade_status = DISCONNECTED if source_connection != "CONNECTED" else (
                STALE if stale_trade_count else WARMING_UP)
            for name in ("buy_aggressor_volume", "sell_aggressor_volume", "buy_sell_volume_imbalance",
                         "trade_count", "trade_activity"):
                metrics[name] = _metric(no_trade_status, None, "base-asset units",
                    "Fresh valid Binance public trades", 0,
                    "No fresh valid trades are present in the bounded buffer")

        books = _field(snapshot, "binance_order_books", {})
        book = _field(books, symbol, {})
        book_status = _field(book, "status", {})
        synchronized = bool(_field(book_status, "synchronized", False))
        book_stale = bool(_field(book_status, "stale", True))
        last_book_update = _number(_field(book_status, "last_update_monotonic"), nonnegative=True)
        if (last_book_update is None or
                time.monotonic() - last_book_update > self.settings.book_freshness_seconds):
            book_stale = True
        book_available = bool(_field(book, "available", False))
        book_issue = ("Binance source disconnected" if source_connection != "CONNECTED" else
                      "Order book is unsynchronized" if not synchronized else
                      "Order book is stale" if book_stale else
                      str(_field(book, "message", "Order book is unavailable")))
        if freshness_status == STALE or source_connection != "CONNECTED" or not book_available or not synchronized or book_stale:
            book_metric_status = DISCONNECTED if source_connection != "CONNECTED" else (
                STALE if freshness_status == STALE or (synchronized and book_stale) else UNAVAILABLE)
            warning = book_issue
            for name in ("bid_depth", "ask_depth", "order_book_imbalance", "bid_top_level_concentration",
                         "ask_top_level_concentration", "bid_depth_change", "ask_depth_change",
                         "bid_depth_change_pct", "ask_depth_change_pct", "bid_depth_anomaly_zscore",
                         "ask_depth_anomaly_zscore"):
                unit = metrics[name].unit
                metrics[name] = _metric(book_metric_status, None, unit, "Requires synchronized fresh displayed book", 0,
                                        warning)
            if freshness_status == STALE:
                warnings.append("Market-manager observation is stale; current order-book metrics are withheld.")
            else:
                warnings.append(warning)
        else:
            self._apply_book_metrics(symbol, book, metrics, warnings)
        self._apply_spread_metrics(symbol, state, metrics)
        metric_statuses = [metric.status for metric in metrics.values()]
        status = (freshness_status if freshness_status == STALE else
                  AVAILABLE if AVAILABLE in metric_statuses else
                  WARMING_UP if WARMING_UP in metric_statuses else UNAVAILABLE)
        return InstrumentLiquidityResult(symbol, "Binance", status,
            timestamp, received_at, _field(state, "tick_timestamp_trust"), freshness, connection,
            _field(state, "order_book_sync_state"), _field(state, "volume_semantics"),
            MappingProxyType(metrics), tuple(warnings))

    def _apply_spread_metrics(self, symbol: str, state: Any,
                              metrics: dict[str, LiquidityMetric]) -> None:
        spread = _number(_field(state, "spread"), nonnegative=True)
        bps = _number(_field(state, "spread_bps"), nonnegative=True)
        freshness = str(_field(state, "freshness", "UNAVAILABLE")).upper()
        if freshness == STALE:
            metrics["spread_price"] = _metric(STALE, None, "price units", "Best bid/ask quote spread", 0,
                                               "Current quote is stale")
            metrics["spread_bps"] = _metric(STALE, None, "basis points", "Quote spread / midpoint * 10,000", 0,
                                             "Current quote is stale")
            self._add_relative_metrics(metrics, "spread", None, [], "price units",
                                       "spread_expansion_ratio", "spread_anomaly_zscore", status=STALE)
            return
        metrics["spread_price"] = (_metric(AVAILABLE, spread, "price units", "Best ask minus best bid", 1)
                                   if spread is not None else _metric(UNAVAILABLE, None, "price units",
                                   "Best ask minus best bid", 0, "Valid spread unavailable"))
        metrics["spread_bps"] = (_metric(AVAILABLE, bps, "basis points", "Spread / midpoint * 10,000", 1)
                                 if bps is not None else _metric(UNAVAILABLE, None, "basis points",
                                 "Spread / midpoint * 10,000", 0, "Valid spread or midpoint unavailable"))
        # Prior spread samples come from the service's bounded book history.
        spread_samples = self._book_history[symbol]
        prior = [item[3] for item in list(spread_samples)[-self.settings.baseline_window:]]
        self._add_relative_metrics(metrics, "spread", spread, prior, "price units",
                                   "spread_expansion_ratio", "spread_anomaly_zscore")

    def _apply_book_metrics(self, symbol: str, book: Any, metrics: dict[str, LiquidityMetric],
                            warnings: list[str]) -> None:
        bids = self._validated_levels(_field(book, "bids", ()), "bid", warnings)
        asks = self._validated_levels(_field(book, "asks", ()), "ask", warnings)
        if not bids or not asks:
            warning = "Synchronized book has no valid levels on both sides"
            for name in ("bid_depth", "ask_depth", "order_book_imbalance", "bid_top_level_concentration",
                         "ask_top_level_concentration"):
                metrics[name] = _metric(UNAVAILABLE, None, metrics[name].unit,
                                        "Top-of-book displayed depth", 0, warning)
            warnings.append(warning)
            return
        bids, asks = bids[:self.settings.book_levels], asks[:self.settings.book_levels]
        bid_total = sum(quantity for _, quantity in bids)
        ask_total = sum(quantity for _, quantity in asks)
        denominator = bid_total + ask_total
        metrics["bid_depth"] = _metric(AVAILABLE, bid_total, "base-asset units",
            f"Sum displayed bid quantities across top {len(bids)} levels", len(bids))
        metrics["ask_depth"] = _metric(AVAILABLE, ask_total, "base-asset units",
            f"Sum displayed ask quantities across top {len(asks)} levels", len(asks))
        metrics["order_book_imbalance"] = _metric(AVAILABLE,
            (bid_total - ask_total) / denominator if denominator else None, "ratio",
            "(top-N bid quantity - top-N ask quantity) / combined top-N quantity",
            len(bids) + len(asks), "Zero depth denominator" if not denominator else None)
        metrics["bid_top_level_concentration"] = _metric(AVAILABLE, bids[0][1] / bid_total,
            "fraction of bid depth", "Best bid level quantity / top-N bid depth", len(bids))
        metrics["ask_top_level_concentration"] = _metric(AVAILABLE, asks[0][1] / ask_total,
            "fraction of ask depth", "Best ask level quantity / top-N ask depth", len(asks))
        best_bid, best_ask = bids[0][0], asks[0][0]
        if best_ask <= best_bid:
            warnings.append("Order book is crossed or locked; depth retained but spread-sensitive metrics are unavailable.")
        else:
            spread = best_ask - best_bid
            mid = (best_ask + best_bid) / 2
            metrics["spread_price"] = _metric(AVAILABLE, spread, "price units",
                "Best ask minus best bid from synchronized displayed book", 1)
            metrics["spread_bps"] = _metric(AVAILABLE, spread / mid * 10000, "basis points",
                "(best ask - best bid) / midpoint * 10,000", 1)
        book_status = _field(book, "status", {})
        update_id = _field(book_status, "update_id")
        key = update_id if update_id is not None else (best_bid, best_ask, bid_total, ask_total)
        with self._lock:
            history = self._book_history[symbol]
            previous = history[-1] if history else None
            if key != self._book_keys.get(symbol):
                self._book_keys[symbol] = key
                history.append((key, bid_total, ask_total,
                                (best_ask - best_bid) if best_ask > best_bid else 0.0))
            prior = list(history)[:-1] if history else []
        prior_bid = prior[-1][1] if prior else None
        prior_ask = prior[-1][2] if prior else None
        self._depth_change_metrics(metrics, "bid", bid_total, prior_bid, prior)
        self._depth_change_metrics(metrics, "ask", ask_total, prior_ask, prior)

    def _depth_change_metrics(self, metrics: dict[str, LiquidityMetric], side: str, current: float,
                              previous: float | None, history: list[tuple[Any, float, float, float]]) -> None:
        change_name, pct_name, anomaly_name = (f"{side}_depth_change", f"{side}_depth_change_pct",
                                               f"{side}_depth_anomaly_zscore")
        depth_index = 1 if side == "bid" else 2
        baselines = [sample[depth_index] for sample in history[-self.settings.baseline_window:]]
        if previous is None:
            metrics[change_name] = _metric(WARMING_UP, None, "base-asset units",
                "Current displayed depth minus previous distinct book snapshot", 1, "Need a prior book snapshot")
            metrics[pct_name] = _metric(WARMING_UP, None, "percent",
                "Depth change / previous displayed depth * 100", 1, "Need a prior book snapshot")
        else:
            delta = current - previous
            metrics[change_name] = _metric(AVAILABLE, delta, "base-asset units",
                "Current displayed depth minus previous distinct book snapshot", 2)
            metrics[pct_name] = (_metric(AVAILABLE, delta / previous * 100, "percent",
                "Depth change / previous displayed depth * 100", 2) if previous else
                _metric(UNAVAILABLE, None, "percent", "Depth change / previous displayed depth * 100", 2,
                        "Previous depth is zero"))
        self._anomaly_metric(metrics, anomaly_name, current, baselines,
                             "standard deviations", "Displayed depth versus prior book snapshots")

    @staticmethod
    def _validated_levels(levels: Any, side: str, warnings: list[str]) -> list[tuple[float, float]]:
        if not isinstance(levels, (list, tuple)):
            warnings.append(f"Invalid {side} level collection")
            return []
        parsed = []
        for item in levels:
            price = _number(_field(item, "price"), positive=True)
            quantity = _number(_field(item, "quantity"), positive=True)
            if price is None or quantity is None:
                warnings.append(f"Invalid {side} book level ignored")
                return []
            parsed.append((price, quantity))
        parsed.sort(key=lambda pair: pair[0], reverse=side == "bid")
        return parsed

    def _add_relative_metrics(self, metrics: dict[str, LiquidityMetric], prefix: str,
                              current: float | None, baseline: list[float], unit: str,
                              ratio_name: str, zscore_name: str, status: str = WARMING_UP) -> None:
        method = f"Current {prefix} compared with prior {self.settings.baseline_window} observations"
        required = self.settings.minimum_baseline_samples
        if current is None:
            metrics[ratio_name] = _metric(UNAVAILABLE, None, "ratio", method, len(baseline),
                                          f"Current {prefix} unavailable")
            metrics[zscore_name] = _metric(UNAVAILABLE, None, "standard deviations", method,
                                           len(baseline), f"Current {prefix} unavailable")
            return
        baseline = baseline[-self.settings.baseline_window:]
        if len(baseline) < required:
            warning = f"Need {required} prior observations; have {len(baseline)}"
            metrics[ratio_name] = _metric(status, None, "ratio", method, len(baseline), warning)
            metrics[zscore_name] = _metric(status, None, "standard deviations", method,
                                             len(baseline), warning)
            return
        mean = float(np.mean(baseline))
        metrics[ratio_name] = (_metric(AVAILABLE, current / mean, "ratio",
            f"Current {prefix} / prior mean", len(baseline)) if mean > 0 else
            _metric(UNAVAILABLE, None, "ratio", f"Current {prefix} / prior mean", len(baseline),
                    "Prior mean is zero"))
        std = float(np.std(baseline, ddof=1)) if len(baseline) > 1 else 0.0
        metrics[zscore_name] = (_metric(AVAILABLE, (current - mean) / std, "standard deviations",
            f"(current {prefix} - prior mean) / prior sample standard deviation", len(baseline))
            if std > 0 else _metric(UNAVAILABLE, None, "standard deviations",
            f"(current {prefix} - prior mean) / prior sample standard deviation", len(baseline),
            "Prior standard deviation is zero"))

    def _anomaly_metric(self, metrics: dict[str, LiquidityMetric], name: str, current: float,
                        baseline: list[float], unit: str, method: str) -> None:
        baseline = baseline[-self.settings.baseline_window:]
        required = self.settings.minimum_baseline_samples
        if len(baseline) < required:
            metrics[name] = _metric(WARMING_UP, None, unit, method, len(baseline),
                                    f"Need {required} prior observations; have {len(baseline)}")
            return
        std = float(np.std(baseline, ddof=1))
        value = (current - float(np.mean(baseline))) / std if std > 0 else None
        metrics[name] = _metric(AVAILABLE if value is not None else UNAVAILABLE, value, unit, method,
                                len(baseline), None if value is not None else "Prior standard deviation is zero")

    def _failed(self, symbol: str, source: str, warning: str, *, state: Any = None,
                status: str = UNAVAILABLE, freshness: str | None = None,
                warnings: list[str] | None = None) -> InstrumentLiquidityResult:
        if source == "MT5":
            names = {name: metric for name, metric in _unavailable_metrics(warning, status).items()
                     if name in {"spread_price", "spread_bps", "spread_expansion_ratio", "spread_anomaly_zscore",
                                 "tick_volume", "tick_volume_activity_ratio", "tick_volume_anomaly_zscore",
                                 "tick_volume_change", "price_move_per_tick_volume"}}
            volume_semantics = _field(state, "volume_semantics") or "Broker-observed tick activity; not centralized traded volume"
        else:
            names = _unavailable_metrics(warning, status)
            volume_semantics = _field(state, "volume_semantics")
        return InstrumentLiquidityResult(symbol, source, status,
            _field(state, "tick_timestamp"), _field(state, "last_update_received_at"),
            _field(state, "tick_timestamp_trust"), freshness or str(_field(state, "freshness", "UNAVAILABLE")),
            str(_field(state, "connection_state", "DISCONNECTED")),
            _field(state, "order_book_sync_state"), volume_semantics,
            MappingProxyType(names), tuple((warnings or []) + [warning]))
