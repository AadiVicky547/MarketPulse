"""Per-instrument anomaly detection over existing MarketPulse analytics.

An anomaly means an observation is statistically unusual relative to that
instrument's prior observations. It does not imply harm, predict price, or
constitute a trading signal. MT5 session-dependent features stay excluded while
its event-time basis is unverified.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from sklearn.ensemble import IsolationForest

from analytics.liquidity_service import LiquidityService
from analytics.volatility_service import VolatilityService
from config import INSTRUMENTS

logger = logging.getLogger(__name__)

AVAILABLE = "AVAILABLE"
WARMING_UP = "WARMING_UP"
UNAVAILABLE = "UNAVAILABLE"
ERROR = "ERROR"

_COMMON_FEATURES = (
    "spread_price", "spread_bps", "spread_expansion_ratio", "spread_anomaly_zscore",
    "atr", "realized_volatility", "ewma_volatility", "rolling_volatility",
    "volatility_zscore", "volatility_percentile",
)
_MT5_FEATURES = (
    "tick_volume", "tick_volume_activity_ratio", "tick_volume_anomaly_zscore",
    "tick_volume_change", "price_move_per_tick_volume",
)
_BINANCE_FEATURES = (
    "buy_aggressor_volume", "sell_aggressor_volume", "buy_sell_volume_imbalance",
    "trade_count", "trade_activity", "bid_depth", "ask_depth", "order_book_imbalance",
    "bid_top_level_concentration", "ask_top_level_concentration", "bid_depth_change",
    "ask_depth_change", "bid_depth_change_pct", "ask_depth_change_pct",
    "bid_depth_anomaly_zscore", "ask_depth_anomaly_zscore",
)
_SPREAD_FEATURES = ("spread_price", "spread_bps", "spread_expansion_ratio", "spread_anomaly_zscore")
_VOLATILITY_FEATURES = (
    "atr", "realized_volatility", "ewma_volatility", "rolling_volatility",
    "volatility_zscore", "volatility_percentile",
)
_SESSION_FEATURES = {
    "adr", "average_daily_range", "intraday_range_expansion",
    "session_range_expansion_ratio",
}


@dataclass(frozen=True)
class AnomalySettings:
    """Isolation Forest and bounded-history settings.

    The default model has 100 trees, a 5% contamination threshold, automatic
    subsampling (up to 256 prior rows), one CPU worker and fixed seed 42. At
    least 50 prior complete rows and two usable features are required. Only
    rows with all features in the current feature schema enter model fitting.
    """

    minimum_training_samples: int = 50
    minimum_features: int = 2
    history_size: int = 500
    max_features: int = 10
    n_estimators: int = 100
    contamination: float = 0.05
    random_state: int = 42
    max_samples: int | str = "auto"
    n_jobs: int = 1
    explanation_z_threshold: float = 2.0
    volatility_timeframe: str = "1h"
    volatility_candle_count: int = 200

    def __post_init__(self) -> None:
        integer_fields = (self.minimum_training_samples, self.minimum_features,
                          self.history_size, self.max_features, self.n_estimators,
                          self.volatility_candle_count)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in integer_fields):
            raise ValueError("anomaly sample counts and model sizes must be positive integers")
        if self.minimum_training_samples > self.history_size:
            raise ValueError("minimum_training_samples cannot exceed history_size")
        if self.minimum_features > self.max_features:
            raise ValueError("minimum_features cannot exceed max_features")
        if not 0 < self.contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")
        if self.explanation_z_threshold <= 0:
            raise ValueError("explanation_z_threshold must be positive")
        if isinstance(self.max_samples, int) and self.max_samples <= 0:
            raise ValueError("max_samples must be positive or 'auto'")


@dataclass(frozen=True)
class AnomalyResult:
    """Anomaly decision and data/model provenance for one instrument.

    ``anomaly_score`` is the negated scikit-learn decision function, so larger
    values indicate a more unusual observation relative to prior samples.
    """

    instrument: str
    provider: str
    status: str
    anomaly_detected: bool | None
    anomaly_score: float | None
    model_decision: int | None
    features_used: tuple[str, ...]
    explanation: str
    sample_count: int
    model_status: str
    timestamp_trust: str | None
    freshness: str
    connection_state: str
    last_update_received_at: Any = None
    feature_values: Mapping[str, float] = field(default_factory=dict)
    data_quality_warnings: tuple[str, ...] = ()
    # Finite current values can exist before prior coverage qualifies them for
    # the Isolation Forest's training schema.
    available_features: tuple[str, ...] = ()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


class AnomalyService:
    """Score per-instrument observations using prior-only Isolation Forests.

    The service reads the manager snapshot, then calls the existing liquidity
    and volatility analytics APIs. It never accesses connectors directly. Each
    instrument has a bounded deque; no current row is used to fit its own model.
    """

    def __init__(self, market_manager: Any, *,
                 liquidity_service: Any | None = None,
                 volatility_service: Any | None = None,
                 settings: AnomalySettings | None = None) -> None:
        self.market_manager = market_manager
        self.settings = settings or AnomalySettings()
        self.liquidity_service = liquidity_service or LiquidityService(market_manager)
        self.volatility_service = volatility_service or VolatilityService()
        self._sources = {item.symbol.upper(): item.source for item in INSTRUMENTS}
        self._history: dict[str, deque[dict[str, float]]] = {
            symbol: deque(maxlen=self.settings.history_size) for symbol in self._sources
        }
        self._last_observation_key: dict[str, Any] = {}
        self._stored_keys: dict[str, set[Any]] = {symbol: set() for symbol in self._sources}
        self._stored_key_order: dict[str, deque[Any]] = {
            symbol: deque() for symbol in self._sources
        }

    def analyze(self) -> Mapping[str, AnomalyResult]:
        """Collect current analytics, score all instruments, and retain valid rows."""
        try:
            snapshot = self.market_manager.get_market_snapshot()
            liquidity_results = self.liquidity_service.analyze()
            volatility_results = self.volatility_service.analyze_all(
                self.settings.volatility_timeframe, self.settings.volatility_candle_count)
        except Exception as exc:
            logger.exception("Could not collect anomaly inputs")
            return MappingProxyType({
                symbol: self._error_result(symbol, source,
                    f"Analytics input collection failed: {type(exc).__name__}: {exc}")
                for symbol, source in self._sources.items()
            })

        results: dict[str, AnomalyResult] = {}
        for symbol, source in self._sources.items():
            try:
                state = _field(_field(snapshot, "instruments", {}), symbol)
                liquidity = _field(liquidity_results, symbol)
                volatility = _field(volatility_results, symbol)
                results[symbol] = self._score_one(symbol, source, state, liquidity, volatility,
                                                  snapshot)
            except Exception as exc:
                logger.exception("Anomaly scoring failed for %s", symbol)
                results[symbol] = self._error_result(
                    symbol, source, f"Anomaly scoring failed: {type(exc).__name__}: {exc}")
        return MappingProxyType(results)

    def _score_one(self, symbol: str, source: str, state: Any, liquidity: Any,
                   volatility: Any, snapshot: Any) -> AnomalyResult:
        warnings: list[str] = []
        if state is None:
            return self._unavailable_result(symbol, source, "Instrument state missing from market snapshot")
        freshness = str(_field(state, "freshness", "UNAVAILABLE")).upper()
        connection = str(_field(state, "connection_state", "DISCONNECTED")).upper()
        trust = (_field(state, "tick_timestamp_trust") or
                 _field(volatility, "timestamp_trust"))
        received_at = _field(state, "last_update_received_at")
        if source == "MT5" and trust != "TRUSTED_UTC":
            warnings.append(
                "MT5 event-time basis is unverified; no session-based timestamp features are used."
            )

        if connection == "DISCONNECTED":
            return self._unavailable_result(symbol, source, "Data source is disconnected", state,
                                            "DISCONNECTED", warnings)
        if freshness in {"STALE", "DISCONNECTED"}:
            return self._unavailable_result(symbol, source, f"Current market observation is {freshness.lower()}",
                                            state, freshness, warnings)
        if freshness == "UNAVAILABLE":
            return self._unavailable_result(symbol, source, "Current market observation is unavailable",
                                            state, freshness, warnings)

        current, feature_warnings = self._extract_features(source, liquidity, volatility)
        warnings.extend(feature_warnings)
        if not current:
            return self._unavailable_result(symbol, source, "No valid analytics features are available",
                                            state, freshness, warnings)

        key = self._observation_key(symbol, state, snapshot, current)
        history = self._history[symbol]
        is_new_observation = key != self._last_observation_key.get(symbol)
        # Score before appending: this observation can never train its own model.
        scored_history = list(history)
        prior_observation_count = len(scored_history)

        features, train_rows = self._select_training_schema(current, scored_history, source)
        training_count = len(train_rows)
        if len(features) < self.settings.minimum_features:
            if current:
                warnings.append(
                    f"Extracted {len(current)} valid current features ({', '.join(current)}); "
                    f"model features remain in warm-up until they have "
                    f"{self.settings.minimum_training_samples} prior observations."
                )
            result = self._warming_result(symbol, source, state, trust, freshness, current,
                features, prior_observation_count, warnings,
                f"Need at least {self.settings.minimum_features} features with "
                f"{self.settings.minimum_training_samples} prior complete training rows; "
                f"{prior_observation_count}/{self.settings.minimum_training_samples} "
                "prior valid observations collected.")
        elif training_count < self.settings.minimum_training_samples:
            result = self._warming_result(symbol, source, state, trust, freshness, current,
                features, prior_observation_count, warnings,
                f"Isolation Forest warm-up: {prior_observation_count}/"
                f"{self.settings.minimum_training_samples} prior complete rows.")
        else:
            try:
                matrix = np.asarray([[row[name] for name in features] for row in train_rows], dtype=float)
                observation = np.asarray([[current[name] for name in features]], dtype=float)
                if not np.isfinite(matrix).all() or not np.isfinite(observation).all():
                    raise ValueError("non-finite model input")
                model = IsolationForest(
                    n_estimators=self.settings.n_estimators,
                    contamination=self.settings.contamination,
                    max_samples=self.settings.max_samples,
                    max_features=1.0,
                    random_state=self.settings.random_state,
                    n_jobs=self.settings.n_jobs,
                )
                model.fit(matrix)
                decision = int(model.predict(observation)[0])
                raw_decision = float(model.decision_function(observation)[0])
                score = -raw_decision
                anomaly = decision < 0
                explanation = self._explain(anomaly, features, current, train_rows,
                                            self.settings.explanation_z_threshold)
                result = AnomalyResult(symbol, source, AVAILABLE, anomaly, score, decision,
                    tuple(features), explanation, training_count, "TRAINED_ON_PRIOR_OBSERVATIONS",
                    trust, freshness, connection, received_at, MappingProxyType(dict(current)),
                    tuple(warnings), tuple(current))
            except Exception as exc:
                result = AnomalyResult(symbol, source, ERROR, None, None, None, tuple(features),
                    f"Isolation Forest scoring failed: {type(exc).__name__}: {exc}", training_count,
                    "ERROR", trust, freshness, connection, received_at,
                    MappingProxyType(dict(current)), tuple(warnings + [str(exc)]), tuple(current))

        if is_new_observation:
            self._last_observation_key[symbol] = key
            # _stored_keys is maintained alongside the bounded deque so repeated
            # snapshots do not inflate the training history with duplicates.
            self._append_observation(symbol, key, current)
        return result

    def _extract_features(self, source: str, liquidity: Any,
                          volatility: Any) -> tuple[dict[str, float], list[str]]:
        values: dict[str, float] = {}
        warnings: list[str] = []
        metric_groups = (
            ("liquidity", _field(liquidity, "metrics", {})),
            ("volatility", _field(volatility, "metrics", {})),
        )
        expected_by_group = {
            "liquidity": set(_SPREAD_FEATURES +
                             (_MT5_FEATURES if source == "MT5" else _BINANCE_FEATURES)),
            "volatility": set(_VOLATILITY_FEATURES),
        }
        allowed = set(_COMMON_FEATURES + (_MT5_FEATURES if source == "MT5" else _BINANCE_FEATURES))
        for group_name, metrics in metric_groups:
            if not isinstance(metrics, Mapping):
                warnings.append(f"{group_name} metrics are missing or malformed")
                continue
            missing = expected_by_group[group_name] - set(metrics)
            if missing:
                warnings.append(
                    f"{group_name} service did not supply feature metrics: "
                    f"{', '.join(sorted(missing))}."
                )
            for name, metric in metrics.items():
                if name not in allowed:
                    if source == "MT5" and name in _SESSION_FEATURES:
                        status = str(_field(metric, "status", "UNAVAILABLE")).upper()
                        if status == "TIMESTAMP_UNVERIFIED":
                            warnings.append(
                                f"Feature {name} excluded because MT5 event-time basis is unverified."
                            )
                    continue
                status = str(_field(metric, "status", "UNAVAILABLE")).upper()
                value = _finite_number(_field(metric, "value"))
                if status != "AVAILABLE" or value is None:
                    warning = _field(metric, "warning")
                    reason = status.lower() if status else "missing status"
                    if status == "AVAILABLE" and value is None:
                        reason = "available status with missing/non-finite value"
                    warnings.append(f"Feature {name} omitted ({reason})" +
                                    (f": {warning}" if warning else ""))
                    continue
                values[name] = value
        if source == "MT5":
            trust = _field(volatility, "timestamp_trust") or _field(liquidity, "timestamp_trust")
            if trust != "TRUSTED_UTC":
                # Never let session-derived features enter an MT5 model even if
                # a future provider/service accidentally marks them available.
                for name in _SESSION_FEATURES:
                    values.pop(name, None)
        return values, warnings

    def _select_training_schema(self, current: Mapping[str, float],
                                history: list[dict[str, float]], source: str
                                ) -> tuple[list[str], list[dict[str, float]]]:
        available_order = _COMMON_FEATURES + (_MT5_FEATURES if source == "MT5" else _BINANCE_FEATURES)
        candidates = [name for name in available_order if name in current]
        coverage = {name: sum(name in row for row in history) for name in candidates}
        candidates.sort(key=lambda name: (-coverage[name], available_order.index(name)))
        selected: list[str] = []
        complete_rows: list[dict[str, float]] = []
        for name in candidates:
            if coverage[name] < self.settings.minimum_training_samples:
                continue
            proposed = selected + [name]
            rows = [row for row in history if all(feature in row for feature in proposed)]
            if len(rows) >= self.settings.minimum_training_samples:
                selected = proposed
                complete_rows = rows
            if len(selected) >= self.settings.max_features:
                break
        return selected, complete_rows

    def _observation_key(self, symbol: str, state: Any, snapshot: Any,
                         current: Mapping[str, float]) -> tuple[Any, ...]:
        trades = _field(_field(snapshot, "binance_trades", {}), symbol, ()) or ()
        latest_trade = _field(trades[-1], "trade_id") if trades else None
        book = _field(_field(snapshot, "binance_order_books", {}), symbol, {})
        book_status = _field(book, "status", {})
        update_id = _field(book_status, "update_id")
        return (_field(state, "tick_timestamp"), _field(state, "bid"), _field(state, "ask"),
                _field(state, "tick_volume"), latest_trade, update_id,
                tuple(sorted(current.items())))

    def _append_observation(self, symbol: str, key: Any, current: Mapping[str, float]) -> None:
        history = self._history[symbol]
        keys = self._stored_keys.setdefault(symbol, set())
        ordered_keys = self._stored_key_order.setdefault(symbol, deque())
        if key in keys:
            return
        if len(history) == history.maxlen:
            history.popleft()
            if ordered_keys:
                keys.discard(ordered_keys.popleft())
        history.append(dict(current))
        ordered_keys.append(key)
        keys.add(key)

    @staticmethod
    def _explain(anomaly: bool, features: list[str], current: Mapping[str, float],
                 rows: list[dict[str, float]], threshold: float = 2.0) -> str:
        if not anomaly:
            return "The observed feature combination is within the model's learned prior pattern. " \
                   "This is a statistical comparison, not a safety assessment."
        unusual = []
        for name in features:
            values = np.asarray([row[name] for row in rows], dtype=float)
            mean, std = float(values.mean()), float(values.std(ddof=1))
            if std > 0:
                zscore = (current[name] - mean) / std
                if abs(zscore) >= threshold:
                    unusual.append(f"{name}={current[name]:.6g} ({zscore:+.1f} prior standard deviations)")
        detail = "; ".join(unusual) if unusual else "the combined feature pattern is unusual"
        return f"Unusual relative to prior observations: {detail}. This is not necessarily harmful or a trading signal."

    @staticmethod
    def _warming_result(symbol: str, source: str, state: Any, trust: Any, freshness: str,
                        current: Mapping[str, float], features: list[str], sample_count: int,
                        warnings: list[str], explanation: str) -> AnomalyResult:
        return AnomalyResult(symbol, source, WARMING_UP, None, None, None, tuple(features),
            explanation, sample_count, "WARMING_UP", trust, freshness,
            str(_field(state, "connection_state", "CONNECTED")),
            _field(state, "last_update_received_at"), MappingProxyType(dict(current)),
            tuple(warnings), tuple(current))

    @staticmethod
    def _unavailable_result(symbol: str, source: str, warning: str, state: Any = None,
                            freshness: str = "UNAVAILABLE",
                            warnings: list[str] | None = None) -> AnomalyResult:
        return AnomalyResult(symbol, source, UNAVAILABLE, None, None, None, (), warning, 0,
            "NOT_FIT", _field(state, "tick_timestamp_trust"), freshness,
            str(_field(state, "connection_state", "DISCONNECTED")),
            _field(state, "last_update_received_at"), MappingProxyType({}),
            tuple((warnings or []) + [warning]))

    def _error_result(self, symbol: str, source: str, warning: str) -> AnomalyResult:
        return AnomalyResult(symbol, source, ERROR, None, None, None, (), warning, 0,
            "ERROR", None, "UNKNOWN", "UNKNOWN", None, MappingProxyType({}), (warning,))

    @property
    def history_counts(self) -> Mapping[str, int]:
        """Read-only current buffer sizes, useful for diagnostics and tests."""
        return MappingProxyType({symbol: len(rows) for symbol, rows in self._history.items()})

