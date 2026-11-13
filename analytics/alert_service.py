"""Rule-based, persisted alerts over existing MarketPulse analytics results.

This service does not collect market data or recompute analytics. It consumes
the existing volatility, liquidity, regime and anomaly service results. Alerts
describe unusual observations; they are not trading instructions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
import threading
from types import MappingProxyType
from typing import Any, Callable, Mapping
from uuid import uuid4

from analytics.anomaly_service import AnomalyService
from analytics.liquidity_service import LiquidityService
from analytics.regime_service import RegimeService
from analytics.volatility_service import VolatilityService
from config import INSTRUMENTS
from storage.database import AlertDatabase

logger = logging.getLogger(__name__)

AVAILABLE = "AVAILABLE"
WARMING_UP = "WARMING_UP"
UNAVAILABLE = "UNAVAILABLE"
STALE = "STALE"
DISCONNECTED = "DISCONNECTED"

VOLATILITY_EXPANSION = "VOLATILITY_EXPANSION"
ABNORMAL_SPREAD = "ABNORMAL_SPREAD"
MT5_TICK_ACTIVITY = "MT5_TICK_ACTIVITY"
BINANCE_ORDER_BOOK_IMBALANCE = "BINANCE_ORDER_BOOK_IMBALANCE"
REGIME_CHANGE = "REGIME_CHANGE"
ISOLATION_FOREST_ANOMALY = "ISOLATION_FOREST_ANOMALY"

INFO = "INFO"
WARNING = "WARNING"
CRITICAL = "CRITICAL"

_ALERT_CATEGORIES = (
    VOLATILITY_EXPANSION, ABNORMAL_SPREAD, MT5_TICK_ACTIVITY,
    BINANCE_ORDER_BOOK_IMBALANCE, REGIME_CHANGE, ISOLATION_FOREST_ANOMALY,
)


@dataclass(frozen=True)
class AlertSettings:
    """Documented thresholds, analysis windows and category cooldowns.

    Expansion thresholds use existing service metrics: volatility percentile
    >= 90 or z-score >= 2.5; spread expansion ratio >= 2 or z-score >= 3;
    MT5 broker tick activity ratio >= 2.5 or z-score >= 3; absolute Binance
    order-book imbalance >= 0.75. CRITICAL cutoffs are percentile/z-score
    98/5 for volatility, ratio/z-score 4/5 for spread and MT5 activity, and
    absolute imbalance 0.90. Re-entry cooldowns default to five minutes except
    regime transitions, whose default is zero because each transition is a
    distinct state change. An unchanged active condition is always suppressed;
    severity escalation can emit immediately.
    """

    volatility_timeframe: str = "1h"
    volatility_candle_count: int = 200
    regime_timeframe: str = "1h"
    regime_candle_count: int = 600
    volatility_percentile_threshold: float = 90.0
    volatility_zscore_threshold: float = 2.5
    critical_volatility_percentile: float = 98.0
    critical_volatility_zscore: float = 5.0
    spread_expansion_ratio_threshold: float = 2.0
    spread_zscore_threshold: float = 3.0
    critical_spread_expansion_ratio: float = 4.0
    critical_spread_zscore: float = 5.0
    mt5_activity_ratio_threshold: float = 2.5
    mt5_activity_zscore_threshold: float = 3.0
    critical_mt5_activity_ratio: float = 4.0
    critical_mt5_activity_zscore: float = 5.0
    binance_imbalance_threshold: float = 0.75
    critical_binance_imbalance: float = 0.90
    volatility_cooldown_seconds: float = 300.0
    spread_cooldown_seconds: float = 300.0
    mt5_activity_cooldown_seconds: float = 300.0
    binance_imbalance_cooldown_seconds: float = 300.0
    regime_cooldown_seconds: float = 0.0
    anomaly_cooldown_seconds: float = 300.0

    def __post_init__(self) -> None:
        thresholds = (
            self.volatility_percentile_threshold, self.volatility_zscore_threshold,
            self.critical_volatility_percentile, self.critical_volatility_zscore,
            self.spread_expansion_ratio_threshold, self.spread_zscore_threshold,
            self.critical_spread_expansion_ratio, self.critical_spread_zscore,
            self.mt5_activity_ratio_threshold, self.mt5_activity_zscore_threshold,
            self.critical_mt5_activity_ratio, self.critical_mt5_activity_zscore,
            self.binance_imbalance_threshold, self.critical_binance_imbalance,
        )
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(value) or value <= 0 for value in thresholds):
            raise ValueError("alert thresholds must be finite and positive")
        if (self.volatility_percentile_threshold > 100 or
                self.critical_volatility_percentile > 100):
            raise ValueError("volatility percentile thresholds must be <= 100")
        if self.critical_binance_imbalance > 1:
            raise ValueError("absolute Binance imbalance thresholds must be <= 1")
        if self.critical_volatility_percentile < self.volatility_percentile_threshold:
            raise ValueError("critical volatility percentile must be >= alert threshold")
        if self.critical_volatility_zscore < self.volatility_zscore_threshold:
            raise ValueError("critical volatility z-score must be >= alert threshold")
        if self.critical_spread_expansion_ratio < self.spread_expansion_ratio_threshold:
            raise ValueError("critical spread ratio must be >= alert threshold")
        if self.critical_spread_zscore < self.spread_zscore_threshold:
            raise ValueError("critical spread z-score must be >= alert threshold")
        if self.critical_mt5_activity_ratio < self.mt5_activity_ratio_threshold:
            raise ValueError("critical MT5 activity ratio must be >= alert threshold")
        if self.critical_mt5_activity_zscore < self.mt5_activity_zscore_threshold:
            raise ValueError("critical MT5 activity z-score must be >= alert threshold")
        if self.critical_binance_imbalance < self.binance_imbalance_threshold:
            raise ValueError("critical Binance imbalance must be >= alert threshold")
        cooldowns = self.cooldowns
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(value) or value < 0 for value in cooldowns.values()):
            raise ValueError("alert cooldowns must be finite and nonnegative")
        if (isinstance(self.volatility_candle_count, bool) or
                isinstance(self.regime_candle_count, bool) or
                not isinstance(self.volatility_candle_count, int) or
                not isinstance(self.regime_candle_count, int) or
                self.volatility_candle_count <= 0 or self.regime_candle_count <= 0):
            raise ValueError("candle counts must be positive")

    @property
    def cooldowns(self) -> Mapping[str, float]:
        return MappingProxyType({
            VOLATILITY_EXPANSION: self.volatility_cooldown_seconds,
            ABNORMAL_SPREAD: self.spread_cooldown_seconds,
            MT5_TICK_ACTIVITY: self.mt5_activity_cooldown_seconds,
            BINANCE_ORDER_BOOK_IMBALANCE: self.binance_imbalance_cooldown_seconds,
            REGIME_CHANGE: self.regime_cooldown_seconds,
            ISOLATION_FOREST_ANOMALY: self.anomaly_cooldown_seconds,
        })


@dataclass(frozen=True)
class Alert:
    """Persisted alert record with distinct detection and market receive time."""

    alert_id: str
    instrument: str
    provider: str
    detected_at_utc: datetime
    timestamp_provenance: str
    market_data_received_at_utc: datetime | None
    category: str
    severity: str
    title: str
    explanation: str
    observed_values: Mapping[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_at_utc: datetime | None = None


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _iso(value: Any) -> str | None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _from_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        result = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if result.tzinfo is None:
        return None
    return result.astimezone(timezone.utc)


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, Mapping):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _safe_value(value.item())
        except (ValueError, TypeError):
            pass
    return str(value)


class AlertService:
    """Evaluate existing analytics results and persist non-flooding alerts.

    The supplied ``MarketDataManager`` must already be running. This service
    never starts connectors. Analytics inputs are read once per ``evaluate``
    call, and stale/disconnected instrument snapshots suppress conclusions.
    """

    def __init__(self, market_manager: Any, *,
                 volatility_service: Any | None = None,
                 liquidity_service: Any | None = None,
                 regime_service: Any | None = None,
                 anomaly_service: Any | None = None,
                 database: AlertDatabase | Any | None = None,
                 settings: AlertSettings | None = None,
                 clock: Callable[[], datetime] | None = None) -> None:
        self.market_manager = market_manager
        self.settings = settings or AlertSettings()
        self.volatility_service = volatility_service or VolatilityService()
        self.liquidity_service = liquidity_service or LiquidityService(market_manager)
        self.regime_service = regime_service or RegimeService(market_manager=market_manager)
        self.anomaly_service = anomaly_service or AnomalyService(market_manager)
        self.database = database or AlertDatabase()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.RLock()
        self._sources = {item.symbol.upper(): item.source for item in INSTRUMENTS}
        self._condition_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def evaluate(self, *, snapshot: Any | None = None,
                 inputs: Mapping[str, Any] | None = None) -> tuple[Alert, ...]:
        """Evaluate all configured instruments using supplied or collected results.

        The application background cycle may supply its already-computed
        ``volatility``, ``liquidity``, ``regime`` and ``anomaly`` mappings to
        avoid duplicate analytics work. Calling ``evaluate()`` without
        arguments retains the standalone service behavior.
        """
        with self._lock:
            if snapshot is None:
                try:
                    snapshot = self.market_manager.get_market_snapshot()
                except Exception:
                    logger.exception("Unable to obtain market snapshot; no alerts evaluated")
                    return ()

            if inputs is None:
                inputs = {
                    "volatility": self._collect(
                        "volatility", lambda: self.volatility_service.analyze_all(
                            self.settings.volatility_timeframe, self.settings.volatility_candle_count)),
                    "liquidity": self._collect("liquidity", self.liquidity_service.analyze),
                    "regime": self._collect(
                        "regime", lambda: self.regime_service.analyze_all(
                            self.settings.regime_timeframe, self.settings.regime_candle_count)),
                    "anomaly": self._collect("anomaly", self.anomaly_service.analyze),
                }
            else:
                # Missing analytics are unavailable. Do not recollect them or
                # substitute values when the shared cycle reports a failure.
                inputs = {name: inputs.get(name) for name in
                          ("volatility", "liquidity", "regime", "anomaly")}
            detected_at = self._clock()
            if not isinstance(detected_at, datetime) or detected_at.tzinfo is None:
                raise ValueError("alert clock must return a timezone-aware datetime")
            detected_at = detected_at.astimezone(timezone.utc)

            alerts: list[Alert] = []
            states = _field(snapshot, "instruments", {})
            for symbol, provider in self._sources.items():
                state = _field(states, symbol)
                if not self._is_live_state(state):
                    continue
                try:
                    candidates = self._build_candidates(symbol, provider, state, inputs)
                    for category in _ALERT_CATEGORIES:
                        condition, known, severity, title, explanation, observed = candidates[category]
                        alert = self._process_condition(
                            symbol, provider, category, condition, known, severity,
                            title, explanation, observed, state, detected_at,
                            state_value=(candidates["__regime_state"]
                                         if category == REGIME_CHANGE else None),
                        )
                        if alert is not None:
                            alerts.append(alert)
                except Exception:
                    logger.exception("Alert evaluation failed for %s", symbol)
            return tuple(alerts)

    @staticmethod
    def _collect(name: str, function: Callable[[], Any]) -> Any:
        try:
            return function()
        except Exception:
            logger.exception("Could not collect %s analytics for alert evaluation", name)
            return None

    @staticmethod
    def _is_live_state(state: Any) -> bool:
        return (state is not None and
                str(_field(state, "connection_state", "")).upper() == "CONNECTED" and
                str(_field(state, "freshness", "")).upper() == "LIVE")

    def _build_candidates(self, symbol: str, provider: str, state: Any,
                          inputs: Mapping[str, Any]) -> dict[str, Any]:
        candidates: dict[str, Any] = {}
        volatility = inputs.get("volatility")
        vol_result = _field(volatility, symbol)
        vol_metrics = _field(vol_result, "metrics", {})
        percentile = self._available_metric(vol_metrics, "volatility_percentile")
        zscore = self._available_metric(vol_metrics, "volatility_zscore")
        vol_known = percentile[1] or zscore[1]
        vol_values = {}
        if percentile[1]:
            vol_values["volatility_percentile"] = percentile[0]
        if zscore[1]:
            vol_values["volatility_zscore"] = zscore[0]
        vol_trigger = ((percentile[1] and percentile[0] >= self.settings.volatility_percentile_threshold) or
                       (zscore[1] and zscore[0] >= self.settings.volatility_zscore_threshold))
        vol_critical = ((percentile[1] and percentile[0] >= self.settings.critical_volatility_percentile) or
                        (zscore[1] and zscore[0] >= self.settings.critical_volatility_zscore))
        candidates[VOLATILITY_EXPANSION] = self._candidate(
            f"EXPANSION:{CRITICAL if vol_critical else WARNING}" if vol_trigger else None, vol_known,
            CRITICAL if vol_critical else WARNING,
            f"Volatility expansion — {symbol}",
            "Volatility percentile or z-score crossed the configured expansion threshold.",
            self._with_provenance(vol_values, state, vol_result, vol_metrics,
                {"volatility_percentile": self.settings.volatility_percentile_threshold,
                 "volatility_zscore": self.settings.volatility_zscore_threshold,
                 "critical_volatility_percentile": self.settings.critical_volatility_percentile,
                 "critical_volatility_zscore": self.settings.critical_volatility_zscore}),
        )

        liquidity = inputs.get("liquidity")
        liq_result = _field(liquidity, symbol)
        liq_metrics = _field(liq_result, "metrics", {})
        spread_ratio = self._available_metric(liq_metrics, "spread_expansion_ratio")
        spread_zscore = self._available_metric(liq_metrics, "spread_anomaly_zscore")
        spread_known = spread_ratio[1] or spread_zscore[1]
        spread_values = {}
        if spread_ratio[1]:
            spread_values["spread_expansion_ratio"] = spread_ratio[0]
        if spread_zscore[1]:
            spread_values["spread_anomaly_zscore"] = spread_zscore[0]
        spread_trigger = ((spread_ratio[1] and spread_ratio[0] >= self.settings.spread_expansion_ratio_threshold) or
                          (spread_zscore[1] and spread_zscore[0] >= self.settings.spread_zscore_threshold))
        spread_critical = ((spread_ratio[1] and spread_ratio[0] >= self.settings.critical_spread_expansion_ratio) or
                           (spread_zscore[1] and spread_zscore[0] >= self.settings.critical_spread_zscore))
        spread_known_values = self._with_provenance(spread_values, state, liq_result, liq_metrics,
            {"spread_expansion_ratio": self.settings.spread_expansion_ratio_threshold,
             "spread_anomaly_zscore": self.settings.spread_zscore_threshold,
             "critical_spread_expansion_ratio": self.settings.critical_spread_expansion_ratio,
             "critical_spread_anomaly_zscore": self.settings.critical_spread_zscore})
        candidates[ABNORMAL_SPREAD] = self._candidate(
            f"WIDE:{CRITICAL if spread_critical else WARNING}" if spread_trigger else None, spread_known,
            CRITICAL if spread_critical else WARNING,
            f"Abnormally wide spread — {symbol}",
            "The spread is elevated relative to its provider-specific recent baseline.",
            spread_known_values,
        )

        if provider == "MT5":
            activity_ratio = self._available_metric(liq_metrics, "tick_volume_activity_ratio")
            activity_z = self._available_metric(liq_metrics, "tick_volume_anomaly_zscore")
            activity_known = activity_ratio[1] or activity_z[1]
            activity_values = {}
            if activity_ratio[1]:
                activity_values["tick_volume_activity_ratio"] = activity_ratio[0]
            if activity_z[1]:
                activity_values["tick_volume_anomaly_zscore"] = activity_z[0]
            activity_trigger = (
                (activity_ratio[1] and activity_ratio[0] >= self.settings.mt5_activity_ratio_threshold) or
                (activity_z[1] and activity_z[0] >= self.settings.mt5_activity_zscore_threshold)
            )
            activity_critical = (
                (activity_ratio[1] and activity_ratio[0] >= self.settings.critical_mt5_activity_ratio) or
                (activity_z[1] and activity_z[0] >= self.settings.critical_mt5_activity_zscore)
            )
            activity_values["volume_semantics"] = (
                _field(liq_result, "volume_semantics") or
                "Broker-observed tick activity; not centralized traded volume")
            candidates[MT5_TICK_ACTIVITY] = self._candidate(
                f"HIGH_ACTIVITY:{CRITICAL if activity_critical else WARNING}" if activity_trigger else None,
                activity_known,
                CRITICAL if activity_critical else WARNING,
                f"Unusual broker-observed tick activity — {symbol}",
                "MT5 tick activity is unusually high versus broker-observed history; it is not centralized traded volume.",
                self._with_provenance(activity_values, state, liq_result, liq_metrics,
                    {"tick_volume_activity_ratio": self.settings.mt5_activity_ratio_threshold,
                     "tick_volume_anomaly_zscore": self.settings.mt5_activity_zscore_threshold,
                     "critical_tick_volume_activity_ratio": self.settings.critical_mt5_activity_ratio,
                     "critical_tick_volume_anomaly_zscore": self.settings.critical_mt5_activity_zscore}),
            )
            candidates[BINANCE_ORDER_BOOK_IMBALANCE] = self._candidate(
                None, False, WARNING, "", "", {})
        else:
            candidates[MT5_TICK_ACTIVITY] = self._candidate(None, False, WARNING, "", "", {})
            imbalance = self._available_metric(liq_metrics, "order_book_imbalance")
            book_sync_state = str(_field(state, "order_book_sync_state", "UNSYNCHRONIZED")).upper()
            imbalance_known = imbalance[1] and book_sync_state == "SYNCHRONIZED"
            imbalance_trigger = imbalance_known and abs(imbalance[0]) >= self.settings.binance_imbalance_threshold
            imbalance_critical = imbalance_known and abs(imbalance[0]) >= self.settings.critical_binance_imbalance
            imbalance_values = {"order_book_imbalance": imbalance[0]} if imbalance_known else {}
            if book_sync_state != "SYNCHRONIZED":
                imbalance_values["order_book_status"] = book_sync_state
            imbalance_values["liquidity_semantics"] = (
                "Displayed Binance Spot liquidity is cancellable and is not guaranteed executable.")
            candidates[BINANCE_ORDER_BOOK_IMBALANCE] = self._candidate(
                (f"BUY_SIDE:{CRITICAL if imbalance_critical else WARNING}" if imbalance_trigger and imbalance[0] > 0 else
                 f"SELL_SIDE:{CRITICAL if imbalance_critical else WARNING}" if imbalance_trigger else None),
                imbalance_known, CRITICAL if imbalance_critical else WARNING,
                f"Order-book imbalance — {symbol}",
                "Synchronized-book imbalance crossed the configured threshold. Displayed depth can be cancelled and is not guaranteed executable.",
                self._with_provenance(imbalance_values, state, liq_result, liq_metrics,
                    {"absolute_order_book_imbalance": self.settings.binance_imbalance_threshold,
                     "critical_absolute_order_book_imbalance": self.settings.critical_binance_imbalance}),
            )

        regime_result = _field(inputs.get("regime"), symbol)
        regime_status = str(_field(regime_result, "status", "UNAVAILABLE")).upper()
        regime = _field(regime_result, "current_regime")
        last_freshness = str(_field(regime_result, "last_observation_freshness", "UNAVAILABLE")).upper()
        regime_known = regime_status == AVAILABLE and isinstance(regime, str) and bool(regime) and last_freshness == "LIVE"
        previous = self._get_condition_state(symbol, REGIME_CHANGE)
        prior_regime = previous.get("state_value")
        regime_changed = regime_known and prior_regime is not None and prior_regime != regime
        regime_values = {"previous_regime": prior_regime, "current_regime": regime}
        candidates[REGIME_CHANGE] = self._candidate(
            f"TO_{regime}" if regime_changed else None, regime_known,
            WARNING, f"Market regime changed — {symbol}",
            f"The available classification changed from {prior_regime} to {regime}. A regime label describes observed conditions, not a prediction or trade signal."
            if regime_changed else "", self._with_provenance(regime_values, state, regime_result, {}, {}),
        )
        candidates["__regime_state"] = regime if regime_known else prior_regime

        anomaly_result = _field(inputs.get("anomaly"), symbol)
        anomaly_status = str(_field(anomaly_result, "status", "UNAVAILABLE")).upper()
        anomaly_freshness = str(_field(anomaly_result, "freshness", "UNAVAILABLE")).upper()
        anomaly_value = _field(anomaly_result, "anomaly_detected")
        anomaly_known = (anomaly_status == AVAILABLE and anomaly_freshness == "LIVE" and
                         isinstance(anomaly_value, bool))
        anomaly_trigger = anomaly_known and anomaly_value
        anomaly_values = {
            "anomaly_score": _finite(_field(anomaly_result, "anomaly_score")),
            "model_decision": _field(anomaly_result, "model_decision"),
            "features_used": _field(anomaly_result, "features_used", ()),
            "sample_count": _field(anomaly_result, "sample_count"),
        }
        anomaly_values["interpretation"] = "Unusual observation; not necessarily harmful and not a trading signal."
        candidates[ISOLATION_FOREST_ANOMALY] = self._candidate(
            "ANOMALOUS" if anomaly_trigger else None, anomaly_known,
            WARNING, f"Unusual statistical observation — {symbol}",
            "The existing Isolation Forest marked this observation unusual relative to prior samples. This does not imply harm or provide a trading signal.",
            self._with_provenance(anomaly_values, state, anomaly_result, {}, {}),
        )
        return candidates

    @staticmethod
    def _candidate(condition: str | None, known: bool, severity: str, title: str,
                   explanation: str, observed: Mapping[str, Any]
                   ) -> tuple[str | None, bool, str, str, str, Mapping[str, Any]]:
        return condition, known, severity, title, explanation, observed

    @staticmethod
    def _available_metric(metrics: Any, name: str) -> tuple[float | None, bool]:
        metric = _field(metrics, name)
        if str(_field(metric, "status", "UNAVAILABLE")).upper() != AVAILABLE:
            return None, False
        value = _finite(_field(metric, "value"))
        return value, value is not None

    def _with_provenance(self, values: Mapping[str, Any], state: Any,
                         result: Any, metrics: Any,
                         thresholds: Mapping[str, float]) -> Mapping[str, Any]:
        provenance = {
            "thresholds": thresholds,
            "data_freshness": _field(result, "freshness", _field(result, "last_observation_freshness",
                                                                       _field(state, "freshness"))),
            "source_connection_state": _field(result, "connection_state", _field(state, "connection_state")),
            "source_event_timestamp": _field(state, "tick_timestamp"),
            "source_timestamp_trust": _field(state, "tick_timestamp_trust"),
            "source_timestamp_field": _field(state, "tick_timestamp_source"),
            "analytics_timestamp_trust": _field(result, "timestamp_trust"),
            "metric_sample_counts": {name: _field(_field(metrics, name), "sample_count")
                                     for name in values if _field(metrics, name) is not None},
        }
        return MappingProxyType({**dict(values), **provenance})

    def _process_condition(self, symbol: str, provider: str, category: str,
                           condition: str | None, known: bool, severity: str,
                           title: str, explanation: str, observed: Mapping[str, Any],
                           market_state: Any, detected_at: datetime,
                           state_value: str | None = None) -> Alert | None:
        previous = self._get_condition_state(symbol, category)
        if not known:
            return None
        active_before = previous.get("active_condition")
        last_emitted = _from_iso(previous.get("last_emitted_at_utc"))
        if condition is None:
            self._set_condition_state(symbol, category,
                active_condition=None, last_emitted_at_utc=last_emitted,
                state_value=state_value if category == REGIME_CHANGE else previous.get("state_value"))
            return None
        if active_before == condition:
            if category == REGIME_CHANGE and state_value != previous.get("state_value"):
                self._set_condition_state(symbol, category,
                    active_condition=active_before, last_emitted_at_utc=last_emitted,
                    state_value=state_value)
            return None
        cooldown = self.settings.cooldowns[category]
        severity_escalation = (active_before is not None and
                               self._severity_rank(condition) > self._severity_rank(active_before))
        if (not severity_escalation and last_emitted is not None and
                (detected_at - last_emitted).total_seconds() < cooldown):
            self._set_condition_state(symbol, category,
                active_condition=condition, last_emitted_at_utc=last_emitted,
                state_value=state_value if category == REGIME_CHANGE else previous.get("state_value"))
            return None

        received_at = _field(market_state, "last_update_received_at")
        if not isinstance(received_at, datetime) or received_at.tzinfo is None:
            received_at = None
        else:
            received_at = received_at.astimezone(timezone.utc)
        alert = Alert(
            alert_id=uuid4().hex,
            instrument=symbol,
            provider=provider,
            detected_at_utc=detected_at,
            timestamp_provenance="PYTHON_SYSTEM_UTC_AT_DETECTION; provider event time is separately recorded",
            market_data_received_at_utc=received_at if isinstance(received_at, datetime) else None,
            category=category,
            severity=severity,
            title=title,
            explanation=explanation,
            observed_values=MappingProxyType(_safe_value(dict(observed))),
        )
        next_state_value = (state_value if category == REGIME_CHANGE
                            else previous.get("state_value"))
        self.database.persist_alert_and_condition(
            self._alert_record(alert), instrument=symbol, category=category,
            active_condition=condition, state_value=next_state_value)
        self._condition_cache[(symbol, category)] = {
            "active_condition": condition,
            "last_emitted_at_utc": detected_at.isoformat(),
            "state_value": next_state_value,
        }
        return alert

    def _get_condition_state(self, instrument: str, category: str) -> dict[str, Any]:
        key = (instrument.upper(), category.upper())
        if key not in self._condition_cache:
            self._condition_cache[key] = dict(self.database.get_condition_state(*key))
        return dict(self._condition_cache[key])

    def _set_condition_state(self, instrument: str, category: str, *,
                             active_condition: str | None,
                             last_emitted_at_utc: datetime | None,
                             state_value: str | None) -> None:
        key = (instrument.upper(), category.upper())
        previous = self._get_condition_state(*key)
        emitted = _iso(last_emitted_at_utc)
        next_state = {
            "active_condition": active_condition,
            "last_emitted_at_utc": emitted,
            "state_value": state_value,
        }
        if next_state != previous:
            self.database.set_condition_state(
                *key, active_condition=active_condition,
                last_emitted_at_utc=last_emitted_at_utc, state_value=state_value)
            self._condition_cache[key] = next_state

    @staticmethod
    def _severity_rank(condition: str | None) -> int:
        if not condition or ":" not in condition:
            return 0
        suffix = condition.rsplit(":", 1)[-1]
        return {INFO: 1, WARNING: 2, CRITICAL: 3}.get(suffix, 0)

    @staticmethod
    def _alert_record(alert: Alert) -> Mapping[str, Any]:
        return {
            "alert_id": alert.alert_id,
            "instrument": alert.instrument,
            "provider": alert.provider,
            "detected_at_utc": alert.detected_at_utc,
            "timestamp_provenance": alert.timestamp_provenance,
            "market_data_received_at_utc": alert.market_data_received_at_utc,
            "category": alert.category,
            "severity": alert.severity,
            "title": alert.title,
            "explanation": alert.explanation,
            "observed_values": dict(alert.observed_values),
            "acknowledged": alert.acknowledged,
            "acknowledged_at_utc": alert.acknowledged_at_utc,
        }

    def get_recent_alerts(self, *, limit: int = 100, instrument: str | None = None,
                          category: str | None = None, severity: str | None = None
                          ) -> tuple[Alert, ...]:
        """Retrieve newest persisted alerts with optional exact filters."""
        rows = self.database.fetch_alerts(limit=limit, instrument=instrument,
                                          category=category, severity=severity)
        return tuple(self._from_record(row) for row in rows)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert by ID; return false when it does not exist."""
        return self.database.acknowledge_alert(alert_id, self._now())

    @staticmethod
    def _from_record(row: Mapping[str, Any]) -> Alert:
        detected_at = _from_iso(row.get("detected_at_utc"))
        if detected_at is None:
            raise ValueError("stored alert has a missing or invalid UTC detection timestamp")
        return Alert(
            alert_id=str(row["alert_id"]), instrument=str(row["instrument"]),
            provider=str(row["provider"]),
            detected_at_utc=detected_at,
            timestamp_provenance=str(row.get("timestamp_provenance", "unknown")),
            market_data_received_at_utc=_from_iso(row.get("market_data_received_at_utc")),
            category=str(row["category"]), severity=str(row["severity"]),
            title=str(row["title"]), explanation=str(row["explanation"]),
            observed_values=MappingProxyType(dict(row.get("observed_values", {}))),
            acknowledged=bool(row.get("acknowledged", False)),
            acknowledged_at_utc=_from_iso(row.get("acknowledged_at_utc")),
        )

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError("alert clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)
