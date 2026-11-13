"""Central configuration for MarketPulse; this module does not collect data."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Instrument:
    """An instrument name and its planned source."""
    symbol: str
    source: str

INSTRUMENTS = (
    Instrument("XAUUSD", "MT5"), Instrument("EURUSD", "MT5"),
    Instrument("GBPUSD", "MT5"), Instrument("USDJPY", "MT5"),
    Instrument("BTCUSDT", "Binance"), Instrument("ETHUSDT", "Binance"),
)
DATA_SOURCES = ("MT5", "Binance")

@dataclass(frozen=True)
class DashboardSettings:
    """Basic dashboard display settings."""
    title: str = "MARKETPULSE"
    subtitle: str = "Multi-Asset Liquidity, Volatility & Market Regime Intelligence Terminal"
    layout: str = "wide"

@dataclass(frozen=True)
class RefreshIntervals:
    """Configurable refresh intervals in seconds for future components."""
    dashboard_seconds: int = 5
    mt5_seconds: int = 1
    binance_seconds: int = 1

DASHBOARD = DashboardSettings()
REFRESH = RefreshIntervals()
