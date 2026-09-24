"""
Central configuration for the regime-gated price/volume + candlestick study.

Every research threshold lives here so that (a) nothing is a magic number in the
logic modules and (b) the dev-period choices that get frozen before the
out-of-sample run are visible in one place.

All values below are *a-priori* research defaults chosen before looking at any
strategy results.  Values that were selected on the 2020-2022 development
period are NOT edited here; they are written to ``frozen_models.json`` by
``run_dev.py`` and read back unchanged by ``run_oos.py``.
"""
from __future__ import annotations

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # per-ticker adjusted OHLCV (committed for universe + benchmarks)
BREADTH_RAW_DIR = DATA_DIR / "breadth_raw"   # ~690 S&P members for breadth (gitignored, rebuildable)
META_DIR = DATA_DIR / "meta"
DERIVED_DIR = DATA_DIR / "derived"
CHART_DIR = ROOT / "charts"
OUT_DIR = ROOT

# ──────────────────────────────────────────────────────────────────────────────
# Dates
# ──────────────────────────────────────────────────────────────────────────────
DOWNLOAD_START = "2018-01-01"      # stock warm-up (only 2019 is needed; extra is harmless)
MARKET_DOWNLOAD_START = "2015-01-01"   # SPY/QQQ/VIX: longer history for 252d VIX percentile
UNIVERSE_SELECTION_START = "2019-01-01"
UNIVERSE_SELECTION_END = "2019-12-31"  # universe chosen with information available on this date
UNIVERSE_MEMBERSHIP_DATE = "2019-12-31"
BACKTEST_START = "2020-01-01"
DEV_END = "2022-12-31"
OOS_START = "2023-01-01"
# BACKTEST_END is "latest available bar"; resolved from the data at runtime.

# ──────────────────────────────────────────────────────────────────────────────
# Universe construction
# ──────────────────────────────────────────────────────────────────────────────
UNIVERSE_MIN_PRICE = 10.0
UNIVERSE_MIN_MEDIAN_DOLLAR_VOLUME = 100e6
# Target counts per sector bucket (sums to 100).  Chosen to stop tech dominating.
SECTOR_BUCKET_TARGETS = {
    "Semiconductors": 8,
    "Software": 8,
    "Tech Hardware & IT Services": 5,
    "Communication/Internet": 8,
    "Consumer Discretionary": 10,
    "Consumer Staples": 7,
    "Financials": 12,
    "Industrials": 8,
    "Transportation": 4,
    "Energy": 7,
    "Healthcare": 9,
    "Biotech": 4,
    "Materials": 4,
    "Utilities": 3,
    "Real Estate": 3,
}

# ──────────────────────────────────────────────────────────────────────────────
# Indicator windows
# ──────────────────────────────────────────────────────────────────────────────
EMA_FAST = 9
EMA_MID = 20
SMA_MID = 50
SMA_LONG = 200
ATR_LEN = 14
VOL_AVG_LEN = 20             # trigger volume ratio denominator (prior 20 sessions, excludes today)
SLOPE_LOOKBACK = 5           # "EMA20 rising" = EMA20 today > EMA20 five sessions ago
SWING_LOOKBACK = 10          # "recent swing low" = lowest low of prior 10 sessions
VIX_PCT_WINDOW = 252
VIX_SMA_LEN = 20
VIX_ROC_LEN = 5

# ──────────────────────────────────────────────────────────────────────────────
# Candle definitions (a-priori; sensitivities are run around these)
# ──────────────────────────────────────────────────────────────────────────────
STRONG_CLOSE_CLV = 0.75       # close in top 25% of the day's range
HAMMER_LOWER_WICK_FRAC = 0.50  # lower wick >= 50% of range
HAMMER_MIN_CLV = 0.65
WRB_RANGE_ATR = 1.25          # wide-range bar: range >= 1.25 x prior ATR14
WRB_MIN_CLV = 0.70
WRB_MIN_BODY_FRAC = 0.50

# ──────────────────────────────────────────────────────────────────────────────
# Setup A (trend pullback + reversal)
# ──────────────────────────────────────────────────────────────────────────────
PULLBACK_WINDOW = 5          # low must have come near EMA20 in the last 5 sessions (incl. trigger)
PULLBACK_DIST = 0.02         # "near" = low <= EMA20 x (1 + 2%)
PB_VOL_DAYS = 3              # pullback volume = mean volume of the 3 sessions before the trigger
PB_VOL_MAX_RATIO = 1.0       # contraction = pullback volume < prior 20d average
TRIGGER_VOL_RATIO = 1.0      # trigger volume >= 1.0 x prior 20d average
EXTENSION_MAX = 0.05         # skip if close > 5% above EMA20

# ──────────────────────────────────────────────────────────────────────────────
# Setup B (failed breakdown / reclaim)
# ──────────────────────────────────────────────────────────────────────────────
SUPPORT_LOOKBACK = 20        # support = lowest low of 20 sessions ending 3 sessions ago
UNDERCUT_WINDOW = 3          # undercut of support must have happened in last 3 sessions
RECLAIM_MIN_CLV = 0.60

# ──────────────────────────────────────────────────────────────────────────────
# Setup C (volatility contraction breakout)
# ──────────────────────────────────────────────────────────────────────────────
BASE_LEN = 10                # consolidation = prior 10 sessions
BASE_MAX_RANGE_ATR = 3.0     # 10-day high-low range <= 3.0 x ATR14 measured before the base
ATR_CONTRACTION = 0.85       # ATR(5)/ATR(20) at T-1 below this = volatility contracting
BREAKOUT_VOL_RATIO = 1.2

# ──────────────────────────────────────────────────────────────────────────────
# Regime definitions
# ──────────────────────────────────────────────────────────────────────────────
BREADTH_STRONG = 60.0
BREADTH_WEAK = 40.0
VIX_HIGH_LEVEL = 25.0
VIX_SHOCK_PCT = 90.0         # VIX 252d percentile >= 90 ...
VIX_SHOCK_ROC = 0.25         # ... or 5-day VIX change >= +25%
VIX_LOW_PCT = 25.0

# ──────────────────────────────────────────────────────────────────────────────
# Trade simulation
# ──────────────────────────────────────────────────────────────────────────────
COST_BPS_PER_SIDE = 5.0      # realistic case; frictionless = 0
BASELINE_ATR_STOP = 1.5      # a-priori baseline initial stop (entry - 1.5 ATR14)
CHANDELIER_ATR = 3.0         # trailing stop for Exit F
FIXED_HORIZONS = (5, 10, 20)  # exit-agnostic forward-return horizons (sessions)
PER_STOCK_CAPITAL = 10_000.0

# ──────────────────────────────────────────────────────────────────────────────
# Portfolio simulation
# ──────────────────────────────────────────────────────────────────────────────
PORTFOLIO_CAPITAL = 50_000.0
RISK_PER_TRADE = 0.01
RISK_GRID = (0.005, 0.01, 0.015)
MAX_POSITION_PCT = 0.25      # single-name notional cap (concentration limit)
MIN_POSITION_PCT = 0.02      # skip dust positions when little room/cash is left
EXPOSURE_CAP_ON = 1.00
EXPOSURE_CAP_REDUCED = 0.50
REDUCED_RISK_MULT = 0.5
