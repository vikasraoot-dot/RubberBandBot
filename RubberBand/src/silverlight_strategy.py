#!/usr/bin/env python3
"""
Silver Light Strategy Module
============================
Core strategy logic for the TQQQ/SQQQ trend-following system.

This module is COMPLETELY ISOLATED from Rubber Band bots.
It implements the "White Light" inspired trend-following strategy.

Indicators:
- 50-Day SMA: Entry trigger (price crosses above)
- 200-Day SMA: Regime filter (SPY > 200 SMA = bullish)
- 14-Day ROC: Profit-taking signal (ROC > 15% = overheated)
- Velocity: Position sizing based on momentum acceleration
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from enum import Enum


class Signal(Enum):
    """Trading signal types."""
    LONG = "LONG"       # Hold TQQQ
    SHORT = "SHORT"     # Hold SQQQ
    CASH = "CASH"       # No position


# Float comparison tolerance (avoid precision artifacts)
PRICE_EPSILON = 0.0001


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def compute_roc(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Rate of Change (momentum velocity).
    ROC = (Current - N periods ago) / N periods ago
    """
    return series.pct_change(periods=period)


def compute_velocity(series: pd.Series, period: int = 5) -> pd.Series:
    """
    Calculate velocity (rate of change of ROC).
    Positive velocity = accelerating trend.
    Negative velocity = decelerating trend.
    """
    roc = compute_roc(series, period=1)  # 1-day ROC
    return roc.diff(periods=period)  # Change in ROC over N days


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index (ADX) for trend strength.
    
    ADX > 25: Strong trend
    ADX 20-25: Weak trend
    ADX < 20: Choppy/sideways market
    """
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    
    # Guard against ATR being zero or NaN
    atr = atr.replace(0, np.nan)
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    # FIX: Create Series with proper index to avoid alignment issues
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0),
        index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0),
        index=high.index
    )
    
    # Smoothed directional indicators
    plus_di = 100 * plus_dm.rolling(window=period, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.rolling(window=period, min_periods=period).mean() / atr
    
    # ADX - guard against divide by zero
    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0, np.nan)
    dx = 100 * abs(plus_di - minus_di) / di_sum
    adx = dx.rolling(window=period, min_periods=period).mean()
    
    # Fill NaN with high value (100) to avoid blocking trades on insufficient data
    adx = adx.fillna(100)
    
    return adx


def attach_indicators(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Attach all strategy indicators to the dataframe.
    
    Args:
        df: DataFrame with OHLCV columns
        config: Strategy configuration dict
        
    Returns:
        DataFrame with indicator columns added
    """
    df = df.copy()
    
    # Get config values
    ind_cfg = config.get("indicators", {})
    sma_fast = int(ind_cfg.get("sma_fast", 50))    # Entry trigger
    sma_exit = int(ind_cfg.get("sma_exit", 20))    # Exit trigger (faster!)
    sma_slow = int(ind_cfg.get("sma_slow", 200))   # Regime filter
    roc_period = int(ind_cfg.get("roc_period", 14))
    velocity_period = int(ind_cfg.get("velocity_period", 5))
    adx_period = int(ind_cfg.get("adx_period", 14))  # ADX period
    
    # Calculate indicators
    df["sma_fast"] = compute_sma(df["close"], sma_fast)   # 50-day for entry
    df["sma_exit"] = compute_sma(df["close"], sma_exit)   # 20-day for exit
    df["sma_slow"] = compute_sma(df["close"], sma_slow)   # 200-day regime
    df["roc"] = compute_roc(df["close"], roc_period)
    df["velocity"] = compute_velocity(df["close"], velocity_period)
    df["adx"] = compute_adx(df["high"], df["low"], df["close"], adx_period)  # Trend strength
    
    # Signal conditions
    df["above_sma_fast"] = df["close"] > df["sma_fast"]   # Entry condition
    df["above_sma_exit"] = df["close"] > df["sma_exit"]   # Exit condition
    df["above_sma_slow"] = df["close"] > df["sma_slow"]   # Regime condition
    
    return df


def generate_signal(
    asset_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    config: Dict[str, Any],
    vix_value: Optional[float] = None
) -> Tuple[Signal, Dict[str, Any]]:
    """
    Generate trading signal based on current market state.
    
    Args:
        asset_df: DataFrame for TQQQ with indicators attached
        regime_df: DataFrame for SPY (regime filter) with indicators attached
        config: Strategy configuration
        vix_value: Current VIX value (optional, for volatility filter)
        
    Returns:
        Tuple of (Signal, metadata dict with reasoning)
    """
    if asset_df.empty or regime_df.empty:
        return Signal.CASH, {"reason": "No data available"}
    
    # Get latest values
    asset_row = asset_df.iloc[-1]
    regime_row = regime_df.iloc[-1]
    
    ind_cfg = config.get("indicators", {})
    entry_cfg = config.get("entry_filters", {})
    
    # Extract values
    price = float(asset_row["close"])
    sma_fast = float(asset_row["sma_fast"]) if not pd.isna(asset_row["sma_fast"]) else 0
    roc = float(asset_row["roc"]) if not pd.isna(asset_row["roc"]) else 0
    adx = float(asset_row["adx"]) if "adx" in asset_row and not pd.isna(asset_row["adx"]) else 100
    regime_above_200 = bool(regime_row["above_sma_slow"]) if not pd.isna(regime_row["above_sma_slow"]) else True
    
    # Entry filter settings
    require_positive_roc = entry_cfg.get("require_positive_roc", True)
    vix_filter_enabled = entry_cfg.get("vix_filter_enabled", True)
    vix_threshold = float(entry_cfg.get("vix_threshold", 25))
    adx_filter_enabled = entry_cfg.get("adx_filter_enabled", True)
    adx_threshold = float(entry_cfg.get("adx_threshold", 20))
    
    metadata = {
        "price": price,
        "sma_fast": sma_fast,
        "roc": roc,
        "roc_pct": roc * 100,
        "adx": adx,
        "regime_bullish": regime_above_200,
        "vix": vix_value,
    }
    
    # --- Decision Logic (with Entry Filters) ---
    
    # 1. Check Regime Filter (SPY > 200 SMA)
    if not regime_above_200:
        metadata["reason"] = "BEARISH REGIME: SPY below 200 SMA"
        return Signal.CASH, metadata
    
    # 2. Check VIX Filter (avoid entering in high volatility)
    if vix_filter_enabled and vix_value is not None and vix_value > vix_threshold:
        metadata["reason"] = f"HIGH VOLATILITY: VIX {vix_value:.1f} > {vix_threshold}. Staying out."
        return Signal.CASH, metadata
    
    # 3. Check ADX Filter (avoid entering in choppy markets)
    if adx_filter_enabled and adx < adx_threshold:
        metadata["reason"] = f"CHOPPY MARKET: ADX {adx:.0f} < {adx_threshold}. Waiting for trend."
        return Signal.CASH, metadata
    
    # 4. Check entry condition (Price > 50 SMA with epsilon tolerance)
    if price > sma_fast + PRICE_EPSILON:
        # 5. Check momentum confirmation (ROC > 0)
        if require_positive_roc and roc <= PRICE_EPSILON:
            metadata["reason"] = f"WEAK MOMENTUM: ROC {roc*100:.1f}% <= 0. Waiting for acceleration."
            return Signal.CASH, metadata
        
        # All conditions met - GO LONG
        metadata["reason"] = f"BULLISH: Price > 50 SMA, ROC={roc*100:.1f}%, ADX={adx:.0f}"
        return Signal.LONG, metadata
    else:
        # Price below 50 SMA - exit or stay out
        metadata["reason"] = f"TREND BREAK: Price {price:.2f} below 50 SMA {sma_fast:.2f}"
        return Signal.CASH, metadata


def calculate_position_size(
    signal: Signal,
    asset_df: pd.DataFrame,
    config: Dict[str, Any],
    current_equity: float
) -> float:
    """
    Calculate target position size based on signal and velocity.
    
    Args:
        signal: The trading signal (LONG, SHORT, CASH)
        asset_df: DataFrame with indicators
        config: Strategy configuration
        current_equity: Current portfolio value
        
    Returns:
        Target position size in dollars
    """
    if signal == Signal.CASH:
        return 0.0
    
    sizing_cfg = config.get("sizing", {})
    max_pct = float(sizing_cfg.get("max_position_pct", 1.0))
    reduced_pct = float(sizing_cfg.get("reduced_position_pct", 0.5))
    
    ind_cfg = config.get("indicators", {})
    roc_threshold = float(ind_cfg.get("roc_threshold", 0.15))
    
    # Get velocity and ROC
    row = asset_df.iloc[-1]
    velocity = float(row["velocity"]) if not pd.isna(row["velocity"]) else 0
    roc = float(row["roc"]) if not pd.isna(row["roc"]) else 0
    
    # Determine sizing
    if roc > roc_threshold:
        # Overheated - reduce size
        pct = reduced_pct
    elif velocity < 0:
        # Decelerating trend - reduce size
        pct = reduced_pct
    else:
        # Full size
        pct = max_pct
    
    return current_equity * pct


def get_target_allocation(
    signal: Signal,
    asset_df: pd.DataFrame,
    config: Dict[str, Any],
    current_equity: float
) -> Dict[str, float]:
    """
    Get target dollar allocation for each asset.
    
    Returns:
        Dict mapping symbol -> target dollar amount
    """
    assets_cfg = config.get("assets", {})
    long_symbol = assets_cfg.get("long", "TQQQ")
    short_symbol = assets_cfg.get("short", "SQQQ")
    
    position_size = calculate_position_size(signal, asset_df, config, current_equity)
    
    if signal == Signal.LONG:
        return {long_symbol: position_size, short_symbol: 0.0}
    elif signal == Signal.SHORT:
        return {long_symbol: 0.0, short_symbol: position_size}
    else:
        return {long_symbol: 0.0, short_symbol: 0.0}
