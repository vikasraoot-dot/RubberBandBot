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
    
    # Calculate indicators
    df["sma_fast"] = compute_sma(df["close"], sma_fast)   # 50-day for entry
    df["sma_exit"] = compute_sma(df["close"], sma_exit)   # 20-day for exit
    df["sma_slow"] = compute_sma(df["close"], sma_slow)   # 200-day regime
    df["roc"] = compute_roc(df["close"], roc_period)
    df["velocity"] = compute_velocity(df["close"], velocity_period)
    
    # Signal conditions
    df["above_sma_fast"] = df["close"] > df["sma_fast"]   # Entry condition
    df["above_sma_exit"] = df["close"] > df["sma_exit"]   # Exit condition
    df["above_sma_slow"] = df["close"] > df["sma_slow"]   # Regime condition
    
    return df


def generate_signal(
    asset_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[Signal, Dict[str, Any]]:
    """
    Generate trading signal based on current market state.
    
    Args:
        asset_df: DataFrame for TQQQ with indicators attached
        regime_df: DataFrame for SPY (regime filter) with indicators attached
        config: Strategy configuration
        
    Returns:
        Tuple of (Signal, metadata dict with reasoning)
    """
    if asset_df.empty or regime_df.empty:
        return Signal.CASH, {"reason": "No data available"}
    
    # Get latest values
    asset_row = asset_df.iloc[-1]
    regime_row = regime_df.iloc[-1]
    
    ind_cfg = config.get("indicators", {})
    roc_threshold = float(ind_cfg.get("roc_threshold", 0.15))
    
    # Extract values
    price = float(asset_row["close"])
    sma_fast = float(asset_row["sma_fast"]) if not pd.isna(asset_row["sma_fast"]) else 0
    sma_exit = float(asset_row["sma_exit"]) if not pd.isna(asset_row["sma_exit"]) else 0
    roc = float(asset_row["roc"]) if not pd.isna(asset_row["roc"]) else 0
    regime_above_200 = bool(regime_row["above_sma_slow"]) if not pd.isna(regime_row["above_sma_slow"]) else True
    
    metadata = {
        "price": price,
        "sma_fast": sma_fast,
        "sma_exit": sma_exit,
        "roc": roc,
        "roc_pct": roc * 100,
        "regime_bullish": regime_above_200,
    }
    
    # --- Decision Logic (Asymmetric SMA) ---
    # Entry: Price > 50 SMA (slow, confirm trend)
    # Exit: Price < 20 SMA (fast, lock in profits)
    
    # 1. Check Regime Filter (SPY > 200 SMA)
    if not regime_above_200:
        metadata["reason"] = "BEARISH REGIME: SPY below 200 SMA"
        return Signal.CASH, metadata  # Could be SQQQ for aggressive short
    
    # 2. Check exit condition first (faster 20 SMA)
    if price < sma_exit:
        metadata["reason"] = f"FAST EXIT: Price {price:.2f} below 20 SMA {sma_exit:.2f}"
        return Signal.CASH, metadata
    
    # 3. Check entry condition (slower 50 SMA)
    if price > sma_fast:
        # In uptrend - check for profit-taking
        if roc > float(config.get("indicators", {}).get("roc_threshold", 0.15)):
            metadata["reason"] = f"OVERHEATED: ROC {roc*100:.1f}% > threshold. Trim position."
            return Signal.LONG, metadata
        else:
            metadata["reason"] = f"BULLISH: Price above 50 SMA. ROC={roc*100:.1f}%"
            return Signal.LONG, metadata
    else:
        # Price between 20 and 50 SMA - hold if already in, don't enter if not
        metadata["reason"] = f"NEUTRAL ZONE: Price between 20 SMA ({sma_exit:.2f}) and 50 SMA ({sma_fast:.2f})"
        return Signal.CASH, metadata  # Conservative: don't enter in neutral zone


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
