# === RubberBand/strategy.py ===
from __future__ import annotations
import pandas as pd
import numpy as np
from RubberBand.src.indicators import (
    ta_add_keltner,
    ta_add_rsi,
    ta_add_vol_dollar,
    ta_add_atr,
    ta_add_sma,
    ta_add_adx_di,
)

def attach_verifiers(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Append indicators and signal columns for Mean Reversion.
    """
    df = df.copy()
    
    # Parameters
    keltner_len = int(cfg.get("keltner_length", 20))
    keltner_mult = float(cfg.get("keltner_mult", 2.0))
    atr_len = int(cfg.get("atr_length", 14))
    rsi_len = int(cfg.get("rsi_length", 14))
    vol_win = int(cfg.get("vol_sma_length", 10))
    
    # Indicators
    df = ta_add_atr(df, length=atr_len)
    df = ta_add_keltner(df, length=keltner_len, mult=keltner_mult, atr_length=atr_len)
    df = ta_add_rsi(df, length=rsi_len)
    df = ta_add_vol_dollar(df, window=vol_win)
    df = ta_add_adx_di(df, period=14)  # Add ADX for trend strength filtering
    
    # Logic: Price below Lower Keltner Channel
    # We use 'close' < 'kc_lower'
    df["below_lower_band"] = df["close"] < df["kc_lower"]
    
    # Logic: RSI Oversold
    rsi_oversold = float(cfg.get("filters", {}).get("rsi_oversold", 30))
    rsi_min = float(cfg.get("filters", {}).get("rsi_min", 15)) # Avoid falling knives
    df["rsi_oversold"] = (df["rsi"] < rsi_oversold) & (df["rsi"] >= rsi_min)
    
    # Logic: Time Filter (Lunch Lull)
    # Exclude entries during 17:00 - 18:00 UTC (12:00 PM - 1:00 PM ET).
    # Since the backtester enters on the NEXT bar, we must block signals that *lead* to entry in this window.
    # Forbidden Entry Times: 17:00, 17:15, 17:30, 17:45.
    # Signals to Block:
    # - 16:45 (Enters at 17:00)
    # - 17:00 (Enters at 17:15)
    # - 17:15 (Enters at 17:30)
    # - 17:30 (Enters at 17:45)
    # We ALLOW 17:45 signal because it enters at 18:00 (which is profitable).
    if isinstance(df.index, pd.DatetimeIndex):
        h = df.index.hour
        m = df.index.minute
        # Block 16:45
        block_pre = (h == 16) & (m == 45)
        # Block 17:00, 17:15, 17:30
        block_main = (h == 17) & (m < 45)
        
        df["time_ok"] = ~(block_pre | block_main)
    else:
        # Fallback if index is not datetime (should not happen in this pipeline)
        df["time_ok"] = True
    
    # Logic: Trend Filter (SMA)
    # If trend_filter_sma > 0, we require close > sma
    sma_period = int(cfg.get("filters", {}).get("trend_filter_sma", 0))
    if sma_period > 0:
        df = ta_add_sma(df, length=sma_period)
        df["trend_ok"] = df["close"] > df[f"sma_{sma_period}"]
    else:
        df["trend_ok"] = True
    
    # Combined Signal
    df["long_signal"] = df["below_lower_band"] & df["rsi_oversold"] & df["trend_ok"] & df["time_ok"]
    
    return df

def check_slope_filter(df: pd.DataFrame, regime_cfg: dict) -> tuple[bool, str]:
    """
    Check if the trade should be SKIPPED based on the slope of the Keltner Middle Band.
    
    Logic Matrix:
    - PANIC Regime (dead_knife_filter=True):
        - Skip if Slope < Threshold (Too Steep / Falling Knife)
    - CALM/NORMAL Regime (dead_knife_filter=False):
        - Skip if Slope > Threshold (Too Flat / Panic Buyer)
        
    Args:
        df: DataFrame with 'kc_middle' column
        regime_cfg: Regime configuration dict from RegimeManager
        
    Returns:
        (should_skip, reason)
    """
    # Defaults
    threshold_pct = -0.12
    is_dead_knife_mode = False
    
    if regime_cfg:
        threshold_pct = regime_cfg.get("slope_threshold_pct", -0.12)
        is_dead_knife_mode = regime_cfg.get("dead_knife_filter", False)
        
    if "kc_middle" not in df.columns or len(df) < 4:
        return False, "" # Cannot calc slope
        
    # Calculate 3-bar slope
    current_slope_3 = (df["kc_middle"].iloc[-1] - df["kc_middle"].iloc[-4]) / 3
    # Normalize to % of Close
    close = df["close"].iloc[-1]
    slope_pct = (current_slope_3 / close) * 100
    
    if is_dead_knife_mode:
        # SAFETY MODE (Panic)
        # Skip if crashing too hard (Falling Knife)
        if slope_pct < threshold_pct:
            return True, f"Safety_Knife_Filter({slope_pct:.4f}% < {threshold_pct}%)"
    else:
        # AGGRESSIVE MODE (Calm)
        # Skip if too flat (Panic Buyer)
        if slope_pct > threshold_pct:
            return True, f"Slope3_Too_Flat({slope_pct:.4f}% > {threshold_pct}%)"
            
    return False, ""
