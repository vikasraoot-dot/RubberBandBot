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
    
    # Logic: Price below Lower Keltner Channel
    # We use 'close' < 'kc_lower'
    df["below_lower_band"] = df["close"] < df["kc_lower"]
    
    # Logic: RSI Oversold
    rsi_oversold = float(cfg.get("filters", {}).get("rsi_oversold", 30))
    rsi_min = float(cfg.get("filters", {}).get("rsi_min", 15)) # Avoid falling knives
    df["rsi_oversold"] = (df["rsi"] < rsi_oversold) & (df["rsi"] >= rsi_min)
    
    # Logic: Time Filter (Lunch Lull)
    # Exclude 12:00 PM - 1:00 PM ET (approx 17:00 - 18:00 UTC)
    # Data shows 17:00 UTC is the primary loser (-$115 PnL, 41% WR).
    # 18:00 UTC is profitable (+$27 PnL, 46% WR), so we keep it.
    # We assume the index is a DatetimeIndex in UTC.
    if isinstance(df.index, pd.DatetimeIndex):
        # Only exclude hour 17
        df["time_ok"] = ~df.index.hour.isin([17])
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
