#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
from datetime import timedelta

# Ensure repo root on sys.path
_THIS = os.path.abspath(os.path.dirname(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from RubberBand.src.utils import load_config, read_tickers
from RubberBand.src.data import fetch_latest_bars
from RubberBand.src.indicators import (
    ta_add_rsi, ta_add_adx_di, ta_add_atr, ta_add_sma, 
    ta_add_keltner, ta_add_macd, ta_add_vol_dollar
)

def prepare_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Calculate all technical indicators needed for ML features.
    """
    df = df.copy()
    
    # Basic Params
    keltner_len = int(cfg.get("keltner_length", 20))
    keltner_mult = float(cfg.get("keltner_mult", 2.0))
    atr_len = int(cfg.get("atr_length", 14))
    rsi_len = int(cfg.get("rsi_length", 14))
    
    # 1. Standard Indicators
    df = ta_add_atr(df, length=atr_len)
    df = ta_add_keltner(df, length=keltner_len, mult=keltner_mult, atr_length=atr_len)
    df = ta_add_rsi(df, length=rsi_len)
    df = ta_add_vol_dollar(df, window=20)
    
    # 2. ML Specific Indicators
    df = ta_add_adx_di(df, period=14)
    df = ta_add_macd(df, fast=12, slow=26, signal=9)
    df = ta_add_sma(df, length=50) # For trend slope
    
    # 3. Feature Engineering
    # Relative Volatility
    df["feat_atr_pct"] = df["atr"] / df["close"]
    
    # Distance from Lower Band (How deep is the dip?)
    # Negative value means below band
    df["feat_dist_lower"] = (df["close"] - df["kc_lower"]) / df["close"]
    
    # RSI Trends
    df["feat_rsi"] = df["rsi"]
    df["feat_rsi_5"] = ta_add_rsi(df, length=5)["rsi"] # Short term RSI
    
    # Trend Strength
    df["feat_adx"] = df["adx"]
    df["feat_macd_hist"] = df["macd_hist"]
    
    # Slope of SMA50 (Trend direction)
    df["feat_sma50_slope"] = df["sma_50"].diff(3) / df["sma_50"]
    
    # Volume Spike
    df["feat_vol_rel"] = df["volume"] / df["volume"].rolling(20).mean()
    
    # Time of Day (Hour)
    df["feat_hour"] = df.index.hour
    
    return df

def simulate_trade_outcome(df: pd.DataFrame, idx: int, cfg: dict) -> int:
    """
    Look forward from index `idx` to determine if trade hits TP before SL.
    Returns: 1 (Win), 0 (Loss)
    """
    entry_px = df["open"].iloc[idx] # Enter on Open of NEXT bar (idx is signal bar)
    atr = df["atr"].iloc[idx-1]     # ATR from signal bar
    
    if atr <= 0: return 0
    
    # Risk Params
    bcfg = cfg.get("brackets", {})
    sl_mult = float(bcfg.get("atr_mult_sl", 2.5))
    tp_mult = float(bcfg.get("take_profit_r", 1.5))
    
    stop_px = entry_px - (sl_mult * atr)
    take_px = entry_px + (tp_mult * atr)
    
    # Mean Reversion Exit (Optional, but let's stick to brackets for clear labeling)
    # If we want to match live bot exactly, we should check mean exit too.
    # But for "Falling Knife" detection, pure TP/SL is a good proxy for "Did it bounce?"
    
    # Scan forward up to N bars (e.g., 5 days max hold)
    max_bars = 390 # approx 10 days of 15m bars
    
    for i in range(idx, min(len(df), idx + max_bars)):
        row = df.iloc[i]
        
        # Check Low vs SL
        if row["low"] <= stop_px:
            return 0 # Loss
            
        # Check High vs TP
        if row["high"] >= take_px:
            return 1 # Win
            
    return 0 # Timed out / Flat (Treat as Loss for strictness)

def main():
    print("Loading config...")
    cfg = load_config("RubberBand/config.yaml")
    tickers = read_tickers("RubberBand/tickers.txt")
    
    # Override for strict RSI signal generation
    # We want to capture ALL "dip" signals (RSI < 30) to train the model 
    # to separate the good ones (<30 & bounce) from bad ones (<30 & crash).
    signal_rsi_threshold = 30 
    
    print(f"Fetching data for {len(tickers)} tickers (90 days)...")
    bars_map, _ = fetch_latest_bars(
        tickers, 
        timeframe="15Min", 
        history_days=90, 
        bar_limit=10000,
        verbose=True
    )
    
    dataset = []
    
    for sym, df in bars_map.items():
        if df.empty or len(df) < 100: continue
        
        # 1. Engineer Features
        df = prepare_features(df, cfg)
        
        # 2. Scan for Signals
        # Signal logic: Close < Lower KC AND RSI < 30
        # We use a slightly looser RSI (30) than live (15) to get more training examples
        # and teach the model to filter the "marginal" ones too.
        
        signals = (df["close"] < df["kc_lower"]) & (df["rsi"] < signal_rsi_threshold)
        
        # Get indices where signal is True
        # We need to be careful: Signal at `i` means entry at `i+1`
        sig_indices = np.where(signals)[0]
        
        for i in sig_indices:
            if i >= len(df) - 1: continue # Cannot enter next bar
            if i < 50: continue # Not enough warmup
            
            # Feature Vector (from Signal Bar `i`)
            row = df.iloc[i]
            
            # Label (Outcome from `i+1` onwards)
            label = simulate_trade_outcome(df, i+1, cfg)
            
            data_point = {
                "symbol": sym,
                "timestamp": str(df.index[i]),
                "label": label,
                # Features
                "rsi": row["feat_rsi"],
                "rsi_5": row["feat_rsi_5"],
                "adx": row["feat_adx"],
                "macd_hist": row["feat_macd_hist"],
                "atr_pct": row["feat_atr_pct"],
                "dist_lower": row["feat_dist_lower"],
                "sma50_slope": row["feat_sma50_slope"],
                "vol_rel": row["feat_vol_rel"],
                "hour": row["feat_hour"],
            }
            dataset.append(data_point)
            
    # Save to CSV
    if not dataset:
        print("No training data generated.")
        return
        
    df_train = pd.DataFrame(dataset)
    df_train.to_csv("training_data.csv", index=False)
    
    print(f"\n=== Data Collection Complete ===")
    print(f"Total Samples: {len(df_train)}")
    print(f"Win Rate (Baseline): {df_train['label'].mean():.2%}")
    print(f"Saved to: training_data.csv")

if __name__ == "__main__":
    main()
