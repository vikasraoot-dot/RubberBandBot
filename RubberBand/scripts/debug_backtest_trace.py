#!/usr/bin/env python3
"""
Debug: Full backtest trace for TSLA with verbose output.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from RubberBand.src.data import fetch_latest_bars
from RubberBand.strategy import attach_verifiers
import yaml
import pandas as pd
from datetime import datetime

def main():
    print("\n" + "="*80)
    print(" FULL BACKTEST TRACE: TSLA Dec 24")
    print("="*80)
    
    # Load config
    with open("RubberBand/config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    # Fetch 15-minute bars
    print("\n[1] Fetching 15-minute bars...")
    bars_15m, _ = fetch_latest_bars(["TSLA"], "15Min", history_days=10, feed="iex", verbose=False)
    df = bars_15m.get("TSLA")
    
    if df is None or df.empty:
        print("ERROR: No data for TSLA")
        return
    
    df = attach_verifiers(df, cfg)
    print(f"    Loaded {len(df)} bars")
    
    # Fetch daily bars for SMA
    print("\n[2] Fetching daily bars for SMA-20...")
    daily_bars, _ = fetch_latest_bars(["TSLA"], "1Day", history_days=100, feed="iex", verbose=False)
    df_daily = daily_bars.get("TSLA")
    
    if df_daily is None or df_daily.empty:
        print("ERROR: No daily data for TSLA")
        return
    
    print(f"    Loaded {len(df_daily)} daily bars")
    
    # Calculate SMA-20
    sma_period = 20
    df_daily["sma"] = df_daily["close"].rolling(sma_period).mean()
    daily_sma = float(df_daily["sma"].iloc[-1])
    latest_close = float(df_daily["close"].iloc[-1])
    
    print(f"\n[3] SMA-20 Filter:")
    print(f"    Latest Daily Close: ${latest_close:.2f}")
    print(f"    SMA-20: ${daily_sma:.2f}")
    print(f"    Trend: {'BULL ✅' if latest_close >= daily_sma else 'BEAR ❌'}")
    trend_ok = latest_close >= daily_sma
    
    # Fetch VIXY
    print("\n[4] Fetching VIXY for Regime...")
    vixy_bars, _ = fetch_latest_bars(["VIXY"], "1Day", history_days=100, feed="iex", verbose=False)
    vixy_df = vixy_bars.get("VIXY")
    
    if vixy_df is None or vixy_df.empty:
        print("WARNING: No VIXY data")
        vixy_val = None
        slope_threshold = -0.08  # Fallback
    else:
        vixy_val = float(vixy_df["close"].iloc[-1])
        print(f"    VIXY Close: ${vixy_val:.2f}")
        if vixy_val < 35:
            slope_threshold = -0.08  # CALM
            regime = "CALM"
        elif vixy_val > 55:
            slope_threshold = -0.20  # PANIC
            regime = "PANIC"
        else:
            slope_threshold = -0.12  # NORMAL
            regime = "NORMAL"
        print(f"    Regime: {regime}")
        print(f"    Slope Threshold: {slope_threshold}%")
    
    # Simulate backtest loop for Dec 24
    print("\n[5] Simulating Backtest Loop for Dec 24:")
    print("-" * 100)
    
    signal_bars = df[df["long_signal"] == True]
    print(f"    Total long_signal bars: {len(signal_bars)}")
    
    for i, (idx, row) in enumerate(signal_bars.iterrows()):
        if idx.date() != datetime(2025, 12, 24).date():
            continue
        
        bar_idx = df.index.get_loc(idx)
        close = float(row["close"])
        
        # Calculate slope_3
        if bar_idx >= 4 and "kc_middle" in df.columns:
            slope_3 = (df["kc_middle"].iloc[bar_idx-1] - df["kc_middle"].iloc[bar_idx-4]) / 3
            slope_3_pct = (slope_3 / close) * 100
        else:
            slope_3_pct = 0
        
        slope_pass = slope_3_pct <= slope_threshold
        
        print(f"\n    Bar: {idx}")
        print(f"    Close: ${close:.2f}")
        print(f"    Slope_3_pct: {slope_3_pct:.4f}% (threshold: {slope_threshold}%)")
        print(f"    Slope Pass: {'✅' if slope_pass else '❌'}")
        print(f"    Trend Pass (SMA-20): {'✅' if trend_ok else '❌'}")
        print(f"    WOULD ENTER: {'🚀 YES' if (slope_pass and trend_ok) else '❌ NO'}")
    
    print("\n" + "="*80)
    print(" END TRACE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
