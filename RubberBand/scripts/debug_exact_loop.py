#!/usr/bin/env python3
"""
Debug: Trace the exact backtest loop logic bar-by-bar for TSLA.
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
    print(" EXACT BACKTEST LOOP TRACE: TSLA")
    print("="*80)
    
    # Load config
    with open("RubberBand/config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    # Fetch 15-minute bars
    print("\n[1] Fetching 15-minute bars (10 days)...")
    bars_15m, _ = fetch_latest_bars(["TSLA"], "15Min", history_days=10, feed="iex", verbose=False)
    df = bars_15m.get("TSLA")
    
    if df is None or df.empty or len(df) < 50:
        print(f"ERROR: Not enough data for TSLA (len={len(df) if df is not None else 0})")
        return
    
    df = attach_verifiers(df, cfg).copy()
    print(f"    Loaded {len(df)} bars")
    print(f"    len(df) < 50 check: {len(df) < 50}")
    
    # Fetch daily SMA
    print("\n[2] Fetching daily bars for SMA-20...")
    daily_bars, _ = fetch_latest_bars(["TSLA"], "1Day", history_days=100, feed="iex", verbose=False)
    df_daily = daily_bars.get("TSLA")
    
    if df_daily is not None and len(df_daily) >= 20:
        df_daily["sma"] = df_daily["close"].rolling(20).mean()
        daily_sma = float(df_daily["sma"].iloc[-1])
        print(f"    Daily SMA-20: ${daily_sma:.2f}")
    else:
        daily_sma = None
        print("    Daily SMA-20: None (not enough data)")
    
    # Simulate the exact backtest loop
    print("\n[3] Simulating EXACT backtest loop (line 302-350)...")
    print("-" * 100)
    
    slope_threshold = -0.08  # CALM regime
    trend_filter = True
    
    for i in range(20, len(df)):
        cur = df.iloc[i]
        date_obj = cur.name.date() if hasattr(cur.name, "date") else None
        
        # Skip non-Dec-24
        if date_obj != datetime(2025, 12, 24).date():
            continue
        
        prev = df.iloc[i - 1]
        
        # Check for long_signal on PREV bar (this is how backtest works)
        has_signal = prev.get("long_signal", False)
        
        if has_signal:
            print(f"\n    PREV bar (i-1): {prev.name}")
            print(f"      long_signal: {prev.get('long_signal', False)}")
            print(f"      RSI: {prev.get('rsi', 0):.1f}")
            print(f"    CUR bar (i): {cur.name}")
            print(f"      Entry price (open): ${cur.get('open', 0):.2f}")
            print(f"      ATR: {prev.get('atr', 0):.2f}")
            
            # Check slope filter
            if "kc_middle" in df.columns and i >= 4:
                slope_3 = (df["kc_middle"].iloc[i-1] - df["kc_middle"].iloc[i-4]) / 3
                slope_3_pct = (slope_3 / float(cur["open"])) * 100
                slope_pass = slope_3_pct <= slope_threshold
                print(f"      Slope_3_pct: {slope_3_pct:.4f}% (threshold: {slope_threshold}%)")
                print(f"      Slope Pass: {'✅' if slope_pass else '❌'}")
            else:
                slope_pass = True
                print(f"      Slope: Not calculated (insufficient data)")
            
            # Check SMA trend filter
            if daily_sma is not None and trend_filter:
                close_price = float(prev.get("close", 0))
                trend_pass = close_price >= daily_sma
                print(f"      PREV Close: ${close_price:.2f} vs SMA: ${daily_sma:.2f}")
                print(f"      Trend_pass: {'✅' if trend_pass else '❌'}")
            else:
                trend_pass = True
                print(f"      Trend: Filter disabled or no SMA")
            
            # Final decision
            if slope_pass and trend_pass:
                print(f"      >>> WOULD ENTER 🚀")
            else:
                print(f"      >>> WOULD SKIP ❌ (slope={slope_pass}, trend={trend_pass})")
    
    print("\n" + "="*80)
    print(" END TRACE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
