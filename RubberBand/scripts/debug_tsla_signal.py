#!/usr/bin/env python3
"""
Debug: Trace TSLA signal generation in backtest.
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
    print("\n" + "="*70)
    print(" DEBUG: TSLA Signal Generation for Dec 24, 2025")
    print("="*70)
    
    # Load config
    with open("RubberBand/config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    # Override RSI threshold to match signal generation
    print(f"\n[Config] RSI Oversold Threshold: {cfg.get('filters', {}).get('rsi_oversold', 30)}")
    print(f"[Config] RSI Min (falling knife): {cfg.get('filters', {}).get('rsi_min', 15)}")
    
    # Fetch 15-minute bars
    print("\n[1] Fetching 15-minute bars (10 days)...")
    bars_15m, _ = fetch_latest_bars(["TSLA"], "15Min", history_days=10, feed="iex", verbose=False)
    df = bars_15m.get("TSLA")
    
    if df is None or df.empty:
        print("ERROR: No data for TSLA")
        return
    
    print(f"    Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    # Apply attach_verifiers
    print("\n[2] Running attach_verifiers (strategy.py)...")
    df = attach_verifiers(df, cfg)
    
    # Check Dec 24
    df_dec24 = df[df.index.date == datetime(2025, 12, 24).date()]
    print(f"    Dec 24 bars: {len(df_dec24)}")
    
    # Show all signal components for Dec 24
    print("\n[3] Signal Components for Dec 24:")
    print("-" * 100)
    
    for idx, row in df_dec24.iterrows():
        time_str = idx.strftime("%H:%M")
        below_band = "✅" if row.get("below_lower_band", False) else "❌"
        rsi_os = "✅" if row.get("rsi_oversold", False) else "❌"
        trend = "✅" if row.get("trend_ok", True) else "❌"
        time_ok = "✅" if row.get("time_ok", True) else "❌"
        signal = "🚀" if row.get("long_signal", False) else "  "
        
        rsi = row.get("rsi", 0)
        close = row.get("close", 0)
        kc_lower = row.get("kc_lower", 0)
        
        print(f"{time_str} | RSI={rsi:5.1f} | Close=${close:7.2f} | KC_Low=${kc_lower:7.2f} | "
              f"<KC={below_band} RSI_OS={rsi_os} Trend={trend} Time={time_ok} | Signal={signal}")
    
    # Count signals
    signal_count = df_dec24.get("long_signal", pd.Series()).sum()
    print("-" * 100)
    print(f"\n[RESULT] Dec 24 long_signal count: {signal_count}")
    
    # Check 10:31 AM entry time (15:31 UTC)
    target_time = "2025-12-24 15:15"  # Bar before 10:31 entry
    print(f"\n[4] Signal at ~10:15 AM ET (bar before live entry):")
    for idx, row in df_dec24.iterrows():
        if "15:15" in str(idx):
            print(f"    Time: {idx}")
            print(f"    Close: ${row.get('close', 0):.2f}")
            print(f"    KC_Lower: ${row.get('kc_lower', 0):.2f}")
            print(f"    RSI: {row.get('rsi', 0):.1f}")
            print(f"    below_lower_band: {row.get('below_lower_band', False)}")
            print(f"    rsi_oversold: {row.get('rsi_oversold', False)}")
            print(f"    trend_ok: {row.get('trend_ok', True)}")
            print(f"    time_ok: {row.get('time_ok', True)}")
            print(f"    **long_signal: {row.get('long_signal', False)}**")
    
    print("\n" + "="*70)
    print(" END DEBUG")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
