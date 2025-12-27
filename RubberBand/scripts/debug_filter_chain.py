#!/usr/bin/env python3
"""
Debug: Exactly replicate backtest loop to find why TSLA trade is skipped.
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
    print(" EXACT BACKTEST FILTER TRACE FOR TSLA")
    print("="*80)
    
    # Load config
    with open("RubberBand/config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    # Fetch data
    bars_15m, _ = fetch_latest_bars(["TSLA"], "15Min", history_days=16, feed="iex", verbose=False)
    df = bars_15m.get("TSLA")
    df = attach_verifiers(df, cfg).copy()
    print(f"15m bars: {len(df)}")
    
    daily_bars, _ = fetch_latest_bars(["TSLA"], "1Day", history_days=100, feed="iex", verbose=False)
    df_daily = daily_bars.get("TSLA")
    daily_sma = float(df_daily["close"].rolling(20).mean().iloc[-1])
    print(f"Daily SMA-20: ${daily_sma:.2f}")
    
    vixy_bars, _ = fetch_latest_bars(["VIXY"], "1Day", history_days=100, feed="iex", verbose=False)
    vixy_df = vixy_bars.get("VIXY")
    daily_vix_map = {}
    if vixy_df is not None and not vixy_df.empty:
        vixy_df["vixy_prev_close"] = vixy_df["close"].shift(1)
        daily_vix_map = {k.date(): v for k, v in vixy_df["vixy_prev_close"].dropna().to_dict().items()}
    print(f"VIXY data points: {len(daily_vix_map)}")
    
    opts = {"slope_threshold": -0.08, "trend_filter": True, "slope_threshold_10": None, "adx_max": 0}
    
    # Find Dec 24 signal bar
    for i in range(20, len(df)):
        cur = df.iloc[i]
        prev = df.iloc[i - 1]
        date_obj = cur.name.date() if hasattr(cur.name, "date") else None
        
        # Skip non-Dec-24
        if date_obj != datetime(2025, 12, 24).date():
            continue
        
        has_signal = prev.get("long_signal", False)
        if not has_signal:
            continue
        
        # Signal found!
        print(f"\n>>> SIGNAL FOUND at i={i}:")
        print(f"    Prev bar: {prev.name}")
        print(f"    Cur bar: {cur.name}")
        
        # Regime logic
        current_slope_threshold = float(opts.get("slope_threshold") or -0.08)
        if daily_vix_map and date_obj:
            vix_val = daily_vix_map.get(date_obj, float('nan'))
            if not pd.isna(vix_val):
                if vix_val < 35:
                    current_slope_threshold = -0.08
                elif vix_val > 55:
                    current_slope_threshold = -0.20
                else:
                    current_slope_threshold = -0.12
        print(f"    Regime threshold: {current_slope_threshold}%")
        
        entry_price = float(cur["open"])
        atr = float(prev.get("atr", 0))
        print(f"    Entry price: ${entry_price:.2f}")
        print(f"    ATR: {atr:.2f}")
        
        # ATR filter
        if atr <= 0 or entry_price <= 0:
            print("    >>> SKIPPED: ATR or entry_price <= 0")
            continue
        print("    ATR check: PASS")
        
        # Slope calculation
        slope_3 = 0.0
        if "kc_middle" in df.columns and i >= 4:
            slope_3 = (df["kc_middle"].iloc[i-1] - df["kc_middle"].iloc[i-4]) / 3
        entry_price_ref = float(cur["open"])
        slope_3_pct = (slope_3 / entry_price_ref) * 100 if entry_price_ref > 0 else 0
        print(f"    Slope_3_pct: {slope_3_pct:.4f}%")
        
        # Slope filter
        if slope_3_pct > current_slope_threshold:
            print(f"    >>> SKIPPED: slope {slope_3_pct:.4f}% > threshold {current_slope_threshold}%")
            continue
        print("    Slope check: PASS")
        
        # SMA filter
        if daily_sma is not None and opts.get("trend_filter", True):
            close_price = float(prev.get("close", 0))
            if close_price < daily_sma:
                print(f"    >>> SKIPPED: close ${close_price:.2f} < SMA ${daily_sma:.2f}")
                continue
        print(f"    SMA check: PASS (close ${float(prev.get('close', 0)):.2f} >= ${daily_sma:.2f})")
        
        # ADX filter
        adx_max = opts.get("adx_max", 0)
        if adx_max > 0:
            entry_adx = float(prev.get("adx", 0) or prev.get("ADX", 0))
            if entry_adx > adx_max:
                print(f"    >>> SKIPPED: ADX {entry_adx:.1f} > {adx_max}")
                continue
        print("    ADX check: PASS (disabled)")
        
        print("\n    ✅ ALL FILTERS PASSED - TRADE SHOULD BE GENERATED!")
    
    print("\n" + "="*80)
    print(" END TRACE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
