#!/usr/bin/env python3
"""
Debug: Run actual backtest loop manually with verbose output.
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
    print(" FULL BACKTEST SIMULATION WITH VERBOSE TRACE")
    print("="*80)
    
    # Load config
    with open("RubberBand/config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    # Fetch 15-minute bars (matching backtest's fetch_days = days * 1.6 = 16)
    print("\n[1] Fetching 15-minute bars (16 days ~= 10 trading days)...")
    bars_15m, _ = fetch_latest_bars(["TSLA"], "15Min", history_days=16, feed="iex", verbose=False)
    df = bars_15m.get("TSLA")
    
    print(f"    Raw bars: {len(df) if df is not None else 0}")
    if df is None or df.empty or len(df) < 50:
        print(f"    ERROR: Not enough data (need >= 50 bars)")
        return
    
    df = attach_verifiers(df, cfg).copy()
    print(f"    After attach_verifiers: {len(df)} bars")
    
    # Fetch daily SMA
    print("\n[2] Fetching daily bars (100 days)...")
    daily_bars, _ = fetch_latest_bars(["TSLA"], "1Day", history_days=100, feed="iex", verbose=False)
    df_daily = daily_bars.get("TSLA")
    daily_sma = None
    if df_daily is not None and len(df_daily) >= 20:
        sma_val = df_daily["close"].rolling(20).mean().iloc[-1]
        if not pd.isna(sma_val):
            daily_sma = float(sma_val)
            print(f"    Daily SMA-20: ${daily_sma:.2f}")
    
    # Fetch VIXY  
    print("\n[3] Fetching VIXY...")
    vixy_bars, _ = fetch_latest_bars(["VIXY"], "1Day", history_days=100, feed="iex", verbose=False)
    vixy_df = vixy_bars.get("VIXY")
    daily_vix_map = {}
    if vixy_df is not None and not vixy_df.empty:
        vixy_df["vixy_prev_close"] = vixy_df["close"].shift(1)
        daily_vix_map = {k.date(): v for k, v in vixy_df["vixy_prev_close"].dropna().to_dict().items()}
        print(f"    VIXY data points: {len(daily_vix_map)}")
    
    # Simulate backtest loop
    print("\n[4] Running backtest loop (i from 20 to len(df))...")
    print("-" * 100)
    
    opts = {"slope_threshold": -0.08, "trend_filter": True}
    trades = []
    skip_reasons = {"len<50": 0, "in_trade": 0, "no_signal": 0, "bad_atr": 0, "slope": 0, "trend": 0}
    
    in_trade_until = -1
    
    for i in range(20, len(df)):
        cur = df.iloc[i]
        date_obj = cur.name.date() if hasattr(cur.name, "date") else None
        
        # Skip if in trade
        if i <= in_trade_until:
            skip_reasons["in_trade"] += 1
            continue
        
        prev = df.iloc[i - 1]
        
        # Check for long signal on PREV
        if not prev.get("long_signal", False):
            skip_reasons["no_signal"] += 1
            continue
        
        # Calculate slope
        current_slope_threshold = -0.08  # CALM default
        if daily_vix_map and date_obj:
            vix_val = daily_vix_map.get(date_obj, float('nan'))
            if not pd.isna(vix_val):
                if vix_val < 35:
                    current_slope_threshold = -0.08
                elif vix_val > 55:
                    current_slope_threshold = -0.20
                else:
                    current_slope_threshold = -0.12
        
        entry_price = float(cur.get("open", 0))
        atr = float(prev.get("atr", 0))
        
        if atr <= 0 or entry_price <= 0:
            skip_reasons["bad_atr"] += 1
            continue
        
        # Slope check
        slope_3_pct = 0
        if "kc_middle" in df.columns and i >= 4:
            slope_3 = (df["kc_middle"].iloc[i-1] - df["kc_middle"].iloc[i-4]) / 3
            slope_3_pct = (slope_3 / entry_price) * 100
        
        if slope_3_pct > current_slope_threshold:
            skip_reasons["slope"] += 1
            continue
        
        # Trend check
        if daily_sma is not None and opts.get("trend_filter", True):
            close_price = float(prev.get("close", 0))
            if close_price < daily_sma:
                skip_reasons["trend"] += 1
                continue
        
        # PASSED ALL FILTERS!
        print(f"\n    >>> SIGNAL AT i={i}: {prev.name}")
        print(f"        Entry on: {cur.name}, price=${entry_price:.2f}")
        print(f"        RSI: {prev.get('rsi', 0):.1f}")
        print(f"        Slope_pct: {slope_3_pct:.4f}% (thresh: {current_slope_threshold}%)")
        print(f"        Trend: close=${float(prev.get('close', 0)):.2f} vs SMA=${daily_sma:.2f}")
        trades.append({"entry_time": str(cur.name), "price": entry_price})
    
    print("\n" + "-" * 100)
    print(f"\n[RESULTS]")
    print(f"    Total bars processed: {len(df) - 20}")
    print(f"    Skip reasons:")
    for reason, count in skip_reasons.items():
        if count > 0:
            print(f"      - {reason}: {count}")
    print(f"    TRADES FOUND: {len(trades)}")
    
    for t in trades:
        print(f"      - {t['entry_time']} @ ${t['price']:.2f}")
    
    print("\n" + "="*80)
    print(" END SIMULATION")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
