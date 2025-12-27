#!/usr/bin/env python3
"""
Debug script to trace why TSLA was not entered in backtest but was entered in live.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from RubberBand.src.data import fetch_latest_bars
from RubberBand.src.indicators import ta_add_rsi, ta_add_atr, ta_add_keltner
import pandas as pd
from datetime import datetime, timedelta

def main():
    symbol = "TSLA"
    
    print(f"\n{'='*60}")
    print(f" DEBUG: Tracing {symbol} for Dec 24, 2025")
    print(f"{'='*60}\n")
    
    # Fetch 15-minute bars
    print("[1] Fetching 15-minute bars...")
    bars_15m, _ = fetch_latest_bars([symbol], "15Min", history_days=5, feed="iex", verbose=False)
    df_15m = bars_15m.get(symbol)
    
    if df_15m is None or df_15m.empty:
        print(f"ERROR: No 15m data for {symbol}")
        return
    
    print(f"    Loaded {len(df_15m)} bars from {df_15m.index[0]} to {df_15m.index[-1]}")
    
    # Fetch daily bars for SMA
    print("\n[2] Fetching daily bars for SMA-50...")
    bars_daily, _ = fetch_latest_bars([symbol], "1Day", history_days=60, feed="iex", verbose=False)
    df_daily = bars_daily.get(symbol)
    
    if df_daily is None or df_daily.empty:
        print(f"ERROR: No daily data for {symbol}")
        return
    
    print(f"    Loaded {len(df_daily)} daily bars")
    
    # Calculate SMA-50
    df_daily["sma_50"] = df_daily["close"].rolling(50).mean()
    latest_daily = df_daily.iloc[-1]
    sma_50 = float(latest_daily["sma_50"])
    latest_close = float(latest_daily["close"])
    
    print(f"\n[3] SMA-50 Trend Filter:")
    print(f"    Latest Daily Close: ${latest_close:.2f}")
    print(f"    SMA-50:             ${sma_50:.2f}")
    print(f"    Trend:              {'BULL ✅' if latest_close >= sma_50 else 'BEAR ❌'}")
    
    # Calculate Keltner Channels and RSI on 15m data
    print("\n[4] Adding indicators to 15m data...")
    df_15m = ta_add_atr(df_15m, length=14)
    df_15m = ta_add_rsi(df_15m, length=14)
    df_15m = ta_add_keltner(df_15m, length=20, mult=2.0)
    
    # Filter to Dec 24 only
    df_dec24 = df_15m[df_15m.index.date == datetime(2025, 12, 24).date()]
    print(f"    Dec 24 bars: {len(df_dec24)}")
    
    # Check for RSI oversold bars
    print("\n[5] Checking for RSI < 30 (oversold) bars on Dec 24:")
    oversold = df_dec24[df_dec24["rsi"] < 30]
    if oversold.empty:
        print("    ❌ No RSI oversold bars found on Dec 24")
    else:
        for idx, row in oversold.iterrows():
            print(f"    ✅ {idx} | RSI={row['rsi']:.1f} | Close=${row['close']:.2f} | KC_Lower=${row['kc_lower']:.2f}")
    
    # Check slope at each potential entry point
    print("\n[6] Slope calculation at oversold bars:")
    for i, (idx, row) in enumerate(oversold.iterrows()):
        loc = df_15m.index.get_loc(idx)
        if loc >= 4:
            slope_3 = (df_15m["kc_middle"].iloc[loc-1] - df_15m["kc_middle"].iloc[loc-4]) / 3
            slope_3_pct = (slope_3 / float(row["open"])) * 100
            print(f"    {idx} | Slope3_pct={slope_3_pct:.4f}% | Threshold=-0.08% | {'PASS ✅' if slope_3_pct < -0.08 else 'FAIL ❌'}")
    
    # Check what the live bot saw
    print("\n[7] Live Bot Entry Details:")
    print("    Symbol: TSLA")
    print("    Long:   TSLA260102C00477500 ($477.50 Call)")
    print("    Spread Width: $2.50")
    print("    Net Debit: $1.11")
    print("    Entry Time: ~10:31 AM ET")
    
    # Cross-reference   
    entry_time = "2025-12-24 10:31"
    print(f"\n[8] Finding bar closest to live entry ({entry_time})...")
    df_15m_str = df_15m.copy()
    df_15m_str["time_str"] = df_15m_str.index.strftime("%Y-%m-%d %H:%M")
    
    for idx, row in df_15m.iterrows():
        time_str = idx.strftime("%Y-%m-%d %H:%M")
        if "2025-12-24 10" in time_str or "2025-12-24 15:3" in str(idx):
            rsi = row.get("rsi", 0)
            close = row.get("close", 0)
            kc_lower = row.get("kc_lower", 0)
            loc = df_15m.index.get_loc(idx)
            slope_3_pct = 0
            if loc >= 4:
                slope_3 = (df_15m["kc_middle"].iloc[loc-1] - df_15m["kc_middle"].iloc[loc-4]) / 3
                slope_3_pct = (slope_3 / float(row["open"])) * 100 if row["open"] > 0 else 0
            print(f"    {idx} | RSI={rsi:.1f} | Close=${close:.2f} | KC_Lower=${kc_lower:.2f} | Slope_pct={slope_3_pct:.4f}")
    
    print(f"\n{'='*60}")
    print(" END DEBUG")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
