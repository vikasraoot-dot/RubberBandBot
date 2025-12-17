
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import sys
import os

# Ensure we can import from RubberBand
sys.path.append(os.getcwd())

from RubberBand.src.data import fetch_latest_bars
from RubberBand.src.indicators import ta_add_keltner, ta_add_adx_di, ta_add_rsi, ta_add_atr

def check_indicators():
    symbols = ["CRM", "COP", "KKR"]
    # 12/15 was a Monday. We need data leading up to 9:56 ET (14:56 UTC).
    # We'll fetch 5 days of history to ensure indicators warm up.
    
    print(f"Fetching data for {symbols}...")
    bars_map, _ = fetch_latest_bars(
        symbols=symbols,
        timeframe="15Min",
        history_days=5,
        feed="iex", # or sip if available, iex is fine
        verbose=False
    )
    
    for sym in symbols:
        df = bars_map.get(sym)
        if df is None or df.empty:
            print(f"No data for {sym}")
            continue
            
        # Add indicators
        # Config used in live bot: Keltner(20, 2.0), ATR(14)
        df = ta_add_atr(df, length=14)
        df = ta_add_keltner(df, length=20, mult=2.0)
        df = ta_add_adx_di(df, period=14)
        df = ta_add_rsi(df, length=14)
        
        # Calculate Slope (3-bar diff of kc_middle / 3) roughly matches bot
        # Bot logic: (current - prev3) / 3
        df["slope"] = (df["kc_middle"] - df["kc_middle"].shift(3)) / 3
        
        # Filter for the specific time range of the crash (approx 14:00 UTC to 16:00 UTC on 12/15)
        # 12/15 09:30 ET is 14:30 Z.
        # Target: 12/15
        
        # Filter by index date
        day_df = df[df.index.strftime('%Y-%m-%d') == '2025-12-15'].copy()
        
        print(f"\n--- {sym} Analysis (12/15) ---")
        # Show 9:45, 10:00, 10:15 ET (approx signals were 9:56, 10:11)
        # Times in index are UTC. 14:45, 15:00, 15:15.
        
        target_times = ["14:45", "15:00", "15:15", "15:30"]
        
        print(f"{'Time(UTC)':<10} {'Close':<8} {'RSI':<6} {'Slope':<8} {'ADX':<6} {'+DI':<6} {'-DI':<6} {'Result'}")
        
        for idx in day_df.index:
            t_str = idx.strftime('%H:%M')
            if t_str in target_times:
                row = day_df.loc[idx]
                
                # Replicate logic
                slope_fail = row["slope"] < -0.20
                adx_filter = (row["ADX"] > 25) and (row["-DI"] > row["+DI"])
                
                res = "OPEN"
                if slope_fail: res = "BLOCKED(Slope)"
                elif adx_filter: res = "BLOCKED(ADX)"
                
                print(f"{t_str:<10} {row['close']:<8.2f} {row['rsi']:<6.1f} {row['slope']:<8.4f} {row['ADX']:<6.1f} {row['+DI']:<6.1f} {row['-DI']:<6.1f} {res}")

if __name__ == "__main__":
    check_indicators()
