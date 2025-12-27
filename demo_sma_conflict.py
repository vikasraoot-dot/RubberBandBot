
from __future__ import annotations
import os
import sys
import pandas as pd

_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.utils import load_config
from RubberBand.src.data import fetch_latest_bars

def check_sma_conflict():
    cfg = load_config("RubberBand/config.yaml")
    
    # 1. Fetch Daily Data for SMA20
    # SMA20 means 20 DAYS.
    key = os.getenv("APCA_API_KEY_ID", "")
    secret = os.getenv("APCA_API_SECRET_KEY", "")
    feed = cfg.get("feed", "iex")
    
    daily_map, _ = fetch_latest_bars(["IBM"], "1Day", 40, feed, False, key=key, secret=secret)
    df_daily = daily_map.get("IBM")
    
    # Calculate SMA20
    sma20 = df_daily["close"].rolling(20).mean().iloc[-1]
    sma200 = df_daily["close"].rolling(200).mean().iloc[-1] # if we had enough data
    
    # 2. Fetch 15m Data for Signal
    bars_map, _ = fetch_latest_bars(["IBM"], "15Min", 2, feed, True, key=key, secret=secret)
    df_15m = bars_map.get("IBM")
    
    # Get 15:15 bar
    target = df_15m[df_15m.index.strftime('%Y-%m-%d %H:%M').str.contains('2025-12-18 15:15')]
    
    if not target.empty:
        row = target.iloc[0]
        close = row["close"]
        
        print("\n=== IBM CONFLICT DEMO (Dec 18 15:15) ===")
        print(f"Price (Close): ${close:.2f}")
        print(f"SMA20 (Trend): ${sma20:.2f}")
        print(f"Result: {close} < {sma20} -> {close < sma20}")
        if close < sma20:
            print("VERDICT: REJECTED because Price is below SMA20 (Bear Trend)")
        else:
            print("VERDICT: ACCEPTED")
            
        print("\nExplanation:")
        print("To satisfy RSI < 25, Price MUST crash hard.")
        print("When Price crashes hard, it usually falls BELOW the short-term average (SMA20).")
        print("Therefore, requiring Price > SMA20 creates a 'Catch-22':")
        print("   'You must crash (to get signal) ... but don't crash below the average (to pass filter)!'")
    else:
        print("Could not find 15:15 bar.")

if __name__ == "__main__":
    check_sma_conflict()
