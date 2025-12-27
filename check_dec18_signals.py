
from __future__ import annotations
import os
import sys
import pandas as pd
from collections import Counter

# Setup Path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.utils import load_config, read_tickers
from RubberBand.src.data import fetch_latest_bars
from RubberBand.strategy import attach_verifiers

def check_dec18(df, cfg, sym):
    if df.empty: return []

    try:
        df = attach_verifiers(df, cfg).copy()
    except:
        return []

    valid_signals = []
    
    # Filter for Dec 18 only
    # Assuming UTC
    target_date = pd.Timestamp("2025-12-18").date()
    
    # Iterate
    for i in range(11, len(df)):
        cur = df.iloc[i]
        chk_date = cur.name.date()
        
        if chk_date != target_date:
            continue
            
        # 1. RSI / Signal Check
        rsi = cur.get("rsi", 100)
        kc_lower = cur.get("kc_lower", 0)
        rsi_threshold = cfg.get("filters", {}).get("rsi_oversold", 30)
        
        if rsi < rsi_threshold and cur["close"] < kc_lower:
            # 2. Trend Check
            trend_sma = cur.get("trend_sma", float('nan'))
            is_bull_trend = False
            if not pd.isna(trend_sma) and cur["close"] > trend_sma:
                is_bull_trend = True
            
            trend_enabled = cfg.get("trend_filter", {}).get("enabled", True)
            if trend_enabled and not is_bull_trend:
                continue 
            
            # 3. Slope Check
            slope_threshold = cfg.get("slope_threshold", -0.20)
            slope_threshold_10 = cfg.get("slope_threshold_10", -0.15)
            
            slope3 = (df["kc_middle"].iloc[i-1] - df["kc_middle"].iloc[i-4]) / 3
            slope10 = (df["kc_middle"].iloc[i-1] - df["kc_middle"].iloc[i-11]) / 10
            
            if slope3 > slope_threshold or slope10 > slope_threshold_10:
                continue 
            
            # PASSED
            valid_signals.append(cur.name)

    return valid_signals

def main():
    cfg = load_config("RubberBand/config.yaml")
    tickers = read_tickers("RubberBand/tickers.txt")
    extras = ["FIX", "CLH"]
    for e in extras:
        if e not in tickers: tickers.append(e)

    print(f"Checking {len(tickers)} tickers for Dec 18 signals...")
    
    from RubberBand.src.data import fetch_latest_bars
    
    key = os.getenv("APCA_API_KEY_ID", "")
    secret = os.getenv("APCA_API_SECRET_KEY", "")
    feed = cfg.get("feed", "iex")
    
    results = {}
    
    for sym in tickers:
        try:
            # Load enough data
            bars_map, _ = fetch_latest_bars([sym], "15Min", 3, feed, True, key=key, secret=secret, verbose=False)
            df = bars_map.get(sym, pd.DataFrame())
            
            daily_map, _ = fetch_latest_bars([sym], "1Day", 400, feed, False, key=key, secret=secret, verbose=False)
            df_daily = daily_map.get(sym, pd.DataFrame())
            
            if not df.empty and not df_daily.empty:
                df_daily["trend_sma"] = df_daily["close"].rolling(200).mean().shift(1)
                df_daily["date_only"] = df_daily.index.date
                df["date_only"] = df.index.date
                sma_map = df_daily.set_index("date_only")["trend_sma"].to_dict()
                df["trend_sma"] = df["date_only"].map(sma_map)
                
                sigs = check_dec18(df, cfg, sym)
                if sigs:
                    results[sym] = sigs
                
        except Exception:
            continue

    print("\n=== CONFIRMED DEC 18 SIGNALS ===")
    if not results:
        print("None.")
    for sym, times in results.items():
        print(f"{sym}: {len(times)} signals ({[t.strftime('%H:%M') for t in times]})")

if __name__ == "__main__":
    main()
