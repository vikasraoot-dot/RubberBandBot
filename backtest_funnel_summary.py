
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

STATS = Counter()

def simulate_funnel_agg(df, cfg, sym):
    global STATS
    if df.empty: return

    try:
        df = attach_verifiers(df, cfg).copy()
    except:
        return

    # Focus on last 2 days (approx 78 bars)
    tail_len = min(len(df), 78) 
    df_run = df.tail(tail_len)

    for i in range(len(df_run)):
        cur = df_run.iloc[i]
        
        # 1. RSI / Signal Check
        rsi = cur.get("rsi", 100)
        kc_lower = cur.get("kc_lower", 0)
        rsi_threshold = cfg.get("filters", {}).get("rsi_oversold", 30)
        
        # Check basic conditions
        if rsi < rsi_threshold and cur["close"] < kc_lower:
            # 2. Trend Check
            trend_sma = cur.get("trend_sma", float('nan'))
            is_bull_trend = False
            if not pd.isna(trend_sma) and cur["close"] > trend_sma:
                is_bull_trend = True
            
            trend_enabled = cfg.get("trend_filter", {}).get("enabled", True)
            if trend_enabled and not is_bull_trend:
                continue # Rejected by Trend
            
            # 3. Slope Check
            # Need previous bars in original DF. 
            # Use index to find location in original DF
            orig_idx = df.index.get_loc(cur.name)
            if orig_idx < 11: continue

            slope_threshold = cfg.get("slope_threshold", -0.20)
            slope_threshold_10 = cfg.get("slope_threshold_10", -0.15)
            
            slope3 = (df["kc_middle"].iloc[orig_idx-1] - df["kc_middle"].iloc[orig_idx-4]) / 3
            slope10 = (df["kc_middle"].iloc[orig_idx-1] - df["kc_middle"].iloc[orig_idx-11]) / 10
            
            if slope3 > slope_threshold or slope10 > slope_threshold_10:
                continue # Rejected by Slope
            
            # PASSED
            STATS[sym] += 1
            # print(f"  {sym} pass")

def main():
    cfg = load_config("RubberBand/config.yaml")
    tickers = read_tickers("RubberBand/tickers.txt")
    extras = ["FIX", "CLH"]
    for e in extras:
        if e not in tickers: tickers.append(e)

    print(f"Scanning {len(tickers)} tickers for signals in last 2 days...")
    
    # Needs dummy load func from previous script or import
    # I'll just rely on `backtest.py` being in path if I could, but `load_bars` is local there.
    # Re-copying compact load logic for speed
    from RubberBand.src.data import fetch_latest_bars
    
    key = os.getenv("APCA_API_KEY_ID", "")
    secret = os.getenv("APCA_API_SECRET_KEY", "")
    feed = cfg.get("feed", "iex")
    
    # Batch loading for speed? No, robustness first.
    
    for sym in tickers:
        try:
            # 1. Load Daily for Trend
            # 2. Load 15Min for Signal
            # Doing minimal
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
                simulate_funnel_agg(df, cfg, sym)
                
        except Exception:
            continue

    print("\n=== Valid Signals Per Ticker (Dec 17-18) ===")
    for sym, count in STATS.most_common():
        print(f"{sym}: {count}")

if __name__ == "__main__":
    main()
