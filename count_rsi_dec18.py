
from __future__ import annotations
import os
import sys
import pandas as pd

# Setup Path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.utils import load_config, read_tickers
from RubberBand.src.data import fetch_latest_bars
from RubberBand.strategy import attach_verifiers

def count_rsi(df, cfg, sym):
    if df.empty: return 0
    try:
        df = attach_verifiers(df, cfg).copy()
    except:
        return 0
    
    cnt = 0
    target = pd.Timestamp("2025-12-18").date()
    # Check RSI < 30
    cutoff = 30
    
    for i in range(len(df)):
        if df.iloc[i].name.date() == target:
            if df["rsi"].iloc[i] < cutoff:
                cnt += 1
                # print(f"{sym} {df.iloc[i].name} RSI={df['rsi'].iloc[i]:.1f}")
    return cnt

def main():
    cfg = load_config("RubberBand/config.yaml")
    tickers = read_tickers("RubberBand/tickers.txt")
    extras = ["FIX", "CLH"]
    for e in extras:
        if e not in tickers: tickers.append(e)

    print(f"Checking {len(tickers)} tickers for RSI < 30 on Dec 18...")
    
    from RubberBand.src.data import fetch_latest_bars
    key = os.getenv("APCA_API_KEY_ID", "")
    secret = os.getenv("APCA_API_SECRET_KEY", "")
    feed = cfg.get("feed", "iex")
    
    hits = {}
    
    for sym in tickers:
        try:
            bars_map, _ = fetch_latest_bars([sym], "15Min", 2, feed, True, key=key, secret=secret, verbose=False)
            df = bars_map.get(sym, pd.DataFrame())
            c = count_rsi(df, cfg, sym)
            if c > 0:
                hits[sym] = c
        except: continue
        
    print("\n=== RSI < 30 COUNTS (Dec 18) ===")
    if not hits:
        print("None.")
    for s, c in hits.items():
        print(f"{s}: {c}")

if __name__ == "__main__":
    main()
