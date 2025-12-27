
from __future__ import annotations
import os
import sys
import pandas as pd

# Setup Path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.utils import load_config
from RubberBand.src.data import fetch_latest_bars
from RubberBand.strategy import attach_verifiers

def inspect(sym, cfg):
    try:
        from RubberBand.src.data import fetch_latest_bars
        key = os.getenv("APCA_API_KEY_ID", "")
        secret = os.getenv("APCA_API_SECRET_KEY", "")
        feed = cfg.get("feed", "iex")
        
        bars_map, _ = fetch_latest_bars([sym], "15Min", 3, feed, True, key=key, secret=secret, verbose=False)
        df = bars_map.get(sym, pd.DataFrame())
        
        daily_map, _ = fetch_latest_bars([sym], "1Day", 400, feed, False, key=key, secret=secret, verbose=False)
        df_daily = daily_map.get(sym, pd.DataFrame())
    
        if df.empty or df_daily.empty:
            print(f"{sym}: No Data")
            return

        df = attach_verifiers(df, cfg).copy()
        
        df_daily["trend_sma"] = df_daily["close"].rolling(200).mean().shift(1)
        df_daily["date_only"] = df_daily.index.date
        df["date_only"] = df.index.date
        sma_map = df_daily.set_index("date_only")["trend_sma"].to_dict()
        df["trend_sma"] = df["date_only"].map(sma_map)
        
        target = pd.Timestamp("2025-12-18").date()
        
        for i in range(11, len(df)):
            cur = df.iloc[i]
            if cur.name.date() != target: continue
            
            # Check RSI < 30 (Broad)
            if cur["rsi"] < 30:
                reason = "Unknown"
                
                # Check 1: Price < KC
                kc_pass = cur["close"] < cur["kc_lower"]
                
                # Check 2: Trend
                trend_val = cur.get("trend_sma", float('nan'))
                trend_pass = False
                if not pd.isna(trend_val) and cur["close"] > trend_val:
                    trend_pass = True
                    
                # Check 3: Slope
                slope3 = (df["kc_middle"].iloc[i-1] - df["kc_middle"].iloc[i-4]) / 3
                slope10 = (df["kc_middle"].iloc[i-1] - df["kc_middle"].iloc[i-11]) / 10
                slope_thresh = -0.20
                slope_pass = True
                if slope3 > slope_thresh: slope_pass = False
                
                print(f"{sym} {cur.name.strftime('%H:%M')} | RSI={cur['rsi']:.1f} | <KC? {kc_pass} | Trend? {trend_pass} (SMA={trend_val:.1f} C={cur['close']:.1f}) | Slope? {slope_pass} ({slope3:.3f})")

    except Exception as e:
        print(f"{sym} Error: {e}")

def main():
    cfg = load_config("RubberBand/config.yaml")
    targets = ["NFLX", "CME", "RBLX", "KMI", "TJX", "BMY", "IBM"]
    print("Inspection Results (Dec 18):")
    for t in targets:
        inspect(t, cfg)

if __name__ == "__main__":
    main()
