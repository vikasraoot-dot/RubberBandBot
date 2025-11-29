import os
import sys
import argparse
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RubberBand.src.data import fetch_latest_bars
from RubberBand.src.utils import load_config

def _broker_creds(cfg):
    base_url = cfg.get("broker", {}).get("base_url") or os.getenv("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets"
    key = cfg.get("broker", {}).get("key") or os.getenv("APCA_API_KEY_ID")
    secret = cfg.get("broker", {}).get("secret") or os.getenv("APCA_API_SECRET_KEY")
    return base_url, key, secret

def get_all_assets(cfg):
    """Fetch all active, tradeable US equity assets from Alpaca."""
    import requests
    base_url, key, secret = _broker_creds(cfg)
    if not key or not secret:
        print("Error: API keys not found in config or env.")
        return []
    
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret
    }
    
    print("Fetching all assets from Alpaca...")
    r = requests.get(f"{base_url}/v2/assets?status=active&asset_class=us_equity", headers=headers)
    if r.status_code != 200:
        print(f"Error fetching assets: {r.text}")
        return []
    
    assets = r.json()
    # Filter for tradeable and marginable (optional)
    symbols = [a["symbol"] for a in assets if a.get("tradable") and a.get("status") == "active"]
    print(f"Found {len(symbols)} total assets.")
    return symbols

def calculate_metrics(df, benchmark_df):
    """Calculate Beta, ATR%, and Avg Dollar Volume."""
    if df.empty or len(df) < 60:
        return None
    
    # Align dates
    df = df.copy()
    df["pct_change"] = df["close"].pct_change()
    
    # Dollar Volume (Avg 20)
    df["dollar_vol"] = df["close"] * df["volume"]
    avg_dollar_vol = df["dollar_vol"].rolling(20).mean().iloc[-1]
    
    # ATR% (14)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_pct = (atr / df["close"]) * 100
    current_atr_pct = atr_pct.iloc[-1]
    
    # Beta (vs Benchmark)
    # Join with benchmark on index (date)
    joined = df[["pct_change"]].join(benchmark_df[["pct_change"]], lsuffix="_stock", rsuffix="_bench").dropna()
    
    if len(joined) < 30:
        beta = 0.0
    else:
        cov = np.cov(joined["pct_change_stock"], joined["pct_change_bench"])[0, 1]
        var = np.var(joined["pct_change_bench"])
        beta = cov / var if var > 0 else 0.0
        
    return {
        "beta": beta,
        "atr_pct": current_atr_pct,
        "dollar_vol": avg_dollar_vol,
        "close": df["close"].iloc[-1]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="RubberBand/config.yaml")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--days", type=int, default=100)
    parser.add_argument("--min-vol", type=float, default=20_000_000) # $20M min
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-beta", type=float, default=1.5)
    parser.add_argument("--min-atr", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tickers to scan (for testing)")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    # 1. Get Universe
    all_symbols = get_all_assets(cfg)
    
    # Filter out obvious junk (optional: length check, exchange check if available)
    # For now, just take a slice if limit is set
    if args.limit > 0:
        import random
        random.shuffle(all_symbols)
        all_symbols = all_symbols[:args.limit]
        
    # Always include benchmark
    if args.benchmark not in all_symbols:
        all_symbols.append(args.benchmark)
        
    print(f"Scanning {len(all_symbols)} symbols...")
    
    # 2. Fetch Data (Daily)
    # We use fetch_latest_bars but with '1D' timeframe
    # Note: fetch_latest_bars might be optimized for 15m, let's check.
    # It uses 'timeframe' arg.
    
    # Fetch in chunks to avoid memory explosion/timeouts
    chunk_size = 200
    results = []
    
    # First, fetch benchmark
    print(f"Fetching benchmark {args.benchmark}...")
    bench_res = fetch_latest_bars([args.benchmark], "1Day", args.days, cfg.get("feed", "iex"), rth_only=False)
    if not bench_res or not bench_res[0] or args.benchmark not in bench_res[0]:
        print(f"Failed to fetch benchmark data. Result: {bench_res}")
        return
    
    bench_df = bench_res[0][args.benchmark]
    bench_df["pct_change"] = bench_df["close"].pct_change()
    
    # Chunk loop
    for i in range(0, len(all_symbols), chunk_size):
        chunk = all_symbols[i:i+chunk_size]
        print(f"Processing chunk {i} to {i+len(chunk)}...")
        
        try:
            data_tuple = fetch_latest_bars(chunk, "1Day", args.days, cfg.get("feed", "iex"), rth_only=False)
            if not data_tuple:
                continue
            bars_map, _ = data_tuple
            
            for sym, df in bars_map.items():
                if sym == args.benchmark: continue
                if df.empty: continue
                
                # Pre-filter by price
                if df["close"].iloc[-1] < args.min_price:
                    continue
                
                metrics = calculate_metrics(df, bench_df)
                if not metrics:
                    continue
                
                # Filter Logic
                if (metrics["dollar_vol"] >= args.min_vol and
                    metrics["beta"] >= args.min_beta and
                    metrics["atr_pct"] >= args.min_atr):
                    
                    res_row = {"symbol": sym, **metrics}
                    results.append(res_row)
                    # print(f"  FOUND: {sym} Beta={metrics['beta']:.2f} ATR={metrics['atr_pct']:.1f}% Vol=${metrics['dollar_vol']/1M:.1f}M")
                    
        except Exception as e:
            print(f"Error in chunk: {e}")
            continue

    # 3. Sort and Save
    results.sort(key=lambda x: x["beta"], reverse=True)
    
    print(f"\nScan Complete. Found {len(results)} candidates.")
    
    # Save to CSV
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        print(df_res.head(20))
        df_res.to_csv("candidates.csv", index=False)
        print("Saved to candidates.csv")
        
        # Save simple list for backtester
        with open("candidates.txt", "w") as f:
            for sym in df_res["symbol"]:
                f.write(f"{sym}\n")
        print("Saved list to candidates.txt")
    else:
        print("No candidates found.")

if __name__ == "__main__":
    main()
