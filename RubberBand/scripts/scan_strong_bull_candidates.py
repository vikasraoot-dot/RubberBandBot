#!/usr/bin/env python3
"""
Scan the ticker universe for "Strong Bull" candidates suitable for RubberBandBot.

Criteria:
1. Strong Bull: Price > SMA 120 AND Price > SMA 22
2. Liquidity: Avg Daily Dollar Volume > $2,000,000
3. Volatility: ATR% (ATR/Price) > 1.5%

Usage:
    python RubberBand/scripts/scan_candidates.py --tickers tickers_full_list.txt --limit 15
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ensure repo root is on path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.data import fetch_latest_bars, load_symbols_from_file

def calculate_indicators(df: pd.DataFrame) -> dict:
    """Calculate SMA 100, ATR 14, Dollar Vol."""
    if len(df) < 100:
        return {}

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # SMAs
    sma_100 = close.rolling(window=100).mean().iloc[-1]

    # ATR 14
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14).mean().iloc[-1]

    # Dollar Volume (20 day avg)
    dollar_vol = (close * volume).rolling(window=20).mean().iloc[-1]

    current_price = close.iloc[-1]
    
    return {
        "price": current_price,
        "sma_100": sma_100,
        "atr_14": atr_14,
        "atr_pct": (atr_14 / current_price) * 100.0 if current_price > 0 else 0.0,
        "dollar_vol": dollar_vol
    }

def main():
    parser = argparse.ArgumentParser(description="Scan tickers for RubberBandBot candidates.")
    parser.add_argument("--tickers", default="tickers_full_list.txt", help="Path to ticker list")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tickers to scan (for testing)")
    parser.add_argument("--output", default="candidates.csv", help="Output CSV file")
    args = parser.parse_args()

    # Load Tickers
    ticker_path = args.tickers
    if not os.path.isabs(ticker_path):
        ticker_path = os.path.join(_REPO_ROOT, ticker_path)
    
    if not os.path.exists(ticker_path):
        print(f"Error: Ticker file not found at {ticker_path}")
        return

    tickers = load_symbols_from_file(ticker_path)
    print(f"Loaded {len(tickers)} tickers from {ticker_path}")

    if args.limit > 0:
        tickers = tickers[:args.limit]
        print(f"Limiting scan to first {args.limit} tickers.")

    # DEBUG: Check for API Keys
    key = os.getenv("ALPACA_KEY_ID", "") or os.getenv("APCA_API_KEY_ID", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "") or os.getenv("APCA_API_SECRET_KEY", "")
    print(f"DEBUG: ALPACA_KEY_ID present? {bool(key)}")
    if key:
        print(f"DEBUG: Key starts with: {key[:4]}...")
    
    # Fetch Data (Daily Bars)
    # We need ~200 days to calculate SMA 120 safely with buffer
    print("Fetching daily bars...")
    bars_map, meta = fetch_latest_bars(
        symbols=tickers,
        timeframe="1Day",
        history_days=200, # 200 calendar days is sufficient for SMA 100
                          # SMA 100 needs 100 trading days (~145 calendar). 200 is plenty.
        feed="iex",       # Use IEX for free tier compatibility if needed, or sip if available
        rth_only=False,
        verbose=True
    )
    
    # DEBUG: Print Fetch Metadata (Errors)
    if meta.get("http_errors"):
        print(f"DEBUG: HTTP Errors: {meta['http_errors']}")
    if meta.get("stale_symbols"):
        print(f"DEBUG: Stale Symbols: {len(meta['stale_symbols'])}")

    results = []
    print(f"Processing {len(bars_map)} symbols...")

    for sym, df in bars_map.items():
        if df.empty:
            continue

        try:
            inds = calculate_indicators(df)
            if not inds:
                continue

            price = inds["price"]
            sma_100 = inds["sma_100"]
            dollar_vol = inds["dollar_vol"]
            atr_pct = inds["atr_pct"]

            # --- FILTERS ---
            # 1. Trend: Price > SMA 100 (aligned with config.yaml)
            # We REMOVE the SMA 22 check because we want to allow stocks that are dipping (Price < SMA 20).
            is_uptrend = (price > sma_100)

            # 2. Liquidity: > $2M
            is_liquid = (dollar_vol > 2_000_000)

            # 3. Volatility: ATR% > 1.5%
            is_volatile = (atr_pct > 1.5)

            if is_uptrend and is_liquid and is_volatile:
                results.append({
                    "symbol": sym,
                    "price": round(price, 2),
                    "sma_100": round(sma_100, 2),
                    "atr_pct": round(atr_pct, 2),
                    "dollar_vol_m": round(dollar_vol / 1_000_000, 2)
                })

        except Exception as e:
            print(f"Error processing {sym}: {e}")
            continue

    # Output
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values("dollar_vol_m", ascending=False)
        print(f"\nFound {len(df_res)} candidates matching criteria.")
        print(df_res.head(10).to_string(index=False))
        
        df_res.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")
    else:
        print("\nNo candidates found matching criteria.")

if __name__ == "__main__":
    main()
