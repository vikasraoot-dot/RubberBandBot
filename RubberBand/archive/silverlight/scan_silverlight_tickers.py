#!/usr/bin/env python3
"""
Silver Light Ticker Scanner
============================
Scans a universe of tickers to find the best candidates for the Silver Light strategy.

Criteria for good candidates:
1. Strong trend-following characteristics (above 50 SMA most of the time)
2. Liquid (adequate daily volume)
3. Good risk/reward when applying the strategy
4. Not too correlated with TQQQ/SQQQ (diversification)

Usage:
    python RubberBand/scripts/scan_silverlight_tickers.py --input tickers_full_list.txt --top 10
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

# --- Ensure repo root on sys.path ---
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.data import fetch_latest_bars


def load_tickers(path: str) -> List[str]:
    """Load tickers from file, filter to valid symbols."""
    with open(path, "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]
    
    # Filter out warrants, units, preferred shares, etc.
    valid = []
    for t in tickers:
        if len(t) > 5:  # Skip long symbols (warrants, etc.)
            continue
        if any(c in t for c in ["W", "U", "R", "P"]) and len(t) > 4:
            continue
        valid.append(t)
    
    return valid


def compute_trend_score(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute trend-following score for a ticker.
    
    Returns dict with:
    - trend_score: % of time price was above 50 SMA
    - avg_volume: Average daily dollar volume
    - volatility: Annualized volatility
    - max_drawdown: Maximum drawdown
    - return_1yr: 1-year return
    """
    if df is None or df.empty or len(df) < 200:
        return None
    
    # Calculate indicators
    df = df.copy()
    df["sma_50"] = df["close"].rolling(window=50, min_periods=50).mean()
    df["sma_200"] = df["close"].rolling(window=200, min_periods=200).mean()
    
    # Skip warmup
    df = df.iloc[200:]
    if len(df) < 50:
        return None
    
    # Trend score: % of time above 50 SMA
    above_sma = (df["close"] > df["sma_50"]).sum() / len(df)
    
    # Average dollar volume
    df["dollar_vol"] = df["close"] * df["volume"]
    avg_volume = df["dollar_vol"].mean()
    
    # Volatility (annualized)
    df["daily_return"] = df["close"].pct_change()
    volatility = df["daily_return"].std() * np.sqrt(252)
    
    # Max drawdown
    df["peak"] = df["close"].cummax()
    df["drawdown"] = (df["close"] - df["peak"]) / df["peak"]
    max_drawdown = abs(df["drawdown"].min())
    
    # 1-year return
    if len(df) >= 252:
        return_1yr = (df["close"].iloc[-1] / df["close"].iloc[-252] - 1)
    else:
        return_1yr = (df["close"].iloc[-1] / df["close"].iloc[0] - 1)
    
    # Trend consistency: how often above 200 SMA (bull market indicator)
    above_200 = (df["close"] > df["sma_200"]).sum() / len(df)
    
    return {
        "trend_score": round(above_sma * 100, 1),
        "trend_200": round(above_200 * 100, 1),
        "avg_volume": round(avg_volume / 1e6, 2),  # In millions
        "volatility": round(volatility * 100, 1),
        "max_drawdown": round(max_drawdown * 100, 1),
        "return_1yr": round(return_1yr * 100, 1),
    }


def scan_tickers(tickers: List[str], batch_size: int = 50) -> List[Dict[str, Any]]:
    """Scan tickers in batches and compute scores."""
    results = []
    
    # Filter to reasonable list
    print(f"Scanning {len(tickers)} tickers...")
    
    # Fetch data in batches
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"  Batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}: {batch[:3]}...")
        
        try:
            bars_map, _ = fetch_latest_bars(
                symbols=batch,
                timeframe="1Day",
                history_days=500,  # ~2 years
                feed="iex",
                rth_only=False,
                verbose=False
            )
            
            for symbol, df in bars_map.items():
                if df is not None and not df.empty:
                    score = compute_trend_score(df)
                    if score:
                        score["symbol"] = symbol
                        results.append(score)
        except Exception as e:
            print(f"  Error fetching batch: {e}")
            continue
    
    return results


def rank_candidates(results: List[Dict[str, Any]], top_n: int = 10) -> pd.DataFrame:
    """Rank candidates by Silver Light suitability."""
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Filter criteria
    # 1. Minimum volume ($1M/day)
    df = df[df["avg_volume"] >= 1.0]
    
    # 2. Trend-following (above 50 SMA > 50% of time)
    df = df[df["trend_score"] >= 50]
    
    # 3. Not too volatile (< 100% annual volatility)
    df = df[df["volatility"] <= 100]
    
    # 4. Positive 1-year return
    df = df[df["return_1yr"] > 0]
    
    if df.empty:
        return df
    
    # Composite score:
    # - Higher trend_score = better
    # - Lower volatility = better
    # - Lower max_drawdown = better
    # - Higher return = better
    df["composite"] = (
        df["trend_score"] * 0.3 +
        df["return_1yr"] * 0.3 +
        (100 - df["volatility"]) * 0.2 +
        (100 - df["max_drawdown"]) * 0.2
    )
    
    # Sort by composite score
    df = df.sort_values("composite", ascending=False)
    
    return df.head(top_n)


def main():
    parser = argparse.ArgumentParser(description="Silver Light Ticker Scanner")
    parser.add_argument("--input", default="tickers_full_list.txt", help="Input ticker file")
    parser.add_argument("--top", type=int, default=10, help="Number of top candidates")
    parser.add_argument("--max-scan", type=int, default=200, help="Max tickers to scan (first N)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("SILVER LIGHT TICKER SCANNER")
    print("=" * 60)
    
    # Load tickers
    tickers = load_tickers(args.input)
    print(f"Loaded {len(tickers)} valid tickers")
    
    # Limit scan size for speed
    if len(tickers) > args.max_scan:
        print(f"Limiting to first {args.max_scan} tickers for speed")
        tickers = tickers[:args.max_scan]
    
    # Scan
    results = scan_tickers(tickers)
    print(f"\nSuccessfully analyzed {len(results)} tickers")
    
    # Rank
    top = rank_candidates(results, args.top)
    
    if top.empty:
        print("No suitable candidates found")
        return
    
    # Display results
    print("\n" + "=" * 60)
    print(f"TOP {len(top)} SILVER LIGHT CANDIDATES")
    print("=" * 60)
    
    print(f"\n{'Symbol':<8} {'Trend%':<8} {'Vol$M':<8} {'Volatility':<10} {'MaxDD':<8} {'1Y Ret':<8} {'Score':<8}")
    print("-" * 66)
    
    for _, row in top.iterrows():
        print(f"{row['symbol']:<8} {row['trend_score']:<8.1f} {row['avg_volume']:<8.1f} {row['volatility']:<10.1f} {row['max_drawdown']:<8.1f} {row['return_1yr']:<8.1f} {row['composite']:<8.1f}")
    
    # Save results
    output_file = "silverlight_candidates.csv"
    top.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    print("\n" + "=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
