#!/usr/bin/env python3
"""
Weekly Strategy Candidate Scanner

Scans the full ticker universe for stocks matching the "Elite DNA" criteria:
1. High Volatility: ATR% > 4.0%
2. High Liquidity: Avg Dollar Volume > $50M
3. Price: > $40
4. Trend Filter: High volatility tickers (ATR>8%) must be in uptrend

Usage:
    python RubberBand/scripts/scan_weekly_candidates.py --tickers tickers_full_list.txt --output weekly_candidates.csv
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Ensure repo root is on path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.data import fetch_latest_bars, load_symbols_from_file


def calculate_weekly_indicators(df: pd.DataFrame) -> dict:
    """Calculate ATR%, Dollar Vol, and Trend for weekly strategy screening."""
    if len(df) < 50:  # Need at least 50 days for SMA50
        return {}

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ATR 14
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14).mean().iloc[-1]

    # Dollar Volume (20 day avg)
    dollar_vol = (close * volume).rolling(window=20).mean().iloc[-1]

    current_price = close.iloc[-1]
    
    # Trend: SMA50 vs SMA200 (or just SMA50 slope if not enough data)
    sma_50 = close.rolling(window=50).mean().iloc[-1]
    if len(df) >= 200:
        sma_200 = close.rolling(window=200).mean().iloc[-1]
        trend = 1 if sma_50 > sma_200 else 0
    else:
        # Fallback: is price above SMA50?
        trend = 1 if current_price > sma_50 else 0
    
    return {
        "price": current_price,
        "atr_14": atr_14,
        "atr_pct": (atr_14 / current_price) * 100.0 if current_price > 0 else 0.0,
        "dollar_vol_m": dollar_vol / 1_000_000,  # In millions
        "trend": trend  # 1 = bullish, 0 = bearish/sideways
    }


def main():
    parser = argparse.ArgumentParser(description="Scan tickers for Weekly RubberBand candidates.")
    parser.add_argument("--tickers", default="tickers_full_list.txt", help="Path to ticker list")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tickers to scan (for testing)")
    parser.add_argument("--output", default="RubberBand/tickers_weekly_candidates.txt", help="Output file")
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
        print(f"Limited to first {args.limit} tickers")

    # Load current elite list to identify NEW candidates
    elite_path = os.path.join(_REPO_ROOT, "RubberBand/tickers_weekly.txt")
    current_elite = set()
    if os.path.exists(elite_path):
        current_elite = set(load_symbols_from_file(elite_path))
        print(f"Current elite list has {len(current_elite)} tickers")

    print("="*80)
    print("WEEKLY STRATEGY CANDIDATE SCANNER")
    print("="*80)
    print("Looking for 'Elite DNA': ATR > 4.0%, Avg Vol > $50M, Price > $40")

    # Fetch data in batches (Alpaca has rate limits)
    BATCH_SIZE = 50
    candidates = []
    
    # Diagnostic counters
    stats = {
        "total_tickers": len(tickers),
        "tickers_with_data": 0,
        "tickers_with_indicators": 0,
        "passed_price_filter": 0,
        "passed_atr_filter": 0,
        "passed_volume_filter": 0,
        "filtered_by_trend": 0,
        "passed_trend_filter": 0,
    }

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i+BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(tickers) + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch)} tickers)...")

        try:
            bars_map, meta = fetch_latest_bars(
                symbols=batch,
                timeframe="1Day",  # Daily bars for scanning
                history_days=100,  # 100 calendar days = ~70 trading days (need 50 for SMA50)
                feed="sip",
                rth_only=False,    # CRITICAL: Must be False for daily bars
                verbose=False
            )
        except Exception as e:
            print(f"Error fetching batch: {e}")
            continue

        for sym in batch:
            try:
                df = bars_map.get(sym)
                if df is None or df.empty:
                    continue
                
                stats["tickers_with_data"] += 1

                indicators = calculate_weekly_indicators(df)
                if not indicators:
                    continue
                
                stats["tickers_with_indicators"] += 1

                price = indicators["price"]
                atr_pct = indicators["atr_pct"]
                dollar_vol_m = indicators["dollar_vol_m"]

                # Apply filters with tracking
                if price < 40:
                    continue
                stats["passed_price_filter"] += 1
                
                if atr_pct < 4.0:
                    continue
                stats["passed_atr_filter"] += 1
                
                if dollar_vol_m < 50:  # $50M (lowered from $500M)
                    continue
                stats["passed_volume_filter"] += 1
                
                # HIGH VOLATILITY TREND FILTER: If ATR > 8%, require bullish trend
                trend = indicators.get("trend", 1)
                if atr_pct > 8.0 and trend == 0:
                    stats["filtered_by_trend"] += 1
                    continue
                stats["passed_trend_filter"] += 1

                candidates.append({
                    "symbol": sym,
                    "price": round(price, 2),
                    "atr_pct": round(atr_pct, 2),
                    "dollar_vol_m": round(dollar_vol_m, 1),
                    "trend": trend,
                    "is_new": sym not in current_elite
                })

            except Exception as e:
                continue

    # Print diagnostic summary
    print(f"\n{'='*80}")
    print("DIAGNOSTIC SUMMARY:")
    print(f"  Total tickers scanned: {stats['total_tickers']}")
    print(f"  Tickers with data: {stats['tickers_with_data']}")
    print(f"  Tickers with valid indicators: {stats['tickers_with_indicators']}")
    print(f"  Passed price filter (>$40): {stats['passed_price_filter']}")
    print(f"  Passed ATR filter (>4%): {stats['passed_atr_filter']}")
    print(f"  Passed volume filter (>$50M): {stats['passed_volume_filter']}")
    print(f"  Filtered by trend (ATR>10% + bearish): {stats['filtered_by_trend']}")
    print(f"  FINAL CANDIDATES: {stats['passed_trend_filter']}")
    print(f"{'='*80}")

    # Results
    if not candidates:
        print("\nNo candidates found matching Elite criteria.")
        return

    results_df = pd.DataFrame(candidates)
    results_df = results_df.sort_values("atr_pct", ascending=False)

    print(f"\n{'='*80}")
    print(f"FOUND {len(results_df)} ELITE CANDIDATES:")
    print(results_df.to_string(index=False))

    # Identify new candidates
    new_candidates = results_df[results_df["is_new"] == True]
    if not new_candidates.empty:
        print(f"\n>>> {len(new_candidates)} NEW CANDIDATES (not in current list) <<<")
        print(new_candidates[["symbol", "price", "atr_pct", "dollar_vol_m"]].to_string(index=False))

        # Save new candidates
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(_REPO_ROOT, output_path)
        
        with open(output_path, "w") as f:
            for sym in new_candidates["symbol"]:
                f.write(f"{sym}\n")
        print(f"\nSaved {len(new_candidates)} new candidates to {output_path}")
    else:
        print("\nNo NEW candidates found outside current elite list.")

    # Also save full CSV for reference
    csv_path = output_path.replace(".txt", ".csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved full results to {csv_path}")


if __name__ == "__main__":
    main()
