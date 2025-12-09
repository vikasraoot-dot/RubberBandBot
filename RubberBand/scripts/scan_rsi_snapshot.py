#!/usr/bin/env python3
"""
Quick RSI snapshot scan for all tickers.
Shows current RSI values to understand how close tickers got to the 25 threshold.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

from RubberBand.src.data import fetch_latest_bars, load_symbols_from_file

ET = ZoneInfo("US/Eastern")

def load_tickers(path: str) -> list:
    """Load tickers from file."""
    with open(path) as f:
        return [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]

def get_rsi_snapshot(symbols: list, timeframe: str = "15Min", rsi_period: int = 14) -> pd.DataFrame:
    """Fetch bars and calculate RSI for all symbols."""
    
    print(f"Fetching {timeframe} bars for {len(symbols)} symbols...")
    bars_map, meta = fetch_latest_bars(
        symbols=symbols,
        timeframe=timeframe,
        history_days=10,  # Need enough data for RSI calculation
        feed="iex",
        verbose=False
    )
    
    results = []
    for sym in symbols:
        if sym not in bars_map or bars_map[sym].empty:
            results.append({"symbol": sym, "rsi": None, "close": None, "status": "NO_DATA"})
            continue
        
        df = bars_map[sym].copy()
        if len(df) < rsi_period + 1:
            results.append({"symbol": sym, "rsi": None, "close": None, "status": "INSUFFICIENT_BARS"})
            continue
        
        # Calculate RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        current_close = df["close"].iloc[-1]
        
        # Determine status
        if current_rsi < 25:
            status = "OVERSOLD"
        elif current_rsi < 30:
            status = "NEAR_THRESHOLD"
        elif current_rsi > 70:
            status = "OVERBOUGHT"
        else:
            status = "NEUTRAL"
        
        results.append({
            "symbol": sym,
            "rsi": round(current_rsi, 1),
            "close": round(current_close, 2),
            "status": status
        })
    
    return pd.DataFrame(results)

def main():
    # Load tickers from the main ticker file
    ticker_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tickers.txt")
    if not os.path.exists(ticker_path):
        print(f"Ticker file not found: {ticker_path}")
        return
    
    symbols = load_tickers(ticker_path)
    print(f"Loaded {len(symbols)} tickers from {ticker_path}\n")
    
    # Get RSI snapshot
    df = get_rsi_snapshot(symbols)
    
    # Sort by RSI (lowest first - closest to oversold)
    df_valid = df[df["rsi"].notna()].sort_values("rsi")
    
    print("\n" + "="*70)
    print("RSI SNAPSHOT - Sorted by RSI (lowest first)")
    print("="*70)
    print(f"{'Symbol':<8} {'RSI':>8} {'Close':>10} {'Status':<20}")
    print("-"*70)
    
    for _, row in df_valid.iterrows():
        print(f"{row['symbol']:<8} {row['rsi']:>8.1f} {row['close']:>10.2f} {row['status']:<20}")
    
    # Summary stats
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    oversold = df_valid[df_valid["rsi"] < 25]
    near_threshold = df_valid[(df_valid["rsi"] >= 25) & (df_valid["rsi"] < 30)]
    overbought = df_valid[df_valid["rsi"] > 70]
    
    print(f"🟢 Oversold (RSI < 25):     {len(oversold)}")
    print(f"🟡 Near threshold (25-30): {len(near_threshold)}")
    print(f"🔴 Overbought (RSI > 70):  {len(overbought)}")
    print(f"⚪ Neutral (30-70):        {len(df_valid) - len(oversold) - len(near_threshold) - len(overbought)}")
    
    if len(oversold) > 0:
        print(f"\n✅ Oversold tickers: {', '.join(oversold['symbol'].tolist())}")
    else:
        print(f"\n❌ No oversold tickers (RSI < 25)")
        if len(near_threshold) > 0:
            print(f"   Closest to threshold: {near_threshold.iloc[0]['symbol']} (RSI={near_threshold.iloc[0]['rsi']})")

if __name__ == "__main__":
    main()
