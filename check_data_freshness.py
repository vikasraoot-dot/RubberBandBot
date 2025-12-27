
import os
import sys
import pandas as pd
# Ensure repo root is in path
sys.path.insert(0, os.path.abspath("."))
from RubberBand.src.data import fetch_latest_bars

def check_freshness(symbol):
    print(f"Checking data for {symbol}...")
    # Mimic backtest call
    bars_map, _ = fetch_latest_bars(
        symbols=[symbol],
        timeframe="15Min",
        history_days=5,
        feed="iex", # or whatever config uses, usually 'iex' or 'sip'
        verbose=True
    )
    
    if symbol in bars_map:
        df = bars_map[symbol]
        if not df.empty:
            print(f"Latest 5 bars for {symbol}:")
            print(df.index[-5:])
            print(f"Last bar timestamp: {df.index[-1]}")
        else:
            print(f"No data found for {symbol}")
    else:
        print(f"Symbol {symbol} not returned in map")

if __name__ == "__main__":
    check_freshness("FIX")
