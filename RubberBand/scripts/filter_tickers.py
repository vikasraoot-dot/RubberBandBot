
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RubberBand.src.data import fetch_latest_bars
from RubberBand.src.indicators import ta_add_atr

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
INPUT_FILE = "tickers_full_list.txt"
OUTPUT_FILE = "tickers_volatile.csv"
MIN_PRICE = 5.0
MIN_DOLLAR_VOL = 10_000_000  # $10M daily liquidity
HISTORY_DAYS = 14
TOP_N = 50

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    # 1. Load Tickers
    try:
        with open(INPUT_FILE, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loaded {len(tickers)} tickers from {INPUT_FILE}")

    # 2. Check Existing Progress
    processed_symbols = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            processed_symbols = set(existing_df["symbol"].unique())
            print(f"Resuming... {len(processed_symbols)} tickers already processed.")
        except:
            pass
            
    tickers_to_process = [t for t in tickers if t not in processed_symbols]
    print(f"Remaining tickers to process: {len(tickers_to_process)}")

    # 3. Fetch Data in Chunks
    CHUNK_SIZE = 100 # Smaller chunks for more frequent saves
    total_chunks = (len(tickers_to_process) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    for i in range(0, len(tickers_to_process), CHUNK_SIZE):
        chunk = tickers_to_process[i : i + CHUNK_SIZE]
        print(f"Processing chunk {i//CHUNK_SIZE + 1}/{total_chunks} ({len(chunk)} tickers)...")
        
        chunk_results = []
        try:
            # Fetch data
            data_map, _ = fetch_latest_bars(chunk, history_days=HISTORY_DAYS)
            
            for symbol, df in data_map.items():
                if df.empty or len(df) < 5:
                    continue
                
                # Calculate Metrics
                last_close = df["close"].iloc[-1]
                if last_close < MIN_PRICE:
                    continue
                
                df["dollar_vol"] = df["close"] * df["volume"]
                avg_dollar_vol = df["dollar_vol"].mean()
                
                if avg_dollar_vol < MIN_DOLLAR_VOL:
                    continue
                
                df = ta_add_atr(df, length=14)
                last_atr = df["atr"].iloc[-1]
                atr_pct = (last_atr / last_close) * 100
                
                chunk_results.append({
                    "symbol": symbol,
                    "price": last_close,
                    "atr_pct": atr_pct,
                    "dollar_vol": avg_dollar_vol
                })
                
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
        
        # Save Chunk
        if chunk_results:
            df_chunk = pd.DataFrame(chunk_results)
            # Append to CSV
            write_header = not os.path.exists(OUTPUT_FILE)
            df_chunk.to_csv(OUTPUT_FILE, mode='a', header=write_header, index=False)
            print(f"Saved {len(chunk_results)} tickers to {OUTPUT_FILE}")

    # 4. Final Rank and Report
    if os.path.exists(OUTPUT_FILE):
        df_res = pd.read_csv(OUTPUT_FILE)
        df_res = df_res.sort_values("atr_pct", ascending=False)
        top_n = df_res.head(TOP_N)
        
        print(f"\n=== TOP {TOP_N} VOLATILE LIQUID STOCKS ===")
        print(f"{'Symbol':<10} {'Price':<10} {'ATR%':<10} {'Avg $Vol (M)':<15}")
        for _, row in top_n.iterrows():
            print(f"{row['symbol']:<10} {row['price']:<10.2f} {row['atr_pct']:<10.2f} {row['dollar_vol']/1_000_000:<15.2f}")
            
        # Save list
        with open("tickers_volatile_list.txt", "w") as f:
            for sym in top_n["symbol"]:
                f.write(f"{sym}\n")
        print("Top symbols saved to tickers_volatile_list.txt")

if __name__ == "__main__":
    main()
