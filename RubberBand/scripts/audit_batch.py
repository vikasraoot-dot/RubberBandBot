
import sys
import os
import argparse
import pandas as pd
import numpy as np
import time

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import existing modules
from RubberBand.src.data import fetch_latest_bars
from RubberBand.scripts.audit_framework import run_simulation, get_ma_slope

def analyze_portfolio(tickers_file):
    print(f"=== Starting Batch Slope Analysis ===")
    
    # Load Tickers
    with open(tickers_file, 'r') as f:
        tickers = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]
    
    print(f"Loaded {len(tickers)} tickers from {tickers_file}")
    
    # Master Store
    all_slopes = []
    all_returns = []
    all_drawdowns = []
    
    # Metrics
    LOOK_FORWARD = 12 # 3 hours
    
    # Batch Fetching (to be efficient, but fetch_latest_bars handles lists)
    # However, fetching 120 tickers at once might be heavy. Let's do chunks of 20.
    CHUNK_SIZE = 20
    
    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i:i+CHUNK_SIZE]
        print(f"Processing chunk {i+1}-{min(i+CHUNK_SIZE, len(tickers))}...")
        
        try:
            bars_map, _ = fetch_latest_bars(
                symbols=chunk, 
                timeframe="15Min", 
                history_days=60, 
                verbose=False
            )
        except Exception as e:
            print(f"  Error fetching chunk: {e}")
            continue
            
        for ticker in chunk:
            if ticker not in bars_map or bars_map[ticker].empty:
                continue
                
            df = bars_map[ticker]
            
            # Run Algo
            try:
                res = run_simulation(df, {"length": 20, "mult": 2.0})
                trades = res[res["long_signal"]]
                
                for entry_time, row in trades.iterrows():
                    loc = df.index.get_loc(entry_time)
                    if loc + LOOK_FORWARD >= len(df):
                        continue
                        
                    entry_price = row["close"]
                    future_price = df.iloc[loc + LOOK_FORWARD]["close"]
                    min_future_price = df.iloc[loc : loc + LOOK_FORWARD]["low"].min()
                    
                    fwd_ret = (future_price - entry_price) / entry_price
                    dd = (min_future_price - entry_price) / entry_price
                    slope = row["mean_slope"]
                    
                    all_slopes.append(slope)
                    all_returns.append(fwd_ret)
                    all_drawdowns.append(dd)
                    
            except Exception as e:
                print(f"  Error processing {ticker}: {e}")
                
    # Analysis
    slopes_arr = np.array(all_slopes)
    rets_arr = np.array(all_returns)
    dd_arr = np.array(all_drawdowns)
    
    total_trades = len(slopes_arr)
    print(f"\n=== GLOBAL RESULTS ({total_trades} trades across {len(tickers)} tickers) ===")
    
    print(f"{'Threshold':<10} | {'WinRate':<8} | {'AvgRet':<8} | {'Crashes Avoided':<15} | {'PnL Saved (Avoided Loss)':<25} | {'Profit Missed':<15} | {'NET PnL IMPACT':<15}")
    print("-" * 140)
    
    total_crashes = np.sum(dd_arr < -0.02)

    for thresh in [0.0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40]:
        # Trades ACCEPTED by filter (Slope > Thresh)
        accepted_mask = slopes_arr > thresh
        rejected_mask = ~accepted_mask
        
        if np.sum(accepted_mask) == 0:
            continue
            
        acc_rets = rets_arr[accepted_mask]
        rej_rets = rets_arr[rejected_mask]
        
        # Stats of the Strategy (Accepted Trades)
        win_rate = np.sum(acc_rets > 0) / len(acc_rets)
        avg_ret = np.mean(acc_rets)
        
        # Stats of what we BLOCKED (Rejected Trades)
        crashes_avoided = np.sum((rejected_mask) & (dd_arr < -0.02))
        
        # Financial Impact
        # "Profit Missed" = Sum of all POSITIVE returns in the rejected pile
        profit_missed = np.sum(rej_rets[rej_rets > 0])
        
        # "Loss Avoided" = Sum of all NEGATIVE returns in the rejected pile
        loss_avoided = np.sum(rej_rets[rej_rets < 0])
        
        # "Net PnL Impact" = The total return of the rejected pile.
        # If this is NEGATIVE, the filter is PROFITABLE (we avoided more loss than gain).
        net_impact_of_filter = np.sum(rej_rets)
        
        # Formatting explanation:
        # If Net PnL Impact is -50.0, it means by REJECTING these trades, we saved 50 units of loss.
        # So "Value Added" is -1 * Net Impact.
        
        print(f"{thresh:<10.2f} | {win_rate:<8.1%} | {avg_ret:<8.2%} | {crashes_avoided:<15} | {loss_avoided:<25.2f} | {profit_missed:<15.2f} | {net_impact_of_filter:<15.2f}")

    print("\nNote: 'NET PnL IMPACT' is the sum of returns of all REJECTED trades.")
    print("If 'NET PnL IMPACT' is NEGATIVE, the filter SAVED you money overall (Losses Avoided > Profits Missed).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default="RubberBand/tickers_options.txt")
    args = parser.parse_args()
    
    analyze_portfolio(args.tickers)
