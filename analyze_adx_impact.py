
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Usage: python analyze_adx_impact.py

def analyze_impact():
    print("Loading detailed_trades.csv...")
    try:
        df_trades = pd.read_csv("detailed_trades.csv")
    except FileNotFoundError:
        print("detailed_trades.csv not found.")
        return

    # Helper to parse ADX/DI from "entry_reason" or recreate if missing
    # Since we can't easily recreate exact history for 1000 trades quickly, 
    # we will look for 'adx' column if it exists, or infer from context.
    # WAIT - detailed_trades.csv might not have ADX column.
    # Let's check columns first.
    
    print(f"Columns: {df_trades.columns.tolist()}")
    
    # If no ADX column, we have to fetch history for each trade... that's slow.
    # ALTERNATIVE: run a quick backtest re-simulation with and without the filter.
    # But user wants to know about "profitable trades would have gone".
    
    # Let's try to map the trades to the 15m bar data we can fetch.
    # We will fetch 30 days of 15m data for the top tickers and cross-reference.
    
    # 1. Identify distinct symbols in winners
    winners = df_trades[df_trades['pnl'] > 0]
    symbols = winners['symbol'].unique().tolist()
    print(f"Analyzing {len(winners)} winning trades across {len(symbols)} symbols...")
    
    # Load historical data
    from RubberBand.src.data import fetch_latest_bars
    from RubberBand.src.indicators import ta_add_adx_di
    
    print("Fetching 60 days of history to cover trades...")
    bars_map, _ = fetch_latest_bars(
        symbols=symbols[:10], # Limit to top 10 distinct symbols for speed/quota if needed, or all if feasible
        timeframe="15Min",
        history_days=60,
        feed="iex",
        verbose=False
    )
    
    false_positives = 0
    total_checked = 0
    
    print("\nChecking Top Winners against ADX Filter (ADX > 25 & -DI > +DI)...")
    print(f"{'Symbol':<8} {'Entry Time':<20} {'PnL':<10} {'ADX':<6} {'-DI':<6} {'+DI':<6} {'Result'}")
    
    for idx, row in winners.iterrows():
        sym = row['symbol']
        if sym not in bars_map:
            continue
            
        entry_time_str = str(row['entry_time'])
        # Try to parse entry time
        try:
            # Format usually: 2025-12-15 09:30:00-05:00
            # Pandas timestamp to string
            entry_ts = pd.Timestamp(entry_time_str)
        except:
            continue
            
        df = bars_map[sym]
        if df.empty: continue
        
        # Add ADX
        if "ADX" not in df.columns:
            df = ta_add_adx_di(df)
            
        # Find bar at or before entry time
        # Ensure index is tz-aware compatible
        if df.index.tz is None:
             df.index = df.index.tz_localize('UTC') # Assuming data comes as UTC
        
        # Convert entry_ts to match df index tz
        try:
            target_ts = entry_ts.tz_convert(df.index.tz)
        except:
            target_ts = entry_ts # hope for best
            
        # Get bar
        # We need the bar *leading up to* entry.
        # If entry is 09:45, signal was likely closed 09:45 bar or 09:30 bar? 
        # Signals are usually "on close of previous bar".
        # So look for bar timestamp <= target_ts
        
        try:
            loc_idx = df.index.get_indexer([target_ts], method='pad')[0]
            if loc_idx == -1: continue
            
            bar = df.iloc[loc_idx]
            
            # Check filter
            adx = bar['ADX']
            pdi = bar['+DI']
            mdi = bar['-DI']
            
            total_checked += 1
            
            # Filter condition: ADX > 25 AND -DI > +DI
            if adx > 25 and mdi > pdi:
                print(f"{sym:<8} {entry_time_str[:19]:<20} ${row['pnl']:<9.2f} {adx:<6.1f} {mdi:<6.1f} {pdi:<6.1f} BLOCKED (Profitable Trade Lost)")
                false_positives += 1
            #else:
            #    print(f"{sym:<8} {entry_time_str[:19]:<20} ${row['pnl']:<9.2f} {adx:<6.1f} {mdi:<6.1f} {pdi:<6.1f} KEPT")
                
        except Exception as e:
            pass
            
    print(f"\nAnalysis Complete.")
    print(f"Total Winners Checked: {total_checked}")
    print(f"Profitable Trades Blocked by ADX Filter: {false_positives}")
    print(f"Percentage Lost: {false_positives/max(1, total_checked)*100:.1f}%")

if __name__ == "__main__":
    analyze_impact()
