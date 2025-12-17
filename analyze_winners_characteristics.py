
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Usage: python analyze_winners_characteristics.py

def analyze_winners():
    print("Loading detailed_trades.csv...")
    try:
        df_trades = pd.read_csv("detailed_trades.csv")
    except FileNotFoundError:
        print("detailed_trades.csv not found.")
        return

    # Filter for winners
    # CRITICAL: Only filter winners from the last 50 days because fetch_latest_bars only gets recent history.
    # Older trades won't find their bars.
    cutoff_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=50)
    df_trades['entry_dt'] = pd.to_datetime(df_trades['entry_time'], utc=True)
    
    winners = df_trades[
        (df_trades['pnl'] > 0) & 
        (df_trades['entry_dt'] > cutoff_date)
    ]
    
    # Debug: Print unique dates to ensure we match format
    unique_dates = df_trades['entry_dt'].dt.date.unique()
    print(f"Available dates in data: {unique_dates[:5]}...")
    
    target_date = pd.Timestamp("2025-12-15").date()
    losers_12_15 = df_trades[
        (df_trades['entry_dt'].dt.date == target_date) & 
        (df_trades['pnl'] < 0)
    ]
    
    print(f"Analyzing {len(winners)} winning trades and {len(losers_12_15)} losers from 12/15...")
    
    # Get distinct symbols
    all_syms = list(set(winners['symbol'].unique().tolist() + losers_12_15['symbol'].unique().tolist()))
    
    # Load historical data
    from RubberBand.src.data import fetch_latest_bars
    from RubberBand.src.indicators import ta_add_keltner, ta_add_vol_dollar
    
    print("Fetching 60 days of history...")
    bars_map, _ = fetch_latest_bars(
        symbols=all_syms[:50], # Increase limit to 50
        timeframe="15Min",
        history_days=60,
        feed="iex",
        verbose=False
    )
    
    print(f"Fetched data for {len(bars_map)} symbols.")
    
    results = []
    
    for df_subset, label in [(winners, "WINNER"), (losers_12_15, "LOSER_12_15")]:
        for idx, row in df_subset.iterrows():
            sym = row['symbol']
            if sym not in bars_map: continue
            
            df = bars_map[sym]
            if df.empty: continue
            
            # Ensure indicators
            if "kc_middle" not in df.columns:
                df = ta_add_keltner(df, length=20)
            if "dollar_vol_avg" not in df.columns:
                df = ta_add_vol_dollar(df)
            
            # Find entry bar
            try:
                entry_ts = row['entry_dt']
                target_ts = entry_ts.tz_convert(df.index.tz)
                loc_idx = df.index.get_indexer([target_ts], method='pad')[0]
                
                if loc_idx < 4: continue # Need history for slope
                
                # Calculate metrics at entry
                # 1. Slope: (KC_Mid[i-1] - KC_Mid[i-4]) / 3
                # Bot uses signal bar (i-1) vs (i-4) relative to current bar i.
                # Here loc_idx is the current bar (or close to it).
                
                # Let's assume prediction was made at loc_idx
                # So we look at loc_idx-1 backwards
                
                kc_now = df['kc_middle'].iloc[loc_idx]
                kc_prev3 = df['kc_middle'].iloc[loc_idx-3]
                slope = (kc_now - kc_prev3) / 3
                
                # Normalize slope by price? The bot uses raw absolute difference.
                # If slope is -0.20, it means KC dropped $0.60 in 3 bars.
                # Wait, -0.20 is raw price. For a $200 stock, -0.20 is tiny (-0.1%).
                # For a $20 stock, -0.20 is huge (-1%).
                # This seems like a bug in the bot? It uses fixed threshold for all tickers?
                # Let's check if the bot normalizes.
                
                # Reviewing live_spreads_loop.py:
                # current_slope = (df["kc_middle"].iloc[-1] - df["kc_middle"].iloc[-4]) / 3
                # It is RAW PRICE CHANGE.
                # This suggests the filter is biased against high priced stocks!
                # NVDA ($135) vs F ($10).
                
                # 2. Acceleration (Change in Slope)
                # Slope_prev = (KC[i-2] - KC[i-5]) / 3 ?? 
                # Let's just do (Slope_now - Slope_prev)
                kc_prev1 = df['kc_middle'].iloc[loc_idx-1]
                kc_prev4 = df['kc_middle'].iloc[loc_idx-4]
                slope_prev = (kc_prev1 - kc_prev4) / 3
                acceleration = slope - slope_prev
                
                # 3. Volume Ratio relative to SMA
                vol_now = df['volume'].iloc[loc_idx]
                vol_avg = df['volume'].rolling(20).mean().iloc[loc_idx]
                vol_ratio = vol_now / vol_avg if vol_avg > 0 else 1.0
                
                results.append({
                    "Type": label,
                    "Symbol": sym,
                    "Price": df['close'].iloc[loc_idx],
                    "Slope": slope,
                    "Acceleration": acceleration,
                    "VolRatio": vol_ratio,
                    "PnL": row['pnl']
                })
                
            except Exception as e:
                pass
                
    # Output analysis
    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("No results generated.")
        return

    print("\n--- METRIC DISTRIBUTION ---")
    print(res_df.groupby("Type")[["Slope", "Acceleration", "VolRatio"]].describe().T)
    
    print("\n--- IMPACT OF SLOPE THRESHOLDS ON WINNERS ---")
    winners_df = res_df[res_df["Type"] == "WINNER"]
    total_w = len(winners_df)
    
    for thresh in [-0.10, -0.12, -0.14, -0.20, -0.30]:
        # Filter logic: Skip if Slope < Thresh
        blocked = winners_df[winners_df["Slope"] < thresh]
        count = len(blocked)
        pct = (count / total_w) * 100
        print(f"Threshold {thresh:>6}: Blocks {count:>3} winners ({pct:.1f}%)")

if __name__ == "__main__":
    analyze_winners()
