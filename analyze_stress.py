
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_stress_test():
    csv_path = "results/spread_backtest_trades.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("CSV not found. Wait for backtest to finish.")
        return

    if df.empty:
        print("No trades in CSV.")
        return
        
    # Convert entry_time to datetime
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    
    # Define windows
    end_date = df['entry_time'].max()
    windows = [5, 20, 30, 60, 90]
    
    print(f"Stress Test Analysis (End Date: {end_date.date()})")
    print("="*80)
    print(f"{'Window':<10} | {'Strategy':<20} | {'Trades':<6} | {'Win%':<6} | {'Total PnL':<10} | {'Avg PnL':<8} | {'ROI%':<6}")
    print("-" * 80)
    
    for days in windows:
        start_date = end_date - timedelta(days=days)
        mask = df['entry_time'] >= start_date
        sub_df = df[mask].copy()
        
        if sub_df.empty:
            continue
            
        # Strategy 1: Baseline (No Slope Filter)
        # Note: We ran backtest with slope_threshold=100, effectively no filter.
        stats_base = get_stats(sub_df)
        print_row(f"{days}d", "Baseline", stats_base)
        
        # Strategy 2: Anti-Falling Knife (Old Logic)
        # Skip if Slope 3 < -0.20 (Avoid Crashes)
        # Implies: Keep if Slope 3 >= -0.20
        # Wait, if we 'Skip Crashes', we keep flat/gentle.
        # Logic was: if slope < -0.20: SKIP. 
        # So we KEEP where slope >= -0.20.
        mask_old = sub_df['entry_slope'] >= -0.20
        stats_old = get_stats(sub_df[mask_old])
        print_row(f"{days}d", "Anti-Knife (Old)", stats_old)
        
        # Strategy 3: Panic Persistency (New Logic)
        # Catch Crashes (Slope 3 < -0.20) AND Avoid Drifts (Slope 10 < -0.15)
        # Assuming we want Negative slopes. 
        # "Slope > -0.20: SKIP" -> We need Slope <= -0.20.
        # "Slope 10 > -0.15: SKIP" -> We need Slope 10 <= -0.15.
        mask_new = (sub_df['entry_slope'] <= -0.20) & (sub_df['entry_slope_10'] <= -0.15)
        stats_new = get_stats(sub_df[mask_new])
        print_row(f"{days}d", "Dual-Slope (New)", stats_new)
        
        print("-" * 80)

def get_stats(df):
    if df.empty:
        return {"count": 0, "wins": 0, "pnl": 0.0, "cost": 0.0}
    
    count = len(df)
    wins = len(df[df['pnl'] > 0])
    pnl = df['pnl'].sum()
    cost = df['cost'].sum()
    
    return {
        "count": count,
        "wins": wins,
        "pnl": pnl,
        "cost": cost
    }

def print_row(window, name, stats):
    count = stats["count"]
    if count == 0:
        print(f"{window:<10} | {name:<20} | 0      | 0.0%   | $0.00      | $0.00    | 0.0%")
        return
        
    win_rate = (stats["wins"] / count) * 100
    avg_pnl = stats["pnl"] / count
    roi = (stats["pnl"] / stats["cost"] * 100) if stats["cost"] > 0 else 0
    
    print(f"{window:<10} | {name:<20} | {count:<6} | {win_rate:<6.1f} | ${stats['pnl']:<9.0f} | ${avg_pnl:<7.2f} | {roi:<5.1f}")

if __name__ == "__main__":
    analyze_stress_test()
