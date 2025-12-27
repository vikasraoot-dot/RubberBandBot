
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import sys
import os

# Add repo root to path
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.data import fetch_latest_bars

def analyze_efficiency(trade_files):
    all_trades = []
    
    # 1. Load Trades
    for f in trade_files:
        if not os.path.exists(f):
            print(f"Skipping missing file: {f}")
            continue
        try:
            df = pd.read_csv(f)
            df['source_file'] = f
            all_trades.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    if not all_trades:
        print("No trades loaded.")
        return

    df_trades = pd.concat(all_trades, ignore_index=True)
    
    # Standardize Date Column
    if 'entry_time' in df_trades.columns:
        df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
    elif 'entry_date' in df_trades.columns:
        df_trades['entry_time'] = pd.to_datetime(df_trades['entry_date'])
    else:
        print("Error: Could not find entry_time or entry_date column")
        return

    # Ensure UTC
    if df_trades['entry_time'].dt.tz is None:
        df_trades['entry_time'] = df_trades['entry_time'].dt.tz_localize('UTC')
    else:
         df_trades['entry_time'] = df_trades['entry_time'].dt.tz_convert('UTC')

    print(f"Loaded {len(df_trades)} trades.")
    
    # 2. Fetch VIXY Data
    min_date = df_trades['entry_time'].min() - timedelta(days=5)
    max_date = df_trades['entry_time'].max() + timedelta(days=5)
    
    # Calculate days for fetch
    days_needed = (datetime.now(pytz.UTC) - min_date).days + 5
    
    print(f"Fetching VIXY data for last {days_needed} days...")
    bars_map, _ = fetch_latest_bars(['VIXY'], '1Day', days_needed, feed='iex', verbose=False)
    vix_df = bars_map.get('VIXY')
    
    if vix_df is None or vix_df.empty:
        print("Error: Could not fetch VIXY data.")
        return

    # optimize lookup
    vix_df.index = pd.to_datetime(vix_df.index).date

    # 3. Enrich Trades
    enriched = []
    
    for idx, row in df_trades.iterrows():
        entry_date = row['entry_time'].date()
        
        # Get VIX
        # Look for exact date, or closest previous date
        vix_val = np.nan
        if entry_date in vix_df.index:
             vix_val = vix_df.loc[entry_date]['close']
        else:
            # Fallback to recent history (e.g. previous friday for monday trade)
            for lag in range(1, 5):
                 d = entry_date - timedelta(days=lag)
                 if d in vix_df.index:
                     vix_val = vix_df.loc[d]['close']
                     break
        
        # Calculate Slope %
        # entry_slope is usually absolute
        # entry_price is price
        # slope% = (entry_slope / entry_price) * 100
        
        slope_abs = row.get('entry_slope', np.nan)
        price = row.get('entry_price', np.nan)
        
        slope_pct = np.nan
        if not pd.isna(slope_abs) and not pd.isna(price) and price > 0:
            slope_pct = (slope_abs / price) * 100
            
        rec = row.to_dict()
        rec['vix'] = vix_val
        rec['slope_pct'] = slope_pct
        enriched.append(rec)
        
    df_final = pd.DataFrame(enriched)
    
    # 4. Analyze Regimes
    # Define Regimes: Calm (<15), Normal (15-25), Volatile (>25)
    # Note: VIXY price != VIX index. VIXY 25 is roughly VIX 20-30 depending on decay.
    # For now we categorize by quartiles or raw values for analysis.
    
    print("\n" + "="*60)
    print("DATA-DRIVEN ANALYSIS: Slope % Efficiency")
    print("="*60)
    
    # Filter valid data
    valid = df_final.dropna(subset=['slope_pct', 'pnl'])
    
    if valid.empty:
        print("No valid data for analysis (missing slope or pnl).")
        return

    # Bucket by Slope %
    # We want to find the "Cliff" where Win Rate drops.
    # Buckets: 0 to -0.05, -0.05 to -0.10, etc.
    
    bins = [0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.50, -1.0]
    labels = ["0 to -0.05%", "-0.05% to -0.10%", "-0.10% to -0.15%", "-0.15% to -0.20%", "-0.20% to -0.25%", "-0.25% to -0.50%", "< -0.50%"]
    
    valid['slope_bucket'] = pd.cut(valid['slope_pct'], bins=[float(x) for x in bins[::-1]] + [0.1], right=True) # Reversed for negatives
    
    # Group by VIX Regime (Rough buckets for VIXY)
    # VIXY Low: < 20
    # VIXY Med: 20-30
    # VIXY High: > 30
    
    def get_regime(v):
        if pd.isna(v): return "Unknown"
        if v < 20: return "Low Vol (<20)"
        if v < 30: return "Med Vol (20-30)"
        return "High Vol (>30)"
        
    valid['regime'] = valid['vix'].apply(get_regime)
    
    print(f"Total Trades Analyzed: {len(valid)}")
    
    # Pivot Table: PnL by Slope Bucket & Regime
    summary = valid.groupby(['regime', pd.cut(valid['slope_pct'], bins=[-1.0, -0.5, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0])]).agg({
        'pnl': ['count', 'mean', 'sum'],
        'pnl_pct': 'mean'
    })
    
    print("\n--- Performance Matrix (PnL per Trade) ---")
    # Manual Markdown Table
    # print(summary.to_markdown()) 
    print("| Regime | Slope Bucket | Count | Mean PnL | Sum PnL | Mean PnL% |")
    print("|---|---|---|---|---|---|")
    for idx, row in summary.iterrows():
        regime, bucket = idx
        count = row[('pnl', 'count')]
        mean = row[('pnl', 'mean')]
        total = row[('pnl', 'sum')]
        mean_pct = row[('pnl_pct', 'mean')]
        if count > 0:
            print(f"| {regime} | {bucket} | {count} | ${mean:.2f} | ${total:.2f} | {mean_pct:.2f}% |")

    # Find "Golden Number"
    # Where does PnL turn positive/reliable?
    
    print("\n--- Correlation Checks ---")
    corr_slope = valid['pnl'].corr(valid['slope_pct'])
    corr_vix = valid['pnl'].corr(valid['vix'])
    print(f"Correlation PnL vs Slope%: {corr_slope:.4f} (Expect Negative - steeper is better?)")
    print(f"Correlation PnL vs VIX: {corr_vix:.4f}")

    # Top Winners vs Losers Analysis
    print("\n--- Top 10 Winners ---")
    print("| Symbol | Entry Time | Slope% | VIX | PnL |")
    print("|---|---|---|---|---|")
    for _, row in valid.nlargest(10, 'pnl').iterrows():
        print(f"| {row['symbol']} | {row['entry_time']} | {row['slope_pct']:.4f}% | {row['vix']:.2f} | ${row['pnl']:.2f} |")
    
    print("\n--- Top 10 Losers ---")
    print("| Symbol | Entry Time | Slope% | VIX | PnL |")
    print("|---|---|---|---|---|")
    for _, row in valid.nsmallest(10, 'pnl').iterrows():
         print(f"| {row['symbol']} | {row['entry_time']} | {row['slope_pct']:.4f}% | {row['vix']:.2f} | ${row['pnl']:.2f} |")

    # Save to CSV for user inspection
    df_final.to_csv("analysis_slope_regime.csv", index=False)
    print("\nFull details saved to 'analysis_slope_regime.csv'")

if __name__ == "__main__":
    # List of files to analyze (add more if available)
    files = [
        "trades_opt_Baseline_-0.12.csv",
        "trades_opt_Aggressive_-0.18_No_DKF.csv", 
        "weekly_stock_backtest.csv",
        "weekly_options_backtest.csv",
        "trades_telemetry_exhaustive.csv"        
    ]
    
    # Check for stock files
    if os.path.exists("trades_stock_-0.12.csv"):
        files.append("trades_stock_-0.12.csv")
        
    analyze_efficiency(files)
