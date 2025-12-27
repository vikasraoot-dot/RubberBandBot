
import pandas as pd
import json
import os
import datetime

WINDOWS = [30, 90, 120, 220]
SLOPES = [-0.12, -0.15, -0.18, -0.20]

def analyze_csv(path, label, bot_type):
    if not os.path.exists(path):
        return None
    
    try:
        df = pd.read_csv(path)
        col_date = 'entry_time' if 'entry_time' in df.columns else 'date'
        if col_date not in df.columns and 'entry_date' in df.columns: col_date = 'entry_date'
        
        if col_date not in df.columns or df.empty:
            return None
            
        df[col_date] = pd.to_datetime(df[col_date], utc=True)
        if df[col_date].dt.tz is None:
             df[col_date] = df[col_date].dt.tz_localize('UTC')

        max_date = df[col_date].max()
        
        row = {"Bot": bot_type, "Config": label}
        
        for w in WINDOWS:
            cutoff = max_date - datetime.timedelta(days=w)
            mask = df[col_date] > cutoff
            subset = df[mask]
            
            pnl_col = 'pnl' if 'pnl' in subset.columns else 'net'
            if pnl_col not in subset.columns and 'net_pnl' in subset.columns: pnl_col = 'net_pnl'
            
            pnl = subset[pnl_col].sum() if not subset.empty else 0
            count = len(subset)
            wins = len(subset[subset[pnl_col] > 0])
            wr = (wins / count * 100) if count > 0 else 0.0
            
            # Daily Freq
            subset_daily = subset.groupby(subset[col_date].dt.date).size()
            avg_daily = subset_daily.mean() if not subset_daily.empty else 0.0
            
            # Format: "$PnL (N trades, WR%)"
            val = f"${pnl:,.2f} ({count}, {wr:.0f}%, {avg_daily:.1f}/day)"
            row[f"{w}d"] = val
            
        return row
    except Exception as e:
        print(e)
        return None

rows = []

# Options Scenarios
scenarios = [
    {"label": "Baseline (-0.12)", "file": "trades_opt_Baseline_-0.12.csv"},
    {"label": "Aggressive (-0.18, No DKF)", "file": "trades_opt_Aggressive_-0.18_No_DKF.csv"},
    {"label": "Proposed (-0.18 + DKF)", "file": "trades_opt_Proposed_-0.18_+_DKF.csv"},
    {"label": "Strict (-0.20 + DKF)", "file": "trades_opt_Strict_-0.20_+_DKF.csv"}
]

for scen in scenarios:
    r = analyze_csv(scen["file"], scen["label"], "15m Options")
    if r: rows.append(r)
    else: rows.append({"Bot": "15m Options", "Config": scen["label"], "30d": "Pending/Error", "90d": "-", "120d": "-", "220d": "-"})

# Stock (Check if any exist)
for s in [-0.12]: # Only check basic stock if generated
    if os.path.exists(f"trades_stock_{s}.csv"):
        r = analyze_csv(f"trades_stock_{s}.csv", f"Slope {s}", "15m Stock")
        if r: rows.append(r)

# Weekly
r_wk_stk = analyze_csv("weekly_stock_backtest.csv", "Default", "Weekly Stock")
if not r_wk_stk: r_wk_stk = analyze_csv("weekly_detailed_trades.csv", "Default", "Weekly Stock") # Fallback
if r_wk_stk: rows.append(r_wk_stk)

r_wk_opt = analyze_csv("weekly_options_backtest.csv", "Default", "Weekly Options")
if r_wk_opt: rows.append(r_wk_opt)

# Print Table Manual MD
headers = ["Bot", "Config", "30d", "90d", "120d", "220d"]
print("| " + " | ".join(headers) + " |")
print("|" + "|".join(["---"] * len(headers)) + "|")

for row in rows:
    line = f"| {row['Bot']} | {row['Config']} | {row.get('30d', '-')} | {row.get('90d', '-')} | {row.get('120d', '-')} | {row.get('220d', '-')} |"
    print(line)

# Also print Daily Frequency Data for Best Options Config (e.g. -0.12)
print("\n\n### Daily Trade Counts (Options -0.12, Last 30 Days)")
if os.path.exists("trades_options_-0.12.csv"):
    df = pd.read_csv("trades_options_-0.12.csv")
    print(f"Loaded: {len(df)} trades total.")
    col_date = 'entry_time' if 'entry_time' in df.columns else 'date' 
    # normalize... (simplified for brevity in this script print)
    # Just output the csv path so I can read it if needed.
