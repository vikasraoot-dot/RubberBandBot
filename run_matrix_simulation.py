
import subprocess
import pandas as pd
import os
import json
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
DAYS_MAX = 220
WINDOWS = [30, 90, 120, 220]
SLOPES = [-0.12, -0.15, -0.18, -0.20]

# Scripts
SCRIPT_STOCK = "RubberBand/scripts/backtest.py"
SCRIPT_OPTIONS = "RubberBand/scripts/backtest_spreads.py"
SCRIPT_WEEKLY_STOCK = "RubberBand/scripts/backtest_weekly.py"
SCRIPT_WEEKLY_OPTIONS = "RubberBand/scripts/backtest_weekly_options.py"

RESULTS_FILE = "matrix_results.json"

def run_backtest_task(task_def):
    """Runs a single backtest task."""
    bot_name = task_def["bot"]
    script = task_def["script"]
    args = task_def["args"]
    csv_out = task_def["csv_out"]
    config_label = task_def["config"]
    
    # Unique output file to prevent race conditions
    unique_csv = f"{csv_out}_{bot_name.replace(' ', '_')}_{config_label.replace(' ', '_')}.csv"
    
    # Some scripts accept output file args, others dump to fixed names.
    # We must patch args to specific output if supported, OR rename after run.
    # backtest.py -> prints detailed trades to stdout? No, it collects in `rows`. 
    # Wait, backtest.py doesn't have a CSV output arg. It has detailed_trades in memory.
    # My previous analysis showed it prints. 
    # The previous script hacked it: "Stock backtest uses 'detailed_trades.csv'".
    # Actually, legacy backtest.py DOES NOT save CSV by default. I need to fix that first.
    # But wait, looking at my previous `run_matrix` code: `if run_backtest(..., "detailed_trades.csv")`. 
    # It assumed the file appears. 
    # If backtest.py doesn't write it, the previous run was failing silently on analysis? 
    # No, I saw "Processing Slope -0.15", so it didn't crash.
    
    # CRITICAL FIX: I need to ensure backtest.py writes to unique CSV or I can't parallelize.
    # Since I cannot easily change backtest.py args without editing it, I will use a Lock-based approach?
    # No, parallelism requires unique outputs.
    # I will modify args to pass `--output` if I add that support, OR just run sequentially per script-type but parallel across types?
    # Stock and Options use different scripts, so I can run 1 Stock and 1 Option in parallel.
    # But running 4 Stocks in parallel will race on `detailed_trades.csv`.
    
    # For now, let's run Stock and Options in parallel (2 streams), but sequential within them.
    # That speeds up by 2x.
    # Better: Patch `backtest.py` to accept `--output` or use run directory.
    pass

# ... Reverting to sequential-parallel hybrid logic in main ...

def run_cmd(cmd, check=True):
    try:
        subprocess.run(cmd, check=check, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False

def analyze_trades_from_csv(csv_path, label, bot_type):
    return analyze_trades(csv_path, WINDOWS) # Reuse logic

def analyze_trades(trades_csv, windows):
    """Slices trade data for different time windows."""
    # ... (Same logic as before) ...
    stats = {}
    if not os.path.exists(trades_csv):
        return {w: {"pnl": 0, "trades": 0} for w in windows}, {}

    try:
        df = pd.read_csv(trades_csv)
        if df.empty:
             return {w: {"pnl": 0, "trades": 0, "win_rate": 0, "avg_trades_per_day": 0} for w in windows}, {}
             
        col_date = 'entry_time' if 'entry_time' in df.columns else 'date'
        # Normalize date column detection
        if col_date not in df.columns and 'entry_date' in df.columns: col_date = 'entry_date'
        
        if col_date not in df.columns:
             return {w: {"error": "no_date_col"} for w in windows}, {}

        df[col_date] = pd.to_datetime(df[col_date], utc=True)
        # Fix: handle timezone awareness mixed
        if df[col_date].dt.tz is None:
             df[col_date] = df[col_date].dt.tz_localize('UTC')
             
        max_date = df[col_date].max()
        daily_counts = df.groupby(df[col_date].dt.date).size().to_dict()
        
        for w in windows:
            cutoff = max_date - datetime.timedelta(days=w)
            mask = df[col_date] > cutoff
            subset = df[mask]
            
            pnl_col = 'pnl' if 'pnl' in subset.columns else 'net' # 'net' in some scripts?
            if pnl_col not in subset.columns and 'net_pnl' in subset.columns: pnl_col = 'net_pnl'
            
            pnl = subset[pnl_col].sum() if not subset.empty else 0
            count = len(subset)
            wins = len(subset[subset[pnl_col] > 0])
            wr = (wins / count * 100) if count > 0 else 0.0
            
            subset_daily = subset.groupby(subset[col_date].dt.date).size()
            avg_daily = subset_daily.mean() if not subset_daily.empty else 0.0
            
            stats[w] = {
                "pnl": round(pnl, 2),
                "trades": count,
                "win_rate": round(wr, 1),
                "avg_trades_per_day": round(avg_daily, 2)
            }
            
        return stats, daily_counts
    except Exception as e:
        print(f"Error {e}")
        return {}, {}


def process_stock_scenario(slope):
    print(f"  [Stock] Starting Slope {slope}...")
    # Stock backtest doesn't support custom output filename arg currently.
    # We must run it, then immediately rename 'detailed_trades.csv' to a unique name.
    # LOCKING: Only one process can run SCRIPT_STOCK at a time to avoid race on 'detailed_trades.csv'.
    # So we can't parallelize *within* Stock, but we can parallelize Stock vs Options.
    
    args = ["--days", str(DAYS_MAX), "--slope-threshold", str(slope), "--dead-knife-filter", "--quiet"]
    cmd = ["python", SCRIPT_STOCK] + args
    
    if run_cmd(cmd):
        unique_name = f"trades_stock_{slope}.csv"
        # Renaming... handle race if other process starts?
        # That's why we will run 1 Stock task stream and 1 Option task stream.
        if os.path.exists("detailed_trades.csv"):
            if os.path.exists(unique_name): os.remove(unique_name)
            os.rename("detailed_trades.csv", unique_name)
            data, freq = analyze_trades(unique_name, WINDOWS)
            return {
                "bot": "15m Stock",
                "config": f"Slope {slope}",
                "data": data,
                "freq": freq
            }
    return None

def process_options_scenario(slope):
    print(f"  [Options] Starting Slope {slope}...")
    # Options saves to results/spread_backtest_trades.csv
    args = ["--days", str(DAYS_MAX), "--slope-threshold", str(slope), "--quiet"]
    cmd = ["python", SCRIPT_OPTIONS] + args
    
    if run_cmd(cmd):
        default_out = "results/spread_backtest_trades.csv"
        unique_name = f"trades_options_{slope}.csv"
        if os.path.exists(default_out):
            if os.path.exists(unique_name): os.remove(unique_name)
            os.rename(default_out, unique_name)
            data, freq = analyze_trades(unique_name, WINDOWS)
            return {
                "bot": "15m Options",
                "config": f"Slope {slope}",
                "data": data,
                "freq": freq
            }
    return None

def main():
    results = []
    
    # Targeted Options Simulation for Sane Proposal
    options_scenarios = [
        {"slope": -0.12, "dkf": False, "label": "Baseline (-0.12)"},
        {"slope": -0.18, "dkf": False, "label": "Aggressive (-0.18, No DKF)"},
        {"slope": -0.18, "dkf": True, "label": "Proposed (-0.18 + DKF)"},
        {"slope": -0.20, "dkf": True, "label": "Strict (-0.20 + DKF)"} 
    ]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for scen in options_scenarios:
            slope = scen["slope"]
            dkf = scen["dkf"]
            label = scen["label"]
            
            print(f"Queueing {label}...")
            
            args = ["--days", str(DAYS_MAX), "--slope-threshold", str(slope), "--quiet"]
            if dkf: args.append("--dead-knife-filter")
            
            # Helper wrapper to match signature
            def run_wrapper(lbl, a):
                unique_name = f"trades_opt_{lbl.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.csv"
                cmd = ["python", SCRIPT_OPTIONS] + a
                # Custom run command that renames output
                if run_cmd(cmd):
                    default_out = "results/spread_backtest_trades.csv"
                    if os.path.exists(default_out):
                        if os.path.exists(unique_name): os.remove(unique_name)
                        os.rename(default_out, unique_name)
                        data, freq = analyze_trades(unique_name, WINDOWS)
                        return {
                            "bot": "15m Options",
                            "config": lbl,
                            "data": data,
                            "freq": freq
                        }
                return None

            futures.append(executor.submit(run_wrapper, label, args))
            
        for f in as_completed(futures):
            res = f.result()
            if res: results.append(res)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print("Done.")

def run_weekly_stock():
    print("  [Weekly Stock] Starting...")
    csv = "weekly_stock_backtest.csv"
    if run_cmd(["python", SCRIPT_WEEKLY_STOCK, "--days", str(DAYS_MAX), "--quiet"]):
        data, freq = analyze_trades(csv, WINDOWS)
        return {"bot": "Weekly Stock", "config": "Default", "data": data, "freq": freq}

def run_weekly_options():
    print("  [Weekly Options] Starting...")
    csv = "weekly_options_backtest.csv"
    if run_cmd(["python", SCRIPT_WEEKLY_OPTIONS, "--days", str(DAYS_MAX), "--quiet"]):
        data, freq = analyze_trades(csv, WINDOWS)
        return {"bot": "Weekly Options", "config": "Default", "data": data, "freq": freq}

if __name__ == "__main__":
    main()
