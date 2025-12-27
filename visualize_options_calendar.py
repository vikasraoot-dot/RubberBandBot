
from __future__ import annotations
import os
import sys
import shutil
import subprocess
import pandas as pd
import yaml
import datetime as dt
import calendar

# Setup Path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))

def generate_options_calendar():
    print("\n--- Generating Options Trade Calendar (SMA 50 | Last 220 Days) ---")
    
    config_path = "RubberBand/config.yaml"
    backtest_script = "RubberBand/scripts/backtest_spreads.py"
    tickers_file = "RubberBand/tickers_options.txt"
    results_file = "results/spread_backtest_trades.csv"
    
    # 1. Backup Config
    shutil.copy(config_path, config_path + ".bak")
    
    try:
        # 2. Modify Config for SMA 50
        with open(config_path, 'r') as f:
            cfg_data = yaml.safe_load(f)
            
        if "trend_filter" not in cfg_data: cfg_data["trend_filter"] = {}
        cfg_data["trend_filter"]["sma_period"] = 50
        cfg_data["trend_filter"]["enabled"] = True
        
        with open(config_path, 'w') as f:
            yaml.dump(cfg_data, f)
            
        # 3. Run Backtest
        print("Running Options Simulation (120 days to avoid data limits)...")
        cmd = [sys.executable, backtest_script, "--days", "120", "--tickers", tickers_file]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # 4. Parse Results
        if not os.path.exists(results_file):
            print(f"Error: Results file not found at {results_file}")
            return

        df = pd.read_csv(results_file)
        if df.empty:
            print("No trades found.")
            return
            
        # DataFrame columns: entry_date, entry_time, exit_time, symbol, side, pnl, ...
        # Check actual columns. Usually 'entry_time' is the timestamp.
        
        trades_by_date = {}
        
        for _, row in df.iterrows():
            # Parse entry_time
            # Format usually "2025-12-18 19:30:00"
            try:
                ts = pd.to_datetime(row["entry_time"])
                date_str = str(ts.date())
            except:
                continue
                
            pnl = row["pnl"]
            
            if date_str not in trades_by_date:
                trades_by_date[date_str] = {"count": 0, "pnl": 0.0, "wins": 0}
                
            trades_by_date[date_str]["count"] += 1
            trades_by_date[date_str]["pnl"] += pnl
            if pnl > 0:
                trades_by_date[date_str]["wins"] += 1
                
        # 5. Print Calendar
        dates = [dt.datetime.strptime(d, "%Y-%m-%d").date() for d in trades_by_date.keys()]
        if not dates:
            print("No valid dates found.")
            return

        start_date = min(dates)
        end_date = max(dates)
        curr_date = start_date.replace(day=1)

        print("\nLEGEND: [Trades] PnL_Symbol (W=Win, L=Loss, M=Mixed)")
        print("=" * 60)

        while curr_date <= end_date:
            year = curr_date.year
            month = curr_date.month
            month_name = calendar.month_name[month]
            month_matrix = calendar.monthcalendar(year, month)

            print(f"\n{month_name.upper()} {year}")
            print("Mon        Tue        Wed        Thu        Fri")
            print("-" * 50)

            for week in month_matrix:
                line_str = ""
                week_relevant = False
                for day in week[:5]:
                    if day != 0: week_relevant = True
                if not week_relevant: continue

                for i, day in enumerate(week[:5]): # Only Mon-Fri
                    if day == 0:
                        line_str += "           "
                        continue

                    d_obj = dt.date(year, month, day)
                    d_str = str(d_obj)
                    cell = f"{day:02d} | ...  "

                    if d_str in trades_by_date:
                        data = trades_by_date[d_str]
                        cnt = data["count"]
                        pnl = data["pnl"]
                        
                        if pnl > 0: icon = "✅"
                        elif pnl < 0: icon = "❌"
                        else: icon = "➖"
                        
                        cell = f"{day:02d} | {cnt:<1}{icon}  "

                    line_str += f"{cell:<11}"
                print(line_str)

            if month == 12:
                curr_date = dt.date(year + 1, 1, 1)
            else:
                curr_date = dt.date(year, month + 1, 1)
        
        print("\nSummary:")
        print(f"Total Days Traded: {len(trades_by_date)}")
        print(f"Total Trades: {len(df)}")
        print(f"Total PnL: ${df['pnl'].sum():,.2f}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Restore Config
        if os.path.exists(config_path + ".bak"):
            shutil.copy(config_path + ".bak", config_path)
            os.remove(config_path + ".bak")

if __name__ == "__main__":
    generate_options_calendar()
