
from __future__ import annotations
import os
import sys
import shutil
import subprocess
import pandas as pd
import yaml
import re
import datetime as dt

# Setup Path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.utils import load_config, read_tickers
from RubberBand.src.data import fetch_latest_bars
# Import Stock Simulation
from RubberBand.scripts.backtest import simulate_mean_reversion
from RubberBand.src.ticker_health import TickerHealthManager

def run_stock_timeline(days_list, sma_modes):
    print("\n--- Running Stock Bot Multi-Timeline Comparison ---")
    cfg = load_config("RubberBand/config.yaml")
    tickers = read_tickers("RubberBand/tickers.txt")
    
    max_days = max(days_list)
    print(f"Loading {len(tickers)} tickers for max {max_days} days...")
    
    key = os.getenv("APCA_API_KEY_ID", "")
    secret = os.getenv("APCA_API_SECRET_KEY", "")
    feed = cfg.get("feed", "iex")
    
    # Pre-fetch simulation data (Max Range)
    # 1.6 factor for calculation buffer
    stock_df_map, _ = fetch_latest_bars(tickers, "15Min", int(max_days*1.6), feed, True, key=key, secret=secret, verbose=False)
    daily_df_map, _ = fetch_latest_bars(tickers, "1Day", max_days + 250, feed, False, key=key, secret=secret, verbose=False)
    
    # Pre-process Data Caches
    data_cache = {}
    for sym in tickers:
        if sym in stock_df_map and sym in daily_df_map:
            df = stock_df_map[sym]
            df_d = daily_df_map[sym]
            if not df.empty and not df_d.empty:
                # Calculate SMAs
                df_d["sma20"] = df_d["close"].rolling(20).mean().shift(1)
                df_d["sma50"] = df_d["close"].rolling(50).mean().shift(1)
                df_d["sma80"] = df_d["close"].rolling(80).mean().shift(1)
                df_d["sma120"] = df_d["close"].rolling(120).mean().shift(1)
                
                # Map to Intraday
                df_d["date"] = df_d.index.date
                df["date"] = df.index.date
                
                # Pre-map
                sma20_map = df_d.set_index("date")["sma20"].to_dict()
                sma50_map = df_d.set_index("date")["sma50"].to_dict()
                sma80_map = df_d.set_index("date")["sma80"].to_dict()
                sma120_map = df_d.set_index("date")["sma120"].to_dict()
                
                data_cache[sym] = {
                    "df": df,
                    "sma20": sma20_map,
                    "sma50": sma50_map,
                    "sma80": sma80_map,
                    "sma120": sma120_map
                }

    results = [] # list of dicts: {Days, SMA, Trades, PnL}

    for days in days_list:
        print(f"  Simulating Timeline: {days} Days...")
        # Calculate start cutoff
        cutoff_date = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).date()

        for sma in sma_modes:
            total_pnl = 0
            total_trades = 0
            total_wins = 0
            
            cfg["trend_filter"]["enabled"] = True
            
            for sym, data in data_cache.items():
                # Slice Dataframe for this timeline
                full_df = data["df"]
                df_slice = full_df[full_df["date"] >= cutoff_date].copy()
                
                if df_slice.empty: continue

                # Inject Trend
                sma_map = data[f"sma{sma}"]
                df_slice["trend_sma"] = df_slice["date"].map(sma_map)
                df_slice.drop(columns=["date"], inplace=True)
                
                # Run Simulation
                # Use unique health file to avoid cross-contamination
                hm = TickerHealthManager(f"temp_health_{days}_{sma}.json", cfg.get("resilience", {}))
                res = simulate_mean_reversion(df_slice, cfg, hm, sym, start_cash=10000, risk_pct=0.01, verbose=False)
                
                total_pnl += res["net"]
                total_trades += res["trades"]
                
                # Calculate Wins
                wins = round(res["trades"] * (res["win_rate"] / 100))
                total_wins += wins
                
                # Cleanup
                if os.path.exists(f"temp_health_{days}_{sma}.json"):
                    try:
                        os.remove(f"temp_health_{days}_{sma}.json")
                    except Exception as e:
                        print(f"  [WARN] Failed to delete health file: {e}")

            win_pct = (total_wins / total_trades * 100) if total_trades > 0 else 0.0

            results.append({
                "Bot": "Stock",
                "Days": days,
                "SMA": f"SMA {sma}",
                "Trades": total_trades,
                "PnL": total_pnl,
                "Win%": win_pct
            })
            print(f"  >> Stock | {days}d | SMA {sma} | Trades: {total_trades} | Win%: {win_pct:.1f}% | PnL: ${total_pnl:.2f}")

    return results

def run_options_timeline(days_list, sma_modes):
    print("\n--- Running Options Bot Multi-Timeline Comparison ---")
    config_path = "RubberBand/config.yaml"
    backtest_script = "RubberBand/scripts/backtest_spreads.py"
    # Use Sample file for speed
    tickers_file = "RubberBand/tickers_options_sample.txt"
    
    if not os.path.exists(tickers_file):
        print("Sample file not found, using full list (SLOW!)")
        tickers_file = "RubberBand/tickers_options.txt"
        
    shutil.copy(config_path, config_path + ".bak")
    
    results = []
    
    try:
        with open(config_path, 'r') as f:
            cfg_data = yaml.safe_load(f)

        for days in days_list:
            print(f"  Simulating Timeline: {days} Days...")
            
            for sma in sma_modes:
                # Modify Config
                if "trend_filter" not in cfg_data: cfg_data["trend_filter"] = {}
                cfg_data["trend_filter"]["sma_period"] = sma
                
                with open(config_path, 'w') as f:
                    yaml.dump(cfg_data, f)
                
                # Run
                cmd = [sys.executable, backtest_script, "--days", str(days), "--tickers", tickers_file]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                output = proc.stdout
                
                trades = 0
                pnl = 0.0
                m_trades = re.search(r"Total Trades:\s+(\d+)", output)
                if m_trades: trades = int(m_trades.group(1))
                m_pnl = re.search(r"Total P&L:\s+\$([\d,.-]+)", output)
                if m_pnl: pnl = float(m_pnl.group(1).replace(",", ""))
                
                results.append({
                    "Bot": "Options",
                    "Days": days,
                    "SMA": f"SMA {sma}",
                    "Trades": trades,
                    "PnL": pnl
                })
                
    finally:
        shutil.copy(config_path + ".bak", config_path)
        os.remove(config_path + ".bak")
        
    return results

def main():
    days_range = [30, 60, 90, 120, 220]
    # Stock: 20 vs 50 vs 80 vs 120
    stock_res = run_stock_timeline(days_range, [20, 50, 80, 120])
    # stock_res = [ ... ] (Hardcoded removed)
    
    # Options: 20 vs 50 vs 80 vs 120
    opt_res = run_options_timeline(days_range, [20, 50, 80, 120])
    
    all_res = stock_res + opt_res
    df = pd.DataFrame(all_res)
    
    # Save CSV FIRST
    df.to_csv("sma_comparison_results.csv", index=False)
    print("\nSaved to sma_comparison_results.csv")

    print("\n" + "="*60)
    print("FINAL MULTI-TIMELINE REPORT")
    print("="*60)
    
    # Manual Table Printing
    print(f"{'Bot':<10} | {'Days':<6} | {'SMA':<8} | {'Trades':<8} | {'Win%':<8} | {'PnL':<12}")
    print("-" * 65)
    for _, row in df.iterrows():
        print(f"{row['Bot']:<10} | {row['Days']:<6} | {row['SMA']:<8} | {row['Trades']:<8} | {row['Win%']:<8.1f} | ${row['PnL']:<12.2f}")

if __name__ == "__main__":
    main()
