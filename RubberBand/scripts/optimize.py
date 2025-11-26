
import sys
import os
import pandas as pd
import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
import yaml

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RubberBand.src.data import fetch_latest_bars
from RubberBand.scripts.backtest import simulate_mean_reversion

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
TICKERS = ["TQQQ", "SOXL", "NVDA", "TSLA", "COIN", "MSTR", "MARA", "AMD", "META", "AMZN"]
DAYS = 30

# Parameter Grid
PARAM_GRID = {
    "rsi_oversold": [20, 25, 30, 35],
    "keltner_mult": [1.5, 2.0, 2.5],
    "atr_mult_sl": [1.0, 1.5, 2.0],
    "take_profit_r": [2.0, 3.0, 5.0],
    "exit_at_mean": [True],
    "exit_at_upper": [False, True] # Test holding to upper band
}

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def load_data(tickers, days):
    print(f"Loading data for {len(tickers)} tickers over last {days} days...")
    data_map = {}
    for symbol in tickers:
        try:
            df, _ = fetch_latest_bars([symbol], history_days=days)
            if symbol in df and not df[symbol].empty:
                data_map[symbol] = df[symbol]
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
    print(f"Loaded {len(data_map)} tickers.")
    return data_map

def run_single_backtest(args):
    params, data_map = args
    
    # Construct config override
    cfg = {
        "keltner_length": 20,
        "keltner_mult": params["keltner_mult"],
        "atr_length": 14,
        "rsi_length": 14,
        "filters": {
            "rsi_oversold": params["rsi_oversold"],
            "rsi_overbought": 70,
            "min_price": 5.0,
            "min_dollar_vol": 0
        },
        "brackets": {
            "enabled": True,
            "atr_mult_sl": params["atr_mult_sl"],
            "take_profit_r": params["take_profit_r"]
        },
        "exit_at_mean": params["exit_at_mean"],
        "exit_at_upper": params["exit_at_upper"],
        
        # Fixed params
        "qty": 10000,
        "max_shares_per_trade": 10000,
        "max_notional_per_trade": 2000,
        "max_open_trades_per_ticker": 1,
        "flatten_minutes_before_close": 15,
        "entry_cutoff_min": 20,
        "allow_shorts": False
    }

    total_trades = 0
    total_net = 0.0
    wins = 0
    
    for symbol, df in data_map.items():
        res = simulate_mean_reversion(df.copy(), cfg)
        total_trades += res["trades"]
        total_net += res["net"]
        wins += int(res["win_rate"] * res["trades"] / 100)
        
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    expectancy = (total_net / total_trades) if total_trades > 0 else 0.0
    
    return {
        "params": params,
        "trades": total_trades,
        "net": total_net,
        "win_rate": win_rate,
        "expectancy": expectancy
    }

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    # 1. Load Data
    data_map = load_data(TICKERS, DAYS)
    if not data_map:
        print("No data loaded. Exiting.")
        return

    # 2. Generate Combinations
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Testing {len(combinations)} combinations...")

    # 3. Run Grid Search
    results = []
    # Running sequentially for simplicity and to avoid pickling issues with complex objects if any
    # Can switch to ProcessPoolExecutor if too slow
    for i, params in enumerate(combinations):
        if i % 10 == 0:
            print(f"Processing {i}/{len(combinations)}...")
        
        # Skip invalid combos (e.g. exit at mean AND upper? Logic handles priority but let's be clean)
        if params["exit_at_mean"] and params["exit_at_upper"]:
             # If both true, let's assume we want to test "Exit at Mean OR Upper" (whichever comes first)
             # But backtest logic prioritizes Upper if I wrote it that way? 
             # Let's check backtest.py logic:
             # elif hit_upper: ... elif hit_mean: ...
             # So Upper is checked first. If Upper is hit, we exit. If not, we check Mean.
             # This effectively means "Exit at Upper OR Mean".
             pass

        res = run_single_backtest((params, data_map))
        results.append(res)

    # 4. Sort and Report
    # Sort by Net PnL
    results.sort(key=lambda x: x["net"], reverse=True)

    print("\n=== TOP 10 CONFIGURATIONS (by Net PnL) ===")
    print(f"{'Rank':<5} {'Net PnL':<10} {'Trades':<8} {'Win%':<6} {'Exp($)':<8} {'Params'}")
    for i, r in enumerate(results[:10]):
        p = r["params"]
        p_str = f"RSI<{p['rsi_oversold']} KC={p['keltner_mult']} SL={p['atr_mult_sl']} TP={p['take_profit_r']} Mean={p['exit_at_mean']} Upper={p['exit_at_upper']}"
        print(f"{i+1:<5} {r['net']:<10.2f} {r['trades']:<8} {r['win_rate']:<6.1f} {r['expectancy']:<8.2f} {p_str}")

    # Save to CSV
    df_res = pd.DataFrame(results)
    df_res["params"] = df_res["params"].apply(str)
    df_res.to_csv("optimization_results.csv", index=False)
    print("\nFull results saved to optimization_results.csv")

if __name__ == "__main__":
    main()
