import os
import subprocess
import pandas as pd
import json
import itertools

# Define parameter grid
param_grid = {
    "atr_mult_sl": [1.0, 1.5, 2.0, 2.5],
    "take_profit_r": [1.5, 2.0, 2.5, 3.0],
    "rsi_oversold": [20, 25, 30]
}

# Baseline config file
BASE_CONFIG = "RubberBand/config.yaml"
TEMP_CONFIG = "RubberBand/config_temp.yaml"
TICKERS_FILE = "repro_tickers.txt"
RESULTS_FILE = "optimization_results_new.csv"

def run_backtest(config_path):
    cmd = [
        "python", "RubberBand/scripts/backtest.py",
        "--tickers", TICKERS_FILE,
        "--days", "30",
        "--config", config_path,
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Check if backtest_summary.json exists
        if os.path.exists("backtest_summary.json"):
            with open("backtest_summary.json", "r") as f:
                data = json.load(f)
                # Aggregate results
                total_net = sum(d['net'] for d in data)
                total_trades = sum(d['trades'] for d in data)
                wins = sum(d['win_rate'] * d['trades'] / 100 for d in data)
                win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
                return total_net, win_rate, total_trades
        else:
            print("backtest_summary.json not found")
            return -9999, 0, 0
            
    except subprocess.CalledProcessError as e:
        print(f"Error running backtest: {e}")
        print(f"Stderr: {e.stderr}")
        return -9999, 0, 0
    except Exception as e:
        print(f"General Error: {e}")
        return -9999, 0, 0

def update_config(params):
    with open(BASE_CONFIG, "r") as f:
        lines = f.readlines()
    
    with open(TEMP_CONFIG, "w") as f:
        for line in lines:
            if "atr_mult_sl:" in line:
                f.write(f"  atr_mult_sl: {params['atr_mult_sl']}\n")
            elif "take_profit_r:" in line:
                f.write(f"  take_profit_r: {params['take_profit_r']}\n")
            elif "rsi_oversold:" in line:
                f.write(f"  rsi_oversold: {params['rsi_oversold']} # Modified\n")
            else:
                f.write(line)

results = []
keys = list(param_grid.keys())
combinations = list(itertools.product(*param_grid.values()))

print(f"Starting optimization with {len(combinations)} combinations...")

for i, combo in enumerate(combinations):
    params = dict(zip(keys, combo))
    print(f"Testing {params}...")
    
    update_config(params)
    net_pnl, win_rate, trades = run_backtest(TEMP_CONFIG)
    
    result = {**params, "Net PnL": net_pnl, "Win Rate": win_rate, "Trades": trades}
    results.append(result)
    print(f"Result: PnL=${net_pnl:.2f}, WR={win_rate:.1f}%, Trades={trades}")

# Save results
df = pd.DataFrame(results)
df.to_csv(RESULTS_FILE, index=False)
print(f"\nOptimization complete. Results saved to {RESULTS_FILE}")
print("\nTop 5 Configurations:")
print(df.nlargest(5, "Net PnL"))
