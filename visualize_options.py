import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import os

# Parameters
OPT1_PARAMS = {"atr_mult_sl": 1.0, "take_profit_r": 2.0, "rsi_oversold": 25}
OPT2_PARAMS = {"atr_mult_sl": 2.5, "take_profit_r": 1.5, "rsi_oversold": 25}

BASE_CONFIG = "RubberBand/config.yaml"
TEMP_CONFIG = "RubberBand/config_viz.yaml"
TICKERS_FILE = "repro_tickers.txt"

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
                f.write(f"  rsi_oversold: {params['rsi_oversold']}\n")
            else:
                f.write(line)

def run_backtest_and_get_equity(params, label):
    print(f"Running backtest for {label}...")
    update_config(params)
    cmd = [
        "python", "RubberBand/scripts/backtest.py",
        "--tickers", TICKERS_FILE,
        "--days", "30",
        "--config", TEMP_CONFIG
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    
    # Read daily stats
    if os.path.exists("daily_stats.csv"):
        df = pd.read_csv("daily_stats.csv")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df['cumulative_pnl'] = df['net_pnl'].cumsum()
        return df
    return None

# Run both
df1 = run_backtest_and_get_equity(OPT1_PARAMS, "Option 1 (Max Profit)")
df2 = run_backtest_and_get_equity(OPT2_PARAMS, "Option 2 (Consistency)")

# Plot
plt.figure(figsize=(12, 6))
if df1 is not None:
    plt.plot(df1['date'], df1['cumulative_pnl'], label=f"Option 1: Profit (SL=1.0, TP=2.0)", linewidth=2)
if df2 is not None:
    plt.plot(df2['date'], df2['cumulative_pnl'], label=f"Option 2: Consistency (SL=2.5, TP=1.5)", linewidth=2, linestyle="--")

plt.title("Equity Curve Comparison: Profit vs Consistency")
plt.xlabel("Date")
plt.ylabel("Cumulative Net PnL ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("equity_comparison.png")
print("Chart saved to equity_comparison.png")
