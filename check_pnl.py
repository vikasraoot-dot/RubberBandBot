import subprocess
import pandas as pd
import os
import json

OPT1_PARAMS = {"atr_mult_sl": 1.0, "take_profit_r": 2.0, "rsi_oversold": 20}
OPT2_PARAMS = {"atr_mult_sl": 2.5, "take_profit_r": 1.5, "rsi_oversold": 25}

BASE_CONFIG = "RubberBand/config.yaml"
TEMP_CONFIG = "RubberBand/config_check.yaml"
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

def run_check(params, label):
    update_config(params)
    cmd = [
        "python", "RubberBand/scripts/backtest.py",
        "--tickers", TICKERS_FILE,
        "--days", "30",
        "--config", TEMP_CONFIG
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    
    if os.path.exists("backtest_summary.json"):
        with open("backtest_summary.json", "r") as f:
            data = json.load(f)
            total_net = sum(d['net'] for d in data)
            print(f"{label}: ${total_net:.2f}")
    else:
        print(f"{label}: Failed to read results")

run_check(OPT1_PARAMS, "Option 1 (Blue)")
run_check(OPT2_PARAMS, "Option 2 (Orange)")
