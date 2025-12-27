import json
import csv
import os

RESULTS_DIR = 'backtest_results_aggressive'
SUMMARY_FILE = os.path.join(RESULTS_DIR, 'backtest_summary.json')
TRADES_FILE = os.path.join(RESULTS_DIR, 'detailed_trades.csv')

if not os.path.exists(SUMMARY_FILE):
    print(f"Error: {SUMMARY_FILE} not found.")
    exit(1)

with open(SUMMARY_FILE, 'r') as f:
    data = json.load(f)

total_trades = sum(d['trades'] for d in data)
total_net = sum(d['net'] for d in data)

wins = 0
losses = 0
if os.path.exists(TRADES_FILE):
    with open(TRADES_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pnl = float(row['pnl'])
            if pnl > 0:
                wins += 1
            else:
                losses += 1

real_total = wins + losses
wr = (wins / real_total * 100) if real_total > 0 else 0.0

print(f"--- AGGRESSIVE BACKTEST RESULTS ---")
print(f"Total Trades: {total_trades}")
print(f"Total Net PnL: ${total_net:.2f}")
print(f"Win Rate: {wr:.1f}% ({wins}/{losses})")
print(f"----------------------------------")
