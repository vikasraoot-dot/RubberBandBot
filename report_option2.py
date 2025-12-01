import json
import pandas as pd

with open('backtest_summary.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

print("### Option 2 Performance Report (Existing Tickers)")
print("| Timeframe | Net PnL | Win Rate | Trades | Best Ticker | Worst Ticker |")
print("| :--- | :--- | :--- | :--- | :--- | :--- |")

timeframes = [30, 60, 90, 120, 240]

for days in timeframes:
    df_days = df[df['days'] == days]
    if df_days.empty:
        continue
        
    total_net = df_days['net'].sum()
    total_trades = df_days['trades'].sum()
    
    # Weighted win rate
    wins = (df_days['win_rate'] / 100) * df_days['trades']
    avg_win_rate = (wins.sum() / total_trades * 100) if total_trades > 0 else 0
    
    best = df_days.loc[df_days['net'].idxmax()]
    worst = df_days.loc[df_days['net'].idxmin()]
    
    print(f"| {days} Days | ${total_net:.2f} | {avg_win_rate:.1f}% | {total_trades} | {best['symbol']} (${best['net']:.2f}) | {worst['symbol']} (${worst['net']:.2f}) |")
