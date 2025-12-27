import pandas as pd

df = pd.read_csv("artifacts/stock-90d/backtest-results/detailed_trades.csv")

# Calculate stats
total_trades = len(df)
winners = len(df[df['pnl'] > 0])
losers = total_trades - winners
win_rate = winners / total_trades * 100
total_pnl = df['pnl'].sum()
avg_pnl = df['pnl'].mean()

print(f"\n{'='*60}")
print("15m STOCK BOT BACKTEST SUMMARY (90 Days)")
print(f"{'='*60}")
print(f"Total Trades: {total_trades}")
print(f"Winners: {winners} ({win_rate:.1f}%)")
print(f"Losers: {losers}")
print(f"Total PnL: ${total_pnl:,.2f}")
print(f"Avg PnL: ${avg_pnl:.2f}")
print(f"\nAvg Win: ${df[df['pnl'] > 0]['pnl'].mean():.2f}")
print(f"Avg Loss: ${df[df['pnl'] <= 0]['pnl'].mean():.2f}")
print(f"{'='*60}")
