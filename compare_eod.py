"""
Compare backtest results WITH and WITHOUT EOD flattening.
"""
import pandas as pd
import os

print("="*70)
print("EOD FLATTENING COMPARISON ANALYSIS")
print("="*70)

# Check if we have both result sets
if not os.path.exists('detailed_trades.csv'):
    print("ERROR: No backtest results found. Run backtest first.")
    exit(1)

# Load current results (should be WITHOUT EOD flattening)
df_current = pd.read_csv('detailed_trades.csv')

print(f"\n--- CURRENT RUN (Check flatten_eod setting) ---")
print(f"Total Trades: {len(df_current)}")
print(f"Total PnL: ${df_current['pnl'].sum():.2f}")
print(f"Win Rate: {(df_current['pnl'] > 0).sum() / len(df_current) * 100:.1f}%")

print(f"\nExit Reason Breakdown:")
exit_counts = df_current['exit_reason'].value_counts()
for reason, count in exit_counts.items():
    pct = count / len(df_current) * 100
    reason_df = df_current[df_current['exit_reason'] == reason]
    wr = (reason_df['pnl'] > 0).sum() / len(reason_df) * 100
    total_pnl = reason_df['pnl'].sum()
    avg_pnl = reason_df['pnl'].mean()
    print(f"  {reason:10s}: {count:4d} ({pct:5.1f}%) | WR: {wr:5.1f}% | Total: ${total_pnl:8.2f} | Avg: ${avg_pnl:6.2f}")

# Analyze hold duration
print(f"\n--- HOLD DURATION ANALYSIS ---")
print(f"Average Hold Duration: {df_current['hold_duration_days'].mean():.2f} days")
print(f"Median Hold Duration: {df_current['hold_duration_days'].median():.2f} days")
print(f"Max Hold Duration: {df_current['hold_duration_days'].max():.2f} days")

# Overnight holds
overnight = df_current[df_current['hold_duration_days'] > 0.5]
print(f"\nOvernight Holds (>12h): {len(overnight)} ({len(overnight)/len(df_current)*100:.1f}%)")
if len(overnight) > 0:
    print(f"  Win Rate: {(overnight['pnl'] > 0).sum() / len(overnight) * 100:.1f}%")
    print(f"  Avg PnL: ${overnight['pnl'].mean():.2f}")
    print(f"  Total PnL: ${overnight['pnl'].sum():.2f}")

# Multi-day holds
multiday = df_current[df_current['hold_duration_days'] > 1.0]
print(f"\nMulti-Day Holds (>1d): {len(multiday)} ({len(multiday)/len(df_current)*100:.1f}%)")
if len(multiday) > 0:
    print(f"  Win Rate: {(multiday['pnl'] > 0).sum() / len(multiday) * 100:.1f}%")
    print(f"  Avg PnL: ${multiday['pnl'].mean():.2f}")
    print(f"  Total PnL: ${multiday['pnl'].sum():.2f}")

print("\n" + "="*70)
print("INSTRUCTIONS FOR COMPARISON:")
print("="*70)
print("1. Save these results")
print("2. Run backtest WITH EOD flattening:")
print("   python RubberBand/scripts/backtest.py --tickers RubberBand/candidates_tickers.txt --days 30 --flatten-eod")
print("3. Compare the two sets of results")
print("="*70)
