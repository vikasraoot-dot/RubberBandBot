
import pandas as pd
from datetime import timedelta

# Load CSV
df = pd.read_csv('weekly_options_backtest.csv', parse_dates=['entry_date'])
df = df.sort_values('entry_date')

# Calculate trades per week
df['week'] = df['entry_date'].dt.to_period('W').apply(lambda r: r.start_time)
weekly_counts = df.groupby('week').size()

print(f"Total Weeks with Trades: {len(weekly_counts)}")
print(f"Total Trades: {len(df)}")
print(f"Average Trades per Active Week: {weekly_counts.mean():.2f}")

# Find gaps > 2 weeks
print("\n--- Significant Idle Periods (No Trades > 2 Weeks) ---")
dates = sorted(df['entry_date'].unique())
for i in range(len(dates)-1):
    d1 = dates[i]
    d2 = dates[i+1]
    gap = (d2 - d1).days
    if gap > 14:
        print(f"Gap: {gap} days ({d1.date()} to {d2.date()})")

# Check specifically for recent period (Dec 2024)
print("\n--- Recent Activity (Nov-Dec 2024) ---")
recent = df[(df['entry_date'] >= '2024-11-01') & (df['entry_date'] <= '2024-12-31')]
if recent.empty:
    print("No trades found in Nov-Dec 2024")
else:
    print(recent[['symbol', 'entry_date', 'reason']])
