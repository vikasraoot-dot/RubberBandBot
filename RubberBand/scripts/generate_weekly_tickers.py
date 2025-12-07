
import pandas as pd

# Load the analysis results
trades = pd.read_csv('weekly_detailed_trades.csv')
analysis = pd.read_csv('RubberBand/ticker_analysis.csv')

# Aggregate PnL
perf = trades.groupby('symbol')['pnl'].sum().reset_index()

# Merge with analysis (for volume data)
merged = perf.merge(analysis, left_on='symbol', right_on='Ticker', how='left')

# Filter Criteria
# 1. Must be Profitable (PnL > 0)
# 2. Must be Liquid (Avg Dollar Vol > $500M) - Avoiding illiquid traps
# 3. Optional: Price > $20 (Avoid strict penny stocks, though most liquid ones are >$20)

filtered = merged[
    (merged['pnl'] > 0) & 
    (merged['Avg_Dollar_Vol_M'] > 500)
]

# Sort by PnL
filtered = filtered.sort_values('pnl', ascending=False)

# Get the list
ticker_list = filtered['symbol'].tolist()

# Write to file
with open('RubberBand/tickers_weekly.txt', 'w') as f:
    for ticker in ticker_list:
        f.write(f"{ticker}\n")

print(f"Created RubberBand/tickers_weekly.txt with {len(ticker_list)} tickers.")
print(f"Top 5: {ticker_list[:5]}")
print(f"Removed {len(merged) - len(filtered)} tickers (Unprofitable or Illiquid).")

# Print the list of removed tickers for transparency
removed = merged[~merged['symbol'].isin(ticker_list)]
print("\nRemoved Tickers (Losers/Illiquid):")
print(removed['symbol'].tolist())
