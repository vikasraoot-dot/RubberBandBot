
import pandas as pd

# Load both datasets
trades = pd.read_csv('weekly_detailed_trades.csv')
analysis = pd.read_csv('RubberBand/ticker_analysis.csv')

# Aggregate trade performance by ticker
perf = trades.groupby('symbol').agg({
    'pnl': 'sum',
    'qty': 'count', # Using qty roughly as trade count proxy if 1 trade at a time
    'entry_price': 'mean' # Just for reference
}).reset_index()

# Note: 'qty' col in detailed trades is actual shares, so we need to count rows
perf_counts = trades['symbol'].value_counts().reset_index()
perf_counts.columns = ['symbol', 'trade_count']
perf = perf.merge(perf_counts, on='symbol')

# Calculate Win Rate
wins = trades[trades['pnl'] > 0].groupby('symbol').size().reset_index(name='wins')
perf = perf.merge(wins, on='symbol', how='left')
perf['wins'] = perf['wins'].fillna(0)
perf['win_rate'] = (perf['wins'] / perf['trade_count']) * 100

# Merge with attributes
merged = perf.merge(analysis, left_on='symbol', right_on='Ticker', how='left')

# Drop unanalyzed tickers (if any)
merged = merged.dropna(subset=['ATR%'])

# Sort by PnL
merged = merged.sort_values('pnl', ascending=False)

print("="*80)
print("CORRELATION ANALYSIS: What makes a winning Weekly RubberBand Ticker?")
print("="*80)

# Sort by PnL
merged = merged.sort_values('pnl', ascending=False)

print("="*100)
print("FULL TICKER PERFORMANCE ANALYSIS (Sorted by Profit)")
print("Cols: PnL, WinRate, TradeCount, Price, ATR%, Avg$Vol(M), MaxDD")
print("="*100)

pd.set_option('display.max_rows', None)
print(merged[['Ticker', 'pnl', 'win_rate', 'trade_count', 'Price', 'ATR%', 'Avg_Dollar_Vol_M', 'Max_Drawdown']].to_string(index=False))

# Group by Price Buckets
merged['Price_Bucket'] = pd.qcut(merged['Price'], 3, labels=["Low Price", "Mid Price", "High Price"])
grouped_price = merged.groupby('Price_Bucket').agg({
    'pnl': 'mean',
    'win_rate': 'mean',
    'trade_count': 'mean',
    'ATR%': 'mean'
})
print(f"\nPerformance by Price Bucket:")
print(grouped_price.to_string())

# Group by Volatility Buckets
merged['Volatility_Bucket'] = pd.qcut(merged['ATR%'], 3, labels=["Low Vol", "Mid Vol", "High Vol"])
grouped = merged.groupby('Volatility_Bucket').agg({
    'pnl': 'mean',
    'win_rate': 'mean',
    'trade_count': 'mean',
    'ATR%': 'mean'
})
print(f"\nPerformance by Volatility Bucket:")
print(grouped.to_string())

# Group by Volume Buckets
merged['Volume_Bucket'] = pd.qcut(merged['Avg_Dollar_Vol_M'], 3, labels=["Low Liq", "Mid Liq", "High Liq"])
grouped_vol = merged.groupby('Volume_Bucket').agg({
    'pnl': 'mean',
    'win_rate': 'mean',
    'trade_count': 'mean',
    'Avg_Dollar_Vol_M': 'mean'
})
print(f"\nPerformance by Liquidity Bucket:")
print(grouped_vol.to_string())
