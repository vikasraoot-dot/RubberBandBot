
import pandas as pd
import numpy as np
from datetime import datetime

# Load data
trades = pd.read_csv('weekly_detailed_trades.csv')
with open('RubberBand/tickers_weekly.txt', 'r') as f:
    elite_tickers = [t.strip() for t in f if t.strip()]

# Filter for elite tickers only
df = trades[trades['symbol'].isin(elite_tickers)].copy()

# Convert dates
df['entry_date'] = pd.to_datetime(df['entry_time'])
df['year'] = df['entry_date'].dt.year

# Calculate Stats per Ticker
results = []

for ticker in elite_tickers:
    t_df = df[df['symbol'] == ticker]
    if t_df.empty:
        continue
        
    # Yearly PnL
    pnl_by_year = t_df.groupby('year')['pnl'].sum()
    
    # Capital Usage (Max Capital Deployed at once)
    # Since we don't have overlapped trade checking here easily without simulating time, 
    # we'll approximate "Working Capital" as the max single position size deployed 
    # (since this strat usually holds 1 pos per ticker). 
    # The user asked for "working capital needed for this profit to be generated" which per ticker 
    # is roughly the position size ($5000 in our config) x max concurrent trades. 
    # But since we're looking at per-ticker stats, it's just the max cost basis.
    max_capital = (t_df['entry_price'] * t_df['qty']).max()
    
    # Total PnL
    total_pnl = t_df['pnl'].sum()
    
    # ROI
    roi = (total_pnl / max_capital) * 100 if max_capital > 0 else 0
    
    row = {
        'Symbol': ticker,
        'Total_PnL': total_pnl,
        'Max_Capital': max_capital,
        'ROI_%': roi,
        'Trades': len(t_df)
    }
    
    # Add yearly columns dynamically
    years = sorted(df['year'].unique())
    for y in years:
        row[f'PnL_{y}'] = pnl_by_year.get(y, 0.0)
        
    results.append(row)

# Create DataFrame
res_df = pd.DataFrame(results)

# Sort by Total PnL
res_df = res_df.sort_values('Total_PnL', ascending=False)

# Format for display
display_cols = ['Symbol', 'Total_PnL', 'Max_Capital', 'ROI_%', 'Trades'] + [c for c in res_df.columns if 'PnL_' in c]

print("="*100)
print("ELITE 56 TICKER PERFORMANCE MATRIX (3-YEAR BREAKDOWN)")
print("="*100)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.0f' % x)

print(res_df[display_cols].to_string(index=False))

# Export to CSV for user
res_df[display_cols].to_csv('RubberBand/elite_ticker_yearly_performance.csv', index=False)
print("\nSaved to RubberBand/elite_ticker_yearly_performance.csv")
