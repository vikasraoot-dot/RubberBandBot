import pandas as pd

# Read scan results  
scan_df = pd.read_csv('scan_artifacts/candidates-csv/candidates.csv')

# Read current tickers
current_tickers = set()
with open('RubberBand/tickers.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            current_tickers.add(line)

# Find delta
scan_symbols = set(scan_df['symbol'].tolist())
delta_symbols = scan_symbols - current_tickers
in_both = scan_symbols & current_tickers

# Filter to delta
delta_df = scan_df[scan_df['symbol'].isin(delta_symbols)].sort_values('dollar_vol_m', ascending=False)

print(f'SCAN SUMMARY:')
print(f'  Total in scan: {len(scan_symbols)}')
print(f'  Current tickers: {len(current_tickers)}')
print(f'  Already in list: {len(in_both)}')
print(f'  NEW (delta): {len(delta_symbols)}')
print()

if len(delta_df) > 0:
    print('NEW CANDIDATES (not in current list):')
    print('-' * 70)
    print(f"{'Symbol':6} {'Price':>8} {'SMA120':>8} {'SMA22':>8} {'ATR%':>6} {'Vol($M)':>8}")
    print('-' * 70)
    for _, row in delta_df.iterrows():
        print(f"{row['symbol']:6} ${row['price']:>7.2f} ${row['sma_120']:>7.2f} ${row['sma_22']:>7.2f} {row['atr_pct']:>5.2f}% ${row['dollar_vol_m']:>6.2f}M")
else:
    print('No new candidates found.')
