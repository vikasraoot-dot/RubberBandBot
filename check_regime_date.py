"""Test regime for a specific date."""
import sys
import os

sys.path.insert(0, 'C:/Users/vraoo/GitHub/RubberBandBot/RubberBandBot')
from RubberBand.src.data import fetch_latest_bars

def check_regime_for_date(end_date):
    bars, _ = fetch_latest_bars(['VIXY'], '1Day', 35, feed='iex', end=end_date, verbose=False)
    df = bars.get('VIXY')

    if df is None or len(df) < 20:
        print('Insufficient data')
        return

    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['std_20'] = df['close'].rolling(window=20).std()
    df['vol_sma_20'] = df['volume'].rolling(window=20).mean()
    df['upper_band'] = df['sma_20'] + (2.0 * df['std_20'])
    df['prev_close'] = df['close'].shift(1)
    df['delta_pct'] = ((df['close'] - df['prev_close']) / df['prev_close']) * 100.0

    latest = df.iloc[-1]
    print(f'Date: {latest.name.date()}')
    print(f'VIXY Price: ${latest.close:.2f} (Delta: {latest.delta_pct:+.2f}%)')
    print(f'SMA20: ${latest.sma_20:.2f} | UpperBand: ${latest.upper_band:.2f}')
    print(f'Volume: {int(latest.volume):,} (Avg: {int(latest.vol_sma_20):,})')

    is_panic_price = (latest.close > latest.upper_band) or (latest.delta_pct > 8.0)
    is_high_volume = (latest.volume > 1.5 * latest.vol_sma_20)

    if is_panic_price and is_high_volume:
        regime = 'PANIC'
    elif is_panic_price:
        regime = 'NORMAL (High Price but Low Vol)'
    else:
        subset = df.iloc[-3:]
        all_below = all(row.close < row.sma_20 for _, row in subset.iterrows())
        regime = 'CALM' if all_below else 'NORMAL'

    print(f'Regime Verdict: {regime}')

if __name__ == '__main__':
    date = sys.argv[1] if len(sys.argv) > 1 else '2026-01-20'
    print(f'\n=== Checking Regime for {date} ===\n')
    check_regime_for_date(date)
