import pandas as pd

# Load data
df = pd.read_csv(r"C:\Users\vraoo\GitHub\RubberBandBot\RubberBandBot\latest runs\12_16_2025\15m-options-backtest-20279741528\spread_backtest_trades.csv")

# Split by win/loss
wins = df[df['pnl'] > 0]
losses = df[df['pnl'] <= 0]

print("=" * 60)
print("SLOPE VS PNL ANALYSIS")
print("=" * 60)
print(f"\nTotal: {len(df)} trades, {len(wins)} wins ({len(wins)/len(df)*100:.1f}%), {len(losses)} losses ({len(losses)/len(df)*100:.1f}%)")

print(f"\n--- Average Entry Slope ---")
print(f"Wins avg slope:   {wins['entry_slope'].mean():.4f}")
print(f"Losses avg slope: {losses['entry_slope'].mean():.4f}")

print(f"\n--- Slope Ranges ---")
print(f"Win slope range:  {wins['entry_slope'].min():.4f} to {wins['entry_slope'].max():.4f}")
print(f"Loss slope range: {losses['entry_slope'].min():.4f} to {losses['entry_slope'].max():.4f}")

# Analyze by slope buckets
print(f"\n--- Win Rate by Slope Bucket ---")
buckets = [
    (-2.0, -0.50),
    (-0.50, -0.30),
    (-0.30, -0.20),
    (-0.20, -0.10),
    (-0.10, 0.00),
]

for low, high in buckets:
    bucket = df[(df['entry_slope'] >= low) & (df['entry_slope'] < high)]
    if len(bucket) > 0:
        bucket_wins = len(bucket[bucket['pnl'] > 0])
        bucket_wr = bucket_wins / len(bucket) * 100
        bucket_pnl = bucket['pnl'].sum()
        print(f"  Slope [{low:.2f} to {high:.2f}]: {len(bucket):3d} trades, WR={bucket_wr:.1f}%, Total PnL=${bucket_pnl:,.0f}")

# Best trades by slope
print(f"\n--- Top 5 Winners (by PnL) ---")
top_wins = wins.nlargest(5, 'pnl')[['symbol', 'entry_slope', 'pnl', 'pnl_pct']]
print(top_wins.to_string(index=False))

print(f"\n--- Top 5 Losers (by PnL) ---")
top_losses = losses.nsmallest(5, 'pnl')[['symbol', 'entry_slope', 'pnl', 'pnl_pct']]
print(top_losses.to_string(index=False))
