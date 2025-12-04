import pandas as pd

df = pd.read_csv('detailed_trades.csv')

print(f"Total trades: {len(df)}")
print(f"\nExit Reason Distribution:")
print(df['exit_reason'].value_counts())

eod = df[df['exit_reason'] == 'EOD']
print(f"\n--- EOD FLATTENING ANALYSIS ---")
print(f"EOD exits: {len(eod)} ({len(eod)/len(df)*100:.1f}%)")
if len(eod) > 0:
    print(f"EOD Win Rate: {(eod['pnl'] > 0).sum() / len(eod) * 100:.1f}%")
    print(f"EOD Avg PnL: ${eod['pnl'].mean():.2f}")
    print(f"EOD Total PnL: ${eod['pnl'].sum():.2f}")
else:
    print("No EOD exits found!")

print(f"\n--- OVERALL STATS ---")
print(f"Total PnL: ${df['pnl'].sum():.2f}")
print(f"Win Rate: {(df['pnl'] > 0).sum() / len(df) * 100:.1f}%")
