import pandas as pd

df = pd.read_csv("optimization_results_new.csv")
print("Top 10 Configurations by Net PnL:")
print(df.nlargest(10, "Net PnL"))

print("\nBaseline Comparison (approx SL=1.5, TP=2.0, RSI=25):")
baseline = df[(df['atr_mult_sl'] == 1.5) & (df['take_profit_r'] == 2.0) & (df['rsi_oversold'] == 25)]
print(baseline)

print("\nBest Win Rate (with positive PnL):")
positive_pnl = df[df['Net PnL'] > 0]
if not positive_pnl.empty:
    print(positive_pnl.nlargest(5, "Win Rate"))
else:
    print("No positive PnL configs found.")
