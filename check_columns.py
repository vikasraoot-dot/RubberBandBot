
import pandas as pd
try:
    df = pd.read_csv("results/spread_backtest_trades.csv")
    print("Columns:", df.columns.tolist())
    if not df.empty:
        print("First row:", df.iloc[0].to_dict())
except Exception as e:
    print(e)
