import json
import pandas as pd

def rank_tickers():
    with open("backtest_summary.json", "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Pivot to get columns for each timeframe
    pivot = df.pivot(index="symbol", columns="days", values="net")
    pivot["Total_240d"] = pivot[240]
    pivot["WinRate_240d"] = df[df["days"] == 240].set_index("symbol")["win_rate"]
    
    # Sort by 240d Net PnL
    pivot = pivot.sort_values("Total_240d", ascending=False)
    
    print("| Rank | Ticker | 240d Net | 120d Net | 90d Net | 60d Net | 30d Net | 240d Win Rate |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    rank = 1
    for sym, row in pivot.iterrows():
        p240 = f"${row[240]:.2f}"
        p120 = f"${row[120]:.2f}"
        p90 = f"${row[90]:.2f}"
        p60 = f"${row[60]:.2f}"
        p30 = f"${row[30]:.2f}"
        wr = f"{row['WinRate_240d']:.1f}%"
        print(f"| {rank} | **{sym}** | {p240} | {p120} | {p90} | {p60} | {p30} | {wr} |")
        rank += 1

if __name__ == "__main__":
    rank_tickers()
