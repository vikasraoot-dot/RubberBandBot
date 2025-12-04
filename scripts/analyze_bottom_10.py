import json
import pandas as pd

def main():
    with open("backtest_summary.json", "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Pivot table for the bottom 10
    pivot_pnl = df.pivot(index="symbol", columns="days", values="net")
    
    # Sort by 30-day performance to see recent trends
    pivot_pnl = pivot_pnl.sort_values(30, ascending=False)
    
    print("### Bottom 10 (3-Year Losers) Performance by Timeframe")
    print(pivot_pnl.to_string(float_format="%.2f"))
    
    print("\n### Win Rate %")
    pivot_wr = df.pivot(index="symbol", columns="days", values="win_rate")
    pivot_wr = pivot_wr.reindex(pivot_pnl.index)
    print(pivot_wr.to_string(float_format="%.1f"))

if __name__ == "__main__":
    main()
