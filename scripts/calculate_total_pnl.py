import json
import pandas as pd

def main():
    # Load the backtest results
    try:
        with open("backtest_summary.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: backtest_summary.json not found.")
        return

    df = pd.DataFrame(data)
    
    # Filter for the 3-year backtest (1070 days)
    # The file might contain mixed results if not cleared, but the last run was specific.
    # Let's assume the file was overwritten or we filter by the 'days' parameter if needed.
    # The last run used --days 1070.
    
    df_3yr = df[df["days"] == 1070]
    
    if df_3yr.empty:
        print("No 3-year (1070 days) backtest data found.")
        return

    total_pnl = df_3yr["net"].sum()
    total_trades = df_3yr["trades"].sum()
    avg_win_rate = df_3yr["win_rate"].mean()
    
    # Get top winners and losers
    winners = df_3yr[df_3yr["net"] > 0].sort_values("net", ascending=False)
    losers = df_3yr[df_3yr["net"] < 0].sort_values("net", ascending=True)
    
    print(f"### Total 3-Year Performance (Jan 2023 - Present)")
    print(f"**Total Net PnL:** ${total_pnl:,.2f}")
    print(f"**Total Trades:** {total_trades}")
    print(f"**Average Win Rate:** {avg_win_rate:.1f}%")
    print(f"**Ticker Count:** {len(df_3yr)}")
    
    print("\n**Top 5 Contributors:**")
    print(winners.head(5)[["symbol", "net", "win_rate"]].to_string(index=False))
    
    print("\n**Bottom 10 Drags:**")
    bottom_10 = losers.head(10)
    print(bottom_10[["symbol", "net", "win_rate"]].to_string(index=False))
    
    # Save bottom 10 symbols to file for backtesting
    with open("tickers_bottom_10.txt", "w") as f:
        for sym in bottom_10["symbol"]:
            f.write(f"{sym}\n")
    print("\nSaved bottom 10 symbols to tickers_bottom_10.txt")

if __name__ == "__main__":
    main()
