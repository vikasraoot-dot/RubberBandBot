import json
import pandas as pd

file_path = r"C:\Users\vraoo\GitHub\RubberBandBot\RubberBandBot\latest runs\full_scan_results.json"

try:
    with open(file_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    
    # Filter for 30-day results
    df_30 = df[df['days'] == 30]
    
    if df_30.empty:
        print("No 30-day results found.")
    else:
        total_tickers = len(df_30)
        profitable_tickers = len(df_30[df_30['net'] > 0])
        avg_pnl = df_30['net'].mean()
        avg_win_rate = df_30['win_rate'].mean()
        median_pnl = df_30['net'].median()
        
        print(f"Total Tickers Tested (30 days): {total_tickers}")
        print(f"Profitable Tickers: {profitable_tickers} ({profitable_tickers/total_tickers*100:.2f}%)")
        print(f"Average PnL: ${avg_pnl:.2f}")
        print(f"Median PnL: ${median_pnl:.2f}")
        print(f"Average Win Rate: {avg_win_rate:.2f}%")
        
        # Top 5 best and worst
        print("\nTop 5 Performers (30d):")
        print(df_30.nlargest(5, 'net')[['symbol', 'net', 'win_rate']])
        
        print("\nBottom 5 Performers (30d):")
        print(df_30.nsmallest(5, 'net')[['symbol', 'net', 'win_rate']])

except Exception as e:
    print(f"Error: {e}")
