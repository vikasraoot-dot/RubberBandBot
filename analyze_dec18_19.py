
import pandas as pd
import sys

def main():
    try:
        df = pd.read_csv("detailed_trades.csv")
    except FileNotFoundError:
        print("Error: detailed_trades.csv not found.")
        return

    # Filter for Dec 18 and 19 (2025)
    # The 'date' column is exit date. We should probably look at entry or exit depending on what user wants.
    # Usually strategy performance "on a day" means realized PnL on that day (exit).
    # But let's show both if relevant.
    
    # 2025-12-18 and 2025-12-19
    target_dates = ["2025-12-18", "2025-12-19"]
    
    # Ensure date column is string or datetime
    df['date'] = df['date'].astype(str)
    
    filtered = df[df['date'].isin(target_dates)].copy()
    
    if filtered.empty:
        print("No trades closed on Dec 18 or 19.")
        return

    print(f"=== Trades Closed on Dec 18 & 19 (With Dead Knife Filter) ===")
    print(f"{'Date':<12} {'Symbol':<6} {'Side':<6} {'Entry':<8} {'Exit':<8} {'PnL':<8} {'Reason':<10}")
    print("-" * 65)
    
    total_pnl = 0
    total_trades = 0
    wins = 0
    
    for _, row in filtered.iterrows():
        pnl = row['pnl']
        total_pnl += pnl
        total_trades += 1
        if pnl > 0: wins += 1
        
        print(f"{row['date']:<12} {row['symbol']:<6} {row['side']:<6} {row['entry_price']:<8.2f} {row['exit_price']:<8.2f} {pnl:<8.2f} {row['exit_reason']:<10}")

    print("-" * 65)
    print(f"Total PnL: ${total_pnl:.2f}")
    print(f"Trades: {total_trades} (Win Rate: {wins/total_trades*100:.1f}%)")

if __name__ == "__main__":
    main()
