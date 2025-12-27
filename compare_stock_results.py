import pandas as pd

def compare_results():
    print("\n" + "="*70)
    print("STOCK BOT COMPARISON: Dual-Slope vs 3-Bar Only")
    print("="*70)
    
    # Load both datasets
    dual_slope = pd.read_csv("artifacts/stock-90d/backtest-results/detailed_trades.csv")
    no_slope10 = pd.read_csv("artifacts/stock-90d-no-slope10/backtest-results/detailed_trades.csv")
    
    def stats(df, label):
        total = len(df)
        winners = len(df[df['pnl'] > 0])
        losers = total - winners
        win_rate = winners / total * 100 if total > 0 else 0
        total_pnl = df['pnl'].sum()
        avg_pnl = df['pnl'].mean() if total > 0 else 0
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winners > 0 else 0
        avg_loss = df[df['pnl'] <= 0]['pnl'].mean() if losers > 0 else 0
        
        print(f"\n{label}")
        print("-"*40)
        print(f"  Trades:    {total:>6}")
        print(f"  Winners:   {winners:>6} ({win_rate:.1f}%)")
        print(f"  Losers:    {losers:>6}")
        print(f"  Total PnL: ${total_pnl:>9,.2f}")
        print(f"  Avg PnL:   ${avg_pnl:>9.2f}")
        print(f"  Avg Win:   ${avg_win:>9.2f}")
        print(f"  Avg Loss:  ${avg_loss:>9.2f}")
        
        return {
            'trades': total,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl
        }
    
    dual = stats(dual_slope, "WITH Dual-Slope Filter (slope_3 + slope_10)")
    single = stats(no_slope10, "WITHOUT 10-Bar Slope (slope_3 only)")
    
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Metric':<20} | {'Dual-Slope':>15} | {'3-Bar Only':>15} | {'Difference':>15}")
    print("-"*70)
    print(f"{'Trades':<20} | {dual['trades']:>15} | {single['trades']:>15} | {single['trades'] - dual['trades']:>+15}")
    print(f"{'Win Rate %':<20} | {dual['win_rate']:>14.1f}% | {single['win_rate']:>14.1f}% | {single['win_rate'] - dual['win_rate']:>+14.1f}%")
    print(f"{'Total PnL':<20} | ${dual['total_pnl']:>13,.2f} | ${single['total_pnl']:>13,.2f} | ${single['total_pnl'] - dual['total_pnl']:>+13,.2f}")
    print(f"{'Avg PnL':<20} | ${dual['avg_pnl']:>13.2f} | ${single['avg_pnl']:>13.2f} | ${single['avg_pnl'] - dual['avg_pnl']:>+13.2f}")
    
    if single['total_pnl'] > dual['total_pnl']:
        print(f"\n✅ RECOMMENDATION: DISABLE 10-bar slope filter for Stock Bot")
    else:
        print(f"\n⚠️ RECOMMENDATION: KEEP 10-bar slope filter for Stock Bot")

if __name__ == "__main__":
    compare_results()
