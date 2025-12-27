import pandas as pd
import os

def analyze_losses():
    # Load all three CSVs
    periods = {
        "90d": "artifacts/options-90d/15m-options-backtest-20326921935/spread_backtest_trades.csv",
        "60d": "artifacts/options-60d/15m-options-backtest-20326920825/spread_backtest_trades.csv",
        "30d": "artifacts/options-30d/15m-options-backtest-20326916934/spread_backtest_trades.csv",
    }
    
    all_trades = []
    for period, path in periods.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['period'] = period
            all_trades.append(df)
    
    if not all_trades:
        print("No trade files found")
        return
    
    df = pd.concat(all_trades, ignore_index=True)
    
    # Filter losers
    losers = df[df['pnl'] < 0].copy()
    winners = df[df['pnl'] > 0].copy()
    
    print("="*70)
    print("LOSS ANALYSIS REPORT")
    print("="*70)
    print(f"\nTotal Trades: {len(df)}")
    print(f"Winners: {len(winners)} ({len(winners)/len(df)*100:.1f}%)")
    print(f"Losers: {len(losers)} ({len(losers)/len(df)*100:.1f}%)")
    
    # ---- Slope Analysis ----
    print("\n" + "-"*70)
    print("SLOPE COMPARISON (Winners vs Losers)")
    print("-"*70)
    
    if 'entry_slope' in df.columns:
        print(f"\n3-Bar Slope (entry_slope):")
        print(f"  Winners Avg:  {winners['entry_slope'].mean():.4f}")
        print(f"  Losers Avg:   {losers['entry_slope'].mean():.4f}")
        print(f"  Winners Range: {winners['entry_slope'].min():.4f} to {winners['entry_slope'].max():.4f}")
        print(f"  Losers Range:  {losers['entry_slope'].min():.4f} to {losers['entry_slope'].max():.4f}")
    
    if 'entry_slope_10' in df.columns:
        print(f"\n10-Bar Slope (entry_slope_10):")
        print(f"  Winners Avg:  {winners['entry_slope_10'].mean():.4f}")
        print(f"  Losers Avg:   {losers['entry_slope_10'].mean():.4f}")
        print(f"  Winners Range: {winners['entry_slope_10'].min():.4f} to {winners['entry_slope_10'].max():.4f}")
        print(f"  Losers Range:  {losers['entry_slope_10'].min():.4f} to {losers['entry_slope_10'].max():.4f}")
    
    # ---- RSI Analysis ----
    print("\n" + "-"*70)
    print("RSI COMPARISON")
    print("-"*70)
    if 'entry_rsi' in df.columns:
        print(f"  Winners Avg RSI: {winners['entry_rsi'].mean():.1f}")
        print(f"  Losers Avg RSI:  {losers['entry_rsi'].mean():.1f}")
    
    # ---- ADX Analysis ----
    print("\n" + "-"*70)
    print("ADX COMPARISON")
    print("-"*70)
    if 'entry_adx' in df.columns:
        print(f"  Winners Avg ADX: {winners['entry_adx'].mean():.1f}")
        print(f"  Losers Avg ADX:  {losers['entry_adx'].mean():.1f}")
    
    # ---- Symbol Analysis ----
    print("\n" + "-"*70)
    print("SYMBOL LOSS DISTRIBUTION (Top 10 Losers)")
    print("-"*70)
    symbol_stats = losers.groupby('symbol').agg(
        count=('pnl', 'count'),
        total_loss=('pnl', 'sum'),
        avg_loss=('pnl', 'mean')
    ).sort_values('total_loss')
    print(symbol_stats.head(10).to_string())
    
    # ---- Exit Reason Analysis ----
    print("\n" + "-"*70)
    print("EXIT REASON (Losers Only)")
    print("-"*70)
    if 'reason' in losers.columns:
        print(losers['reason'].value_counts().to_string())
    
    # ---- Bars Held Analysis ----
    print("\n" + "-"*70)
    print("BARS HELD COMPARISON")
    print("-"*70)
    if 'bars_held' in df.columns:
        print(f"  Winners Avg Bars: {winners['bars_held'].mean():.1f}")
        print(f"  Losers Avg Bars:  {losers['bars_held'].mean():.1f}")
    
    # ---- Print All Losing Trades ----
    print("\n" + "-"*70)
    print(f"ALL LOSING TRADES ({len(losers)} total)")
    print("-"*70)
    cols_to_show = ['symbol', 'entry_time', 'pnl', 'reason', 'entry_slope', 'entry_slope_10', 'entry_rsi', 'entry_adx', 'bars_held']
    cols_available = [c for c in cols_to_show if c in losers.columns]
    print(losers[cols_available].sort_values('pnl').to_string())

if __name__ == "__main__":
    analyze_losses()
