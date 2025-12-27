import pandas as pd
import os

def analyze_adx_threshold():
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
    
    # Remove duplicates (same trade appears in multiple time windows)
    df = df.drop_duplicates(subset=['symbol', 'entry_time', 'pnl'])
    
    print("="*80)
    print("ADX THRESHOLD IMPACT ANALYSIS")
    print("="*80)
    print(f"\nTotal Unique Trades: {len(df)}")
    
    # Test multiple thresholds
    thresholds = [30, 35, 40, 45, 50]
    
    print("\n" + "-"*80)
    print(f"{'ADX Thresh':<12} | {'Trades':<8} | {'Winners':<8} | {'Losers':<8} | {'Win%':<8} | {'Total PnL':<12} | {'Avg PnL':<10}")
    print("-"*80)
    
    # Baseline (no filter)
    baseline_trades = len(df)
    baseline_winners = len(df[df['pnl'] > 0])
    baseline_losers = len(df[df['pnl'] <= 0])
    baseline_pnl = df['pnl'].sum()
    baseline_wr = baseline_winners / baseline_trades * 100
    baseline_avg = df['pnl'].mean()
    print(f"{'None':<12} | {baseline_trades:<8} | {baseline_winners:<8} | {baseline_losers:<8} | {baseline_wr:<7.1f}% | ${baseline_pnl:<11,.2f} | ${baseline_avg:<9.2f}")
    
    # With each threshold
    for thresh in thresholds:
        filtered = df[df['entry_adx'] <= thresh]
        n_trades = len(filtered)
        n_winners = len(filtered[filtered['pnl'] > 0])
        n_losers = n_trades - n_winners
        total_pnl = filtered['pnl'].sum()
        avg_pnl = filtered['pnl'].mean() if n_trades > 0 else 0
        wr = n_winners / n_trades * 100 if n_trades > 0 else 0
        
        # Calculate impact vs baseline
        trades_lost = baseline_trades - n_trades
        pnl_lost = baseline_pnl - total_pnl
        winners_lost = baseline_winners - n_winners
        losers_filtered = baseline_losers - n_losers
        
        print(f"ADX <= {thresh:<5} | {n_trades:<8} | {n_winners:<8} | {n_losers:<8} | {wr:<7.1f}% | ${total_pnl:<11,.2f} | ${avg_pnl:<9.2f}")
    
    print("-"*80)
    
    # Detailed impact for ADX <= 40
    print("\n" + "="*80)
    print("DETAILED IMPACT: ADX <= 40")
    print("="*80)
    
    thresh = 40
    kept = df[df['entry_adx'] <= thresh]
    dropped = df[df['entry_adx'] > thresh]
    
    kept_winners = kept[kept['pnl'] > 0]
    kept_losers = kept[kept['pnl'] <= 0]
    dropped_winners = dropped[dropped['pnl'] > 0]
    dropped_losers = dropped[dropped['pnl'] <= 0]
    
    print(f"\n--- TRADES KEPT (ADX <= {thresh}) ---")
    print(f"Total: {len(kept)} trades")
    print(f"Winners: {len(kept_winners)} (Total: ${kept_winners['pnl'].sum():,.2f})")
    print(f"Losers: {len(kept_losers)} (Total: ${kept_losers['pnl'].sum():,.2f})")
    print(f"Net PnL: ${kept['pnl'].sum():,.2f}")
    
    print(f"\n--- TRADES DROPPED (ADX > {thresh}) ---")
    print(f"Total: {len(dropped)} trades")
    print(f"Winners SACRIFICED: {len(dropped_winners)} (Total: ${dropped_winners['pnl'].sum():,.2f})")
    print(f"Losers AVOIDED: {len(dropped_losers)} (Total: ${dropped_losers['pnl'].sum():,.2f})")
    print(f"Net Impact: ${dropped_winners['pnl'].sum() + dropped_losers['pnl'].sum():,.2f} (Positive = We lost profit)")
    
    if len(dropped_winners) > 0:
        print(f"\n--- WINNERS WE WOULD SACRIFICE ---")
        cols = ['symbol', 'entry_time', 'pnl', 'entry_adx', 'entry_slope']
        cols_avail = [c for c in cols if c in dropped_winners.columns]
        print(dropped_winners[cols_avail].sort_values('pnl', ascending=False).head(10).to_string())
    
    if len(dropped_losers) > 0:
        print(f"\n--- LOSERS WE WOULD AVOID ---")
        cols = ['symbol', 'entry_time', 'pnl', 'entry_adx', 'entry_slope']
        cols_avail = [c for c in cols if c in dropped_losers.columns]
        print(dropped_losers[cols_avail].sort_values('pnl').to_string())

if __name__ == "__main__":
    analyze_adx_threshold()
