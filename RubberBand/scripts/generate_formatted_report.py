
import pandas as pd
import os

# Paths
baseline_csv = r"C:\Users\vraoo\GitHub\RubberBandBot\RubberBandBot\latest runs\12_16_2025\BacktestRuns\Live_baseline_20291353797\15m-options-backtest-20291353797\spread_backtest_trades.csv"
adx_csv = r"C:\Users\vraoo\GitHub\RubberBandBot\RubberBandBot\latest runs\12_16_2025\BacktestRuns\With_ADX_60_20291376695\15m-options-backtest-20291376695\spread_backtest_trades.csv"

def generate_block(csv_path, title):
    if not os.path.exists(csv_path):
        return f"Error: {csv_path} not found"
        
    df = pd.read_csv(csv_path)
    
    # Metrics
    total_trades = len(df)
    
    # PnL & Cost
    total_pnl = df['pnl'].sum()
    
    # Cost is explicitly in the CSV
    if 'cost' in df.columns:
        total_cost = df['cost'].sum()
    else:
        # Fallback if cost missing (shouldn't happen based on CSV check)
        entry_debit = df['entry_debit'] if 'entry_debit' in df.columns else 0
        total_cost = (entry_debit * 100).sum() # Assume qty 1
    
    roi = (total_pnl / total_cost * 100) if total_cost > 0 else 0
    
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]
    
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    
    avg_win = wins['pnl'].mean() if not wins.empty else 0
    avg_loss = losses['pnl'].mean() if not losses.empty else 0
    
    # average bars held
    avg_bars = df['bars_held'].mean() if 'bars_held' in df.columns else 0
    
    # Exit Reasons
    # Group by 'reason'
    exits = df['reason'].value_counts()
    
    # Map to requested format
    # CSV reasons seen: MAX_PROFIT, BARS_STOP, STOP_LOSS
    
    sl_count = exits.get('STOP_LOSS', 0)
    tp_count = exits.get('MAX_PROFIT', 0)
    mean_rev_count = exits.get('MEAN_REVERSION', 0) # Guessing key
    bars_stop_count = exits.get('BARS_STOP', 0)
    eod_count = exits.get('EOD', 0)
    
    time_exit_count = bars_stop_count + eod_count
    
    return f"""============================================================
{title}
============================================================
Period: 30 days
Symbols: 120
Spread Config: DTE=2, Width=1.5 ATR, ADX Filter={'Enabled (<60)' if 'ADX' in title else 'Disabled'}
------------------------------------------------------------
Total Trades: {total_trades}
Total Cost: ${total_cost:,.2f}
Total P&L: ${total_pnl:,.2f}
ROI: {roi:.1f}%
Win Rate: {win_rate:.1f}% ({win_count}W / {loss_count}L)
Avg Win: ${avg_win:.2f}
Avg Loss: ${avg_loss:.2f}
Avg Bars Held: {avg_bars:.1f}
------------------------------------------------------------
Exit @ Max Profit: {tp_count} ({tp_count/total_trades*100:.1f}%)
Exit @ Max Loss: {sl_count} ({sl_count/total_trades*100:.1f}%)
Exit @ Time Limit: {time_exit_count} ({time_exit_count/total_trades*100:.1f}%)
============================================================
"""

print(generate_block(baseline_csv, "BASELINE RESULTS (NO FILTER)"))
print("\n")
print(generate_block(adx_csv, "ADX FILTER RESULTS (MAX 60)"))
