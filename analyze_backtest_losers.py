
import pandas as pd
import numpy as np
import os
import sys

def main():
    # Paths
    trades_path = "latest runs/2025-12-20/15m_Stock_Backtest_Fixed/backtest-results/detailed_trades.csv"
    meta_path = "RubberBand/ticker_analysis.csv"
    
    if not os.path.exists(trades_path):
        print(f"Error: {trades_path} not found")
        return
        
    # Load Trades
    df_trades = pd.read_csv(trades_path)
    print(f"Loaded {len(df_trades)} trades")
    
    # Load Metadata (if available)
    if os.path.exists(meta_path):
        df_meta = pd.read_csv(meta_path)
        # normalize columns
        if 'Ticker' in df_meta.columns:
            df_meta = df_meta.rename(columns={'Ticker': 'symbol'})
        elif 'ticker' in df_meta.columns:
             df_meta = df_meta.rename(columns={'ticker': 'symbol'})
             
        # Select relevant cols
        useful_cols = ['symbol', 'Beta', 'ATR%', 'Sector', 'Industry', 'Avg_Dollar_Vol_M', 'Market_Cap_B', 'Volatility_Bucket']
        existing_cols = [c for c in useful_cols if c in df_meta.columns]
        df_meta = df_meta[existing_cols]
        
        # Merge
        df_full = df_trades.merge(df_meta, on='symbol', how='left')
    else:
        print("Metadata file not found, proceeding without Beta/Sector")
        df_full = df_trades.copy()

    # Define "Loser" tickers (Net PnL < 0)
    ticker_pnl = df_full.groupby('symbol')['pnl'].sum().sort_values()
    losing_tickers = ticker_pnl[ticker_pnl < 0].index.tolist()
    winning_tickers = ticker_pnl[ticker_pnl > 0].index.tolist()
    
    print("\n" + "="*80)
    print(f"TOTAL ANALYSIS: {len(winning_tickers)} Winners vs {len(losing_tickers)} Losers")
    print("="*80)
    
    # Group Trades by outcome
    df_full['is_loss'] = df_full['pnl'] < 0
    
    # metrics to analyze
    metrics = ['entry_rsi', 'entry_atr']
    if 'Beta' in df_full.columns: metrics.append('Beta')
    if 'ATR%' in df_full.columns: metrics.append('ATR%')
    
    print("\n--- Correlation of Metrics with Trade Outcome (Win vs Loss) ---")
    print(df_full.groupby('is_loss')[metrics].mean())
    
    print("\n--- Correlation of Metrics with Ticker Overall Performance (Winning Ticker vs Losing Ticker) ---")
    df_full['ticker_type'] = np.where(df_full['symbol'].isin(winning_tickers), 'Winner', 'Loser')
    print(df_full.groupby('ticker_type')[metrics].mean())
    
    # Deep Dive into specific Losers
    target_losers = ['NVDA', 'TQQQ', 'ARM']
    # Add top 3 other losers
    other_losers = [t for t in losing_tickers if t not in target_losers][:3]
    deep_dive_list = target_losers + other_losers
    
    print("\n" + "="*80)
    print("DEEP DIVE: SPECIFIC LOSERS")
    print("="*80)
    
    for sym in deep_dive_list:
        if sym not in df_full['symbol'].values: continue
        
        subset = df_full[df_full['symbol'] == sym]
        total_pnl = subset['pnl'].sum()
        trades_count = len(subset)
        
        print(f"\n[{sym}] Total PnL: ${total_pnl:.2f} ({trades_count} trades)")
        
        # Show metadata if available
        if 'Beta' in subset.columns:
            beta = subset['Beta'].iloc[0] if not pd.isna(subset['Beta'].iloc[0]) else "N/A"
            atr_pct = subset['ATR%'].iloc[0] if 'ATR%' in subset.columns and not pd.isna(subset['ATR%'].iloc[0]) else "N/A"
            sector = subset['Sector'].iloc[0] if 'Sector' in subset.columns and not pd.isna(subset['Sector'].iloc[0]) else "N/A"
            print(f"   Beta: {beta} | ATR%: {atr_pct} | Sector: {sector}")
            
        print(subset[['date', 'entry_time', 'entry_price', 'exit_reason', 'pnl', 'entry_rsi', 'entry_atr']].to_string(index=False))

    # --- HYPOTHESIS TESTING ---
    print("\n" + "="*80)
    print("HYPOTHESIS TESTING")
    print("="*80)
    
    # 1. RSI Floor (Avoid Crash Mode)
    # Filter: Entry RSI must be > 20
    df_rsi_filtered = df_full[df_full['entry_rsi'] > 20]
    dropped_trades = len(df_full) - len(df_rsi_filtered)
    orig_pnl = df_full['pnl'].sum()
    new_pnl = df_rsi_filtered['pnl'].sum()
    
    print(f"\n[Hypothesis 1] RSI > 20 Filter (Avoid 'Crash Mode')")
    print(f"   Original PnL: ${orig_pnl:.2f} ({len(df_full)} trades)")
    print(f"   New PnL:      ${new_pnl:.2f} ({len(df_rsi_filtered)} trades)")
    print(f"   Impact:       ${(new_pnl - orig_pnl):.2f} (Dropped {dropped_trades} trades)")
    if dropped_trades > 0:
        dropped = df_full[df_full['entry_rsi'] <= 20]
        print(f"   Dropped Trades PnL: ${dropped['pnl'].sum():.2f} (Win Rate: {len(dropped[dropped['pnl']>0])/len(dropped)*100:.1f}%)")

    # 2. Daily Loss Limit (Max 1 Loss per Ticker per Day)
    print(f"\n[Hypothesis 2] Daily Loss Limit (Stop ticker after 1 loss in a day)")
    
    # Sort by time to process chronologically
    df_sorted = df_full.sort_values(['symbol', 'date', 'entry_time'])
    
    kept_indices = []
    
    # Iterate and simulate state
    # We need to track "has_loss_today" per symbol+date
    loss_tracker = {} # Key: (symbol, date) -> bool
    
    # 3. 2-Hour Cooldown after Loss
    print(f"\n[Hypothesis 3] 2-Hour Cooldown (Skip trades within 2h of a loss)")
    
    kept_indices_cd = []
    # Map: symbol -> datetime of last loss
    last_loss_time = {} 
    
    # Ensure date/time columns are proper datetime objects
    if not np.issubdtype(df_sorted['entry_time'].dtype, np.datetime64):
        # Infer format or assume ISO
        # detailed trades csv has ISO format usually
        df_sorted['entry_dt'] = pd.to_datetime(df_sorted['entry_time'], utc=True)
    else:
        df_sorted['entry_dt'] = df_sorted['entry_time']

    for idx, row in df_sorted.iterrows():
        sym = row['symbol']
        entry_time = row['entry_dt']
        
        # Check cooldown
        last_loss = last_loss_time.get(sym)
        if last_loss is not None:
            # Check if within 2 hours
            diff = (entry_time - last_loss).total_seconds() / 3600.0
            if diff < 2.0:
                # Skip
                continue
                
        # Keep trade
        kept_indices_cd.append(idx)
        
        # Update loss time if this is a loss
        if row['pnl'] < 0:
            last_loss_time[sym] = entry_time
            
    df_cd_filtered = df_full.loc[kept_indices_cd]
    cd_pnl = df_cd_filtered['pnl'].sum()
    
    print(f"   Original PnL: ${orig_pnl:.2f}")
    print(f"   New PnL:      ${cd_pnl:.2f} ({len(df_cd_filtered)} trades)")
    print(f"   Impact:       ${(cd_pnl - orig_pnl):.2f} (Dropped {len(df_full) - len(df_cd_filtered)} trades)")
    
    # 4. Smart Cooldown (Skip re-entry within 2h ONLY if RSI is LOWER than previous loss RSI)
    # Rationale: If RSI is lower, it's a falling knife. If RSI is higher, it might be divergence/recovery.
    print(f"\n[Hypothesis 4] Smart Cooldown (Skip re-entry < 2h if RSI < Prev Loss RSI)")
    
    kept_indices_smart = []
    # Map: symbol -> {time, rsi} of last loss
    last_loss_info = {} 
    
    for idx, row in df_sorted.iterrows():
        sym = row['symbol']
        entry_time = row['entry_dt']
        entry_rsi = row['entry_rsi']
        
        skip = False
        last_loss = last_loss_info.get(sym)
        if last_loss:
            # Check time diff
            diff = (entry_time - last_loss['time']).total_seconds() / 3600.0
            if diff < 2.0:
                # Check RSI condition: If new RSI is LOWER (or equal) to loss RSI, skip
                if entry_rsi <= last_loss['rsi']:
                    skip = True
        
        if skip:
            continue
            
        kept_indices_smart.append(idx)
        
        if row['pnl'] < 0:
            last_loss_info[sym] = {'time': entry_time, 'rsi': entry_rsi}
            
    df_smart_filtered = df_full.loc[kept_indices_smart]
    smart_pnl = df_smart_filtered['pnl'].sum()
    
    print(f"   Original PnL: ${orig_pnl:.2f}")
    print(f"   New PnL:      ${smart_pnl:.2f} ({len(df_smart_filtered)} trades)")
    print(f"   Impact:       ${(smart_pnl - orig_pnl):.2f} (Dropped {len(df_full) - len(df_smart_filtered)} trades)")

    
    # 5. Low RSI (< 20) Winners vs Losers
    print("\n" + "="*80)
    print("LOW RSI (< 20) ANALYSIS")
    print("="*80)
    
    df_low_rsi = df_full[df_full['entry_rsi'] <= 20]
    print(f"Total Low RSI Trades: {len(df_low_rsi)}")
    print(f"Total PnL: ${df_low_rsi['pnl'].sum():.2f}")
    print(f"Win Rate: {len(df_low_rsi[df_low_rsi['pnl']>0]) / len(df_low_rsi) * 100:.1f}%")
    
    print("\nTop Low RSI Winners:")
    # Group by ticker
    low_rsi_by_ticker = df_low_rsi.groupby('symbol').agg({'pnl': 'sum', 'entry_rsi': 'count'}).sort_values('pnl', ascending=False)
    print(low_rsi_by_ticker.head(10))
    
    print("\nTop Low RSI Losers:")
    print(low_rsi_by_ticker.tail(10).sort_values('pnl'))

    # 6. Dead Knife Filter (Skip re-entry if Current RSI < 20 AND Previous Loss RSI < 20)
    print(f"\n[Hypothesis 6] Dead Knife Filter (Skip if RSI < 20 AND Prev Loss RSI < 20)")
    
    kept_indices_dk = []
    # Map: (symbol, date) -> rsi of last loss
    dk_loss_tracker = {} 
    
    for idx, row in df_sorted.iterrows():
        sym = row['symbol']
        dt = row['date']
        key = (sym, dt)
        entry_rsi = row['entry_rsi']
        
        skip = False
        last_rsi = dk_loss_tracker.get(key)
        
        if last_rsi is not None:
            # If current is Extreme AND previous loss was Extreme
            if entry_rsi < 20 and last_rsi < 20:
                skip = True
                
        if skip:
            continue
            
        kept_indices_dk.append(idx)
        
        if row['pnl'] < 0:
            dk_loss_tracker[key] = entry_rsi
            
    df_dk_filtered = df_full.loc[kept_indices_dk]
    dk_pnl = df_dk_filtered['pnl'].sum()
    
    print(f"   Original PnL: ${orig_pnl:.2f}")
    print(f"   New PnL:      ${dk_pnl:.2f} ({len(df_dk_filtered)} trades)")
    print(f"   Impact:       ${(dk_pnl - orig_pnl):.2f} (Dropped {len(df_full) - len(df_dk_filtered)} trades)")

    # Check specific losers with this filter
    print("\n   Impact on Specific Losers & Winners (NVDA, TSLA, AMD):")
    for sym in ['NVDA', 'TSLA', 'TQQQ', 'AMD', 'MRVL']:
        orig_pnl_sym = df_full[df_full['symbol']==sym]['pnl'].sum()
        new_pnl_sym = df_dk_filtered[df_dk_filtered['symbol']==sym]['pnl'].sum()
        print(f"   {sym}: ${orig_pnl_sym:.2f} -> ${new_pnl_sym:.2f} (Change: ${new_pnl_sym - orig_pnl_sym:.2f})")

if __name__ == "__main__":
    main()
