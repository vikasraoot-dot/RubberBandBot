
import pandas as pd
import sys
import os
from datetime import datetime
from zoneinfo import ZoneInfo

# Ensure repo root is on path
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

from RubberBand.src.data import fetch_latest_bars
from RubberBand.src.indicators import ta_add_keltner

def analyze_dual_slope():
    # 1. Load Trades
    csv_path = "results/spread_backtest_trades.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df_trades = pd.read_csv(csv_path)
    if df_trades.empty:
        print("No trades found.")
        return

    # Parse timestamps
    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
    
    # Get unique symbols
    symbols = df_trades['symbol'].unique().tolist()
    print(f"Analyzing {len(df_trades)} trades across {len(symbols)} symbols...")

    # 2. Fetch Data (Buffer 15 days to cover backtest)
    bars_map, _ = fetch_latest_bars(
        symbols=symbols,
        timeframe="15Min",
        history_days=15, 
        feed="iex",
        verbose=False
    )

    # 3. Calculate Slopes
    slope10_results = []
    slope3_check = []

    for idx, row in df_trades.iterrows():
        sym = row['symbol']
        entry_time = row['entry_time']
        
        # Get dataframe for symbol
        df_bars = bars_map.get(sym)
        if df_bars is None or df_bars.empty:
            slope10_results.append(None)
            slope3_check.append(None)
            continue

        # Add indicators if not present (fetch_latest_bars returns raw OHLC typically)
        # Note: ta_add_keltner requires 'close', 'high', 'low'
        if "kc_middle" not in df_bars.columns:
            df_bars = ta_add_keltner(df_bars, length=20, mult=2.0)
        
        # Calculate Slopes
        # Slope 3 (Verification): (KC[-1] - KC[-4]) / 3
        df_bars["slope_3"] = (df_bars["kc_middle"] - df_bars["kc_middle"].shift(3)) / 3
        
        # Slope 10 (New): (KC[-1] - KC[-11]) / 10
        df_bars["slope_10"] = (df_bars["kc_middle"] - df_bars["kc_middle"].shift(10)) / 10

        # Find the row at entry_time (closest match <= entry_time)
        # Entry times in CSV are likely fill times, bars are candle times.
        # We need the bar whose close time was *before* or *at* entry.
        # Actually, bot uses closed bars. So if entry is 10:00:05, it used 9:45-10:00 bar (timestamp 9:45 or 10:00 depending on convention).
        # Alpaca bars usually indexed by START time.
        # A trade at 10:00 would have decided based on 9:45 bar.
        # Let's try `asof` logic or exact match after truncation.
        
        # Truncate entry time to 15m floor to match bar start time? 
        # Actually safer to looking at the last available bar before entry_time
        mask = df_bars.index <= entry_time
        if not mask.any():
            slope10_results.append(None)
            slope3_check.append(None)
            continue
            
        # Get last available bar
        last_bar = df_bars[mask].iloc[-1]
        
        slope10_results.append(last_bar.get("slope_10"))
        slope3_check.append(last_bar.get("slope_3"))

    df_trades['slope_10'] = slope10_results
    df_trades['slope_3_calc'] = slope3_check

    # 4. Analysis
    # Filter valid rows
    df_valid = df_trades.dropna(subset=['slope_10'])
    
    wins = df_valid[df_valid['pnl'] > 0]
    losses = df_valid[df_valid['pnl'] <= 0]

    print("\n" + "="*60)
    print("DUAL SLOPE ANALYSIS PROPOSAL")
    print("="*60)
    
    print(f"\n--- Short-Term Slope (3-Bar) [Revert Confirmation] ---")
    print(f"Wins Avg:   {wins['entry_slope'].mean():.4f}")
    print(f"Losses Avg: {losses['entry_slope'].mean():.4f}")
    print(f"Rec Threshold: -0.20 seems safe (Wins go down to {wins['entry_slope'].min():.4f})")

    print(f"\n--- Long-Term Slope (10-Bar) [New Guard] ---")
    print(f"Wins Avg:   {wins['slope_10'].mean():.4f}")
    print(f"Losses Avg: {losses['slope_10'].mean():.4f}")
    print(f"Win Range:  {wins['slope_10'].min():.4f} to {wins['slope_10'].max():.4f}")
    print(f"Loss Range: {losses['slope_10'].min():.4f} to {losses['slope_10'].max():.4f}")

    # Analyze hypothetical threshold for Slope 10
    # Proposed: -0.10. Let's see how many wins we kill.
    threshold = -0.10
    killed_wins = wins[wins['slope_10'] > threshold] # Panic Buyer logic? No, Anti-Falling Knife.
    # Anti-Slow Bleed: "If slope_10 < -0.10 (too steep down for too long), SKIP".
    # Wait, "Slow Bleed" implies gentle down trend. 
    # If slope is -0.05, that's a slow bleed. If slope is -0.20, that's a crash.
    # We want to AVOID slow bleeds.
    # Actually, usually "Slow Bleed" means price < SMA (Trend Filter).
    # If we want to filter persistent downtrends, maybe we want slope > -0.05?
    # No, that would be "Only buy if trend is flat/up".
    # User said: "gradual decrease that on an extended scale amounts to a crash".
    # This implies a consistent negative slope.
    # So we want to ensure Slope_10 is NOT too negative? Or NOT negative at all?
    # If we want to catch Reversals, we EXPECT negative slope.
    # But maybe not "Deeply Negative for Long Time".
    
    # Interpretation:
    # If Slope_3 is -0.40 (Crash) -> Buy! (Reversal)
    # But if Slope_10 is ALSO -0.40 -> It's been crashing for 2.5 hours -> Falling Knife?
    # Or if Slope_10 is -0.10 -> It's been slowly bleeding for 2.5 hours -> Don't catch.
    
    print(f"\n--- Hypothesis Testing (Threshold = {threshold}) ---")
    print(f"IF we skip trades where Slope_10 < {threshold} (Anti-Long-Term-Trend):")
    kept_wins = wins[wins['slope_10'] >= threshold]
    kept_losses = losses[losses['slope_10'] >= threshold]
    
    print(f"Wins Kept:   {len(kept_wins)}/{len(wins)} ({len(kept_wins)/len(wins)*100:.1f}%)")
    print(f"Losses Kept: {len(kept_losses)}/{len(losses)} ({len(kept_losses)/len(losses)*100:.1f}%)")
    
    if len(kept_losses) > 0:
        print(f"Kept Loss Avg PnL: ${kept_losses['pnl'].mean():.2f}")
    
    print("\n--- Detailed Data (Slope 3 vs Slope 10) ---")
    print(df_valid[['symbol', 'pnl', 'entry_slope', 'slope_10']].sort_values('pnl', ascending=False).to_string())

if __name__ == "__main__":
    analyze_dual_slope()
