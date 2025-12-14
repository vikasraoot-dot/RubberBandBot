"""
Rubber Band Strategy Audit Framework ("The Glass Box")

This script performs forensic validation of the Rubber Band strategy without modifying the core logic.
It addresses "Catching a Falling Knife" (Visually) and "Overfitting" (Statistically).

Components:
1. Visual Extension Check: Plots entries with bands and MA slope.
2. Stability Heatmap: Tests parameter robustness.
3. Walk-Forward Integrity: Checks if logic holds on unseen data.

Usage:
    python -m RubberBand.scripts.audit_framework [TICKER]
    
    Example:
    python -m RubberBand.scripts.audit_framework SPY
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import existing modules (ReadOnly access)
from RubberBand.src.data import fetch_latest_bars
from RubberBand.strategy import attach_verifiers

def get_ma_slope(series, period=3):
    """Calculates the slope of a series over `period` bars."""
    return series.diff(period) / period

def run_simulation(df, params):
    """
    Runs the strategy on a DataFrame with specific parameters.
    Returns the dataframe with signals and verification metrics.
    """
    # Create a config dict for the strategy wrapper
    cfg = {
        "keltner_length": params.get("length", 20),
        "keltner_mult": params.get("mult", 2.0),
        "atr_length": 14,
        "rsi_length": 14,
        "filters": {
            "trend_filter_sma": 0, # Disable external trend filter for pure rubber band test
            "rsi_oversold": 30,
            "rsi_min": 15
        }
    }
    
    # Run strategy logic
    res = attach_verifiers(df.copy(), cfg)
    
    # Add Slope Check for "Falling Knife" detection
    # Slope of the Keltner Basis (Central Mean)
    if "kc_middle" in res.columns:
        res["mean_slope"] = get_ma_slope(res["kc_middle"])
    
    return res

def calculate_max_drawdown(df):
    """Simple Max Drawdown calculation based on cumulative Close price changes during trades."""
    # Note: This is a simplified proxy. 
    # Real DD requires a full backtest engine with account balance.
    # Here we assume we enter on 'long_signal' and exit after N bars or simple mean reversion for estimation.
    # For the Heatmap, we will use 'Sum of Adverse Excursions' as a proxy for risk if full PnL isn't available.
    
    # Logic: precise PnL require entry/exit logic. 
    # Let's use a simpler metric for "Bad Trades": 
    # Average Maximum Adverse Excursion (MAE) of signals over next 10 bars.
    
    if "long_signal" not in df.columns or df["long_signal"].sum() == 0:
        return 0.0

    signals = df[df["long_signal"]].index
    maes = []
    
    for entry_time in signals:
        # Look forward 10 bars
        if entry_time not in df.index: continue
        try:
            loc = df.index.get_loc(entry_time)
            forward_window = df.iloc[loc+1 : loc+11] # Next 10 bars
            if forward_window.empty: continue
            
            entry_price = df.loc[entry_time, "close"]
            min_price = forward_window["low"].min()
            
            # Drawdown for this trade %
            dd = (min_price - entry_price) / entry_price
            maes.append(dd)
        except Exception:
            pass
            
    if not maes:
        return 0.0
        
    # Return Average MAE (Negative number)
    return np.mean(maes)

# ==============================================================================
# 1. Visual Extension Check
# ==============================================================================
def visual_extension_check(ticker, df_res, target_date=None):
    """
    Plots trades to visual inspect 'Falling Knife' scenarios.
    If target_date is provided, only plots trades on that day.
    Otherwise plots last 3 trades.
    """
    print(f"\n[Visual Extension] Generating forensic plots for {ticker}...")
    
    trades = df_res[df_res["long_signal"]]
    if trades.empty:
        print("  No trades found to plot.")
        return

    # Filter by date if requested
    if target_date:
        # target_date string format "YYYY-MM-DD"
        # Filter trades where the date part matches
        day_str = str(target_date)
        trades = trades[trades.index.astype(str).str.startswith(day_str)]
        print(f"  Filtering for date {day_str}... Found {len(trades)} trades.")
        if trades.empty:
            return

    # If no date filter, plot last 3. If date filter, plot all on that day (up to 5 max)
    trade_indices = trades.index[-5:] if target_date else trades.index[-3:] 
    
    plot_count = 0
    for entry_time in trade_indices:
        plot_count += 1
        
        # Define zoom window (20 bars before, 10 bars after)
        try:
            loc = df_res.index.get_loc(entry_time)
            start_loc = max(0, loc - 30)
            end_loc = min(len(df_res), loc + 15)
            snippet = df_res.iloc[start_loc:end_loc]
            
            entry_price = snippet.loc[entry_time, "close"]
            slope = snippet.loc[entry_time, "mean_slope"]
            
            # Assessment
            knife_status = "FALLING KNIFE (DANGER)" if slope < -0.05 else "STABILIZING" if slope < 0 else "REVERSING UP"
            
            plt.figure(figsize=(10, 6))
            plt.plot(snippet.index, snippet['close'], label='Close', color='black', alpha=0.7)
            
            if 'kc_upper' in snippet.columns:
                plt.plot(snippet.index, snippet['kc_upper'], color='gray', linestyle='--', alpha=0.3)
                plt.plot(snippet.index, snippet['kc_lower'], color='gray', linestyle='--', alpha=0.3)
                plt.plot(snippet.index, snippet['kc_middle'], color='blue', label='Mean (Basis)', alpha=0.5)
                
            # Plot Entry
            plt.scatter([entry_time], [entry_price], color='lime', s=100, marker='^', label='Buy Signal', zorder=5)
            
            plt.title(f"Trade Forensic: {entry_time} | Slope: {slope:.4f} ({knife_status})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save or Show? For script usage, show or save to file.
            # We will save to a file for review.
            fname = f"audit_plot_{ticker}_{plot_count}.png"
            plt.savefig(fname)
            print(f"  Saved plot: {fname} | Slope: {slope:.5f}")
            plt.close()
            
        except Exception as e:
            print(f"  Error plotting trade at {entry_time}: {e}")

# ==============================================================================
# 2. Parameter Stability Heatmap
# ==============================================================================
def stability_heatmap(df_raw, center_len=20, center_mult=2.0):
    """
    Generates a heatmap of Risk (Avg MAE) across parameter variations.
    """
    print(f"\n[Stability Heatmap] Testing variations around Length={center_len}, Mult={center_mult}...")
    
    # Variations +/- 15%ish
    lengths = sorted(list(set([int(center_len * x) for x in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]])))
    mults = sorted([round(center_mult * x, 1) for x in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]])
    
    results = [] # (len, mult, score)
    
    for l in lengths:
        row = []
        for m in mults:
            df_test = run_simulation(df_raw, {"length": l, "mult": m})
            risk_score = calculate_max_drawdown(df_test)
            # We want to maximize "Safety", so close to 0 is better. 
            # Risk score is negative (drawdown).
            row.append(risk_score)
        results.append(row)
        
    results_np = np.array(results)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    sns.heatmap(results_np, xticklabels=mults, yticklabels=lengths, annot=True, fmt=".2%", cmap="RdYlGn", center=-0.02)
    plt.xlabel("Multipliers (Stretch)")
    plt.ylabel("Lengths (Mean)")
    plt.title("Parameter Stability: Avg Trade Drawdown (MAE)")
    plt.savefig("audit_heatmap.png")
    plt.close()
    print("  Saved heatmap: audit_heatmap.png")
    
    # Text Analysis
    # Check if center is surrounded by similar values (Plateau) or drop-offs (Cliff)
    center_l_idx = lengths.index(center_len) if center_len in lengths else len(lengths)//2
    center_m_idx = mults.index(center_mult) if center_mult in mults else len(mults)//2
    
    center_val = results_np[center_l_idx, center_m_idx]
    avg_val = np.mean(results_np)
    
    print(f"  Center Perf (Drawdown): {center_val:.2%}")
    print(f"  Region Avg (Drawdown):  {avg_val:.2%}")
    
    if abs(center_val - avg_val) < 0.01:
        print("  > STATUS: STABLE (Plateau detected).")
    else:
        print("  > STATUS: VOLATILE (Performance varies significantly).")

# ==============================================================================
# 3. Walk-Forward Integrity
# ==============================================================================
def walk_forward_integrity(df):
    """
    Split 70/30. Optimize on 70. Test on 30. Check for Trend Failure.
    """
    print("\n[Walk-Forward] Running Trend Integrity Test (70/30 split)...")
    
    split_idx = int(len(df) * 0.70)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"  Training Bars: {len(train_df)} | Testing Bars: {len(test_df)}")
    
    # 1. Optimize on Training (Simple Grid Search for best frequency/safety balance)
    # We will pick the param set with best Ratio: (NumTrades / Abs(AvgDrawdown))
    best_score = -1
    best_params = {"length": 20, "mult": 2.0}
    
    param_grid = [
        {"length": 20, "mult": 2.0},
        {"length": 20, "mult": 2.5},
        {"length": 30, "mult": 2.0},
        {"length": 14, "mult": 2.2},
    ]
    
    for p in param_grid:
        res = run_simulation(train_df, p)
        num_trades = res["long_signal"].sum()
        dd = calculate_max_drawdown(res) # negative percent
        
        if num_trades < 5: continue # Too few samples
        
        # Simple score: We want trades, but low DD.
        # Avoid div by zero
        safe_dd = abs(dd) if abs(dd) > 0.001 else 0.001
        score = num_trades / safe_dd
        
        if score > best_score:
            best_score = score
            best_params = p
            
    print(f"  Optimized Params (In-Sample): {best_params}")
    
    # 2. Apply to Out-Of-Sample
    oos_res = run_simulation(test_df, best_params)
    oos_trades = oos_res["long_signal"].sum()
    oos_dd = calculate_max_drawdown(oos_res)
    
    print(f"  Out-of-Sample Result: {oos_trades} trades, Avg Drawdown: {oos_dd:.2%}")
    
    # 3. Trend Failure Check
    # If OOS DD is > 2x Training DD, flag it.
    
    # Re-calc training DD for best params
    train_res = run_simulation(train_df, best_params)
    train_dd = calculate_max_drawdown(train_res)
    
    print(f"  In-Sample Drawdown:   {train_dd:.2%}")
    
    if abs(oos_dd) > abs(train_dd) * 1.5:
        print("  > RESULT: FAILED TREND FILTERING.")
        print("    The bot performed significantly worse on unseen data.")
        print("    likely due to catching falling knives in strong trends.")
    else:
        print("  > RESULT: PASSED.")
        print("    Performance was consistent across regimes.")

# ==============================================================================
# MAIN
# ==============================================================================

# ==============================================================================
# 4. Slope Impact Analysis
# ==============================================================================
def analyze_slope_impact(ticker, df_raw):
    """
    Analyzes the correlation between Entry Slope and Forward Return.
    Generates a scatter plot and recommends a threshold.
    """
    print(f"\n[Slope Analysis] Analyzing Slope vs. Outcome for {ticker}...")
    
    # Run with default 'center' parameters for consistency
    df = run_simulation(df_raw, {"length": 20, "mult": 2.0})
    
    trades = df[df["long_signal"]].copy()
    if trades.empty:
        print("  No trades found to analyze.")
        return

    slopes = []
    returns = []
    outcomes = [] # 'Win', 'Loss', 'Crash'

    # Look forward 12 bars (approx 3 hours) for outcome
    LOOK_FORWARD = 12 
    
    for entry_time, row in trades.iterrows():
        entry_idx = df.index.get_loc(entry_time)
        if entry_idx + LOOK_FORWARD >= len(df):
            continue
            
        entry_price = row["close"]
        future_price = df.iloc[entry_idx + LOOK_FORWARD]["close"]
        min_future_price = df.iloc[entry_idx : entry_idx + LOOK_FORWARD]["low"].min()
        
        # Metrics
        fwd_return = (future_price - entry_price) / entry_price
        max_drawdown = (min_future_price - entry_price) / entry_price
        slope = row["mean_slope"]
        
        slopes.append(slope)
        returns.append(fwd_return)
        
        if max_drawdown < -0.02:
            outcomes.append('Crash')
        elif fwd_return > 0:
            outcomes.append('Win')
        else:
            outcomes.append('Loss')

    if not slopes:
        print("  Not enough data for forward analysis.")
        return

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Scatter plot with color coding
    colors = {'Win': 'green', 'Loss': 'orange', 'Crash': 'red'}
    c_map = [colors[o] for o in outcomes]
    
    plt.scatter(slopes, returns, c=c_map, alpha=0.7, edgecolors='k')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Threshold Lines for visual reference
    plt.axvline(-0.1, color='blue', linestyle=':', label='Threshold -0.1')
    plt.axvline(-0.3, color='purple', linestyle=':', label='Threshold -0.3')
    
    plt.title(f"Entry Slope vs 3hr Forward Return ({ticker})\nGreen=Win, Orange=Loss, Red=Crash (>2% DD)")
    plt.xlabel("Keltner Mean Slope (at Entry)")
    plt.ylabel("Forward Return (12 bars)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = f"audit_slope_analysis_{ticker}.png"
    plt.savefig(plot_path)
    print(f"  Saved plot: {plot_path}")
    plt.close()
    
    # Statistical Recommendation
    slopes_arr = np.array(slopes)
    rets_arr = np.array(returns)
    
    print("\n  >>> Threshold Impact Analysis <<<")
    for thresh in [0.0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30]:
        mask = slopes_arr > thresh
        if np.sum(mask) == 0:
            continue
        
        filtered_rets = rets_arr[mask]
        win_rate = np.sum(filtered_rets > 0) / len(filtered_rets)
        avg_ret = np.mean(filtered_rets)
        bad_trades_avoided = np.sum((slopes_arr <= thresh) & (rets_arr < -0.01)) # Avoiding losses > 1%
        good_trades_missed = np.sum((slopes_arr <= thresh) & (rets_arr > 0.01))  # Missing wins > 1%
        
        print(f"  Thresh {thresh:>5.2f}: WinRate {win_rate:>6.1%} | AvgRet {avg_ret:>6.2%} | Avoided {bad_trades_avoided} Crashes | Missed {good_trades_missed} Wins")

def main():
    parser = argparse.ArgumentParser(description="Rubber Band Audit Framework")
    parser.add_argument("ticker", nargs="?", default="SPY", help="Ticker to audit")
    parser.add_argument("--date", help="Target date for forensic plots (YYYY-MM-DD)", default=None)
    parser.add_argument("--analyze-slope", action="store_true", help="Run Slope vs Return analysis")
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    print(f"=== Starting Audit for {ticker} ===")
    
    # 1. Fetch Data
    # Fetch ample history (60 days) to allow for walk forward
    print("Fetching data...")
    bars_map, meta = fetch_latest_bars(
        symbols=[ticker], 
        timeframe="15Min", 
        history_days=60, 
        verbose=False
    )
    
    if ticker not in bars_map or bars_map[ticker].empty:
        print(f"Error: No data found for {ticker}")
        sys.exit(1)
        
    df = bars_map[ticker]
    print(f"Loaded {len(df)} bars.")
    
    # 2. Run Audit Modules
    
    # A. Visual Extension (uses Default Params for baseline)
    default_res = run_simulation(df, {"length": 20, "mult": 2.0})
    visual_extension_check(ticker, default_res, target_date=args.date) 
    
    # B. Stability Heatmap
    stability_heatmap(df, center_len=20, center_mult=2.0)
    
    # C. Walk-Forward Integrity
    walk_forward_integrity(df)

    # D. Slope Impact Analysis
    if args.analyze_slope:
        analyze_slope_impact(ticker, df)
    
    print("\n=== Audit Complete ===")

if __name__ == "__main__":
    main()
