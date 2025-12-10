#!/usr/bin/env python3
"""
Silver Light Trade Loss Analyzer
=================================
Analyzes losing trades to identify patterns and suggest config improvements.

Usage:
    python RubberBand/scripts/analyze_silverlight_losses.py silverlight_backtest_results.csv
"""

import sys
import pandas as pd
import numpy as np
from collections import defaultdict


def analyze_trades(csv_path: str):
    """Analyze trades and categorize losses."""
    
    # Read trades
    df = pd.read_csv(csv_path)
    
    print("=" * 70)
    print("SILVER LIGHT TRADE LOSS ANALYSIS")
    print("=" * 70)
    
    # Basic stats
    total_trades = len(df)
    winners = df[df["pnl"] > 0]
    losers = df[df["pnl"] <= 0]
    
    print(f"\n📊 OVERALL STATISTICS")
    print("-" * 40)
    print(f"Total Trades: {total_trades}")
    print(f"Winners: {len(winners)} ({len(winners)/total_trades*100:.1f}%)")
    print(f"Losers: {len(losers)} ({len(losers)/total_trades*100:.1f}%)")
    print(f"Total P/L: ${df['pnl'].sum():,.2f}")
    print(f"Avg Win: ${winners['pnl'].mean():,.2f}" if len(winners) > 0 else "Avg Win: N/A")
    print(f"Avg Loss: ${losers['pnl'].mean():,.2f}" if len(losers) > 0 else "Avg Loss: N/A")
    
    # Win/Loss ratio
    if len(losers) > 0 and len(winners) > 0:
        avg_win = winners["pnl"].mean()
        avg_loss = abs(losers["pnl"].mean())
        print(f"Win/Loss Ratio: {avg_win/avg_loss:.2f}")
    
    # --- Categorize Losses by Exit Reason ---
    print(f"\n📉 LOSS BREAKDOWN BY EXIT REASON")
    print("-" * 40)
    
    loss_by_reason = defaultdict(list)
    for _, row in losers.iterrows():
        reason = row.get("exit_reason", "UNKNOWN")
        # Categorize
        if "TRAILING STOP" in str(reason):
            category = "TRAILING STOP"
        elif "TREND BREAK" in str(reason):
            category = "TREND BREAK (Price < 50 SMA)"
        elif "BEARISH REGIME" in str(reason):
            category = "REGIME CHANGE (SPY < 200 SMA)"
        elif "HIGH VOLATILITY" in str(reason):
            category = "VIX FILTER"
        else:
            category = "OTHER"
        
        loss_by_reason[category].append(row["pnl"])
    
    for category, losses in sorted(loss_by_reason.items(), key=lambda x: sum(x[1])):
        total_loss = sum(losses)
        count = len(losses)
        avg_loss = total_loss / count if count > 0 else 0
        print(f"{category}:")
        print(f"  Count: {count} trades")
        print(f"  Total Loss: ${total_loss:,.2f}")
        print(f"  Avg Loss: ${avg_loss:,.2f}")
        print()
    
    # --- Analyze Loss Severity ---
    print(f"\n📊 LOSS SEVERITY DISTRIBUTION")
    print("-" * 40)
    
    small_losses = losers[losers["pnl_pct"] > -5]
    medium_losses = losers[(losers["pnl_pct"] <= -5) & (losers["pnl_pct"] > -10)]
    large_losses = losers[losers["pnl_pct"] <= -10]
    
    print(f"Small Losses (0% to -5%): {len(small_losses)} trades, ${small_losses['pnl'].sum():,.2f}")
    print(f"Medium Losses (-5% to -10%): {len(medium_losses)} trades, ${medium_losses['pnl'].sum():,.2f}")
    print(f"Large Losses (< -10%): {len(large_losses)} trades, ${large_losses['pnl'].sum():,.2f}")
    
    # --- Hold Duration Analysis ---
    print(f"\n⏱️ LOSING TRADE DURATION ANALYSIS")
    print("-" * 40)
    
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["hold_days"] = (df["exit_date"] - df["entry_date"]).dt.days
    
    losers_with_days = df[df["pnl"] <= 0]
    
    short_hold = losers_with_days[losers_with_days["hold_days"] <= 3]
    medium_hold = losers_with_days[(losers_with_days["hold_days"] > 3) & (losers_with_days["hold_days"] <= 10)]
    long_hold = losers_with_days[losers_with_days["hold_days"] > 10]
    
    print(f"Short Hold (≤3 days): {len(short_hold)} trades, ${short_hold['pnl'].sum():,.2f}")
    print(f"Medium Hold (4-10 days): {len(medium_hold)} trades, ${medium_hold['pnl'].sum():,.2f}")
    print(f"Long Hold (>10 days): {len(long_hold)} trades, ${long_hold['pnl'].sum():,.2f}")
    
    # --- Top 5 Worst Losses ---
    print(f"\n🔴 TOP 5 WORST LOSSES")
    print("-" * 40)
    worst = losers.nsmallest(5, "pnl")
    for _, row in worst.iterrows():
        print(f"  {row['entry_date'][:10]} to {row['exit_date'][:10]}: ${row['pnl']:,.2f} ({row['pnl_pct']:.1f}%)")
        print(f"    Entry: ${row['entry_price']:.2f}, Exit: ${row['exit_price']:.2f}")
        print(f"    Reason: {row['exit_reason'][:50]}...")
        print()
    
    # --- Recommendations ---
    print("=" * 70)
    print("💡 RECOMMENDATIONS")
    print("=" * 70)
    
    # Check if most losses are from trailing stop
    trailing_stop_losses = loss_by_reason.get("TRAILING STOP", [])
    trailing_stop_pct = len(trailing_stop_losses) / len(losers) * 100 if len(losers) > 0 else 0
    
    if trailing_stop_pct > 50:
        print(f"\n⚠️ {trailing_stop_pct:.0f}% of losses from TRAILING STOP")
        print("   Current: 10% trailing stop")
        print("   Suggestion: Try 12-15% trailing stop to reduce whipsaws")
        print("   Config: risk.trailing_stop_pct: 0.12")
    
    # Check if most losses are short-hold
    if len(short_hold) > len(losers) * 0.5:
        print(f"\n⚠️ {len(short_hold)/len(losers)*100:.0f}% of losses are short-hold (≤3 days)")
        print("   This suggests false breakouts or whipsaws")
        print("   Suggestion 1: Add a confirmation period (wait 2 days after signal)")
        print("   Suggestion 2: Use 55-day SMA instead of 50-day (slower entry)")
        print("   Config: indicators.sma_fast: 55")
    
    # Check average loss vs average win
    if len(winners) > 0 and len(losers) > 0:
        avg_win = winners["pnl"].mean()
        avg_loss = abs(losers["pnl"].mean())
        if avg_loss > avg_win:
            print(f"\n⚠️ Avg Loss (${avg_loss:.2f}) > Avg Win (${avg_win:.2f})")
            print("   Suggestion: Tighten trailing stop or add stop-loss")
            print("   Config: risk.trailing_stop_pct: 0.08 (8% instead of 10%)")
    
    # Check for regime change losses
    regime_losses = loss_by_reason.get("REGIME CHANGE (SPY < 200 SMA)", [])
    if len(regime_losses) > 0:
        print(f"\n⚠️ {len(regime_losses)} losses from REGIME CHANGE")
        print("   These are unavoidable market crashes")
        print("   The VIX filter should help, but crashes are fast")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_silverlight_losses.py <trades_csv>")
        sys.exit(1)
    
    analyze_trades(sys.argv[1])
