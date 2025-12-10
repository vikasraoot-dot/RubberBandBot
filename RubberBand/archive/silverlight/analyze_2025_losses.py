#!/usr/bin/env python3
"""
Silver Light 2025 Loss Analysis Script
========================================
Deep analysis of 2025 trades to identify loss patterns and improvement opportunities.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# --- Ensure repo root on sys.path ---
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml
from RubberBand.src.data import fetch_latest_bars
from RubberBand.src.silverlight_strategy import attach_indicators, generate_signal, Signal


def load_config(path: str = "RubberBand/config_silverlight.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_detailed_backtest(symbol: str, df: pd.DataFrame, spy_df: pd.DataFrame, 
                          config: Dict[str, Any], vixy_df: pd.DataFrame = None) -> List[Dict]:
    """Run backtest and return detailed trade list with market conditions."""
    
    risk_cfg = config.get("risk", {})
    trailing_stop_enabled = risk_cfg.get("trailing_stop_enabled", True)
    trailing_stop_pct = float(risk_cfg.get("trailing_stop_pct", 0.10))
    
    ticker_df = attach_indicators(df.copy(), config)
    regime_df = attach_indicators(spy_df.copy(), config)
    
    common_dates = ticker_df.index.intersection(regime_df.index)
    ticker_df = ticker_df.loc[common_dates]
    regime_df = regime_df.loc[common_dates]
    
    warmup = 200
    if len(ticker_df) <= warmup:
        return []
    
    ticker_df = ticker_df.iloc[warmup:]
    regime_df = regime_df.iloc[warmup:]
    
    # State
    cash = 10000.0
    position_qty = 0
    entry_price = 0.0
    entry_date = None
    peak_price = 0.0
    entry_adx = 0.0
    entry_roc = 0.0
    
    trades = []
    
    for dt, row in ticker_df.iterrows():
        price = float(row["close"])
        adx = float(row["adx"]) if not pd.isna(row["adx"]) else 100
        roc = float(row["roc"]) if not pd.isna(row["roc"]) else 0
        
        # Get VIXY value if available
        vix_value = None
        if vixy_df is not None and dt in vixy_df.index:
            vix_value = float(vixy_df.loc[dt, "close"])
        
        signal, meta = generate_signal(ticker_df.loc[:dt], regime_df.loc[:dt], config, vix_value)
        
        if position_qty > 0:
            if price > peak_price:
                peak_price = price
            
            exit_triggered = False
            exit_reason = ""
            
            if trailing_stop_enabled and peak_price > 0:
                stop_price = peak_price * (1 - trailing_stop_pct)
                if price < stop_price:
                    exit_triggered = True
                    exit_reason = "TRAILING_STOP"
            
            if not exit_triggered and signal != Signal.LONG:
                exit_triggered = True
                exit_reason = meta.get("reason", "SIGNAL")[:30]
            
            if exit_triggered:
                exit_price = price * 0.999
                proceeds = position_qty * exit_price
                pnl = proceeds - (position_qty * entry_price)
                pnl_pct = (exit_price / entry_price - 1) * 100 if entry_price > 0 else 0
                
                trades.append({
                    "symbol": symbol,
                    "entry_date": entry_date,
                    "exit_date": dt,
                    "hold_days": (dt - entry_date).days if entry_date else 0,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "peak_price": peak_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "exit_reason": exit_reason,
                    "entry_adx": entry_adx,
                    "exit_adx": adx,
                    "entry_roc": entry_roc * 100,
                    "exit_roc": roc * 100,
                    "max_gain_pct": (peak_price / entry_price - 1) * 100 if entry_price > 0 else 0,
                    "drawdown_from_peak": (price / peak_price - 1) * 100 if peak_price > 0 else 0,
                })
                
                cash += proceeds
                position_qty = 0
                peak_price = 0.0
        
        elif signal == Signal.LONG:
            entry_price = price * 1.001
            position_qty = int(cash // entry_price)
            
            if position_qty > 0:
                cash -= position_qty * entry_price
                peak_price = entry_price
                entry_date = dt
                entry_adx = adx
                entry_roc = roc
    
    return trades


def main():
    config = load_config()
    
    print("=" * 80)
    print("SILVER LIGHT 2025 DETAILED TRADE ANALYSIS")
    print("=" * 80)
    
    # Fetch 2025 data for TQQQ
    symbols = ["TQQQ", "SPY", "VIXY"]
    
    print("\nFetching data...")
    bars_map, _ = fetch_latest_bars(
        symbols=symbols,
        timeframe="1Day",
        history_days=600,
        feed="iex",
        verbose=True
    )
    
    # Don't filter yet - let backtest handle warmup
    # Check data availability
    print(f"\nData loaded:")
    for sym, df in bars_map.items():
        if df is not None and not df.empty:
            print(f"  {sym}: {len(df)} bars, {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        else:
            print(f"  {sym}: No data")
    
    if "SPY" not in bars_map or bars_map["SPY"] is None or bars_map["SPY"].empty:
        print("ERROR: No SPY data for regime filter")
        return
    
    spy_df = bars_map["SPY"]
    vixy_df = bars_map.get("VIXY")
    
    # Run detailed backtest
    trades = run_detailed_backtest("TQQQ", bars_map["TQQQ"], spy_df, config, vixy_df)
    
    if not trades:
        print("No trades found!")
        return
    
    df = pd.DataFrame(trades)
    
    # Filter to 2025 only
    df = df[df["entry_date"] >= "2025-01-01"]
    
    print(f"\n{'='*80}")
    print("2025 TRADES - DETAILED BREAKDOWN")
    print(f"{'='*80}")
    
    for _, t in df.iterrows():
        win_loss = "✅ WIN" if t["pnl"] > 0 else "❌ LOSS"
        print(f"\n{win_loss}: {t['entry_date'].strftime('%Y-%m-%d')} → {t['exit_date'].strftime('%Y-%m-%d')}")
        print(f"  PnL: ${t['pnl']:+.2f} ({t['pnl_pct']:+.1f}%)")
        print(f"  Hold: {t['hold_days']} days")
        print(f"  Entry ADX: {t['entry_adx']:.0f}, Exit ADX: {t['exit_adx']:.0f}")
        print(f"  Entry ROC: {t['entry_roc']:.1f}%, Exit ROC: {t['exit_roc']:.1f}%")
        print(f"  Max Gain: {t['max_gain_pct']:.1f}%, Drawdown from Peak: {t['drawdown_from_peak']:.1f}%")
        print(f"  Exit Reason: {t['exit_reason']}")
    
    # Summary stats
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total Trades: {len(df)}")
    print(f"Winners: {len(wins)} ({len(wins)/len(df)*100:.0f}%)")
    print(f"Losers: {len(losses)} ({len(losses)/len(df)*100:.0f}%)")
    print(f"Total PnL: ${df['pnl'].sum():+.2f}")
    
    print(f"\n--- LOSS ANALYSIS ---")
    if len(losses) > 0:
        print(f"Average Loss: ${losses['pnl'].mean():.2f}")
        print(f"Average Loss Hold: {losses['hold_days'].mean():.1f} days")
        print(f"Average Entry ADX: {losses['entry_adx'].mean():.1f}")
        print(f"Average Max Gain Before Exit: {losses['max_gain_pct'].mean():.1f}%")
        
        print(f"\n--- EXIT REASONS ---")
        for reason, count in losses["exit_reason"].value_counts().items():
            print(f"  {reason}: {count} trades")
    
    print(f"\n--- WIN ANALYSIS ---")
    if len(wins) > 0:
        print(f"Average Win: ${wins['pnl'].mean():.2f}")
        print(f"Average Win Hold: {wins['hold_days'].mean():.1f} days")
        print(f"Average Entry ADX: {wins['entry_adx'].mean():.1f}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if len(losses) > 0:
        avg_entry_adx_loss = losses['entry_adx'].mean()
        avg_entry_adx_win = wins['entry_adx'].mean() if len(wins) > 0 else 0
        
        if avg_entry_adx_loss < avg_entry_adx_win:
            print(f"➡️ Losing trades entered at lower ADX ({avg_entry_adx_loss:.0f}) than winners ({avg_entry_adx_win:.0f})")
            print(f"   Consider raising ADX threshold from 20 to {int(max(20, avg_entry_adx_loss + 5))}")
        
        avg_hold_loss = losses['hold_days'].mean()
        if avg_hold_loss < 5:
            print(f"➡️ Losses exit quickly ({avg_hold_loss:.1f} days avg) - false breakouts")
            print(f"   Consider requiring 2-day close above SMA for confirmation")
        
        max_gain_before_loss = losses['max_gain_pct'].mean()
        if max_gain_before_loss > 2:
            print(f"➡️ Trades that became losses had avg {max_gain_before_loss:.1f}% max gain")
            print(f"   Consider profit-taking at {int(max_gain_before_loss)}%")


if __name__ == "__main__":
    main()
