#!/usr/bin/env python3
"""
Silver Light Working Capital Analysis
======================================
Analyzes trade entry/exit dates to determine peak capital requirements.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict

# --- Ensure repo root on sys.path ---
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml
from RubberBand.src.data import fetch_latest_bars
from RubberBand.src.silverlight_strategy import attach_indicators, generate_signal, Signal


DEFAULT_TICKERS = ["QQQ", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "COST"]


def load_config(path: str = "RubberBand/config_silverlight.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_backtest_with_trades(symbol: str, df: pd.DataFrame, spy_df: pd.DataFrame, 
                               config: Dict[str, Any], capital_per_ticker: float = 10000.0) -> List[Dict]:
    """Run backtest and return list of trades with entry/exit dates and capital used."""
    
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
    cash = capital_per_ticker
    position_qty = 0
    entry_price = 0.0
    entry_date = None
    peak_price = 0.0
    
    trades = []
    
    for dt, row in ticker_df.iterrows():
        price = float(row["close"])
        
        signal, meta = generate_signal(ticker_df.loc[:dt], regime_df.loc[:dt], config)
        
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
                capital_used = position_qty * entry_price
                
                trades.append({
                    "symbol": symbol,
                    "entry_date": entry_date,
                    "exit_date": dt,
                    "capital_used": capital_used,
                    "pnl": (position_qty * exit_price) - capital_used,
                })
                
                cash += position_qty * exit_price
                position_qty = 0
                peak_price = 0.0
        
        elif signal == Signal.LONG:
            entry_price = price * 1.001
            position_qty = int(cash // entry_price)
            
            if position_qty > 0:
                cash -= position_qty * entry_price
                peak_price = entry_price
                entry_date = dt
    
    # Close any open position
    if position_qty > 0:
        exit_price = float(ticker_df.iloc[-1]["close"])
        capital_used = position_qty * entry_price
        trades.append({
            "symbol": symbol,
            "entry_date": entry_date,
            "exit_date": ticker_df.index[-1],
            "capital_used": capital_used,
            "pnl": (position_qty * exit_price) - capital_used,
        })
    
    return trades


def analyze_capital_requirements(all_trades: List[Dict]) -> Dict[str, Any]:
    """Analyze overlapping positions to find peak capital requirement."""
    
    if not all_trades:
        return {"error": "No trades"}
    
    # Create daily timeline of capital in use
    df = pd.DataFrame(all_trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    
    # Get date range
    min_date = df["entry_date"].min()
    max_date = df["exit_date"].max()
    
    date_range = pd.date_range(min_date, max_date, freq="D")
    
    # Calculate capital in use each day
    daily_capital = []
    daily_positions = []
    
    for date in date_range:
        # Find all trades active on this date
        active = df[(df["entry_date"] <= date) & (df["exit_date"] >= date)]
        capital = active["capital_used"].sum() if len(active) > 0 else 0
        positions = len(active)
        
        daily_capital.append({"date": date, "capital": capital, "positions": positions})
    
    capital_df = pd.DataFrame(daily_capital)
    
    # Analysis
    peak_capital = capital_df["capital"].max()
    peak_date = capital_df.loc[capital_df["capital"].idxmax(), "date"]
    peak_positions = capital_df.loc[capital_df["capital"].idxmax(), "positions"]
    
    avg_capital = capital_df["capital"].mean()
    median_capital = capital_df["capital"].median()
    
    avg_positions = capital_df["positions"].mean()
    max_positions = capital_df["positions"].max()
    
    # Percentiles
    p75_capital = capital_df["capital"].quantile(0.75)
    p90_capital = capital_df["capital"].quantile(0.90)
    p95_capital = capital_df["capital"].quantile(0.95)
    
    total_pnl = df["pnl"].sum()
    
    return {
        "total_trades": len(df),
        "total_pnl": total_pnl,
        "peak_capital": peak_capital,
        "peak_date": peak_date,
        "peak_positions": peak_positions,
        "avg_capital": avg_capital,
        "median_capital": median_capital,
        "p75_capital": p75_capital,
        "p90_capital": p90_capital,
        "p95_capital": p95_capital,
        "avg_positions": avg_positions,
        "max_positions": max_positions,
        "roi_on_peak": (total_pnl / peak_capital * 100) if peak_capital > 0 else 0,
        "roi_on_avg": (total_pnl / avg_capital * 100) if avg_capital > 0 else 0,
    }


def main():
    config = load_config()
    
    print("=" * 80)
    print("SILVER LIGHT WORKING CAPITAL ANALYSIS")
    print("=" * 80)
    
    # Fetch data
    symbols = DEFAULT_TICKERS + ["SPY"]
    
    print(f"\nFetching data for {len(symbols)} symbols...")
    bars_map, _ = fetch_latest_bars(
        symbols=symbols,
        timeframe="1Day",
        history_days=2500,
        feed="iex",
        verbose=True
    )
    
    # Filter to 2020+
    for sym in list(bars_map.keys()):
        if bars_map[sym] is not None and not bars_map[sym].empty:
            bars_map[sym] = bars_map[sym][bars_map[sym].index >= "2020-01-01"]
    
    spy_df = bars_map.get("SPY")
    if spy_df is None or spy_df.empty:
        print("ERROR: No SPY data")
        return
    
    # Run backtests and collect trades
    all_trades = []
    
    print("\nRunning backtests...")
    for symbol in DEFAULT_TICKERS:
        if symbol not in bars_map or bars_map[symbol] is None:
            continue
        
        trades = run_backtest_with_trades(symbol, bars_map[symbol], spy_df, config)
        all_trades.extend(trades)
        print(f"  {symbol}: {len(trades)} trades")
    
    print(f"\nTotal trades across all tickers: {len(all_trades)}")
    
    # Analyze capital requirements
    analysis = analyze_capital_requirements(all_trades)
    
    print("\n" + "=" * 80)
    print("WORKING CAPITAL ANALYSIS")
    print("=" * 80)
    
    print(f"\n📊 CAPITAL REQUIREMENTS")
    print("-" * 40)
    print(f"Peak Capital Required:  ${analysis['peak_capital']:,.0f}")
    print(f"Peak Date:              {analysis['peak_date'].strftime('%Y-%m-%d')}")
    print(f"Peak Positions:         {analysis['peak_positions']}")
    print(f"")
    print(f"Average Capital in Use: ${analysis['avg_capital']:,.0f}")
    print(f"Median Capital in Use:  ${analysis['median_capital']:,.0f}")
    print(f"")
    print(f"75th Percentile:        ${analysis['p75_capital']:,.0f}")
    print(f"90th Percentile:        ${analysis['p90_capital']:,.0f}")
    print(f"95th Percentile:        ${analysis['p95_capital']:,.0f}")
    
    print(f"\n📈 POSITION ANALYSIS")
    print("-" * 40)
    print(f"Max Simultaneous Positions: {analysis['max_positions']}")
    print(f"Avg Simultaneous Positions: {analysis['avg_positions']:.1f}")
    
    print(f"\n💰 RETURN ON CAPITAL")
    print("-" * 40)
    print(f"Total PnL:              ${analysis['total_pnl']:,.0f}")
    print(f"ROI on Peak Capital:    {analysis['roi_on_peak']:.1f}%")
    print(f"ROI on Avg Capital:     {analysis['roi_on_avg']:.1f}%")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"\nTo capture ~95% of the strategy's potential:")
    print(f"  → Recommended Capital: ${analysis['p95_capital']:,.0f}")
    print(f"  → Expected Return: {analysis['total_pnl']/ analysis['p95_capital'] * 100:.1f}%")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
