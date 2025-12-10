#!/usr/bin/env python3
"""
Silver Light Multi-Ticker Backtest
===================================
Runs the Silver Light strategy across multiple tickers and provides
per-ticker breakdown of PnL, wins, losses, etc.

Usage:
    python RubberBand/scripts/silverlight_multi_backtest.py
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# --- Ensure repo root on sys.path ---
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.data import fetch_latest_bars
from RubberBand.src.silverlight_strategy import (
    attach_indicators,
    generate_signal,
    calculate_position_size,
    Signal
)

# Pre-selected tickers for Silver Light
# NOTE: SPY removed - it's used as regime filter, can't backtest SPY with SPY
DEFAULT_TICKERS = [
    "QQQ",    # NASDAQ 100 ETF
    "NVDA",   # GPU leader
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "AMZN",   # Amazon
    "META",   # Meta
    "GOOGL",  # Alphabet
    "AMD",    # Semiconductors
    "AVGO",   # Broadcom
    "COST",   # Costco - stable mega cap
]


def load_config(path: str = "RubberBand/config_silverlight.yaml") -> Dict[str, Any]:
    """Load Silver Light configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_single_ticker_backtest(
    symbol: str,
    df: pd.DataFrame,
    spy_df: pd.DataFrame,
    config: Dict[str, Any],
    initial_capital: float = 10000.0
) -> Dict[str, Any]:
    """
    Run Silver Light backtest for a single ticker.
    
    Returns metrics dict with PnL, wins, losses, etc.
    """
    # Get risk settings
    risk_cfg = config.get("risk", {})
    trailing_stop_enabled = risk_cfg.get("trailing_stop_enabled", True)
    trailing_stop_pct = float(risk_cfg.get("trailing_stop_pct", 0.10))
    
    bt_cfg = config.get("backtest", {})
    slippage_pct = float(bt_cfg.get("slippage_pct", 0.001))
    commission = float(bt_cfg.get("commission_per_trade", 0.0))
    
    # Attach indicators
    ticker_df = attach_indicators(df.copy(), config)
    regime_df = attach_indicators(spy_df.copy(), config)
    
    # Align dates
    common_dates = ticker_df.index.intersection(regime_df.index)
    ticker_df = ticker_df.loc[common_dates]
    regime_df = regime_df.loc[common_dates]
    
    # Skip warmup
    warmup = 200
    if len(ticker_df) <= warmup:
        return {"error": "Not enough data"}
    
    ticker_df = ticker_df.iloc[warmup:]
    regime_df = regime_df.iloc[warmup:]
    
    # State
    cash = initial_capital
    position_qty = 0
    entry_price = 0.0
    entry_date = None  # FIX: Track entry date for trades
    peak_price = 0.0
    
    trades = []
    equity_curve = []
    
    for dt, row in ticker_df.iterrows():
        current_price = float(row["close"])
        
        # Generate signal (using ticker for 50 SMA, SPY for 200 SMA regime)
        signal, meta = generate_signal(
            ticker_df.loc[:dt],
            regime_df.loc[:dt],
            config
        )
        
        # Track equity
        if position_qty > 0:
            equity = cash + (position_qty * current_price)
            if current_price > peak_price:
                peak_price = current_price
        else:
            equity = cash
        
        equity_curve.append({"date": dt, "equity": equity})
        
        # Trading logic
        if position_qty > 0:
            exit_triggered = False
            exit_reason = ""
            
            # Check trailing stop
            if trailing_stop_enabled and peak_price > 0:
                stop_price = peak_price * (1 - trailing_stop_pct)
                if current_price < stop_price:
                    exit_triggered = True
                    exit_reason = "TRAILING_STOP"
            
            # Check signal exit
            if not exit_triggered and signal != Signal.LONG:
                exit_triggered = True
                exit_reason = meta.get("reason", "SIGNAL")[:20]
            
            if exit_triggered:
                exit_price = current_price * (1 - slippage_pct)
                proceeds = position_qty * exit_price - commission
                pnl = proceeds - (position_qty * entry_price)
                
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": dt,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "pnl_pct": (exit_price / entry_price - 1) * 100 if entry_price > 0 else 0,  # FIX: Guard divide-by-zero
                    "exit_reason": exit_reason
                })
                
                cash += proceeds
                position_qty = 0
                peak_price = 0.0
        
        elif signal == Signal.LONG:
            entry_price = current_price * (1 + slippage_pct)
            position_qty = int(cash // entry_price)
            
            if position_qty > 0:
                cost = position_qty * entry_price + commission
                cash -= cost
                peak_price = entry_price
                entry_date = dt  # FIX: Track entry date
    
    # Close any open position
    if position_qty > 0:
        exit_price = float(ticker_df.iloc[-1]["close"])
        proceeds = position_qty * exit_price
        pnl = proceeds - (position_qty * entry_price)
        trades.append({
            "entry_date": entry_date,
            "exit_date": ticker_df.index[-1],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": (exit_price / entry_price - 1) * 100 if entry_price > 0 else 0,
            "exit_reason": "END_OF_BACKTEST"
        })
        cash += proceeds
    
    # Calculate metrics
    final_equity = cash
    total_return = (final_equity / initial_capital - 1) * 100
    
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return {
            "symbol": symbol,
            "final_equity": initial_capital,
            "total_return": 0,
            "max_drawdown": 0,  # FIX: Add missing key
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "avg_win": 0,
            "avg_loss": 0,
        }
    
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    
    # Calculate hold times
    trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
    trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
    trades_df["hold_days"] = (trades_df["exit_date"] - trades_df["entry_date"]).dt.days
    
    wins_with_days = trades_df[trades_df["pnl"] > 0]
    losses_with_days = trades_df[trades_df["pnl"] <= 0]
    
    avg_hold_win = wins_with_days["hold_days"].mean() if len(wins_with_days) > 0 else 0
    avg_hold_loss = losses_with_days["hold_days"].mean() if len(losses_with_days) > 0 else 0
    
    # Max drawdown
    equity_df = pd.DataFrame(equity_curve)
    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["drawdown"] = (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"]
    max_dd = abs(equity_df["drawdown"].min()) * 100
    
    return {
        "symbol": symbol,
        "final_equity": round(final_equity, 2),
        "total_return": round(total_return, 1),
        "max_drawdown": round(max_dd, 1),
        "total_trades": len(trades_df),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades_df) * 100, 1) if len(trades_df) > 0 else 0,
        "total_pnl": round(trades_df["pnl"].sum(), 2),
        "avg_win": round(wins["pnl"].mean(), 2) if len(wins) > 0 else 0,
        "avg_loss": round(losses["pnl"].mean(), 2) if len(losses) > 0 else 0,
        "avg_hold_win": round(avg_hold_win, 1),
        "avg_hold_loss": round(avg_hold_loss, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Silver Light Multi-Ticker Backtest")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="Tickers to test")
    parser.add_argument("--start-date", default="2020-01-01", help="Start date")
    parser.add_argument("--end-date", default="2024-12-31", help="End date")
    parser.add_argument("--capital", type=float, default=10000, help="Capital per ticker")
    args = parser.parse_args()
    
    config = load_config()
    
    print("=" * 80)
    print("SILVER LIGHT MULTI-TICKER BACKTEST")
    print("=" * 80)
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Capital per ticker: ${args.capital:,.0f}")
    print(f"Tickers: {', '.join(args.tickers)}")
    print("=" * 80)
    
    # Fetch all data
    all_symbols = args.tickers + ["SPY"]  # SPY for regime filter
    
    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")
    days = (end - start).days + 300
    
    print(f"\nFetching {days} days of data for {len(all_symbols)} symbols...")
    
    bars_map, _ = fetch_latest_bars(
        symbols=all_symbols,
        timeframe="1Day",
        history_days=days,
        feed="iex",
        rth_only=False,
        verbose=True
    )
    
    # Filter to date range
    for sym, df in bars_map.items():
        if df is not None and not df.empty:
            bars_map[sym] = df[(df.index >= args.start_date) & (df.index <= args.end_date)]
    
    # Check SPY
    if "SPY" not in bars_map or bars_map["SPY"] is None or bars_map["SPY"].empty:
        print("ERROR: No SPY data for regime filter")
        return
    
    spy_df = bars_map["SPY"]
    
    # Run backtest for each ticker
    results = []
    
    print("\n" + "-" * 80)
    print("RUNNING BACKTESTS")
    print("-" * 80)
    
    for symbol in args.tickers:
        if symbol not in bars_map or bars_map[symbol] is None or bars_map[symbol].empty:
            print(f"  {symbol}: No data available")
            continue
        
        print(f"  {symbol}...", end=" ")
        result = run_single_ticker_backtest(
            symbol, bars_map[symbol], spy_df, config, args.capital
        )
        
        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"Return: {result['total_return']:+.1f}%, Trades: {result['total_trades']}, Win Rate: {result['win_rate']:.0f}%")
            results.append(result)
    
    # Summary table
    print("\n" + "=" * 120)
    print("RESULTS BY TICKER")
    print("=" * 120)
    
    print(f"\n{'Symbol':<8} {'Return%':>8} {'MaxDD%':>7} {'Trades':>7} {'Wins':>5} {'Losses':>7} {'WinRate':>8} {'TotalPnL':>11} {'AvgWin':>9} {'AvgLoss':>9} {'HoldWin':>8} {'HoldLoss':>9}")
    print("-" * 120)
    
    total_pnl = 0
    total_trades = 0
    total_wins = 0
    total_losses = 0
    
    for r in results:
        print(f"{r['symbol']:<8} {r['total_return']:>+8.1f} {r['max_drawdown']:>7.1f} {r['total_trades']:>7} {r['wins']:>5} {r['losses']:>7} {r['win_rate']:>7.0f}% ${r['total_pnl']:>10,.0f} ${r['avg_win']:>8,.0f} ${r['avg_loss']:>8,.0f} {r['avg_hold_win']:>7.0f}d {r['avg_hold_loss']:>8.0f}d")
        total_pnl += r['total_pnl']
        total_trades += r['total_trades']
        total_wins += r['wins']
        total_losses += r['losses']
    
    print("-" * 120)
    print(f"{'TOTAL':<8} {'':>8} {'':>7} {total_trades:>7} {total_wins:>5} {total_losses:>7} {total_wins/total_trades*100 if total_trades > 0 else 0:>7.0f}% ${total_pnl:>10,.0f}")
    
    # Portfolio summary
    total_capital = args.capital * len(args.tickers)
    total_final = sum(r['final_equity'] for r in results)
    portfolio_return = (total_final / total_capital - 1) * 100
    
    print("\n" + "=" * 80)
    print("PORTFOLIO SUMMARY")
    print("=" * 80)
    print(f"Total Capital Deployed: ${total_capital:,.0f}")
    print(f"Total Final Equity: ${total_final:,.2f}")
    print(f"Portfolio Return: {portfolio_return:+.1f}%")
    print(f"Total Trades: {total_trades}")
    print(f"Overall Win Rate: {total_wins/total_trades*100 if total_trades > 0 else 0:.1f}%")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("silverlight_multi_results.csv", index=False)
    print(f"\nResults saved to: silverlight_multi_results.csv")
    
    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
