#!/usr/bin/env python3
"""
Silver Light Backtest
=====================
Backtest engine for the TQQQ/SQQQ trend-following strategy.

Usage:
    python RubberBand/scripts/silverlight_backtest.py --start-date 2020-01-01 --end-date 2024-12-31
    python RubberBand/scripts/silverlight_backtest.py --days 365

This script is COMPLETELY ISOLATED from Rubber Band bots.
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# --- Ensure repo root on sys.path ---
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import data fetching (shared utility - read-only)
from RubberBand.src.data import fetch_latest_bars

# Import Silver Light strategy (new module)
from RubberBand.src.silverlight_strategy import (
    attach_indicators,
    generate_signal,
    calculate_position_size,
    Signal
)


def load_config(path: str = "RubberBand/config_silverlight.yaml") -> Dict[str, Any]:
    """Load Silver Light configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fetch_historical_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    feed: str = "iex"
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical daily bars for backtesting.
    
    Args:
        symbols: List of symbols (TQQQ, SQQQ, SPY)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        feed: Data feed (iex, sip)
        
    Returns:
        Dict mapping symbol -> DataFrame with OHLCV
    """
    # Calculate days from start to end
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end - start).days + 300  # Extra buffer for SMA calculation
    
    print(f"Fetching {days} days of data for {symbols}...")
    
    bars_map, meta = fetch_latest_bars(
        symbols=symbols,
        timeframe="1Day",
        history_days=days,
        feed=feed,
        rth_only=False,
        verbose=True
    )
    
    # Filter to date range
    result = {}
    for sym, df in bars_map.items():
        if df is not None and not df.empty:
            df = df[df.index >= start_date]
            df = df[df.index <= end_date]
            result[sym] = df
            print(f"  {sym}: {len(df)} bars")
    
    return result


def run_backtest(
    config: Dict[str, Any],
    data: Dict[str, pd.DataFrame],
    initial_capital: float = 10000.0
) -> Dict[str, Any]:
    """
    Run the Silver Light backtest simulation.
    
    Args:
        config: Strategy configuration
        data: Dict of DataFrames (TQQQ, SQQQ, SPY)
        initial_capital: Starting capital
        
    Returns:
        Dict with backtest results (trades, metrics, equity curve)
    """
    assets_cfg = config.get("assets", {})
    long_sym = assets_cfg.get("long", "TQQQ")
    short_sym = assets_cfg.get("short", "SQQQ")
    regime_sym = assets_cfg.get("regime_index", "SPY")
    vix_sym = assets_cfg.get("volatility_index", "VIXY")
    
    # Get slippage/commission
    bt_cfg = config.get("backtest", {})
    slippage_pct = float(bt_cfg.get("slippage_pct", 0.001))
    commission = float(bt_cfg.get("commission_per_trade", 0.0))
    
    # Get risk management settings
    risk_cfg = config.get("risk", {})
    trailing_stop_enabled = risk_cfg.get("trailing_stop_enabled", True)
    trailing_stop_pct = float(risk_cfg.get("trailing_stop_pct", 0.10))
    
    # Prepare data with indicators
    if long_sym not in data or regime_sym not in data:
        return {"error": f"Missing data for {long_sym} or {regime_sym}"}
    
    tqqq_df = attach_indicators(data[long_sym], config)
    spy_df = attach_indicators(data[regime_sym], config)
    
    # VIX data (optional)
    vix_df = data.get(vix_sym, None)
    if vix_df is not None:
        vix_df = vix_df.copy()
    
    # Align dates
    common_dates = tqqq_df.index.intersection(spy_df.index)
    tqqq_df = tqqq_df.loc[common_dates]
    spy_df = spy_df.loc[common_dates]
    if vix_df is not None:
        vix_common = common_dates.intersection(vix_df.index)
        vix_df = vix_df.loc[vix_common]
    
    # Skip first 200 days (need SMA200 to be valid)
    warmup = 200
    if len(tqqq_df) <= warmup:
        return {"error": f"Not enough data. Need > {warmup} bars."}
    
    tqqq_df = tqqq_df.iloc[warmup:]
    spy_df = spy_df.iloc[warmup:]
    
    # State
    cash = initial_capital
    position_qty = 0
    position_symbol = None
    entry_price = 0.0
    entry_date = None
    peak_price = 0.0  # For trailing stop
    
    trades = []
    equity_curve = []
    
    for i, (dt, tqqq_row) in enumerate(tqqq_df.iterrows()):
        spy_row = spy_df.loc[dt]
        current_price = float(tqqq_row["close"])
        
        # Get VIX value if available
        vix_value = None
        if vix_df is not None and dt in vix_df.index:
            vix_value = float(vix_df.loc[dt]["close"])
        
        # Get signal
        signal, meta = generate_signal(
            tqqq_df.loc[:dt],
            spy_df.loc[:dt],
            config,
            vix_value=vix_value
        )
        
        # Current equity
        if position_qty > 0 and position_symbol:
            equity = cash + (position_qty * current_price)
            # Update peak price for trailing stop
            if current_price > peak_price:
                peak_price = current_price
        else:
            equity = cash
        
        equity_curve.append({
            "date": dt,
            "equity": equity,
            "signal": signal.value,
            "reason": meta.get("reason", "")
        })
        
        # --- Trading Logic ---
        
        # If currently in a position
        if position_qty > 0:
            exit_triggered = False
            exit_reason = ""
            
            # Check trailing stop first
            if trailing_stop_enabled and peak_price > 0:
                stop_price = peak_price * (1 - trailing_stop_pct)
                if current_price < stop_price:
                    exit_triggered = True
                    exit_reason = f"TRAILING STOP: Price {current_price:.2f} < {stop_price:.2f} (10% from peak {peak_price:.2f})"
            
            # Check signal-based exit
            if not exit_triggered and signal != Signal.LONG:
                exit_triggered = True
                exit_reason = meta.get("reason", "Signal changed")
            
            if exit_triggered:
                # EXIT
                exit_price = current_price * (1 - slippage_pct)
                proceeds = position_qty * exit_price - commission
                pnl = proceeds - (position_qty * entry_price)
                
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": dt,
                    "symbol": position_symbol,
                    "qty": position_qty,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round((exit_price / entry_price - 1) * 100, 2),
                    "exit_reason": exit_reason
                })
                
                cash += proceeds
                position_qty = 0
                position_symbol = None
                peak_price = 0.0
        
        # If not in position and signal is LONG
        elif signal == Signal.LONG:
            # ENTER
            target_size = calculate_position_size(
                signal, tqqq_df.loc[:dt], config, cash
            )
            entry_price = current_price * (1 + slippage_pct)
            position_qty = int(target_size // entry_price)
            
            if position_qty > 0:
                cost = position_qty * entry_price + commission
                if cost <= cash:
                    cash -= cost
                    position_symbol = long_sym
                    entry_date = dt
                    peak_price = entry_price  # Initialize peak at entry
                else:
                    position_qty = 0
    
    # Close any open position at end
    if position_qty > 0:
        exit_price = float(tqqq_df.iloc[-1]["close"])
        proceeds = position_qty * exit_price
        pnl = proceeds - (position_qty * entry_price)
        
        trades.append({
            "entry_date": entry_date,
            "exit_date": tqqq_df.index[-1],
            "symbol": position_symbol,
            "qty": position_qty,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": round(pnl, 2),
            "pnl_pct": round((exit_price / entry_price - 1) * 100, 2),
            "exit_reason": "END_OF_BACKTEST"
        })
        cash += proceeds
    
    # Calculate metrics
    equity_df = pd.DataFrame(equity_curve)
    if equity_df.empty:
        return {"error": "No equity curve generated"}
    
    final_equity = cash
    total_return = (final_equity / initial_capital - 1) * 100
    
    # CAGR
    start_date = equity_df["date"].iloc[0]
    end_date = equity_df["date"].iloc[-1]
    years = (end_date - start_date).days / 365.25
    cagr = ((final_equity / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    # Max Drawdown
    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["drawdown"] = (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"]
    max_drawdown = abs(equity_df["drawdown"].min()) * 100
    
    # Sharpe Ratio (annualized)
    equity_df["daily_return"] = equity_df["equity"].pct_change()
    sharpe = (equity_df["daily_return"].mean() / equity_df["daily_return"].std()) * np.sqrt(252) if equity_df["daily_return"].std() > 0 else 0
    
    # Win Rate
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        wins = len(trades_df[trades_df["pnl"] > 0])
        total_trades = len(trades_df)
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    else:
        win_rate = 0
        total_trades = 0
    
    return {
        "metrics": {
            "initial_capital": initial_capital,
            "final_equity": round(final_equity, 2),
            "total_return_pct": round(total_return, 2),
            "cagr_pct": round(cagr, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2),
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate, 1),
        },
        "trades": trades,
        "equity_curve": equity_curve
    }


def run_buy_and_hold_baseline(
    data: Dict[str, pd.DataFrame],
    symbol: str = "TQQQ",
    initial_capital: float = 10000.0
) -> Dict[str, Any]:
    """Run simple buy-and-hold for comparison."""
    if symbol not in data:
        return {"error": f"Missing data for {symbol}"}
    
    df = data[symbol]
    if df.empty:
        return {"error": "Empty dataframe"}
    
    # Skip warmup period to match strategy
    warmup = 200
    if len(df) <= warmup:
        return {"error": "Not enough data"}
    df = df.iloc[warmup:]
    
    start_price = float(df.iloc[0]["close"])
    end_price = float(df.iloc[-1]["close"])
    
    shares = initial_capital / start_price
    final_equity = shares * end_price
    total_return = (final_equity / initial_capital - 1) * 100
    
    # CAGR
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = ((final_equity / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    # Max Drawdown
    df = df.copy()
    df["equity"] = shares * df["close"]
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = (df["equity"] - df["peak"]) / df["peak"]
    max_drawdown = abs(df["drawdown"].min()) * 100
    
    return {
        "metrics": {
            "initial_capital": initial_capital,
            "final_equity": round(final_equity, 2),
            "total_return_pct": round(total_return, 2),
            "cagr_pct": round(cagr, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Silver Light Backtest")
    parser.add_argument("--config", default="RubberBand/config_silverlight.yaml", help="Config file path")
    parser.add_argument("--start-date", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=None, help="Number of days to backtest (alternative to dates)")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--output", default="silverlight_backtest_results.csv", help="Output CSV file")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine date range
    if args.days:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    elif args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        # Use defaults from config
        bt_cfg = config.get("backtest", {})
        start_date = bt_cfg.get("start_date", "2020-01-01")
        end_date = bt_cfg.get("end_date", datetime.now().strftime("%Y-%m-%d"))
    
    print("=" * 70)
    print("SILVER LIGHT BACKTEST")
    print("=" * 70)
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print("=" * 70)
    
    # Fetch data
    assets_cfg = config.get("assets", {})
    symbols = [
        assets_cfg.get("long", "TQQQ"),
        assets_cfg.get("short", "SQQQ"),
        assets_cfg.get("regime_index", "SPY"),
        assets_cfg.get("volatility_index", "VIXY"),  # VIX proxy for volatility filter
    ]
    
    data = fetch_historical_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        feed=config.get("execution", {}).get("feed", "iex")
    )
    
    if not data:
        print("ERROR: No data fetched. Exiting.")
        return
    
    # Run strategy backtest
    print("\n--- Running Silver Light Strategy ---")
    results = run_backtest(config, data, args.capital)
    
    if "error" in results:
        print(f"ERROR: {results['error']}")
        return
    
    # Run buy-and-hold baseline
    print("\n--- Running Buy & Hold Baseline ---")
    baseline = run_buy_and_hold_baseline(data, "TQQQ", args.capital)
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    
    metrics = results["metrics"]
    print(f"\n{'Metric':<25} {'Silver Light':>15} {'Buy & Hold':>15}")
    print("-" * 55)
    print(f"{'Final Equity':<25} ${metrics['final_equity']:>14,.2f} ${baseline['metrics']['final_equity']:>14,.2f}")
    print(f"{'Total Return %':<25} {metrics['total_return_pct']:>14.1f}% {baseline['metrics']['total_return_pct']:>14.1f}%")
    print(f"{'CAGR %':<25} {metrics['cagr_pct']:>14.1f}% {baseline['metrics']['cagr_pct']:>14.1f}%")
    print(f"{'Max Drawdown %':<25} {metrics['max_drawdown_pct']:>14.1f}% {baseline['metrics']['max_drawdown_pct']:>14.1f}%")
    print(f"{'Sharpe Ratio':<25} {metrics['sharpe_ratio']:>15.2f} {'N/A':>15}")
    print(f"{'Total Trades':<25} {metrics['total_trades']:>15} {'1 (Buy & Hold)':>15}")
    print(f"{'Win Rate %':<25} {metrics['win_rate_pct']:>14.1f}% {'N/A':>15}")
    
    # Save trades to CSV
    if results["trades"]:
        trades_df = pd.DataFrame(results["trades"])
        trades_df.to_csv(args.output, index=False)
        print(f"\nTrades saved to: {args.output}")
    
    # Save equity curve
    equity_df = pd.DataFrame(results["equity_curve"])
    equity_file = args.output.replace(".csv", "_equity.csv")
    equity_df.to_csv(equity_file, index=False)
    print(f"Equity curve saved to: {equity_file}")
    
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
