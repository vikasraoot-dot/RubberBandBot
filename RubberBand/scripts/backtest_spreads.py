#!/usr/bin/env python3
"""
Options Spread Backtest: Simulate bull call spreads using 1-3 DTE options.

Bull Call Spread:
- Buy ATM call
- Sell OTM call (e.g., strike + 1 ATR)
- Max profit = spread width - net debit
- Max loss = net debit (defined risk)

Usage:
    python RubberBand/scripts/backtest_spreads.py --config RubberBand/config.yaml --tickers RubberBand/tickers.txt --days 30
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import pandas as pd
from collections import defaultdict
from datetime import datetime

# Ensure repo root is on path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.utils import load_config, read_tickers
from RubberBand.src.data import fetch_latest_bars
from RubberBand.strategy import attach_verifiers

# ──────────────────────────────────────────────────────────────────────────────
# Spread Simulation Parameters
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_OPTS = {
    "dte": 2,                   # Days to expiration (1-3)
    "spread_width_atr": 1.5,    # OTM strike = ATM + this * ATR
    "max_debit": 1.00,          # Max $ per share for the spread
    "contracts": 1,             # Contracts per trade
    "bars_per_day": 26,         # 15m bars per trading day (6.5 hours)
    "sma_period": 20,           # Daily SMA period for trend filter (20 = ~1 month)
    "trend_filter": True,       # Enable/disable SMA trend filter
}


def estimate_spread_value(
    underlying_price: float,
    atm_strike: float,
    otm_strike: float,
    dte_bars_remaining: int,
    total_dte_bars: int,
) -> tuple:
    """
    Estimate call spread value using simplified Black-Scholes-like approximation.
    
    For a bull call spread:
    - Value at entry ≈ intrinsic + time value
    - Value at expiry = max(0, min(underlying - atm_strike, spread_width))
    
    Returns: (long_call_value, short_call_value, spread_value)
    """
    spread_width = otm_strike - atm_strike
    
    # Time decay factor (linear approximation)
    time_factor = dte_bars_remaining / max(total_dte_bars, 1)
    
    # Intrinsic values
    long_intrinsic = max(0, underlying_price - atm_strike)
    short_intrinsic = max(0, underlying_price - otm_strike)
    
    # Time value (rough estimate, decays with sqrt of time for options)
    time_value_factor = (time_factor ** 0.5) * 0.3  # 30% of spread at entry
    
    # Long call value
    long_time_value = spread_width * time_value_factor
    long_value = long_intrinsic + long_time_value
    
    # Short call value (less time value since OTM)
    short_time_value = spread_width * time_value_factor * 0.5
    short_value = short_intrinsic + short_time_value
    
    spread_value = long_value - short_value
    
    return long_value, short_value, spread_value


def simulate_spread_trade(
    df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    atr: float,
    opts: dict,
) -> dict:
    """
    Simulate a bull call spread trade.
    
    Hold for DTE bars or until max profit/loss.
    """
    dte = opts.get("dte", 2)
    spread_width_atr = opts.get("spread_width_atr", 1.5)
    max_debit = opts.get("max_debit", 1.00)
    contracts = opts.get("contracts", 1)
    bars_per_day = opts.get("bars_per_day", 26)
    
    # Calculate strikes
    atm_strike = entry_price
    spread_width = atr * spread_width_atr
    otm_strike = atm_strike + spread_width
    
    # Total bars to hold (DTE * bars per day)
    total_bars = dte * bars_per_day
    exit_idx = min(entry_idx + total_bars, len(df) - 1)
    
    # Initial spread value (entry debit)
    _, _, entry_spread_value = estimate_spread_value(
        entry_price, atm_strike, otm_strike, total_bars, total_bars
    )
    
    # Cap at max debit
    entry_debit = min(entry_spread_value, max_debit)
    if entry_debit <= 0:
        return None
    
    cost = entry_debit * 100 * contracts
    max_profit = (spread_width - entry_debit) * 100 * contracts
    max_loss = -cost
    
    # Simulate bar by bar
    best_pnl = 0
    worst_pnl = 0
    exit_reason = "EXPIRY"
    exit_value = 0
    actual_exit_idx = exit_idx
    
    for i in range(entry_idx + 1, exit_idx + 1):
        bar = df.iloc[i]
        bars_remaining = exit_idx - i
        
        # Check at high and low
        for check_price in [bar["high"], bar["low"], bar["close"]]:
            _, _, current_value = estimate_spread_value(
                float(check_price), atm_strike, otm_strike, bars_remaining, total_bars
            )
            
            current_pnl = (current_value - entry_debit) * 100 * contracts
            
            # Track best/worst
            best_pnl = max(best_pnl, current_pnl)
            worst_pnl = min(worst_pnl, current_pnl)
            
            # Check for max profit (90% of max)
            if current_pnl >= max_profit * 0.9:
                exit_reason = "MAX_PROFIT"
                exit_value = current_value
                actual_exit_idx = i
                break
            
            # Check for max loss (90% of max)
            if current_pnl <= max_loss * 0.9:
                exit_reason = "MAX_LOSS"
                exit_value = current_value
                actual_exit_idx = i
                break
        
        if exit_reason != "EXPIRY":
            break
    
    # If held to expiry, calculate final value
    if exit_reason == "EXPIRY":
        final_bar = df.iloc[exit_idx]
        final_price = float(final_bar["close"])
        
        # At expiry, spread value = intrinsic only
        long_intrinsic = max(0, final_price - atm_strike)
        short_intrinsic = max(0, final_price - otm_strike)
        exit_value = long_intrinsic - short_intrinsic
    
    # Calculate final P&L
    pnl = (exit_value - entry_debit) * 100 * contracts
    pnl = max(pnl, max_loss)  # Can't lose more than debit
    pnl = min(pnl, max_profit)  # Can't win more than spread width - debit
    
    pnl_pct = (pnl / cost) * 100 if cost > 0 else 0
    
    return {
        "entry_price": round(entry_price, 2),
        "atm_strike": round(atm_strike, 2),
        "otm_strike": round(otm_strike, 2),
        "spread_width": round(spread_width, 2),
        "entry_debit": round(entry_debit, 2),
        "exit_value": round(exit_value, 2),
        "cost": round(cost, 2),
        "max_profit": round(max_profit, 2),
        "max_loss": round(max_loss, 2),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 1),
        "reason": exit_reason,
        "bars_held": actual_exit_idx - entry_idx,
        "dte": dte,
    }


def simulate_spreads_for_symbol(
    df: pd.DataFrame,
    cfg: dict,
    sym: str,
    opts: dict,
    daily_sma: Optional[float] = None,
) -> list:
    """
    Run spread simulation for a symbol.
    
    Args:
        daily_sma: If provided, skip signals when close < daily_sma (trend filter)
    """
    if df is None or df.empty or len(df) < 50:
        return []
    
    df = attach_verifiers(df, cfg).copy()
    trades = []
    
    # Track position to avoid overlapping trades
    in_trade_until = -1
    
    for i in range(1, len(df)):
        if i <= in_trade_until:
            continue
        
        prev = df.iloc[i - 1]
        cur = df.iloc[i]
        
        # Check for long signal
        if not prev.get("long_signal", False):
            continue
        
        entry_price = float(cur["open"])
        atr = float(prev.get("atr", 0))
        
        if atr <= 0 or entry_price <= 0:
            continue
        
        # SMA Trend Filter: Skip if close < daily SMA (matching live bot)
        if daily_sma is not None and opts.get("trend_filter", True):
            close_price = float(prev["close"])
            if close_price < daily_sma:
                continue  # Skip - in bear trend
        
        # Simulate the spread trade
        result = simulate_spread_trade(df, i, entry_price, atr, opts)
        
        if result:
            result["symbol"] = sym
            result["entry_time"] = str(cur.name)
            result["atr"] = round(atr, 2)
            trades.append(result)
            
            # Block overlapping trades
            in_trade_until = i + result["bars_held"]
    
    return trades


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Bull Call Spread Backtest")
    ap.add_argument("--config", default="RubberBand/config.yaml")
    ap.add_argument("--tickers", default="RubberBand/tickers.txt")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--dte", type=int, default=2, help="Days to expiration (1-3)")
    ap.add_argument("--spread-width", type=float, default=1.5, help="Spread width in ATR")
    ap.add_argument("--max-debit", type=float, default=1.0, help="Max debit per share")
    ap.add_argument("--sma-period", type=int, default=20, help="Daily SMA period for trend filter")
    ap.add_argument("--no-trend-filter", action="store_true", help="Disable SMA trend filter")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    
    cfg = load_config(args.config)
    
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = read_tickers(args.tickers)
    
    # Debug: Print loaded tickers
    print(f"\n{'='*60}")
    print(f"LOADED TICKERS FROM: {args.tickers}")
    print(f"{'='*60}")
    print(f"Total: {len(symbols)} symbols")
    print(f"Tickers: {symbols}")
    print(f"{'='*60}\n")
    
    opts = {
        "dte": args.dte,
        "spread_width_atr": args.spread_width,
        "max_debit": args.max_debit,
        "contracts": 1,
        "bars_per_day": 26,
        "sma_period": args.sma_period,
        "trend_filter": not args.no_trend_filter,
    }
    
    # Fetch 15-minute data
    timeframe = "15Min"
    feed = cfg.get("feed", "iex")
    fetch_days = int(args.days * 1.6)
    
    print(f"Fetching {len(symbols)} symbols for {args.days} days...", flush=True)
    print(f"SMA Trend Filter: {'ENABLED' if opts['trend_filter'] else 'DISABLED'} (SMA-{args.sma_period})")
    
    bars_map, _ = fetch_latest_bars(
        symbols=symbols,
        timeframe=timeframe,
        history_days=fetch_days,
        feed=feed,
        verbose=not args.quiet,
    )
    
    # Fetch daily data for SMA trend filter
    daily_sma_map = {}
    if opts["trend_filter"]:
        sma_history_days = max(args.sma_period * 2, 60)  # At least 60 days for reliable SMA
        print(f"Fetching daily bars for SMA-{args.sma_period} trend filter (history={sma_history_days} days)...")
        daily_bars_map, _ = fetch_latest_bars(
            symbols=symbols,
            timeframe="1Day",
            history_days=sma_history_days,
            feed=feed,
            verbose=False,
        )
        skipped_no_data = 0
        skipped_short = 0
        for sym in symbols:
            df_daily = daily_bars_map.get(sym)
            if df_daily is None or df_daily.empty:
                skipped_no_data += 1
                continue
            if len(df_daily) < args.sma_period:
                skipped_short += 1
                continue
            sma_val = df_daily["close"].rolling(window=args.sma_period).mean().iloc[-1]
            if not pd.isna(sma_val):
                daily_sma_map[sym] = float(sma_val)
        print(f"  Calculated SMA for {len(daily_sma_map)} symbols")
        if skipped_no_data > 0 or skipped_short > 0:
            print(f"  Skipped: {skipped_no_data} no data, {skipped_short} insufficient history")
    
    # Run simulation
    all_trades = []
    symbol_stats = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0, "losses": 0})
    
    for sym in symbols:
        df = bars_map.get(sym, pd.DataFrame())
        if df.empty:
            continue
        
        # Get daily SMA for this symbol (None if not available)
        daily_sma = daily_sma_map.get(sym)
        
        trades = simulate_spreads_for_symbol(df, cfg, sym, opts, daily_sma=daily_sma)
        all_trades.extend(trades)
        
        for t in trades:
            symbol_stats[sym]["trades"] += 1
            symbol_stats[sym]["pnl"] += t["pnl"]
            if t["pnl"] > 0:
                symbol_stats[sym]["wins"] += 1
            else:
                symbol_stats[sym]["losses"] += 1
    
    if not all_trades:
        print("No trades generated.")
        return
    
    # Summary
    total_trades = len(all_trades)
    total_pnl = sum(t["pnl"] for t in all_trades)
    wins = sum(1 for t in all_trades if t["pnl"] > 0)
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    avg_win = sum(t["pnl"] for t in all_trades if t["pnl"] > 0) / max(wins, 1)
    avg_loss = sum(t["pnl"] for t in all_trades if t["pnl"] <= 0) / max(losses, 1)
    
    max_profit_exits = sum(1 for t in all_trades if t["reason"] == "MAX_PROFIT")
    max_loss_exits = sum(1 for t in all_trades if t["reason"] == "MAX_LOSS")
    expiry_exits = sum(1 for t in all_trades if t["reason"] == "EXPIRY")
    
    avg_bars_held = sum(t["bars_held"] for t in all_trades) / total_trades
    total_cost = sum(t["cost"] for t in all_trades)
    roi = (total_pnl / total_cost * 100) if total_cost > 0 else 0
    
    print("\n" + "=" * 60)
    print("BULL CALL SPREAD BACKTEST RESULTS")
    print("=" * 60)
    print(f"Period: {args.days} days")
    print(f"Symbols: {len(symbols)}")
    print(f"Spread Config: DTE={opts['dte']}, Width={opts['spread_width_atr']} ATR")
    print("-" * 60)
    print(f"Total Trades: {total_trades}")
    print(f"Total Cost: ${total_cost:,.2f}")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"ROI: {roi:.1f}%")
    print(f"Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"Avg Win: ${avg_win:.2f}")
    print(f"Avg Loss: ${avg_loss:.2f}")
    print(f"Avg Bars Held: {avg_bars_held:.1f}")
    print("-" * 60)
    print(f"Exit @ Max Profit: {max_profit_exits} ({max_profit_exits/total_trades*100:.1f}%)")
    print(f"Exit @ Max Loss: {max_loss_exits} ({max_loss_exits/total_trades*100:.1f}%)")
    print(f"Exit @ Expiry: {expiry_exits} ({expiry_exits/total_trades*100:.1f}%)")
    print("=" * 60)
    
    # Top symbols
    sorted_syms = sorted(symbol_stats.items(), key=lambda x: x[1]["pnl"], reverse=True)
    print("\nTop 10 Symbols by P&L:")
    for sym, stats in sorted_syms[:10]:
        wr = stats["wins"] / max(stats["trades"], 1) * 100
        print(f"  {sym}: ${stats['pnl']:,.2f} ({stats['trades']} trades, {wr:.0f}% WR)")
    
    print("\nBottom 5 Symbols:")
    for sym, stats in sorted_syms[-5:]:
        wr = stats["wins"] / max(stats["trades"], 1) * 100
        print(f"  {sym}: ${stats['pnl']:,.2f} ({stats['trades']} trades, {wr:.0f}% WR)")
    
    # Save results
    results_dir = cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    df_trades = pd.DataFrame(all_trades)
    df_trades.to_csv(os.path.join(results_dir, "spread_backtest_trades.csv"), index=False)
    
    summary = {
        "days": args.days,
        "symbols": len(symbols),
        "total_trades": total_trades,
        "total_cost": round(total_cost, 2),
        "total_pnl": round(total_pnl, 2),
        "roi_pct": round(roi, 2),
        "win_rate": round(win_rate, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "opts": opts,
    }
    with open(os.path.join(results_dir, "spread_backtest_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved results to {results_dir}/spread_backtest_*.csv/json")


if __name__ == "__main__":
    main()
