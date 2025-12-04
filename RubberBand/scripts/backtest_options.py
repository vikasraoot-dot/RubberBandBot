#!/usr/bin/env python3
"""
Options Backtest: Simulate 0DTE options trading using stock price data.

This uses a simplified model:
- ATM calls have delta ~0.5
- Premium estimated as ATR * delta_factor
- P&L calculated based on underlying move * delta * leverage

Usage:
    python RubberBand/scripts/backtest_options.py --config RubberBand/config.yaml --tickers RubberBand/tickers.txt --days 30
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
# Options Simulation Parameters
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_OPTS = {
    "delta": 0.50,              # ATM call delta
    "premium_atr_mult": 0.15,   # Premium = ATR * this mult (rough estimate)
    "max_premium": 2.00,        # Max $ per share (so $200 per contract)
    "tp_pct": 30.0,             # Take profit at +30%
    "sl_pct": -50.0,            # Stop loss at -50%
    "contracts": 1,             # Contracts per trade
}


def estimate_premium(close: float, atr: float, delta: float = 0.5) -> float:
    """
    Estimate ATM call premium using ATR.
    
    Rough formula: Premium ≈ ATR * factor * delta
    For volatile stocks, premium is higher.
    """
    # Premium typically 1-3% of stock price for 0DTE ATM
    # Use ATR as volatility proxy
    premium = atr * DEFAULT_OPTS["premium_atr_mult"] / delta
    return min(premium, DEFAULT_OPTS["max_premium"])


def simulate_option_trade(
    entry_price: float,
    high: float,
    low: float,
    close: float,
    atr: float,
    opts: dict,
) -> dict:
    """
    Simulate an option trade for a single bar.
    
    Returns dict with entry, exit, pnl, reason.
    """
    delta = opts.get("delta", 0.5)
    tp_pct = opts.get("tp_pct", 30.0)
    sl_pct = opts.get("sl_pct", -50.0)
    contracts = opts.get("contracts", 1)
    
    # Estimate premium at entry
    premium = estimate_premium(entry_price, atr, delta)
    if premium <= 0:
        return None
    
    # Cost basis
    cost = premium * 100 * contracts  # 100 shares per contract
    
    # Calculate option P&L based on underlying move
    # Option move ≈ Underlying move * delta
    # But we need to check intrabar for TP/SL
    
    # Check stop loss (underlying drops)
    underlying_move_for_sl = (sl_pct / 100) * premium / delta
    sl_price = entry_price + underlying_move_for_sl  # Will be negative move
    
    # Check take profit (underlying rises)
    underlying_move_for_tp = (tp_pct / 100) * premium / delta
    tp_price = entry_price + underlying_move_for_tp
    
    exit_reason = ""
    exit_price = close
    
    # Check if SL hit (low touched SL level)
    if low <= sl_price:
        exit_reason = "SL"
        exit_price = sl_price
    # Check if TP hit (high touched TP level)
    elif high >= tp_price:
        exit_reason = "TP"
        exit_price = tp_price
    else:
        # Exit at close (EOD flatten)
        exit_reason = "EOD"
        exit_price = close
    
    # Calculate option P&L
    underlying_move = exit_price - entry_price
    option_move = underlying_move * delta
    exit_premium = premium + option_move
    
    # P&L
    pnl = (exit_premium - premium) * 100 * contracts
    pnl_pct = (pnl / cost) * 100 if cost > 0 else 0
    
    return {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "premium": round(premium, 2),
        "exit_premium": round(exit_premium, 2),
        "cost": round(cost, 2),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 1),
        "reason": exit_reason,
    }


def simulate_options_for_symbol(
    df: pd.DataFrame,
    cfg: dict,
    sym: str,
    opts: dict,
) -> list:
    """
    Run options simulation for a symbol.
    
    Returns list of trade dicts.
    """
    if df is None or df.empty or len(df) < 30:
        return []
    
    df = attach_verifiers(df, cfg).copy()
    trades = []
    
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        cur = df.iloc[i]
        
        # Check for long signal on previous bar
        if not prev.get("long_signal", False):
            continue
        
        # Entry at open of current bar
        entry_price = float(cur["open"])
        high = float(cur["high"])
        low = float(cur["low"])
        close = float(cur["close"])
        atr = float(prev.get("atr", 0))
        
        if atr <= 0 or entry_price <= 0:
            continue
        
        # Simulate the trade
        result = simulate_option_trade(
            entry_price=entry_price,
            high=high,
            low=low,
            close=close,
            atr=atr,
            opts=opts,
        )
        
        if result:
            result["symbol"] = sym
            result["entry_time"] = str(cur.name)
            result["atr"] = round(atr, 2)
            trades.append(result)
    
    return trades


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Options Backtest (Simulated)")
    ap.add_argument("--config", default="RubberBand/config.yaml")
    ap.add_argument("--tickers", default="RubberBand/tickers.txt")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--tp-pct", type=float, default=30.0)
    ap.add_argument("--sl-pct", type=float, default=-50.0)
    ap.add_argument("--max-premium", type=float, default=2.0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    
    cfg = load_config(args.config)
    
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = read_tickers(args.tickers)
    
    opts = {
        "delta": args.delta,
        "tp_pct": args.tp_pct,
        "sl_pct": args.sl_pct,
        "max_premium": args.max_premium,
        "contracts": 1,
    }
    
    # Fetch data
    timeframe = "15Min"
    feed = cfg.get("feed", "iex")
    fetch_days = int(args.days * 1.6)
    
    print(f"Fetching {len(symbols)} symbols for {args.days} days...", flush=True)
    
    bars_map, _ = fetch_latest_bars(
        symbols=symbols,
        timeframe=timeframe,
        history_days=fetch_days,
        feed=feed,
        verbose=not args.quiet,
    )
    
    # Run simulation
    all_trades = []
    symbol_stats = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0, "losses": 0})
    
    for sym in symbols:
        df = bars_map.get(sym, pd.DataFrame())
        if df.empty:
            continue
        
        trades = simulate_options_for_symbol(df, cfg, sym, opts)
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
    
    tp_exits = sum(1 for t in all_trades if t["reason"] == "TP")
    sl_exits = sum(1 for t in all_trades if t["reason"] == "SL")
    eod_exits = sum(1 for t in all_trades if t["reason"] == "EOD")
    
    print("\n" + "=" * 60)
    print("OPTIONS BACKTEST RESULTS (Simulated)")
    print("=" * 60)
    print(f"Period: {args.days} days")
    print(f"Symbols: {len(symbols)}")
    print(f"Options Config: delta={opts['delta']}, TP={opts['tp_pct']}%, SL={opts['sl_pct']}%")
    print("-" * 60)
    print(f"Total Trades: {total_trades}")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"Avg Win: ${avg_win:.2f}")
    print(f"Avg Loss: ${avg_loss:.2f}")
    print("-" * 60)
    print(f"Exit by TP: {tp_exits} ({tp_exits/total_trades*100:.1f}%)")
    print(f"Exit by SL: {sl_exits} ({sl_exits/total_trades*100:.1f}%)")
    print(f"Exit by EOD: {eod_exits} ({eod_exits/total_trades*100:.1f}%)")
    print("=" * 60)
    
    # Top symbols
    sorted_syms = sorted(symbol_stats.items(), key=lambda x: x[1]["pnl"], reverse=True)
    print("\nTop 10 Symbols by P&L:")
    for sym, stats in sorted_syms[:10]:
        wr = stats["wins"] / max(stats["trades"], 1) * 100
        print(f"  {sym}: ${stats['pnl']:,.2f} ({stats['trades']} trades, {wr:.0f}% WR)")
    
    # Save results
    results_dir = cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    df_trades = pd.DataFrame(all_trades)
    df_trades.to_csv(os.path.join(results_dir, "options_backtest_trades.csv"), index=False)
    
    summary = {
        "days": args.days,
        "symbols": len(symbols),
        "total_trades": total_trades,
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(win_rate, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "tp_exits": tp_exits,
        "sl_exits": sl_exits,
        "eod_exits": eod_exits,
        "opts": opts,
    }
    with open(os.path.join(results_dir, "options_backtest_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved results to {results_dir}/options_backtest_*.csv/json")


if __name__ == "__main__":
    main()
