#!/usr/bin/env python3
"""
Options Spread Backtest: Simulate bull call spreads with variable DTE (matches live bot behavior).

Bull Call Spread:
- Buy ATM call
- Sell OTM call (e.g., strike + 1 ATR)
- Max profit = spread width - net debit
- Max loss = net debit (defined risk)

Variable DTE: Thu/Fri entries roll to next week's Friday (8-10 DTE).

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
from typing import Optional

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
    "dte": 6,                   # Days to expiration (matches live bot default)
    "spread_width_atr": 1.5,    # OTM strike = ATM + this * ATR
    "max_debit": 2.00,          # Max $ per share for the spread (matches production)
    "contracts": 1,             # Contracts per trade
    "bars_per_day": 26,         # 15m bars per trading day (6.5 hours)
    "sma_period": 20,           # Daily SMA period for trend filter (20 = ~1 month)
    "trend_filter": True,       # Enable/disable SMA trend filter
}


def calculate_actual_dte(entry_date: datetime, target_dte: int, min_dte: int) -> int:
    """
    Calculate actual DTE based on entry day of week, matching live bot behavior.
    """
    # Type guard: if entry_date doesn't have weekday(), return target_dte
    if not hasattr(entry_date, 'weekday'):
        return target_dte
    
    # Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4
    day_of_week = entry_date.weekday()
    
    # Days until this Friday (Friday = 4)
    days_to_friday = (4 - day_of_week) if day_of_week <= 4 else (11 - day_of_week)
    
    # If this Friday is >= min_dte away, use it (capped at target_dte if closer)
    if days_to_friday >= min_dte:
        return max(min_dte, min(target_dte, days_to_friday))
    
    # Otherwise, roll to NEXT Friday (add 7 days)
    days_to_next_friday = days_to_friday + 7
    return days_to_next_friday




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
    time_value_factor = (time_factor ** 0.5) * 0.65  # 65% of spread at entry (realistic)
    
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
    entry_time: datetime = None,
) -> dict:
    """
    Simulate a bull call spread trade.
    
    Hold for DTE bars or until max profit/loss.
    Uses variable DTE based on entry day of week (matching live bot behavior).
    """
    target_dte = opts.get("dte", 2)
    min_dte = opts.get("min_dte", target_dte)  # Default to target_dte if not specified
    spread_width_atr = opts.get("spread_width_atr", 1.5)
    max_debit = opts.get("max_debit", 1.00)
    contracts = opts.get("contracts", 1)
    bars_per_day = opts.get("bars_per_day", 26)
    
    # Calculate actual DTE based on entry day of week (Thu/Fri roll to next week)
    if entry_time is not None:
        actual_dte = calculate_actual_dte(entry_time, target_dte, min_dte)
    else:
        actual_dte = target_dte
    
    # Calculate strikes
    atm_strike = entry_price
    spread_width = atr * spread_width_atr
    otm_strike = atm_strike + spread_width
    
    # Total bars to hold (actual DTE * bars per day)
    total_bars = actual_dte * bars_per_day
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
    
    # Get entry day for EOD detection
    entry_bar = df.iloc[entry_idx]
    entry_day = entry_bar.name.date() if hasattr(entry_bar.name, 'date') else None
    flatten_eod = opts.get("flatten_eod", False)
    
    for i in range(entry_idx + 1, exit_idx + 1):
        bar = df.iloc[i]
        bars_remaining = exit_idx - i
        
        # EOD Flatten Check: Exit at end of entry day if enabled
        if flatten_eod and entry_day is not None:
            bar_day = bar.name.date() if hasattr(bar.name, 'date') else None
            if bar_day is not None and bar_day != entry_day:
                # New day started - exit at close of previous bar (last bar of entry day)
                prev_bar = df.iloc[i - 1]
                _, _, exit_value = estimate_spread_value(
                    float(prev_bar["close"]), atm_strike, otm_strike, bars_remaining + 1, total_bars
                )
                exit_reason = "EOD_FLAT"
                actual_exit_idx = i - 1
                break
        
        # Bars Stop: Exit after N bars if no TP/SL (limit slow losers)
        bars_stop = opts.get("bars_stop", 0)
        bars_held = i - entry_idx
        if bars_stop > 0 and bars_held >= bars_stop:
            _, _, exit_value = estimate_spread_value(
                float(bar["close"]), atm_strike, otm_strike, bars_remaining, total_bars
            )
            exit_reason = "BARS_STOP"
            actual_exit_idx = i
            break
        
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
            
            # Check for max loss (Stop Loss)
            sl_pct = opts.get("sl_pct", 0.9) # Default to 90% if not specified
            stop_loss_threshold = -cost * sl_pct
            if current_pnl <= stop_loss_threshold:
                exit_reason = "STOP_LOSS"
                exit_value = current_value
                actual_exit_idx = i
                
                # Update DKF state
                last_loss_date = df.iloc[i].name.date() if hasattr(df.iloc[i].name, 'date') else None
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
        "dte": actual_dte,
    }


def simulate_spreads_for_symbol(
    df: pd.DataFrame,
    cfg: dict,
    sym: str,
    opts: dict,
    daily_sma: Optional[float] = None,
    daily_vix_map: Optional[dict] = None,
    verbose: bool = False,
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
    last_loss_date = None
    
    # Iterate bars
    for i in range(20, len(df)): # Need warmup for SMA/ATR
    
        cur = df.iloc[i]
        date_obj = cur.name.date() if hasattr(cur.name, "date") else None
        
        # --- DYNAMIC REGIME LOGIC ---
        # Default Params
        current_slope_threshold = float(opts.get("slope_threshold") or -0.12)
        current_use_dkf = opts.get("dead_knife_filter", False)
        
        # Check VIXY if map provided
        if daily_vix_map and date_obj:
            # We use 'date_obj' which is today's date.
            # Ideally we want Yesterday's Close known at Open.
            # The map passed in should be constructed such that key=Today -> value=YesterdayClose
            # (Note: In main, we will shift it)
            vix_val = daily_vix_map.get(date_obj, float('nan'))
            
            if not pd.isna(vix_val):
                if vix_val < 35.0:
                    current_slope_threshold = -0.08 # CALM
                    current_use_dkf = False
                elif vix_val > 55.0:
                    current_slope_threshold = -0.20 # PANIC
                    current_use_dkf = True
                else:
                    current_slope_threshold = -0.12 # NORMAL
                    current_use_dkf = False
                    
        # ----------------------------

        # Dead Knife Filter Logic (Pre-check)
        # Skip if we already took a loss today and RSI is deep oversold (<20)
        # This prevents "Doubling Down" on a falling knife.
        if current_use_dkf and last_loss_date == date_obj:
            rsi_check = float(cur.get("rsi", 0))
            if rsi_check < 20:
                continue # Skip "Catching Falling Knife" re-entry 
                 
        # 1. Check for Exits (if in trade)
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
            

        # Slope Filter (Anti-Falling Knife) or (Panic Buyer)
        # We calculate slopes FIRST so we can use them for filtering OR logging.
        
        slope_3 = 0.0
        slope_10 = 0.0
        
        if "kc_middle" in df.columns:
            # 3-bar slope
            if i >= 4:
                slope_3 = (df["kc_middle"].iloc[i-1] - df["kc_middle"].iloc[i-4]) / 3
            # 10-bar slope
            if i >= 11:
                slope_10 = (df["kc_middle"].iloc[i-1] - df["kc_middle"].iloc[i-11]) / 10

        # Normalized Slope (Percentage)
        # -----------------------------
        # Convert absolute slope to percentage of price for consistent filtering.
        # slope_pct = (slope / open_price) * 100
        
        entry_price_ref = float(cur["open"])
        slope_3_pct = 0.0
        slope_10_pct = 0.0
        
        if entry_price_ref > 0:
            slope_3_pct = (slope_3 / entry_price_ref) * 100
            slope_10_pct = (slope_10 / entry_price_ref) * 100

        slope_threshold = current_slope_threshold
        if slope_threshold is not None:
             # Treat threshold as Percentage (e.g. -0.12 means -0.12%)
             if slope_3_pct > float(slope_threshold):
                 continue

        # Check 2: 10-bar slope (sustained crash, 2.5h) - Normalized
        slope_threshold_10 = opts.get("slope_threshold_10")
        if slope_threshold_10 is not None:
            if slope_10_pct > float(slope_threshold_10):
                continue
        # SMA Trend Filter: Skip if close < daily SMA (matching live bot)
        if daily_sma is not None and opts.get("trend_filter", True):
            close_price = float(prev["close"])
            if close_price < daily_sma:
                continue  # Skip - in bear trend
        
        # ADX Filter: Skip if ADX > threshold (strong trend = mean reversion fails)
        adx_max = opts.get("adx_max", 0)
        if adx_max > 0:
            entry_adx = float(prev.get("adx", 0) or prev.get("ADX", 0))
            if entry_adx > adx_max:
                continue  # Skip - trend too strong for mean reversion
        
        # Capture entry signal details
        entry_rsi = float(prev.get("rsi", 0))
        entry_close = float(prev.get("close", 0))
        kc_lower = float(prev.get("kc_lower", 0))
        entry_adx = float(prev.get("adx", 0) or prev.get("ADX", 0))
        
        # Slopes are already calculated above
        entry_slope = slope_3

        
        # Get entry time for variable DTE calculation
        entry_time = cur.name if hasattr(cur.name, 'weekday') else None
        
        # Simulate the spread trade (pass entry_time for variable DTE)
        result = simulate_spread_trade(df, i, entry_price, atr, opts, entry_time=entry_time)
        
        if result:
            result["symbol"] = sym
            result["entry_time"] = str(cur.name)
            result["atr"] = round(atr, 2)
            
            # Add entry signal details
            result["entry_rsi"] = round(entry_rsi, 1)
            result["entry_close"] = round(entry_close, 2)
            result["kc_lower"] = round(kc_lower, 2)
            result["entry_slope"] = round(slope_3_pct, 4) # Log the PERCENTAGE
            result["entry_slope_10"] = round(slope_10_pct, 4)  # Log the PERCENTAGE
            result["entry_adx"] = round(entry_adx, 1)     # NEW: For ADX analysis
            result["entry_reason"] = f"RSI={entry_rsi:.1f}, Close=${entry_close:.2f} < KC_Lower=${kc_lower:.2f}"
            
            # Calculate exit time
            exit_bar_idx = i + result["bars_held"]
            if exit_bar_idx < len(df):
                result["exit_time"] = str(df.iloc[exit_bar_idx].name)
            else:
                result["exit_time"] = "END_OF_DATA"
            
            # Verbose logging
            if verbose:
                pnl_sign = "+" if result["pnl"] >= 0 else ""
                print(f"  [{sym}] ENTRY: {result['entry_time'][:16]} | RSI={entry_rsi:.1f} | Price=${entry_price:.2f}")
                print(f"         EXIT:  {result['exit_time'][:16]} | Reason={result['reason']} | P&L={pnl_sign}${result['pnl']:.2f} ({pnl_sign}{result['pnl_pct']:.1f}%)")
            
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
    ap.add_argument("--dte", type=int, default=6, help="Target days to expiration")
    ap.add_argument("--min-dte", type=int, default=3, help="Minimum DTE (Thu/Fri roll to next week if < min_dte)")
    ap.add_argument("--spread-width", type=float, default=1.5, help="Spread width in ATR")
    ap.add_argument("--max-debit", type=float, default=1.0, help="Max debit per share")
    ap.add_argument("--sma-period", type=int, default=20, help="Daily SMA period for trend filter")
    ap.add_argument("--no-trend-filter", action="store_true", help="Disable SMA trend filter")
    ap.add_argument("--flatten-eod", action="store_true", help="Exit all positions at end of entry day (no overnight holds)")
    ap.add_argument("--adx-max", type=float, default=0, help="Skip entries when ADX > this value (0=disabled, try 25-30)")
    ap.add_argument("--bars-stop", type=int, default=10, help="Time stop: exit after N bars if no TP/SL (default=10)")
    ap.add_argument("--slope-threshold", type=float, default=None, help="Require 3-bar slope to be steeper than this (e.g. -0.20) to enter (Values > thresh are skipped)")
    ap.add_argument("--slope-threshold-10", type=float, default=None, help="Require 10-bar slope to be steeper than this (e.g. -0.15) to enter (Values > thresh are skipped)")
    ap.add_argument("--sl-pct", type=float, default=0.80, help="Stop loss percentage (0.8 = 80% loss). Default 0.80.")
    ap.add_argument("--dead-knife-filter", action="store_true", help="Enable Dead Knife Filter (skip re-entry if RSI<20 and Loss Today)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true", help="Show detailed entry/exit for each trade")
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
    
    # Resolve slope thresholds from CLI > config > None
    slope_threshold_3 = args.slope_threshold if args.slope_threshold is not None else cfg.get("slope_threshold")
    slope_threshold_10 = getattr(args, 'slope_threshold_10', None) if getattr(args, 'slope_threshold_10', None) is not None else cfg.get("slope_threshold_10")
    
    opts = {
        "dte": args.dte,
        "min_dte": args.min_dte,
        "spread_width_atr": args.spread_width,
        "max_debit": args.max_debit,
        "contracts": 1,
        "bars_per_day": 26,
        "sma_period": args.sma_period,
        "trend_filter": not args.no_trend_filter,
        "flatten_eod": args.flatten_eod,
        "adx_max": args.adx_max,
        "bars_stop": args.bars_stop,
        "slope_threshold": slope_threshold_3,
        "slope_threshold_10": slope_threshold_10,
        "slope_threshold_10": slope_threshold_10,
        "sl_pct": args.sl_pct,
        "dead_knife_filter": args.dead_knife_filter,
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
        
    # --- FETCH VIXY FOR REGIME ---
    print(f"Fetching VIXY daily bars for Regime Detection...")
    vix_map, _ = fetch_latest_bars(
        symbols=["VIXY"], timeframe="1Day", history_days=fetch_days, feed=feed, verbose=False
    )
    daily_vix_map = {}
    vix_df = vix_map.get("VIXY")
    if vix_df is not None and not vix_df.empty:
        # Shift VIXY by 1 day so Today's lookup returns Yesterday's Close
        vix_df["vixy_prev_close"] = vix_df["close"].shift(1)
        # Create map: Date -> Prev Close
        daily_vix_map = vix_df["vixy_prev_close"].dropna().to_dict()
        # Convert keys to date objects if needed (index is usually datetime)
        # .to_dict() on Series with DatetimeIndex returns Timestamp keys
        daily_vix_map = {k.date(): v for k, v in daily_vix_map.items()}
        print(f"  Loaded {len(daily_vix_map)} VIXY data points.")
    else:
        print("  WARNING: Could not fetch VIXY data. Dynamic Regime disabled.")
    # -----------------------------
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
        
        trades = simulate_spreads_for_symbol(df, cfg, sym, opts, daily_sma=daily_sma, daily_vix_map=daily_vix_map, verbose=args.verbose)
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
