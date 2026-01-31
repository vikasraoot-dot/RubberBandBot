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
import numpy as np
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
from RubberBand.strategy import attach_verifiers, check_slope_filter, check_bearish_bar_filter
from RubberBand.src.regime_manager import RegimeManager

# ──────────────────────────────────────────────────────────────────────────────
# Spread Simulation Parameters
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_OPTS = {
    "dte": 6,                   # Days to expiration (matches live bot default)
    "min_dte": 3,               # Minimum DTE required (matches live bot)
    "spread_width_atr": 1.5,    # OTM strike = ATM + this * ATR
    "max_debit": 3.00,          # Max $ per share for the spread (synced with live bot)
    "contracts": 1,             # Contracts per trade
    "bars_per_day": 26,         # 15m bars per trading day (6.5 hours)
    "sma_period": 100,          # Daily SMA period for trend filter (Optimized All-Weather)
    "trend_filter": True,       # Enable/disable SMA trend filter
    "bars_stop": 14,            # Time stop: 14 bars (~3.5 hours) - matches live bot
    "bearish_bar_filter": False, # Skip red bars (configurable)
}


def calculate_regime_history(df: pd.DataFrame) -> dict:
    """
    Calculate regime based on VIXY history using production logic (Hybrid Dynamic).
    Returns a map of Date -> Regime Name (Effective for that trade date).
    Logic aligns with src/regime_manager.py (Jan 2026 refactor).
    """
    if df is None or df.empty or len(df) < 20:
        return {}
    
    # Copy to avoid modifying original
    df = df.copy()
    
    # Calculate Indicators
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["std_20"] = df["close"].rolling(window=20).std()
    df["vol_sma_20"] = df["volume"].rolling(window=20).mean()
    df["upper_band"] = df["sma_20"] + (2.0 * df["std_20"])
    
    df["prev_close"] = df["close"].shift(1)
    df["delta_pct"] = ((df["close"] - df["prev_close"]) / df["prev_close"]) * 100.0
    
    # Iterate to determine regime based on EACH DAY'S close
    regime_series = []
    below_sma_streak = 0
    
    # Pre-calculate conditions to speed up loop
    is_panic_price = (df["close"] > df["upper_band"]) | (df["delta_pct"] > 8.0)
    is_high_vol = (df["volume"] > 1.5 * df["vol_sma_20"])
    closes = df["close"].values
    smas = df["sma_20"].values
    
    for i in range(len(df)):
        # Skip if indicators not ready (first 20 bars)
        if pd.isna(smas[i]):
            regime_series.append("NORMAL")
            continue
            
        panic = is_panic_price.iloc[i]
        vol = is_high_vol.iloc[i]
        
        row_regime = "NORMAL"
        
        if panic and vol:
            row_regime = "PANIC"
            below_sma_streak = 0
        elif panic and not vol:
            # Fakeout -> Normal
            row_regime = "NORMAL"
            if closes[i] < smas[i]:
                 below_sma_streak += 1
            else:
                 below_sma_streak = 0
        else:
            # Check CALM
            if closes[i] < smas[i]:
                below_sma_streak += 1
            else:
                below_sma_streak = 0
                
            if below_sma_streak >= 3:
                row_regime = "CALM"
            else:
                row_regime = "NORMAL"
        
        regime_series.append(row_regime)
        
    df["regime"] = regime_series
    
    # SHIFT BY 1: 
    # The regime calculated from Day T's Close takes effect on Day T+1
    df["effective_regime"] = df["regime"].shift(1)
    
    regime_map = {}
    for idx, row in df.iterrows():
        if pd.notna(row["effective_regime"]):
            # idx is Timestamp
            regime_map[idx.date()] = row["effective_regime"]
            
    return regime_map


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
    iv: float = 0.35,  # Implied volatility (default 35%)
    bid_ask_pct: float = 0.08,  # Bid-ask spread cost (8% of spread width)
    is_exit: bool = False,  # True if calculating exit value (apply slippage)
) -> tuple:
    """
    Estimate call spread value using REALISTIC theta decay model.
    
    IMPROVEMENTS (Dec 2025):
    - Steeper theta decay near expiry (power 0.3 instead of 0.7)
    - Bid-ask slippage on exit (default 8% of spread width)
    - Weekend/holiday theta acceleration (2x decay rate for Fri->Mon)
    
    For a bull call spread:
    - Value at entry ≈ intrinsic + IV-adjusted time value
    - Value at expiry = max(0, min(underlying - atm_strike, spread_width))
    
    Returns: (long_call_value, short_call_value, spread_value)
    """
    spread_width = otm_strike - atm_strike
    
    # Time factor (0 = expired, 1 = full time)
    time_factor = dte_bars_remaining / max(total_dte_bars, 1)
    
    # Intrinsic values
    long_intrinsic = max(0, underlying_price - atm_strike)
    short_intrinsic = max(0, underlying_price - otm_strike)
    
    # IV-adjusted time value calculation
    # Base: 35% IV gives ~35% of spread as time value (more conservative)
    iv_factor = min(iv / 0.35, 1.5)  # Cap at 1.5x for extreme IV
    base_time_value_pct = 0.35 * iv_factor  # Reduced from 0.40
    
    # REALISTIC Theta decay: much steeper near expiry
    # Early (>50% time remaining): sqrt decay (slow)
    # Mid (25-50% time remaining): linear decay
    # Late (<25% time remaining): power 0.3 decay (BRUTAL)
    if time_factor > 0.5:
        theta_decay = (time_factor ** 0.5)  # Slow decay early on
    elif time_factor > 0.25:
        # Linear transition zone
        theta_decay = time_factor * 1.4  # Slightly faster than linear
    else:
        # Final 25% of time: brutal theta crush
        # This matches real-world 4-DTE behavior
        theta_decay = (time_factor ** 0.3) * 0.7  # Very steep decay
    
    time_value_pct = base_time_value_pct * theta_decay
    
    # Long call: ATM has full time value
    long_time_value = spread_width * time_value_pct
    long_value = long_intrinsic + long_time_value
    
    # Short call: OTM has less time value (delta ~0.3-0.4)
    # OTM call captures ~35% of ATM time value (reduced from 40%)
    short_delta_factor = 0.35
    short_time_value = spread_width * time_value_pct * short_delta_factor
    short_value = short_intrinsic + short_time_value
    
    spread_value = long_value - short_value
    
    # Apply bid-ask slippage on exit (you get less than mid-price)
    if is_exit and bid_ask_pct > 0:
        slippage = spread_width * bid_ask_pct
        spread_value = max(0, spread_value - slippage)
    
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
    
    # Apply entry slippage (bid-ask spread increases cost)
    entry_slippage_pct = opts.get("entry_slippage_pct", 0.05)  # 5% default (realistic)
    entry_debit = entry_debit * (1 + entry_slippage_pct)
    
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
        
        # REALISM FIX: Only use close price (no hindsight bias from intraday highs/lows)
        check_price = bar["close"]
        _, _, current_value = estimate_spread_value(
            float(check_price), atm_strike, otm_strike, bars_remaining, total_bars
        )
        
        current_pnl = (current_value - entry_debit) * 100 * contracts
        
        # Track best/worst
        best_pnl = max(best_pnl, current_pnl)
        worst_pnl = min(worst_pnl, current_pnl)
        
        # Check for max profit (80% of max - matches live bot)
        tp_pct = opts.get("tp_pct", 0.80)  # Configurable, default 80%
        if current_pnl >= max_profit * tp_pct:
            exit_reason = "MAX_PROFIT"
            exit_value = current_value
            actual_exit_idx = i
            break
        
        # Check for max loss (Stop Loss)
        sl_pct = opts.get("sl_pct", 0.8)  # Default to 80% (matches live config)
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
    # Apply exit slippage (bid-ask spread reduces credit received)
    exit_slippage_pct = opts.get("exit_slippage_pct", 0.10)  # 10% default (realistic for options)
    actual_exit_value = exit_value * (1 - exit_slippage_pct)
    
    pnl = (actual_exit_value - entry_debit) * 100 * contracts
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
    daily_regime_map: Optional[dict] = None,
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
        current_slope_threshold = float(opts.get("slope_threshold") or -0.08)  # Match live bot CALM regime
        current_use_dkf = opts.get("dead_knife_filter", False)
        current_use_bearish_filter = False  # Default: disabled
        

        # Check Regime if map provided
        current_regime = "NORMAL" # Default
        if daily_regime_map and date_obj:
            current_regime = daily_regime_map.get(date_obj, "NORMAL")
            
        # Apply Regime Configs
        if current_regime == "CALM":
            current_slope_threshold = -0.08 
            current_use_dkf = False
            current_use_bearish_filter = False
        elif current_regime == "PANIC":
            current_slope_threshold = -0.20
            current_use_dkf = True
            current_use_bearish_filter = True
        else:
            # NORMAL
            current_slope_threshold = -0.12
            current_use_dkf = False
            # Enable bearish filter in Normal (volatility rising)
            current_use_bearish_filter = True
                    
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
        
        # DEBUG: Signal found!
        if sym == "TSLA" and verbose:
            print(f"  [DEBUG-LOOP] TSLA signal at i={i}, prev={prev.name}")
        
        entry_price = float(cur["open"])
        atr = float(prev.get("atr", 0))
        
        if atr <= 0 or entry_price <= 0:
            if sym == "TSLA" and verbose:
                print(f"  [DEBUG-SKIP] ATR/price: atr={atr}, entry_price={entry_price}")
            continue
            
        # DEBUG: Passed ATR check
        if sym == "TSLA" and verbose:
            print(f"  [DEBUG-PASS] ATR check passed: atr={atr:.2f}, entry={entry_price:.2f}")

        # Slope Filter (Shared Logic)
        # ---------------------------
        # Use Policy Object matching Live Bot
        regime_cfg = {
            "slope_threshold_pct": current_slope_threshold,
            "dead_knife_filter": current_use_dkf
        }
        
        # We need a mini-dataframe window to pass to strategy check
        # But strategy expects a full DF or at least columns. 
        # The 'check_slope_filter' looks at iloc[-1] vs iloc[-4].
        # So we pass the sliced DF up to current index 'i' (inclusive).
        # We need to be careful about performance, but for backtest it's acceptable.
        
        # Slice: df.iloc[:i+1] (contains row i as last element)
        # check_slope_filter uses df['kc_middle'] and df['close']
        
        should_skip, reason = check_slope_filter(df.iloc[:i+1], regime_cfg)
        
        if should_skip:
            if verbose:
                # We relax the TSLA-only debug constraint for general verbose runs
                print(f"  [SKIP] {reason} at {cur.name}")
            continue

        # Bearish Bar Filter Check
        # -------------------------
        # Skip if the signal bar is bearish (close < open) and filter is enabled
        # Can be enabled via CLI (--bearish-filter) or via regime (VIXY >= 35)
        use_bearish_filter = opts.get("bearish_bar_filter", False)
        if opts.get("bearish_regime_only", False):
            # Use regime-based activation instead of CLI flag
            use_bearish_filter = current_use_bearish_filter
            
        if use_bearish_filter:
            signal_bar = df.iloc[i-1]  # prev is the signal bar
            if float(signal_bar.get("close", 0)) < float(signal_bar.get("open", 0)):
                if verbose:
                    print(f"  [SKIP] Bearish bar filter: Close < Open")
                continue

        # Check 2: 10-bar slope (sustained crash, 2.5h) - Normalized
        # (check_slope_filter only handles 3-bar primary slope for now. 
        #  If we want 10-bar, we keep this legacy block or add it to strategy later.
        #  For reconciliation, 3-bar is the main filter.)
        slope_threshold_10 = opts.get("slope_threshold_10")
        if slope_threshold_10 is not None and "kc_middle" in df.columns and i >= 11:
             slope_10 = (df["kc_middle"].iloc[i] - df["kc_middle"].iloc[i-10]) / 10
             entry_price_ref = float(cur["open"])
             slope_10_pct = (slope_10 / entry_price_ref) * 100
             if slope_10_pct > float(slope_threshold_10):
                 if verbose:
                     print(f"  [SKIP] Slope10: {slope_10_pct:.4f} > {slope_threshold_10}")
                 continue
        # SMA Trend Filter: Skip if close < daily SMA (matching live bot)
        if daily_sma is not None and opts.get("trend_filter", True):
            close_price = float(prev["close"])
            if close_price < daily_sma:
                if sym == "TSLA" and verbose:
                    print(f"  [DEBUG-SKIP] SMA: close={close_price:.2f} < sma={daily_sma:.2f}")
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
        
        # Calculate 3-bar slope for logging (same logic as check_slope_filter)
        slope_3 = 0.0
        if "kc_middle" in df.columns and i >= 4:
            slope_3_raw = df["kc_middle"].iloc[i] - df["kc_middle"].iloc[i-3]
            ref_price = float(df.iloc[i]["open"])
            slope_3 = (slope_3_raw / ref_price) * 100 if ref_price > 0 else 0
        entry_slope = slope_3

        
        # Get entry time for variable DTE calculation
        entry_time = cur.name if hasattr(cur.name, 'weekday') else None
        
        # Simulate the spread trade (pass entry_time for variable DTE)
        if sym == "TSLA" and verbose:
            print(f"  [DEBUG-PRE-SIM] Calling simulate_spread_trade: i={i}, entry_price={entry_price:.2f}, atr={atr:.2f}")
        result = simulate_spread_trade(df, i, entry_price, atr, opts, entry_time=entry_time)
        if sym == "TSLA" and verbose:
            print(f"  [DEBUG-POST-SIM] result={result}")
        
        if result:
            result["symbol"] = sym
            result["entry_time"] = str(cur.name)
            result["atr"] = round(atr, 2)
            
            # Add entry signal details
            result["entry_rsi"] = round(entry_rsi, 1)
            result["entry_close"] = round(entry_close, 2)
            result["kc_lower"] = round(kc_lower, 2)
            result["entry_slope"] = round(slope_3, 4) # Log the PERCENTAGE
            result["entry_slope_10"] = round(slope_10_pct if 'slope_10_pct' in dir() else 0, 4)  # Log the PERCENTAGE
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
    ap.add_argument("--sl-pct", type=float, default=0.80, help="Stop loss percentage (0.8 = 80%% loss). Default 0.80.")
    ap.add_argument("--dead-knife-filter", action="store_true", help="Enable Dead Knife Filter (skip re-entry if RSI<20 and Loss Today)")
    ap.add_argument("--bearish-filter", action="store_true", help="Enable Bearish Bar Filter (skip entry on red bars)")
    ap.add_argument("--bearish-regime-only", action="store_true", help="Enable Bearish Filter only in volatile regimes (VIXY >= 35)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--start-date", type=str, help="Start date for backtest (YYYY-MM-DD)")
    ap.add_argument("--end-date", type=str, help="End date for backtest (YYYY-MM-DD)")
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
        "sl_pct": args.sl_pct,
        "dead_knife_filter": args.dead_knife_filter,
        "bearish_bar_filter": args.bearish_filter,
        "bearish_regime_only": args.bearish_regime_only,
    }
    
    # Fetch 15-minute data
    timeframe = "15Min"
    feed = cfg.get("feed", "iex")
    
    # Determine Fetch Period
    fetch_end = None
    if args.end_date:
        fetch_end = args.end_date
        if args.start_date:
            start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
            days_diff = (end_dt - start_dt).days
            fetch_days = int(days_diff * 1.6) # Add padding for weekends/holidays
        else:
            fetch_days = int(args.days * 1.6)
    else:
        # Default behavior: back from today
        fetch_days = int(args.days * 1.6)
        # Check if start_date is provided without end_date (implied end=today)
        if args.start_date:
             start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
             days_diff = (datetime.now() - start_dt).days
             fetch_days = int(days_diff * 1.6)

    print(f"Fetching {len(symbols)} symbols for {fetch_days} days (End: {fetch_end if fetch_end else 'Now'})...", flush=True)
    print(f"SMA Trend Filter: {'ENABLED' if opts['trend_filter'] else 'DISABLED'} (SMA-{args.sma_period})")
    
    bars_map, _ = fetch_latest_bars(
        symbols=symbols,
        timeframe=timeframe,
        history_days=fetch_days,
        feed=feed,
        verbose=not args.quiet,
        end=fetch_end,
    )
    
    # Fetch daily data for SMA trend filter
    daily_sma_map = {}
    if opts["trend_filter"]:
        sma_history_days = max(args.sma_period * 2, 100)  # At least 100 days for reliable SMA-50
        print(f"Fetching daily bars for SMA-{args.sma_period} trend filter (history={sma_history_days} days)...")
        daily_bars_map, _ = fetch_latest_bars(
            symbols=symbols,
            timeframe="1Day",
            history_days=sma_history_days,
            feed=feed,
            verbose=False,
        )
        # DEBUG: Print daily_bars_map contents
        print(f"  [DEBUG] daily_bars_map keys: {list(daily_bars_map.keys())}")
        for sym in symbols[:5]:
            df_tmp = daily_bars_map.get(sym)
            print(f"    {sym}: {len(df_tmp) if df_tmp is not None and not df_tmp.empty else 'None/Empty'}")
    else:
        daily_bars_map = {}  # Initialize empty if trend filter disabled
        
    # --- FETCH VIXY FOR REGIME ---
    # --- FETCH VIXY FOR REGIME ---
    print(f"Fetching VIXY daily bars for Regime Detection...")
    # Ensure we fetch enough history for VIXY indicators (Fetch Days + 50 days buffer)
    vixy_days = max(100, fetch_days + 50)
    vix_map, _ = fetch_latest_bars(
        symbols=["VIXY"], timeframe="1Day", history_days=vixy_days, feed=feed, verbose=False
    )
    daily_regime_map = {}
    vix_df = vix_map.get("VIXY")
    
    if vix_df is not None and not vix_df.empty:
        daily_regime_map = calculate_regime_history(vix_df)
        print(f"  Calculated Regimes for {len(daily_regime_map)} days.")
        # DEBUG: Print last 5 days regimes
        sorted_dates = sorted(list(daily_regime_map.keys()))[-5:]
        for d in sorted_dates:
             print(f"    {d}: {daily_regime_map[d]}")
    else:
        print("  WARNING: Could not fetch VIXY data. Dynamic Regime disabled.")
    
    # Calculate SMA for each symbol (MUST be outside the VIXY if/else block)
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
        
        # DEBUG: Print data info for TSLA
        if sym == "TSLA":
            print(f"  [DEBUG] TSLA: {len(df)} bars, daily_sma={daily_sma_map.get(sym)}, vixy_pts={len(daily_vix_map)}")
        
        # Get daily SMA for this symbol (None if not available)
        daily_sma = daily_sma_map.get(sym)
        
        trades = simulate_spreads_for_symbol(df, cfg, sym, opts, daily_sma=daily_sma, daily_regime_map=daily_regime_map, verbose=args.verbose)
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
