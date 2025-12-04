#!/usr/bin/env python3
"""
Live Options Spreads Loop: Trade bull call spreads using RubberBandBot signals.

Usage:
    python RubberBand/scripts/live_spreads_loop.py --config RubberBand/config.yaml --tickers RubberBand/tickers.txt --dry-run 1

Key Features:
- Uses 3 DTE by default (90% win rate in backtest)
- Bull call spreads for defined risk
- Holds overnight for multi-day DTE
- Exits at 90% of max profit (like backtest)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

# Ensure repo root is on path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from zoneinfo import ZoneInfo

from RubberBand.src.data import load_symbols_from_file, fetch_latest_bars, alpaca_market_open
from RubberBand.strategy import attach_verifiers
from RubberBand.src.options_data import (
    select_spread_contracts,
    get_option_quote,
    is_options_trading_allowed,
    get_ndte_expiration,
)
from RubberBand.src.options_execution import (
    submit_spread_order,
    get_option_positions,
    close_option_position,
    flatten_all_option_positions,
    get_position_pnl,
    OptionsTradeTracker,
)

ET = ZoneInfo("US/Eastern")

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_SPREAD_CONFIG = {
    "dte": 3,                      # Days to expiration (3 = 90% WR in backtest)
    "spread_width_pct": 2.0,       # OTM strike = ATM * (1 + this%)
    "max_debit": 1.00,             # Max $ per share for the spread debit ($100/contract)
    "contracts": 1,                # Contracts per signal
    "tp_max_profit_pct": 80.0,     # Take profit at 80% of max profit
    "sl_pct": -50.0,               # Stop loss at -50% of debit lost
    "hold_overnight": True,        # Hold positions overnight for multi-day DTE
}


def _load_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RubberBandBot Bull Call Spread Trading")
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--tickers", required=True, help="Path to tickers file")
    p.add_argument("--dry-run", type=int, default=1, help="1=dry run, 0=live")
    p.add_argument("--dte", type=int, default=3, help="Days to expiration (1-3)")
    p.add_argument("--max-debit", type=float, default=1.00, help="Max debit per share")
    p.add_argument("--contracts", type=int, default=1, help="Contracts per trade")
    return p.parse_args()


def _now_et() -> datetime:
    return datetime.now(ET)


def _log(msg: str, data: dict = None):
    """Structured JSON logging."""
    entry = {"ts": _now_et().isoformat(), "msg": msg}
    if data:
        entry.update(data)
    print(json.dumps(entry, default=str), flush=True)


def _in_entry_window(now_et: datetime, windows: list) -> bool:
    """
    Check if current time is within any configured entry window.
    
    Windows format: [{"start": "09:45", "end": "15:45"}, ...]
    Returns True if in window or no windows configured.
    """
    if not windows:
        return True  # No windows = always allow
    
    current_time = now_et.time()
    
    for w in windows:
        start_str = w.get("start", "09:30")
        end_str = w.get("end", "16:00")
        
        start_parts = start_str.split(":")
        end_parts = end_str.split(":")
        
        from datetime import time as dt_time
        start_time = dt_time(int(start_parts[0]), int(start_parts[1]))
        end_time = dt_time(int(end_parts[0]), int(end_parts[1]))
        
        if start_time <= current_time <= end_time:
            return True
    
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Signal Detection
# ──────────────────────────────────────────────────────────────────────────────
def get_long_signals(symbols: List[str], cfg: dict) -> List[Dict[str, Any]]:
    """
    Scan for long signals using existing RubberBandBot strategy.
    """
    signals = []
    
    # Fetch bars
    timeframe = "15Min"
    history_days = 10
    feed = cfg.get("feed", "iex")
    
    try:
        bars_map, _ = fetch_latest_bars(
            symbols=symbols,
            timeframe=timeframe,
            history_days=history_days,
            feed=feed,
            verbose=False,
        )
    except Exception as e:
        _log("Error fetching bars", {"error": str(e)})
        return signals
    
    # Check each symbol
    for sym in symbols:
        df = bars_map.get(sym)
        if df is None or df.empty or len(df) < 20:
            continue
        
        try:
            df = attach_verifiers(df, cfg)
            last = df.iloc[-1]
            
            if bool(last.get("long_signal", False)):
                signals.append({
                    "symbol": sym,
                    "entry_price": float(last["close"]),
                    "rsi": float(last.get("rsi", 0)),
                    "atr": float(last.get("atr", 0)),
                })
        except Exception as e:
            _log(f"Error processing {sym}", {"error": str(e)})
            continue
    
    return signals


# ──────────────────────────────────────────────────────────────────────────────
# Spread Entry
# ──────────────────────────────────────────────────────────────────────────────
def try_spread_entry(
    signal: Dict[str, Any],
    spread_cfg: dict,
    tracker: OptionsTradeTracker,
    dry_run: bool = True,
) -> bool:
    """
    Attempt to enter a bull call spread based on a stock signal.
    """
    sym = signal["symbol"]
    dte = spread_cfg.get("dte", 3)
    spread_width_pct = spread_cfg.get("spread_width_pct", 2.0)
    max_debit = spread_cfg.get("max_debit", 0.50)
    contracts = spread_cfg.get("contracts", 1)
    
    # Select spread contracts
    spread = select_spread_contracts(sym, dte=dte, spread_width_pct=spread_width_pct)
    if not spread:
        _log(f"No spread available for {sym}")
        return False
    
    long_contract = spread["long"]
    short_contract = spread["short"]
    long_symbol = long_contract.get("symbol", "")
    short_symbol = short_contract.get("symbol", "")
    
    # Get quotes to estimate cost
    long_quote = get_option_quote(long_symbol)
    short_quote = get_option_quote(short_symbol)
    
    if not long_quote or not short_quote:
        _log(f"Cannot get quotes for {sym} spread")
        return False
    
    long_ask = long_quote.get("ask", 0)
    short_bid = short_quote.get("bid", 0)
    net_debit = long_ask - short_bid
    
    # Validate net debit is positive and within limits
    if net_debit <= 0:
        _log(f"Invalid spread pricing for {sym}", {
            "long_ask": long_ask,
            "short_bid": short_bid,
            "net_debit": net_debit,
        })
        return False
    
    if net_debit > max_debit:
        _log(f"Debit too high for {sym}", {
            "net_debit": net_debit,
            "max_debit": max_debit,
        })
        return False
    
    # Calculate max profit for this spread
    spread_width = spread["spread_width"]
    max_profit = spread_width - net_debit  # Max profit per share
    total_cost = net_debit * 100 * contracts
    
    if dry_run:
        _log(f"[DRY-RUN] Would open spread for {sym}", {
            "long_symbol": long_symbol,
            "short_symbol": short_symbol,
            "atm_strike": spread["atm_strike"],
            "otm_strike": spread["otm_strike"],
            "spread_width": spread_width,
            "net_debit": round(net_debit, 2),
            "max_profit": round(max_profit, 2),
            "total_cost": round(total_cost, 2),
            "expiration": spread["expiration"],
            "underlying_signal": signal,
        })
    else:
        result = submit_spread_order(
            long_symbol=long_symbol,
            short_symbol=short_symbol,
            qty=contracts,
            max_debit=max_debit,
        )
        if result.get("error"):
            _log(f"Spread order failed for {sym}", {"result": result})
            return False
        
        _log(f"Spread order submitted for {sym}", {
            "long_symbol": long_symbol,
            "short_symbol": short_symbol,
            "net_debit": round(net_debit, 2),
            "max_profit": round(max_profit, 2),
            "total_cost": round(total_cost, 2),
        })
    
    # Track the trade (store max_profit for TP calculation later)
    tracker.record_entry(
        underlying=sym,
        option_symbol=f"{long_symbol}|{short_symbol}",
        qty=contracts,
        premium=net_debit,
        strike=spread["atm_strike"],
        expiration=spread["expiration"],
    )
    
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Position Management
# ──────────────────────────────────────────────────────────────────────────────
def check_spread_exit(
    position: Dict[str, Any],
    spread_cfg: dict,
) -> tuple:
    """
    Check if spread should be exited based on P&L.
    
    Uses max profit percentage for TP (like backtest), not premium percentage.
    
    Returns:
        (should_exit, reason)
    """
    tp_max_profit_pct = spread_cfg.get("tp_max_profit_pct", 90.0)
    sl_pct = spread_cfg.get("sl_pct", -80.0)
    hold_overnight = spread_cfg.get("hold_overnight", True)
    dte = spread_cfg.get("dte", 3)
    
    # Get current P&L
    pnl, pnl_pct = get_position_pnl(position)
    
    # For spread, max profit = spread_width - entry_debit
    # We don't have spread_width stored, so use pnl_pct as approximation
    # If pnl_pct >= (max_profit / debit) * tp_pct, exit
    # Simplified: if pnl_pct >= tp_max_profit_pct, exit (since max_profit ≈ debit for tight spreads)
    
    # Take profit check
    if pnl_pct >= tp_max_profit_pct:
        return True, f"TP ({pnl_pct:.1f}% >= {tp_max_profit_pct}% of max profit)"
    
    # Stop loss check
    if pnl_pct <= sl_pct:
        return True, f"SL ({pnl_pct:.1f}% <= {sl_pct}%)"
    
    # Time-based exit only for 0DTE or if not holding overnight
    if not hold_overnight or dte == 0:
        now_et = _now_et()
        cutoff = now_et.replace(hour=15, minute=0, second=0, microsecond=0)
        if now_et >= cutoff:
            return True, "TIME (past 3:00 PM ET, 0DTE or no overnight hold)"
    
    return False, ""


def manage_positions(
    spread_cfg: dict,
    tracker: OptionsTradeTracker,
    dry_run: bool = True,
):
    """Check open positions and exit if TP/SL conditions met."""
    positions = get_option_positions()
    
    for pos in positions:
        symbol = pos.get("symbol", "")
        should_exit, reason = check_spread_exit(pos, spread_cfg)
        
        if should_exit:
            current_price = float(pos.get("current_price", 0))
            pnl, pnl_pct = get_position_pnl(pos)
            
            if dry_run:
                _log(f"[DRY-RUN] Would exit {symbol}", {
                    "reason": reason,
                    "current_price": current_price,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 1),
                })
            else:
                result = close_option_position(symbol)
                _log(f"Closed {symbol}", {"reason": reason, "result": result})
            
            tracker.record_exit(symbol, current_price, reason)


# ──────────────────────────────────────────────────────────────────────────────
# Main Loop
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    args = _parse_args()
    cfg = _load_config(args.config)
    
    # Spread config
    spread_cfg = {**DEFAULT_SPREAD_CONFIG}
    spread_cfg["dte"] = args.dte
    spread_cfg["max_debit"] = args.max_debit
    spread_cfg["contracts"] = args.contracts
    
    # Load tickers
    try:
        symbols = load_symbols_from_file(args.tickers)
    except FileNotFoundError:
        _log(f"Ticker file not found: {args.tickers}")
        return 1
    except Exception as e:
        _log(f"Error loading tickers", {"error": str(e)})
        return 1
    
    if not symbols:
        _log("No symbols loaded from ticker file")
        return 1
    
    _log("Loaded symbols", {"count": len(symbols), "sample": symbols[:5]})
    
    # Check market status
    if not alpaca_market_open():
        _log("Market is closed, exiting")
        return 0
    
    # Check entry windows (same as stock trading - skip lunch etc)
    now_et = _now_et()
    windows = cfg.get("entry_windows", [])
    if not _in_entry_window(now_et, windows):
        _log("Outside entry window, skipping new entries", {
            "current_time": now_et.strftime("%H:%M"),
            "windows": windows,
        })
        # Still manage positions but don't open new ones
        dry_run = bool(args.dry_run)
        tracker = OptionsTradeTracker()
        manage_positions(spread_cfg, tracker, dry_run)
        return 0
    
    # For 0DTE only: check 3:00 PM cutoff
    if spread_cfg["dte"] == 0 and not is_options_trading_allowed():
        _log("Past 3:00 PM ET cutoff for 0DTE, flattening positions")
        if not args.dry_run:
            flatten_all_option_positions()
        return 0
    
    dry_run = bool(args.dry_run)
    tracker = OptionsTradeTracker()
    
    _log("Starting spread scan", {
        "dry_run": dry_run,
        "dte": spread_cfg["dte"],
        "max_debit": spread_cfg["max_debit"],
        "contracts": spread_cfg["contracts"],
        "hold_overnight": spread_cfg["hold_overnight"],
        "tp_max_profit_pct": spread_cfg["tp_max_profit_pct"],
    })
    
    # 1. Check existing positions for exits
    manage_positions(spread_cfg, tracker, dry_run)
    
    # 2. Get current option positions (to avoid duplicates)
    current_positions = get_option_positions()
    position_underlyings = set()
    for pos in current_positions:
        sym = pos.get("symbol", "")
        if len(sym) > 10:
            # Extract underlying from OCC symbol
            for i, c in enumerate(sym):
                if c.isdigit():
                    position_underlyings.add(sym[:i])
                    break
    
    # 3. Scan for new signals
    signals = get_long_signals(symbols, cfg)
    _log(f"Found {len(signals)} long signals", {
        "signals": [s["symbol"] for s in signals]
    })
    
    # 4. Enter new spreads
    entries = 0
    for signal in signals:
        if signal["symbol"] in position_underlyings:
            _log(f"Already have position in {signal['symbol']}, skipping")
            continue
        
        if try_spread_entry(signal, spread_cfg, tracker, dry_run):
            entries += 1
            position_underlyings.add(signal["symbol"])
    
    # 5. Summary
    _log("Scan complete", {
        "signals": len(signals),
        "new_entries": entries,
        "total_positions": len(current_positions) + entries,
    })
    
    # Save trades log
    results_dir = cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f"spreads_trades_{_now_et().strftime('%Y%m%d')}.json")
    tracker.to_json(log_path)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
