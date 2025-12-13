#!/usr/bin/env python3
"""
Live Options Loop: Trade 0DTE/1DTE options using RubberBandBot signals.

Usage:
    python RubberBand/scripts/live_options_loop.py --config RubberBand/config.yaml --tickers RubberBand/tickers.txt --dry-run 1
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
import pandas as pd

from RubberBand.src.data import load_symbols_from_file, fetch_latest_bars, alpaca_market_open
from RubberBand.strategy import attach_verifiers
from RubberBand.src.options_data import (
    select_atm_contract,
    get_option_quote,
    is_options_trading_allowed,
    get_0dte_expiration,
    get_1dte_expiration,
)
from RubberBand.src.options_execution import (
    submit_option_order,
    get_option_positions,
    check_exit_conditions,
    close_option_position,
    flatten_all_option_positions,
    OptionsTradeTracker,
)

ET = ZoneInfo("US/Eastern")

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_OPTIONS_CONFIG = {
    "max_premium": 200.0,        # Max $ per contract
    "max_contracts": 1,          # Contracts per signal
    "tp_pct": 30.0,              # Take profit at +30%
    "sl_pct": -50.0,             # Stop loss at -50%
    "prefer_1dte": False,        # Use 1DTE instead of 0DTE
    "check_interval_sec": 60,    # Seconds between position checks
}


def _load_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RubberBandBot Options Trading Loop")
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--tickers", required=True, help="Path to tickers file")
    p.add_argument("--dry-run", type=int, default=1, help="1=dry run, 0=live")
    p.add_argument("--max-premium", type=float, default=200.0, help="Max premium per contract")
    p.add_argument("--prefer-1dte", action="store_true", help="Use 1DTE instead of 0DTE")
    return p.parse_args()


def _now_et() -> datetime:
    return datetime.now(ET)


def _log(msg: str, data: dict = None):
    """Structured JSON logging."""
    entry = {"ts": _now_et().isoformat(), "msg": msg}
    if data:
        entry.update(data)
    print(json.dumps(entry, default=str), flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Signal Detection
# ──────────────────────────────────────────────────────────────────────────────
def get_long_signals(symbols: List[str], cfg: dict) -> List[Dict[str, Any]]:
    """
    Scan for long signals using existing RubberBandBot strategy.
    
    Returns list of signals with symbol and entry price.
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
                # Safely handle None values (get returns None if key exists but value is None)
                rsi_val = last.get("rsi")
                atr_val = last.get("atr")
                signals.append({
                    "symbol": sym,
                    "entry_price": float(last["close"]),
                    "rsi": float(rsi_val) if rsi_val is not None else 0.0,
                    "atr": float(atr_val) if atr_val is not None else 0.0,
                })
        except Exception as e:
            _log(f"Error processing {sym}", {"error": str(e)})
            continue
    
    return signals


# ──────────────────────────────────────────────────────────────────────────────
# Options Entry
# ──────────────────────────────────────────────────────────────────────────────
def try_option_entry(
    signal: Dict[str, Any],
    opts_cfg: dict,
    tracker: OptionsTradeTracker,
    dry_run: bool = True,
) -> bool:
    """
    Attempt to enter an option position based on a stock signal.
    
    Returns True if entry was made (or would be made in dry-run).
    """
    sym = signal["symbol"]
    max_premium = opts_cfg.get("max_premium", 200.0)
    max_contracts = opts_cfg.get("max_contracts", 1)
    prefer_1dte = opts_cfg.get("prefer_1dte", False)
    
    # Select expiration
    expiration = get_1dte_expiration() if prefer_1dte else get_0dte_expiration()
    
    # Find ATM call
    contract = select_atm_contract(sym, expiration, option_type="call")
    if not contract:
        _log(f"No ATM contract for {sym}", {"expiration": expiration})
        return False
    
    option_symbol = contract.get("symbol", "")
    strike = float(contract.get("strike_price", 0))
    
    # Get quote
    quote = get_option_quote(option_symbol)
    if not quote:
        _log(f"No quote for {option_symbol}")
        return False
    
    ask_price = quote.get("ask", 0)
    premium_cost = ask_price * 100 * max_contracts  # Each contract = 100 shares
    
    # Check max premium
    if premium_cost > max_premium:
        _log(f"Premium too high for {option_symbol}", {
            "ask": ask_price,
            "total_cost": premium_cost,
            "max": max_premium,
        })
        return False
    
    # Entry
    if dry_run:
        _log(f"[DRY-RUN] Would buy {option_symbol}", {
            "qty": max_contracts,
            "ask": ask_price,
            "cost": premium_cost,
            "strike": strike,
            "underlying_signal": signal,
        })
    else:
        result = submit_option_order(
            option_symbol=option_symbol,
            qty=max_contracts,
            side="buy",
            order_type="limit",
            limit_price=ask_price,
        )
        if result.get("error"):
            _log(f"Order failed for {option_symbol}", {"result": result})
            return False
        
        _log(f"Order submitted for {option_symbol}", {
            "order_id": result.get("id"),
            "qty": max_contracts,
            "limit_price": ask_price,
        })
    
    # Track
    tracker.record_entry(
        underlying=sym,
        option_symbol=option_symbol,
        qty=max_contracts,
        premium=ask_price,
        strike=strike,
        expiration=expiration,
    )
    
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Position Management
# ──────────────────────────────────────────────────────────────────────────────
def manage_positions(
    opts_cfg: dict,
    tracker: OptionsTradeTracker,
    dry_run: bool = True,
):
    """Check open positions and exit if TP/SL/Time conditions met."""
    positions = get_option_positions()
    tp_pct = opts_cfg.get("tp_pct", 30.0)
    sl_pct = opts_cfg.get("sl_pct", -50.0)
    
    for pos in positions:
        symbol = pos.get("symbol", "")
        should_exit, reason = check_exit_conditions(pos, tp_pct, sl_pct)
        
        if should_exit:
            current_price = float(pos.get("current_price", 0))
            
            if dry_run:
                _log(f"[DRY-RUN] Would exit {symbol}", {
                    "reason": reason,
                    "current_price": current_price,
                    "pnl_pct": pos.get("unrealized_plpc", 0),
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
    
    # Options config
    opts_cfg = {**DEFAULT_OPTIONS_CONFIG}
    opts_cfg["max_premium"] = args.max_premium
    opts_cfg["prefer_1dte"] = args.prefer_1dte
    
    # Load tickers
    symbols = load_symbols_from_file(args.tickers)
    _log("Loaded symbols", {"count": len(symbols), "sample": symbols[:5]})
    
    # Check market status
    if not alpaca_market_open():
        _log("Market is closed, exiting")
        return 0
    
    # Check options trading window
    if not is_options_trading_allowed():
        _log("Past 3:00 PM ET cutoff, flattening positions")
        if not args.dry_run:
            flatten_all_option_positions()
        return 0
    
    dry_run = bool(args.dry_run)
    tracker = OptionsTradeTracker()
    
    _log("Starting options scan", {
        "dry_run": dry_run,
        "max_premium": opts_cfg["max_premium"],
        "prefer_1dte": opts_cfg["prefer_1dte"],
    })
    
    # 1. Check existing positions for exits
    manage_positions(opts_cfg, tracker, dry_run)
    
    # 2. Get current option positions (to avoid duplicates)
    current_positions = get_option_positions()
    position_underlyings = set()
    for pos in current_positions:
        # Extract underlying from option symbol (first chars before date)
        sym = pos.get("symbol", "")
        if len(sym) > 10:
            # OCC format: SYMBOL + YYMMDD + ... 
            # Find where numbers start
            for i, c in enumerate(sym):
                if c.isdigit():
                    position_underlyings.add(sym[:i])
                    break
    
    # 3. Scan for new signals
    signals = get_long_signals(symbols, cfg)
    _log(f"Found {len(signals)} long signals", {
        "signals": [s["symbol"] for s in signals]
    })
    
    # 4. Enter new positions
    entries = 0
    for signal in signals:
        # Skip if already have position in this underlying
        if signal["symbol"] in position_underlyings:
            _log(f"Already have position in {signal['symbol']}, skipping")
            continue
        
        if try_option_entry(signal, opts_cfg, tracker, dry_run):
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
    log_path = os.path.join(results_dir, f"options_trades_{_now_et().strftime('%Y%m%d')}.json")
    tracker.to_json(log_path)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
