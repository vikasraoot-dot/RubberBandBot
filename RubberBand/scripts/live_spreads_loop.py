#!/usr/bin/env python3
"""
Live Options Spreads Loop: Trade bull call spreads using RubberBandBot signals.

Usage:
    python RubberBand/scripts/live_spreads_loop.py --config RubberBand/config.yaml --tickers RubberBand/tickers.txt --dry-run 1

Key Features:
- Uses 3 DTE by default (90% win rate in backtest)
- Bull call spreads for defined risk
- Holds overnight for multi-day DTE
- Comprehensive trade logging with entry/exit reasons
- EOD summary report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, time as dt_time
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
)
from RubberBand.src.options_execution import (
    submit_spread_order,
    get_option_positions,
    close_option_position,
    flatten_all_option_positions,
    get_position_pnl,
)
from RubberBand.src.options_trade_logger import OptionsTradeLogger

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


def _in_entry_window(now_et: datetime, windows: list) -> bool:
    """Check if current time is within any configured entry window."""
    if not windows:
        return True
    
    current_time = now_et.time()
    
    for w in windows:
        start_str = w.get("start", "09:30")
        end_str = w.get("end", "16:00")
        
        start_parts = start_str.split(":")
        end_parts = end_str.split(":")
        
        start_time = dt_time(int(start_parts[0]), int(start_parts[1]))
        end_time = dt_time(int(end_parts[0]), int(end_parts[1]))
        
        if start_time <= current_time <= end_time:
            return True
    
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Signal Detection
# ──────────────────────────────────────────────────────────────────────────────
def get_long_signals(symbols: List[str], cfg: dict, logger: OptionsTradeLogger) -> List[Dict[str, Any]]:
    """Scan for long signals using existing RubberBandBot strategy."""
    signals = []
    
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
        logger.error(error=str(e), context="fetch_bars")
        return signals
    
    for sym in symbols:
        df = bars_map.get(sym)
        if df is None or df.empty or len(df) < 20:
            continue
        
        try:
            df = attach_verifiers(df, cfg)
            last = df.iloc[-1]
            
            if bool(last.get("long_signal", False)):
                rsi = float(last.get("rsi", 0))
                atr = float(last.get("atr", 0))
                entry_price = float(last["close"])
                
                # Build entry reason from signal components
                entry_reasons = []
                if last.get("rsi_oversold", False):
                    entry_reasons.append(f"RSI_oversold({rsi:.1f})")
                if last.get("ema_ok", False):
                    entry_reasons.append("EMA_aligned")
                if last.get("touch", False):
                    entry_reasons.append("Lower_band_touch")
                
                entry_reason = " + ".join(entry_reasons) if entry_reasons else "RubberBand_long_signal"
                
                logger.spread_signal(
                    underlying=sym,
                    signal_reason=entry_reason,
                    entry_price=entry_price,
                    rsi=rsi,
                    atr=atr
                )
                
                signals.append({
                    "symbol": sym,
                    "entry_price": entry_price,
                    "rsi": rsi,
                    "atr": atr,
                    "entry_reason": entry_reason,
                })
        except Exception as e:
            logger.error(error=str(e), context=f"process_{sym}")
            continue
    
    return signals


# ──────────────────────────────────────────────────────────────────────────────
# Spread Entry
# ──────────────────────────────────────────────────────────────────────────────
def try_spread_entry(
    signal: Dict[str, Any],
    spread_cfg: dict,
    logger: OptionsTradeLogger,
    dry_run: bool = True,
) -> bool:
    """Attempt to enter a bull call spread based on a stock signal."""
    sym = signal["symbol"]
    dte = spread_cfg.get("dte", 3)
    spread_width_pct = spread_cfg.get("spread_width_pct", 2.0)
    max_debit = spread_cfg.get("max_debit", 1.00)
    contracts = spread_cfg.get("contracts", 1)
    entry_reason = signal.get("entry_reason", "RubberBand_signal")
    
    # Select spread contracts
    spread = select_spread_contracts(sym, dte=dte, spread_width_pct=spread_width_pct)
    if not spread:
        logger.spread_skip(underlying=sym, skip_reason="No_contracts_available")
        return False
    
    long_contract = spread["long"]
    short_contract = spread["short"]
    long_symbol = long_contract.get("symbol", "")
    short_symbol = short_contract.get("symbol", "")
    
    # Get quotes
    long_quote = get_option_quote(long_symbol)
    short_quote = get_option_quote(short_symbol)
    
    if not long_quote or not short_quote:
        logger.spread_skip(underlying=sym, skip_reason="Cannot_get_quotes")
        return False
    
    long_ask = long_quote.get("ask", 0)
    short_bid = short_quote.get("bid", 0)
    net_debit = long_ask - short_bid
    
    if net_debit <= 0:
        logger.spread_skip(
            underlying=sym, 
            skip_reason=f"Invalid_pricing(debit={net_debit:.2f})"
        )
        return False
    
    if net_debit > max_debit:
        logger.spread_skip(
            underlying=sym,
            skip_reason=f"Debit_too_high({net_debit:.2f}>{max_debit:.2f})"
        )
        return False
    
    spread_width = spread["spread_width"]
    
    if dry_run:
        # Log entry even in dry-run mode
        logger.spread_entry(
            underlying=sym,
            long_symbol=long_symbol,
            short_symbol=short_symbol,
            atm_strike=spread["atm_strike"],
            otm_strike=spread["otm_strike"],
            spread_width=spread_width,
            net_debit=net_debit,
            contracts=contracts,
            expiration=spread["expiration"],
            entry_reason=f"[DRY-RUN] {entry_reason}",
            signal_rsi=signal.get("rsi", 0),
            signal_atr=signal.get("atr", 0),
        )
    else:
        result = submit_spread_order(
            long_symbol=long_symbol,
            short_symbol=short_symbol,
            qty=contracts,
            max_debit=max_debit,
        )
        if result.get("error"):
            logger.spread_skip(
                underlying=sym,
                skip_reason=f"Order_failed: {result.get('message', 'Unknown')}"
            )
            return False
        
        logger.spread_entry(
            underlying=sym,
            long_symbol=long_symbol,
            short_symbol=short_symbol,
            atm_strike=spread["atm_strike"],
            otm_strike=spread["otm_strike"],
            spread_width=spread_width,
            net_debit=net_debit,
            contracts=contracts,
            expiration=spread["expiration"],
            entry_reason=entry_reason,
            signal_rsi=signal.get("rsi", 0),
            signal_atr=signal.get("atr", 0),
        )
    
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Position Management
# ──────────────────────────────────────────────────────────────────────────────
def check_spread_exit(position: Dict[str, Any], spread_cfg: dict) -> tuple:
    """Check if spread should be exited based on P&L."""
    tp_max_profit_pct = spread_cfg.get("tp_max_profit_pct", 80.0)
    sl_pct = spread_cfg.get("sl_pct", -50.0)
    hold_overnight = spread_cfg.get("hold_overnight", True)
    dte = spread_cfg.get("dte", 3)
    
    pnl, pnl_pct = get_position_pnl(position)
    
    if pnl_pct >= tp_max_profit_pct:
        return True, f"TP_hit({pnl_pct:.1f}%>={tp_max_profit_pct}%)"
    
    if pnl_pct <= sl_pct:
        return True, f"SL_hit({pnl_pct:.1f}%<={sl_pct}%)"
    
    if not hold_overnight or dte == 0:
        now_et = _now_et()
        cutoff = now_et.replace(hour=15, minute=0, second=0, microsecond=0)
        if now_et >= cutoff:
            return True, "EOD_time_exit(3:00PM_cutoff)"
    
    return False, ""


def manage_positions(
    spread_cfg: dict,
    logger: OptionsTradeLogger,
    dry_run: bool = True,
):
    """Check open positions and exit if TP/SL conditions met."""
    positions = get_option_positions()
    
    for pos in positions:
        symbol = pos.get("symbol", "")
        should_exit, exit_reason = check_spread_exit(pos, spread_cfg)
        
        if should_exit:
            current_price = float(pos.get("current_price", 0))
            pnl, pnl_pct = get_position_pnl(pos)
            
            # Extract underlying from OCC symbol
            underlying = ""
            for i, c in enumerate(symbol):
                if c.isdigit():
                    underlying = symbol[:i]
                    break
            
            if dry_run:
                logger.spread_exit(
                    underlying=underlying,
                    long_symbol=symbol,
                    short_symbol="",  # Don't have short symbol in position
                    exit_value=current_price,
                    exit_reason=f"[DRY-RUN] {exit_reason}",
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                )
            else:
                result = close_option_position(symbol)
                logger.spread_exit(
                    underlying=underlying,
                    long_symbol=symbol,
                    short_symbol="",
                    exit_value=current_price,
                    exit_reason=exit_reason,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                )


# ──────────────────────────────────────────────────────────────────────────────
# Single Scan Cycle
# ──────────────────────────────────────────────────────────────────────────────
def run_scan_cycle(
    symbols: List[str],
    cfg: dict,
    spread_cfg: dict,
    logger: OptionsTradeLogger,
    dry_run: bool,
) -> int:
    """Run a single scan cycle. Returns number of new entries."""
    windows = cfg.get("entry_windows", [])
    now_et = _now_et()
    
    # Check entry windows - if outside, only manage positions
    if not _in_entry_window(now_et, windows):
        logger.heartbeat(
            event="outside_entry_window",
            current_time=now_et.strftime("%H:%M"),
        )
        # Still manage positions for exits
        manage_positions(spread_cfg, logger, dry_run)
        return 0
    
    # For 0DTE only: check 3:00 PM cutoff
    if spread_cfg["dte"] == 0 and not is_options_trading_allowed():
        logger.heartbeat(event="0dte_cutoff_reached")
        if not dry_run:
            flatten_all_option_positions()
        return 0
    
    logger.heartbeat(event="scan_start", current_time=now_et.strftime("%H:%M"))
    
    # 1. Check existing positions for exits
    manage_positions(spread_cfg, logger, dry_run)
    
    # 2. Get current option positions (to avoid duplicates)
    current_positions = get_option_positions()
    position_underlyings = set()
    for pos in current_positions:
        sym = pos.get("symbol", "")
        if len(sym) > 10:
            for i, c in enumerate(sym):
                if c.isdigit():
                    position_underlyings.add(sym[:i])
                    break
    
    # 3. Scan for new signals
    signals = get_long_signals(symbols, cfg, logger)
    
    logger.heartbeat(
        event="signals_found",
        count=len(signals),
        symbols=[s["symbol"] for s in signals]
    )
    
    # 4. Enter new spreads
    entries = 0
    for signal in signals:
        if signal["symbol"] in position_underlyings:
            logger.spread_skip(
                underlying=signal["symbol"],
                skip_reason="Already_have_position"
            )
            continue
        
        if try_spread_entry(signal, spread_cfg, logger, dry_run):
            entries += 1
            position_underlyings.add(signal["symbol"])
    
    logger.heartbeat(
        event="scan_complete",
        signals=len(signals),
        new_entries=entries,
        total_positions=len(current_positions) + entries,
    )
    
    return entries


# ──────────────────────────────────────────────────────────────────────────────
# Main Loop
# ──────────────────────────────────────────────────────────────────────────────
SCAN_INTERVAL_SECONDS = 15 * 60  # 15 minutes between scans


def main() -> int:
    import time
    
    args = _parse_args()
    cfg = _load_config(args.config)
    
    # Spread config
    spread_cfg = {**DEFAULT_SPREAD_CONFIG}
    spread_cfg["dte"] = args.dte
    spread_cfg["max_debit"] = args.max_debit
    spread_cfg["contracts"] = args.contracts
    
    # Setup logging
    results_dir = cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f"options_trades_{_now_et().strftime('%Y%m%d')}.jsonl")
    logger = OptionsTradeLogger(log_path)
    
    dry_run = bool(args.dry_run)
    
    logger.heartbeat(
        event="startup",
        dry_run=dry_run,
        dte=spread_cfg["dte"],
        max_debit=spread_cfg["max_debit"],
        contracts=spread_cfg["contracts"],
        tp_pct=spread_cfg["tp_max_profit_pct"],
        sl_pct=spread_cfg["sl_pct"],
        scan_interval_min=SCAN_INTERVAL_SECONDS // 60,
    )
    
    # Load tickers
    try:
        symbols = load_symbols_from_file(args.tickers)
    except FileNotFoundError:
        logger.error(error=f"Ticker file not found: {args.tickers}")
        logger.close()
        return 1
    except Exception as e:
        logger.error(error=str(e), context="load_tickers")
        logger.close()
        return 1
    
    if not symbols:
        logger.error(error="No symbols loaded from ticker file")
        logger.close()
        return 1
    
    logger.heartbeat(event="symbols_loaded", count=len(symbols), sample=symbols[:5])
    
    # ──────────────────────────────────────────────────────────────────────────
    # Main Loop: Run until market close (4:00 PM ET)
    # ──────────────────────────────────────────────────────────────────────────
    market_close_hour = 16  # 4:00 PM ET
    scan_count = 0
    
    while True:
        now_et = _now_et()
        
        # Exit if past market close
        if now_et.hour >= market_close_hour:
            logger.heartbeat(event="market_close_reached", time=now_et.strftime("%H:%M"))
            break
        
        # Check if market is open
        if not alpaca_market_open():
            logger.heartbeat(event="waiting_for_market_open", time=now_et.strftime("%H:%M"))
            time.sleep(60)  # Wait 1 minute and check again
            continue
        
        # Run scan cycle
        scan_count += 1
        logger.heartbeat(event="scan_cycle_start", cycle=scan_count, time=now_et.strftime("%H:%M"))
        
        try:
            entries = run_scan_cycle(symbols, cfg, spread_cfg, logger, dry_run)
            logger.heartbeat(event="scan_cycle_end", cycle=scan_count, new_entries=entries)
        except Exception as e:
            logger.error(error=str(e), context="scan_cycle")
        
        # Check if we should exit (past market close after scan)
        now_et = _now_et()
        if now_et.hour >= market_close_hour:
            logger.heartbeat(event="market_close_reached", time=now_et.strftime("%H:%M"))
            break
        
        # Wait until next scan
        next_scan = now_et.strftime("%H:%M")
        logger.heartbeat(
            event="waiting_for_next_scan",
            next_scan_in_min=SCAN_INTERVAL_SECONDS // 60,
            current_time=next_scan,
        )
        time.sleep(SCAN_INTERVAL_SECONDS)
    
    # ──────────────────────────────────────────────────────────────────────────
    # End of Day
    # ──────────────────────────────────────────────────────────────────────────
    logger.heartbeat(event="eod_processing", scan_count=scan_count)
    
    # Final position management
    manage_positions(spread_cfg, logger, dry_run)
    
    # EOD Summary
    summary = logger.eod_summary()
    logger.close()
    
    print(f"\n{'='*60}", flush=True)
    print(f"EOD SUMMARY: {summary.get('total_trades', 0)} trades, PnL: ${summary.get('total_pnl', 0):.2f}", flush=True)
    print(f"Win Rate: {summary.get('win_rate_pct', 0):.1f}%", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

