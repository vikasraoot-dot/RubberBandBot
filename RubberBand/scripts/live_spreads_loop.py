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
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

# Ensure repo root is on path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from zoneinfo import ZoneInfo

from RubberBand.src.data import (
    load_symbols_from_file,
    fetch_latest_bars,
    alpaca_market_open,
    check_kill_switch,
    KillSwitchTriggered,
    order_exists_today,
)
from RubberBand.strategy import attach_verifiers
from RubberBand.src.options_data import (
    select_spread_contracts,
    get_option_quote,
    is_options_trading_allowed,
)
from RubberBand.src.options_execution import (
    submit_spread_order,
    close_spread,
    get_option_positions,
    close_option_position,
    flatten_all_option_positions,
    get_position_pnl,
)
from RubberBand.src.options_trade_logger import OptionsTradeLogger
from RubberBand.src.position_registry import PositionRegistry

ET = ZoneInfo("US/Eastern")

# Bot tag for position attribution
BOT_TAG = "15M_OPT"

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_SPREAD_CONFIG = {
    "dte": 3,                      # Days to expiration (3 = 90% WR in backtest)
    "min_dte": 3,                   # Minimum DTE allowed (skip if DTE fallback goes below)
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
def get_daily_sma(symbol: str, period: int = 120, feed: str = "iex") -> Optional[float]:
    """Fetch daily data and calculate SMA for trend filter."""
    try:
        daily_map, _ = fetch_latest_bars(
            symbols=[symbol],
            timeframe="1Day",
            history_days=int(period * 1.5),  # Extra history for SMA
            feed=feed,
            verbose=False,
        )
        df = daily_map.get(symbol)
        if df is None or df.empty or len(df) < period:
            return None
        
        # Calculate SMA
        sma = df["close"].rolling(window=period).mean().iloc[-1]
        return float(sma) if not pd.isna(sma) else None
    except Exception:
        return None


def get_long_signals(
    symbols: List[str],
    cfg: dict,
    logger: OptionsTradeLogger,
    min_dte: int = 2,
) -> List[Dict[str, Any]]:
    """
    Scan for long signals using existing RubberBandBot strategy.
    
    Enhanced with:
    - SMA trend filter (skip if close < SMA_120)
    - Min DTE parameter for caller to enforce
    
    Args:
        symbols: List of symbols to scan
        cfg: Full config dict
        logger: Options trade logger
        min_dte: Minimum DTE required (for logging only, actual enforcement in try_spread_entry)
    
    Returns:
        List of signal dicts
    """
    signals = []
    
    timeframe = "15Min"
    history_days = 10
    feed = cfg.get("feed", "iex")
    
    # Get trend filter settings
    trend_cfg = cfg.get("trend_filter", {})
    trend_enabled = trend_cfg.get("enabled", True)  # Default enabled for options
    sma_period = int(trend_cfg.get("sma_period", 120))  # Use 120 for faster calculation
    
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
            close = float(last["close"])
            
            # Check trend filter FIRST (before checking signal)
            if trend_enabled:
                daily_sma = get_daily_sma(sym, sma_period, feed)
                if daily_sma is not None and close < daily_sma:
                    # Skip - in bear trend (below SMA)
                    logger.spread_skip(
                        underlying=sym,
                        skip_reason=f"Bear_trend(close={close:.2f}<SMA{sma_period}={daily_sma:.2f})"
                    )
                    continue
            
            if bool(last.get("long_signal", False)):
                rsi = float(last.get("rsi", 0))
                atr = float(last.get("atr", 0))
                entry_price = close
                
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
    registry: PositionRegistry,
    dry_run: bool = True,
) -> bool:
    """Attempt to enter a bull call spread based on a stock signal."""
    sym = signal["symbol"]
    dte = spread_cfg.get("dte", 3)
    min_dte = spread_cfg.get("min_dte", 2)  # Minimum DTE allowed
    spread_width_pct = spread_cfg.get("spread_width_pct", 2.0)
    max_debit = spread_cfg.get("max_debit", 1.00)
    contracts = spread_cfg.get("contracts", 1)
    entry_reason = signal.get("entry_reason", "RubberBand_signal")
    
    # Select spread contracts (may fallback to different DTE, respecting min_dte)
    spread = select_spread_contracts(sym, dte=dte, spread_width_pct=spread_width_pct, min_dte=min_dte)
    if not spread:
        logger.spread_skip(underlying=sym, skip_reason="No_contracts_available")
        return False
    
    # Check actual DTE vs minimum required
    actual_expiration = spread.get("expiration", "")
    if actual_expiration:
        from datetime import datetime as dt_class
        try:
            exp_date = dt_class.strptime(actual_expiration, "%Y-%m-%d").date()
            today = _now_et().date()
            actual_dte = (exp_date - today).days
            if actual_dte < min_dte:
                logger.spread_skip(
                    underlying=sym,
                    skip_reason=f"DTE_too_low({actual_dte}<min_dte={min_dte})"
                )
                return False
        except ValueError:
            pass  # If we can't parse date, continue anyway
    
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
        # Generate client_order_id for position attribution
        client_order_id = registry.generate_order_id(long_symbol)
        
        # Idempotency check - prevent duplicate orders on restart
        if order_exists_today(client_order_id=client_order_id):
            logger.spread_skip(underlying=sym, skip_reason="Order_already_exists")
            return False
        
        result = submit_spread_order(
            long_symbol=long_symbol,
            short_symbol=short_symbol,
            qty=contracts,
            max_debit=max_debit,
            client_order_id=client_order_id,
        )
        if result.get("error"):
            logger.spread_skip(
                underlying=sym,
                skip_reason=f"Order_failed: {result.get('message', 'Unknown')}"
            )
            return False
        
        # Record in registry for position attribution
        registry.record_entry(
            symbol=long_symbol,
            client_order_id=client_order_id,
            qty=contracts,
            entry_price=net_debit,
            underlying=sym,
            order_id=result.get("order_id", ""),
            short_symbol=short_symbol,
        )
        
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
def parse_occ_symbol(symbol: str) -> Dict[str, Any]:
    """
    Parse OCC option symbol into components.
    
    Format: SYMBOL + YYMMDD + C/P + 00000000 (strike * 1000)
    Example: AAPL251205C00282500 = AAPL Dec 5, 2025 $282.50 Call
    """
    result = {"underlying": "", "expiration": "", "type": "", "strike": 0.0, "raw": symbol}
    
    if len(symbol) < 15:
        return result
    
    # Find where underlying ends (first digit)
    for i, c in enumerate(symbol):
        if c.isdigit():
            result["underlying"] = symbol[:i]
            rest = symbol[i:]
            break
    else:
        return result
    
    if len(rest) < 15:
        return result
    
    # Parse date: YYMMDD
    date_str = rest[:6]
    result["expiration"] = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
    
    # Parse type: C or P
    result["type"] = rest[6]
    
    # Parse strike: 8 digits / 1000
    strike_str = rest[7:15]
    try:
        result["strike"] = int(strike_str) / 1000
    except ValueError:
        pass
    
    return result


def calculate_spread_pnl(
    long_pos: Dict[str, Any],
    short_pos: Optional[Dict[str, Any]],
    entry_debit: float,
) -> Tuple[float, float]:
    """
    Calculate P&L for a spread.
    
    Spread value = long_value - short_value
    P&L = current_spread_value - entry_debit
    """
    # Get current values
    long_value = float(long_pos.get("current_price", 0))
    short_value = float(short_pos.get("current_price", 0)) if short_pos else 0
    
    # Current spread value (what we'd receive if we closed now)
    current_spread_value = long_value - short_value
    
    # P&L vs entry
    pnl = (current_spread_value - entry_debit) * 100  # Per contract
    pnl_pct = ((current_spread_value / entry_debit) - 1) * 100 if entry_debit > 0 else 0
    
    return pnl, pnl_pct


def check_spread_exit_conditions(
    pnl_pct: float,
    spread_cfg: dict,
) -> Tuple[bool, str]:
    """Check if spread should be exited based on P&L percentage."""
    tp_max_profit_pct = spread_cfg.get("tp_max_profit_pct", 80.0)
    sl_pct = spread_cfg.get("sl_pct", -50.0)
    hold_overnight = spread_cfg.get("hold_overnight", True)
    dte = spread_cfg.get("dte", 3)
    
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
    active_spreads: Optional[Dict[str, Dict]] = None,
    registry: Optional["PositionRegistry"] = None,
):
    """
    Check open positions and exit spreads if TP/SL conditions met.
    
    Properly pairs long and short legs by underlying + expiration,
    calculates spread P&L, and closes both legs together.
    
    Args:
        spread_cfg: Spread configuration
        logger: Trade logger
        dry_run: If True, don't actually close
        active_spreads: Dict of {underlying: spread_info} from entries this session
        registry: Position registry to update on successful close
    """
    positions = get_option_positions()
    
    if not positions:
        return
    
    # Group positions by underlying + expiration
    # Key: "UNDERLYING_YYYYMMDD" -> {long: pos, short: pos}
    spreads = {}
    
    for pos in positions:
        symbol = pos.get("symbol", "")
        parsed = parse_occ_symbol(symbol)
        underlying = parsed["underlying"]
        expiration = parsed["expiration"]
        strike = parsed["strike"]
        qty = int(pos.get("qty", 0))
        
        key = f"{underlying}_{expiration}"
        
        if key not in spreads:
            spreads[key] = {"underlying": underlying, "expiration": expiration, "long": None, "short": None}
        
        # Long position (qty > 0), Short position (qty < 0)
        if qty > 0:
            # If we already have a long, keep the lower strike (ATM for bull call spread)
            if spreads[key]["long"] is None or strike < parse_occ_symbol(spreads[key]["long"]["symbol"])["strike"]:
                spreads[key]["long"] = pos
        elif qty < 0:
            # If we already have a short, keep the higher strike (OTM for bull call spread)
            if spreads[key]["short"] is None or strike > parse_occ_symbol(spreads[key]["short"]["symbol"])["strike"]:
                spreads[key]["short"] = pos
    
    # Process each spread
    already_closed = set()
    
    for key, spread in spreads.items():
        underlying = spread["underlying"]
        long_pos = spread["long"]
        short_pos = spread["short"]
        
        if underlying in already_closed:
            continue
        
        if not long_pos:
            # Orphaned short leg - close it individually
            # This can happen if the long leg was closed but short wasn't
            if short_pos:
                short_symbol = short_pos.get("symbol", "")
                print(f"[positions] Orphaned short leg detected for {underlying}")
                print(f"[positions]   Short symbol: {short_symbol}")
                print(f"[positions]   Qty: {short_pos.get('qty', 0)}")
                if not dry_run:
                    result = close_option_position(short_symbol)
                    if result.get("error"):
                        print(f"[positions] ERROR closing orphaned short: {result.get('message', 'Unknown')}")
                        # Will retry next cycle
                    else:
                        print(f"[positions] Orphaned short leg closed successfully")
                else:
                    print(f"[positions] [DRY-RUN] Would close orphaned short leg")
            continue
        
        # Parse symbols
        long_symbol = long_pos.get("symbol", "")
        short_symbol = short_pos.get("symbol", "") if short_pos else ""
        
        # Get entry debit from active_spreads if available
        entry_debit = 1.0  # Default
        if active_spreads and underlying in active_spreads:
            entry_debit = active_spreads[underlying].get("net_debit", 1.0)
        else:
            # Estimate from cost basis
            long_cost = float(long_pos.get("cost_basis", 0)) / 100
            short_cost = abs(float(short_pos.get("cost_basis", 0))) / 100 if short_pos else 0
            entry_debit = long_cost - short_cost
            if entry_debit <= 0:
                entry_debit = long_cost  # Fallback to just long cost
        
        # Calculate spread P&L
        pnl, pnl_pct = calculate_spread_pnl(long_pos, short_pos, entry_debit)
        
        # Check exit conditions
        should_exit, exit_reason = check_spread_exit_conditions(pnl_pct, spread_cfg)
        
        if should_exit:
            # Calculate current spread value for logging
            long_value = float(long_pos.get("current_price", 0))
            short_value = float(short_pos.get("current_price", 0)) if short_pos else 0
            exit_value = long_value - short_value
            
            if dry_run:
                logger.spread_exit(
                    underlying=underlying,
                    long_symbol=long_symbol,
                    short_symbol=short_symbol,
                    exit_value=exit_value,
                    exit_reason=f"[DRY-RUN] {exit_reason}",
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                )
                already_closed.add(underlying)
            else:
                # Close both legs together
                if short_symbol:
                    result = close_spread(long_symbol, short_symbol, qty=1)
                else:
                    # Only have long leg, close individually
                    result = close_option_position(long_symbol)
                
                # Check if close was successful before logging and updating registry
                if result.get("error"):
                    print(f"[positions] ERROR closing {underlying} spread: {result.get('message', 'Unknown error')}")
                    logger.error(
                        error=f"Spread close failed: {result.get('message', 'Unknown')}",
                        context=f"close_{underlying}",
                    )
                    # Don't mark as closed - will retry next cycle
                    continue
                
                # Close was successful
                logger.spread_exit(
                    underlying=underlying,
                    long_symbol=long_symbol,
                    short_symbol=short_symbol,
                    exit_value=exit_value,
                    exit_reason=exit_reason,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                )
                
                # Update registry to remove closed position
                if registry and long_symbol in registry.positions:
                    registry.record_exit(
                        symbol=long_symbol,
                        exit_price=exit_value,
                        exit_reason=exit_reason,
                        pnl=pnl,
                    )
                    print(f"[positions] Registry updated: removed {long_symbol}")
                
                already_closed.add(underlying)


# ──────────────────────────────────────────────────────────────────────────────
# Single Scan Cycle
# ──────────────────────────────────────────────────────────────────────────────
def run_scan_cycle(
    symbols: List[str],
    cfg: dict,
    spread_cfg: dict,
    logger: OptionsTradeLogger,
    registry: PositionRegistry,
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
        manage_positions(spread_cfg, logger, dry_run, registry=registry)
        return 0
    
    # For 0DTE only: check 3:00 PM cutoff
    if spread_cfg["dte"] == 0 and not is_options_trading_allowed():
        logger.heartbeat(event="0dte_cutoff_reached")
        if not dry_run:
            flatten_all_option_positions()
        return 0
    
    logger.heartbeat(event="scan_start", current_time=now_et.strftime("%H:%M"))
    
    # 1. Check existing positions for exits
    manage_positions(spread_cfg, logger, dry_run, registry=registry)
    
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
        
        if try_spread_entry(signal, spread_cfg, logger, registry, dry_run):
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
    
    # Initialize position registry for this bot
    registry = PositionRegistry(bot_tag=BOT_TAG)
    registry.sync_with_alpaca(get_option_positions())
    
    logger.heartbeat(
        event="registry_loaded",
        bot_tag=BOT_TAG,
        my_positions=len(registry.positions),
    )
    
    # Kill Switch Check - halt if daily loss exceeds 25%
    if check_kill_switch(bot_tag=BOT_TAG, max_loss_pct=25.0):
        logger.error(error=f"{BOT_TAG} exceeded 25% daily loss - HALTING")
        logger.close()
        raise KillSwitchTriggered(f"{BOT_TAG} exceeded 25% daily loss")
    
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
            entries = run_scan_cycle(symbols, cfg, spread_cfg, logger, registry, dry_run)
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
    manage_positions(spread_cfg, logger, dry_run, registry=registry)
    
    # Save registry
    registry.save()
    
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

