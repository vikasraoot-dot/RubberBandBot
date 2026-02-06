#!/usr/bin/env python3
"""
Live Weekly Options Loop: Trade 45-DTE ITM Calls using Weekly Mean Reversion signals.

Strategy V3:
- Delta: 0.65 (ITM for less theta decay)
- Entry: RSI < 45, Price < 5% below SMA20 (confirmed previous week)
- Exit: Take Profit (+100%), Stop Loss (-50%)
- Expiration: 45-DTE options naturally expire ~6-7 weeks (replaces time stop)
- No leveraged ETFs (SOXL, TQQQ excluded from tickers)

Usage:
    python RubberBand/scripts/live_weekly_options_loop.py --dry-run 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Ensure repo root is on path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from zoneinfo import ZoneInfo
import pandas as pd
import yaml

from RubberBand.src.data import (
    load_symbols_from_file,
    fetch_latest_bars,
    alpaca_market_open,
    get_daily_fills,
    check_kill_switch,
    check_capital_limit,
    KillSwitchTriggered,
    CapitalLimitExceeded,
    order_exists_today,
)
from RubberBand.scripts.backtest_weekly import attach_indicators
from RubberBand.src.options_data import (
    select_itm_contract,
    get_option_quote,
    is_options_trading_allowed,
)
from RubberBand.src.options_execution import (
    submit_option_order,
    get_option_positions,
    close_option_position,
    OptionsTradeTracker,
)
from RubberBand.src.options_trade_logger import OptionsTradeLogger
from RubberBand.src.position_registry import PositionRegistry
from RubberBand.src.regime_manager import RegimeManager

ET = ZoneInfo("US/Eastern")

# Bot tag for position attribution
BOT_TAG = "WK_OPT"


# ──────────────────────────────────────────────────────────────────────────────
# OCC Symbol Parsing
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

# ──────────────────────────────────────────────────────────────────────────────
# Weekly Options Config (V3 Strategy)
# ──────────────────────────────────────────────────────────────────────────────
WEEKLY_OPTIONS_CONFIG = {
    "target_delta": 0.65,         # ITM for less theta decay
    "target_dte": 45,             # 45 days to expiration
    "max_premium_per_trade": 1000.0,  # Max $ per contract (ITM options cost more)
    "max_contracts": 1,           # Contracts per signal
    "max_positions": 5,           # Max concurrent positions
    "tp_pct": 100.0,              # Take profit at +100% (double)
    "sl_pct": -50.0,              # Stop loss at -50%
    "max_weeks_held": 9,          # Time stop after 9 weeks (via expiration)
    "max_spread_pct": 15.0,       # Max bid-ask spread % (avoid illiquid options)
}


def _load_config(path: str) -> dict:
    """Load YAML config file with validation."""
    if not os.path.exists(path):
        print(f"Warning: Config file not found at {path}, using defaults")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        print(f"Error parsing config file {path}: {e}")
        return {}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weekly Options Trading Loop V3")
    p.add_argument("--config", default="RubberBand/config_weekly.yaml", help="Config file")
    p.add_argument("--tickers", default="RubberBand/tickers_weekly.txt", help="Tickers file")
    p.add_argument("--dry-run", type=int, default=1, help="1=dry run, 0=live")
    p.add_argument("--max-premium", type=float, default=1000.0, help="Max premium per contract ($)")
    p.add_argument("--monitor-only", action="store_true", help="Only check positions, don't scan for new signals")
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
# Weekly Signal Detection
# ──────────────────────────────────────────────────────────────────────────────
def get_weekly_signals(symbols: List[str], cfg: dict, regime_cfg: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Scan for weekly mean reversion signals.
    
    Entry: RSI < 45 AND Price < 5% below SMA20 (previous week confirmed)
    """
    signals = []
    
    # Use Regime Config if available, else fallback
    if regime_cfg:
        rsi_thresh = float(regime_cfg.get("weekly_rsi_oversold", 45))
        regime_dev_pct = regime_cfg.get("weekly_mean_dev_pct", -5.0)
        mean_dev_thresh = float(regime_dev_pct) / 100.0
    else:
        rsi_thresh = float(cfg.get("filters", {}).get("rsi_oversold", 45))
        mean_dev_thresh = float(cfg.get("filters", {}).get("mean_deviation_threshold", -5)) / 100.0
    
    for sym in symbols:
        try:
            # Fetch daily bars and resample to weekly (more robust than direct 1Week fetch)
            bars_map, _ = fetch_latest_bars(
                symbols=[sym],
                timeframe="1Day",
                history_days=400,  # Ensure >52 weeks of data
                feed=cfg.get("feed", "iex"),
                verbose=False,
            )
            
            df_daily = bars_map.get(sym)
            if df_daily is None or df_daily.empty:
                continue

            # Resample to Weekly (Ending Friday)
            df = df_daily.resample('W-FRI').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            if len(df) < 30:
                continue
            
            # Attach indicators
            df = attach_indicators(df, cfg)
            
            # Calculate mean deviation
            df["sma_20"] = df["close"].rolling(20).mean()
            df["mean_dev"] = (df["close"] - df["sma_20"]) / df["sma_20"]
            
            # Use previous bar for confirmed signal
            if len(df) < 2:
                continue
                
            prev = df.iloc[-2]
            cur = df.iloc[-1]
            
            prev_rsi = float(prev["rsi"]) if "rsi" in prev.index else 50
            prev_mean_dev = float(prev["mean_dev"]) if "mean_dev" in prev.index else 0
            
            # Check signal conditions
            if prev_rsi < rsi_thresh and prev_mean_dev < mean_dev_thresh:
                signals.append({
                    "symbol": sym,
                    "entry_price": float(cur["open"]),  # Enter at current week open
                    "close": float(cur["close"]),
                    "rsi": prev_rsi,
                    "mean_dev_pct": prev_mean_dev * 100,
                    "atr": float(cur["atr"]) if "atr" in cur.index else cur["close"] * 0.03,
                })
                
        except Exception as e:
            _log(f"Error scanning {sym}", {"error": str(e)})
            continue
    
    return signals


# ──────────────────────────────────────────────────────────────────────────────
# Options Entry (45-DTE ITM)
# ──────────────────────────────────────────────────────────────────────────────
def get_45dte_expiration() -> str:
    """Get expiration date ~45 days out (nearest Friday)."""
    target = datetime.now() + timedelta(days=45)
    # Find nearest Friday
    days_until_friday = (4 - target.weekday()) % 7
    if days_until_friday == 0:
        days_until_friday = 7
    expiry = target + timedelta(days=days_until_friday)
    return expiry.strftime("%Y-%m-%d")


def try_weekly_option_entry(
    signal: Dict[str, Any],
    opts_cfg: dict,
    tracker: OptionsTradeTracker,
    registry: PositionRegistry,
    dry_run: bool = True,
) -> bool:
    """
    Enter ITM call option for weekly signal.
    
    Target: Delta ~0.65 (strike below current price)
    """
    sym = signal["symbol"]
    entry_price = signal["entry_price"]
    max_premium = opts_cfg.get("max_premium_per_trade", 1000.0)
    max_contracts = opts_cfg.get("max_contracts", 1)
    target_delta = opts_cfg.get("target_delta", 0.65)
    
    # Get 45-DTE expiration
    expiration = get_45dte_expiration()
    
    # Select ITM contract (Delta ~0.65)
    contract = select_itm_contract(sym, expiration, option_type="call", target_delta=target_delta)
    if not contract:
        _log(f"No ITM contract for {sym}", {"expiration": expiration, "target_delta": target_delta})
        return False
    
    option_symbol = contract.get("symbol", "")
    strike = float(contract.get("strike_price", 0))
    
    # Get quote
    quote = get_option_quote(option_symbol)
    if not quote:
        _log(f"No quote for {option_symbol}")
        return False

    bid_price = float(quote.get("bid", 0) or 0)
    ask_price = float(quote.get("ask", 0) or 0)

    # CRITICAL: Check bid-ask spread to avoid illiquidity traps
    # Wide spreads (>20%) indicate poor liquidity - will lose money immediately
    if ask_price > 0 and bid_price > 0:
        mid_price = (bid_price + ask_price) / 2
        spread_pct = (ask_price - bid_price) / mid_price * 100 if mid_price > 0 else 100
        max_spread_pct = opts_cfg.get("max_spread_pct", 20.0)

        if spread_pct > max_spread_pct:
            _log(f"Bid-ask spread too wide for {option_symbol}", {
                "bid": bid_price,
                "ask": ask_price,
                "spread_pct": f"{spread_pct:.1f}%",
                "max_allowed": f"{max_spread_pct}%",
            })
            return False
    elif bid_price <= 0:
        _log(f"No bid for {option_symbol} - illiquid, skipping")
        return False

    premium_cost = ask_price * 100 * max_contracts

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
            "expiration": expiration,
            "signal": signal,
        })
    else:
        # Generate client_order_id for position attribution
        client_order_id = registry.generate_order_id(option_symbol)
        
        # Idempotency check - prevent duplicate orders on restart
        if order_exists_today(client_order_id=client_order_id):
            _log(f"[skip] Order already exists: {client_order_id}")
            return False
        
        # Capital limit check (option cost = premium * contracts * 100)
        trade_value = ask_price * max_contracts * 100
        max_capital = float(opts_cfg.get("max_capital", 100000))
        try:
            check_capital_limit(
                proposed_trade_value=trade_value,
                max_capital=max_capital,
                bot_tag=BOT_TAG,
            )
        except CapitalLimitExceeded as e:
            _log(f"[skip] Capital limit exceeded for {option_symbol}: {e}")
            return False
        
        result = submit_option_order(
            option_symbol=option_symbol,
            qty=max_contracts,
            side="buy",
            order_type="limit",
            limit_price=ask_price,
            client_order_id=client_order_id,
            verify_fill=True,
            verify_timeout=5,
        )
        if result.get("error"):
            _log(f"Order failed for {option_symbol}", {"result": result})
            return False
        
        # Only record if filled or pending (not rejected)
        order_status = result.get("status", "unknown")
        if order_status in ("filled", "new", "pending", "accepted", "pending_new"):
            _log(f"Order submitted for {option_symbol}", {
                "order_id": result.get("id"),
                "client_order_id": client_order_id,
                "qty": max_contracts,
                "limit_price": ask_price,
                "status": order_status,
            })
            
            # Record in registry for position attribution
            registry.record_entry(
                symbol=option_symbol,
                client_order_id=client_order_id,
                qty=max_contracts,
                entry_price=ask_price,
                underlying=sym,
                order_id=result.get("id", ""),
            )
            
            # Track successful entry
            tracker.record_entry(
                underlying=sym,
                option_symbol=option_symbol,
                qty=max_contracts,
                premium=ask_price,
                strike=strike,
                expiration=expiration,
            )
        else:
            _log(f"Order not recorded, status: {order_status}", {"result": result})
            return False  # Don't track failed orders
    
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Position Management (TP/SL/Time Stop)
# ──────────────────────────────────────────────────────────────────────────────
def manage_weekly_positions(
    opts_cfg: dict,
    tracker: OptionsTradeTracker,
    dry_run: bool = True,
    registry: Optional["PositionRegistry"] = None,
):
    """Check open positions and exit if TP/SL conditions met.
    
    Note: Time-based exits are tracked separately since the tracker is
    instantiated fresh each run. For 45-DTE options, expiration date
    serves as the natural time limit.
    
    Args:
        opts_cfg: Options configuration
        tracker: Trade tracker
        dry_run: If True, don't actually close
        registry: Position registry to update on successful close
    """
    positions = get_option_positions()
    tp_pct = opts_cfg.get("tp_pct", 100.0)
    sl_pct = opts_cfg.get("sl_pct", -50.0)
    
    _log(f"Managing {len(positions)} weekly option positions")
    
    for pos in positions:
        symbol = pos.get("symbol", "")
        # Handle None values explicitly (Alpaca may return None for empty fields)
        plpc_val = pos.get("unrealized_plpc")
        pnl_pct = float(plpc_val) * 100 if plpc_val is not None else 0.0
        
        # Check exit conditions (TP and SL only for now)
        # Time stop would require persistent tracking across runs
        should_exit = False
        reason = ""
        
        if pnl_pct >= tp_pct:
            should_exit = True
            reason = "TakeProfit"
        elif pnl_pct <= sl_pct:
            should_exit = True
            reason = "StopLoss"
        
        if should_exit:
            cp_val = pos.get("current_price")
            current_price = float(cp_val) if cp_val is not None else 0.0
            
            if dry_run:
                _log(f"[DRY-RUN] Would exit {symbol}", {
                    "reason": reason,
                    "pnl_pct": pnl_pct,
                    "current_price": current_price,
                })
                tracker.record_exit(symbol, current_price, f"[DRY-RUN] {reason}")
            else:
                result = close_option_position(symbol)
                
                # Check if close was successful before recording exit
                if result.get("error"):
                    _log(f"ERROR closing {symbol}", {
                        "reason": reason,
                        "error": result.get("message", "Unknown"),
                    })
                    # Don't record exit - will retry next cycle
                    continue
                
                _log(f"Closed {symbol}", {"reason": reason, "result": result})
                
                # Only record exit on successful close
                tracker.record_exit(symbol, current_price, reason)
                
                # Update registry to remove closed position
                if registry and symbol in registry.positions:
                    pnl = float(pos.get("unrealized_pl", 0))
                    registry.record_exit(
                        symbol=symbol,
                        exit_price=current_price,
                        exit_reason=reason,
                        pnl=pnl,
                    )
                    _log(f"Registry updated: removed {symbol}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    args = _parse_args()
    cfg = _load_config(args.config)
    
    # --- Dynamic Regime Detection ---
    rm = RegimeManager(verbose=True)
    current_regime = rm.update()
    regime_cfg = rm.get_config_overrides()
    _log(f"Regime: {current_regime} (VIXY={rm.last_vixy_price:.2f})")
    # --------------------------------
    
    # Options config
    opts_cfg = {**WEEKLY_OPTIONS_CONFIG}
    opts_cfg["max_premium_per_trade"] = args.max_premium
    
    # Setup logging (JSONL trade logger + simple JSON logger)
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f"weekly_options_trades_{_now_et().strftime('%Y%m%d')}.jsonl")
    logger = OptionsTradeLogger(log_path)
    
    # Load tickers
    ticker_path = args.tickers
    if not os.path.isabs(ticker_path):
        ticker_path = os.path.join(_REPO_ROOT, ticker_path)
    symbols = load_symbols_from_file(ticker_path)
    
    dry_run = bool(args.dry_run)
    
    logger.heartbeat(
        event="startup",
        dry_run=dry_run,
        monitor_only=args.monitor_only,
        max_premium=opts_cfg["max_premium_per_trade"],
        target_delta=opts_cfg["target_delta"],
        tickers_count=len(symbols),
    )
    _log("Weekly Options V3 Started", {"tickers": len(symbols)})
    
    # Check market status
    if not alpaca_market_open():
        logger.heartbeat(event="market_closed")
        _log("Market is closed, exiting")
        logger.close()
        return 0
    
    # Check options trading cutoff (no trading after 3:00 PM ET)
    if not is_options_trading_allowed():
        logger.heartbeat(event="options_cutoff")
        _log("Options trading cutoff reached (after 3:00 PM ET), exiting")
        logger.close()
        return 0
    
    tracker = OptionsTradeTracker()
    
    # Initialize position registry for this bot
    registry = PositionRegistry(bot_tag=BOT_TAG)
    
    # Kill Switch Check - RE-ENABLED Dec 13, 2025
    # Halts trading if daily loss exceeds 25% of invested capital
    if check_kill_switch(bot_tag=BOT_TAG, max_loss_pct=25.0):
        logger.error(error=f"{BOT_TAG} exceeded 25% daily loss - HALTING")
        logger.close()
        raise KillSwitchTriggered(f"{BOT_TAG} exceeded 25% daily loss")
    
    _log("Starting weekly options scan", {
        "dry_run": dry_run,
        "monitor_only": args.monitor_only,
        "max_premium": opts_cfg["max_premium_per_trade"],
        "target_delta": opts_cfg["target_delta"],
        "bot_tag": BOT_TAG,
        "registry_positions": len(registry.positions),
    })
    
    # 1. Check existing positions for exits (always do this)
    manage_weekly_positions(opts_cfg, tracker, dry_run, registry=registry)
    
    # 2. Get ALL option positions, then filter to only THIS BOT'S positions
    all_option_positions = get_option_positions()

    # PHASE 2 FIX (GAP-008): Use reconcile_or_halt() instead of sync_with_alpaca()
    # This prevents silent cleanup of orphaned positions and alerts on mismatch.
    is_clean, registry_orphans, broker_untracked = registry.reconcile_or_halt(
        all_option_positions,
        auto_clean=False,  # Do NOT auto-clean - we want to know about mismatches
    )

    if not is_clean:
        # Registry has positions that broker doesn't - this is a critical mismatch
        logger.error(
            error=f"POSITION_MISMATCH: Registry has {len(registry_orphans)} orphaned positions",
            context="startup_reconciliation",
        )
        _log("CRITICAL: Registry orphans detected", {
            "orphaned_symbols": registry_orphans,
            "action": "auto_clean_and_alert",
        })

        # Re-run with auto_clean=True to fix the state
        registry.reconcile_or_halt(all_option_positions, auto_clean=True)
    # Get only MY positions (those tracked in my registry)
    my_positions = registry.filter_positions(all_option_positions)
    my_underlyings = registry.get_my_underlyings()
    
    logger.heartbeat(
        event="positions_checked",
        all_positions=len(all_option_positions),
        my_positions=len(my_positions),
        my_underlyings=list(my_underlyings),
    )
    _log(f"Positions: {len(my_positions)} mine / {len(all_option_positions)} total", {
        "my_underlyings": list(my_underlyings),
        "bot_tag": BOT_TAG,
    })
    
    # If monitor-only mode, skip signal scanning and new entries
    if args.monitor_only:
        logger.heartbeat(event="monitor_only_complete", positions=len(my_underlyings))
        _log("Monitor-only mode: skipping signal scan")
        _log("Weekly options monitor complete", {
            "total_positions": len(my_underlyings),
        })
        logger.close()
        registry.save()
        return 0
    
    # Check max positions limit
    max_positions_limit = opts_cfg.get("max_positions", 5)
    if len(my_underlyings) >= max_positions_limit:
        # P1 FIX: Verbose logging for max_positions blocking
        logger.heartbeat(
            event="max_positions_reached",
            positions=len(my_underlyings),
            max_positions=max_positions_limit,
            current_underlyings=list(my_underlyings),
            reason=f"Holding {len(my_underlyings)}/{max_positions_limit} positions - blocking new entries",
        )
        _log("Max positions reached, no new entries", {
            "current_positions": len(my_underlyings),
            "max_positions": max_positions_limit,
            "holding_underlyings": list(my_underlyings),
        })
        logger.close()
        registry.save()
        return 0
    
    # 3. Scan for new weekly signals
    signals = get_weekly_signals(symbols, cfg, regime_cfg=regime_cfg)
    _log(f"Found {len(signals)} weekly signals", {
        "signals": [(s["symbol"], f"RSI={s['rsi']:.1f}") for s in signals]
    })
    
    # 4. Enter new positions
    entries = 0
    for signal in signals:
        if signal["symbol"] in my_underlyings:
            _log(f"Already have position in {signal['symbol']}, skipping")
            continue
        
        if try_weekly_option_entry(signal, opts_cfg, tracker, registry, dry_run):
            entries += 1
            my_underlyings.add(signal["symbol"])
        
        # Respect max positions
        if len(my_underlyings) >= opts_cfg.get("max_positions", 5):
            break
    
    # 5. Summary
    logger.heartbeat(
        event="scan_complete",
        signals=len(signals),
        new_entries=entries,
        total_positions=len(my_underlyings),
    )
    _log("Weekly options scan complete", {
        "signals": len(signals),
        "new_entries": entries,
        "total_positions": len(my_underlyings),
    })
    
    # Save trades log (JSON format for tracker, JSONL already saved by logger)
    tracker.to_json(log_path.replace(".jsonl", "_tracker.json"))
    
    registry.save()
    
    # EOD: Export trades to CSV
    logger.eod_summary()
    csv_date = datetime.now(ET).strftime("%Y%m%d")
    logger.export_trades_csv(f"results/{BOT_TAG}_trades_{csv_date}.csv")
    
    logger.close()
    
    # --- Session Summary ---
    _log("\n=== Weekly Options Session Summary ===")
    try:
        fills = get_daily_fills(bot_tag=BOT_TAG)
        if not fills:
            _log("No trades filled today for WK_OPT.")
        else:
            stats = {}
            for f in fills:
                sym = f.get("symbol")
                side = f.get("side")
                qty = float(f.get("filled_qty", 0))
                px = float(f.get("filled_avg_price", 0))
                if sym not in stats:
                    stats[sym] = {"buy_qty": 0, "buy_val": 0.0, "sell_qty": 0, "sell_val": 0.0}
                if side == "buy":
                    stats[sym]["buy_qty"] += qty
                    stats[sym]["buy_val"] += (qty * px)
                elif side == "sell":
                    stats[sym]["sell_qty"] += qty
                    stats[sym]["sell_val"] += (qty * px)

            header = f"{'Symbol':<25} {'Bought':<8} {'Avg Ent':<10} {'Basis':<12} {'Sold':<8} {'Avg Ex':<10} {'Day PnL':<10}"
            print("-" * len(header), flush=True)
            print(header, flush=True)
            print("-" * len(header), flush=True)

            total_pnl = 0.0
            total_vol = 0.0
            for sym in sorted(stats.keys()):
                s = stats[sym]
                b_qty, b_val = s["buy_qty"], s["buy_val"]
                s_qty, s_val = s["sell_qty"], s["sell_val"]
                avg_ent = (b_val / b_qty) if b_qty > 0 else 0.0
                avg_ex = (s_val / s_qty) if s_qty > 0 else 0.0
                matched_qty = min(b_qty, s_qty)
                realized_pnl = (avg_ex - avg_ent) * matched_qty * 100 if matched_qty > 0 else 0.0  # *100 for contracts
                total_pnl += realized_pnl
                total_vol += (b_val + s_val) * 100
                pnl_str = f"{realized_pnl:,.2f}" if matched_qty > 0 else "-"
                print(f"{sym:<25} {int(b_qty):<8} {avg_ent:<10.2f} {b_val*100:<12.2f} {int(s_qty):<8} {avg_ex:<10.2f} {pnl_str:<10}", flush=True)

            print("-" * len(header), flush=True)
            print(f"TOTAL Day PnL: ${total_pnl:,.2f} | TOTAL VOL: ${total_vol:,.2f}", flush=True)
    except Exception as e:
        _log(f"Failed to generate summary", {"error": str(e)})
    _log("=== End Summary ===")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
