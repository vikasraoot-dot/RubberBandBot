#!/usr/bin/env python3
"""
Silver Light Live Trading Loop (v2 - Production Grade)
=======================================================
Live/Paper trading loop for the TQQQ/SQQQ trend-following strategy.

SAFETY FEATURES:
- Position state persistence (PositionRegistry)
- Order confirmation with retry
- Max daily loss kill switch
- Proper error handling

Usage:
    DRY_RUN=1 python RubberBand/scripts/silverlight_live.py  # Paper trading
    DRY_RUN=0 python RubberBand/scripts/silverlight_live.py  # Live trading

This script is COMPLETELY ISOLATED from Rubber Band bots.
Executes at 3:50 PM ET daily to capture the nearly-closed daily candle.
"""

import os
import sys
import json
import logging
import time
import yaml
import requests
import pandas as pd
from datetime import datetime, date
from typing import Dict, Any, Optional, Tuple
from zoneinfo import ZoneInfo

# --- Ensure repo root on sys.path ---
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import data/order utilities (shared - read-only)
from RubberBand.src.data import (
    fetch_latest_bars,
    get_positions,
    alpaca_market_open,
)

# Import position registry for state persistence
from RubberBand.src.position_registry import PositionRegistry

# Import Silver Light strategy (new module)
from RubberBand.src.silverlight_strategy import (
    attach_indicators,
    generate_signal,
    get_target_allocation,
    Signal
)

# --- Constants ---
ET = ZoneInfo("US/Eastern")
BOT_TAG = "SL_ETF"  # Silver Light ETF bot

# --- Kill Switch Settings ---
MAX_DAILY_LOSS_PCT = 0.05  # 5% max daily loss before stopping
MIN_EQUITY_FLOOR = 1000.0  # Absolute minimum equity to continue trading

# --- Order Settings ---
ORDER_CONFIRM_RETRIES = 10
ORDER_CONFIRM_WAIT_SEC = 0.5

# --- Logging Setup ---
os.makedirs("logs", exist_ok=True)
log_file = f"logs/silverlight_live_{datetime.now(ET).strftime('%Y%m%d')}.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] SILVERLIGHT: %(message)s",
    handlers=[
        logging.FileHandler("silverlight_bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


def load_config(path: str = "RubberBand/config_silverlight.yaml") -> Dict[str, Any]:
    """Load Silver Light configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def log_json(event: Dict[str, Any]):
    """Append JSON event to log file."""
    event["ts"] = datetime.now(ET).isoformat()
    with open(log_file, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")
    print(json.dumps(event, default=str))


def get_alpaca_headers() -> Dict[str, str]:
    """Get Alpaca API headers from environment."""
    return {
        "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID", ""),
        "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY", ""),
    }


def get_alpaca_base_url() -> str:
    """Get Alpaca API base URL."""
    return os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")


def get_account_equity() -> float:
    """Fetch current account equity from Alpaca."""
    try:
        r = requests.get(
            f"{get_alpaca_base_url()}/v2/account",
            headers=get_alpaca_headers(),
            timeout=10
        )
        r.raise_for_status()
        return float(r.json().get("equity", 10000))
    except Exception as e:
        logging.error(f"Failed to get account equity: {e}")
        return 10000.0


def get_order_status(order_id: str) -> Dict[str, Any]:
    """Check status of a submitted order."""
    try:
        r = requests.get(
            f"{get_alpaca_base_url()}/v2/orders/{order_id}",
            headers=get_alpaca_headers(),
            timeout=10
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.error(f"Failed to get order status: {e}")
        return {"error": str(e)}


def submit_market_order_with_confirmation(
    symbol: str,
    qty: int,
    side: str
) -> Tuple[bool, Dict[str, Any]]:
    """
    Submit a market order and wait for fill confirmation.
    
    Returns:
        Tuple of (success: bool, order_result: dict)
    """
    headers = get_alpaca_headers()
    headers["Content-Type"] = "application/json"
    
    payload = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    
    try:
        # Submit order
        r = requests.post(
            f"{get_alpaca_base_url()}/v2/orders",
            headers=headers,
            json=payload,
            timeout=10
        )
        r.raise_for_status()
        order = r.json()
        order_id = order.get("id")
        
        if not order_id:
            return False, {"error": "No order ID returned"}
        
        logging.info(f"Order submitted: {order_id}")
        
        # Wait for confirmation
        for attempt in range(ORDER_CONFIRM_RETRIES):
            time.sleep(ORDER_CONFIRM_WAIT_SEC)
            status = get_order_status(order_id)
            order_status = status.get("status", "")
            
            if order_status == "filled":
                logging.info(f"Order FILLED: {order_id}")
                return True, status
            elif order_status in ["rejected", "cancelled", "expired"]:
                logging.error(f"Order {order_status}: {order_id}")
                return False, status
            
            logging.info(f"Order pending ({attempt+1}/{ORDER_CONFIRM_RETRIES}): {order_status}")
        
        # Timeout - order status unknown
        logging.warning(f"Order confirmation timeout: {order_id}")
        return False, {"error": "Order confirmation timeout", "order_id": order_id}
        
    except requests.exceptions.HTTPError as e:
        # Check for rate limiting
        if e.response.status_code == 429:
            logging.error("API rate limit hit (429). Waiting 60 seconds.")
            time.sleep(60)
            return False, {"error": "Rate limit exceeded"}
        logging.error(f"Order failed (HTTP {e.response.status_code}): {e}")
        return False, {"error": str(e)}
    except Exception as e:
        logging.error(f"Order failed: {e}")
        return False, {"error": str(e)}


def check_kill_switch(config: Dict[str, Any], current_equity: float) -> Tuple[bool, str]:
    """
    Check if trading should be stopped due to kill switch conditions.
    
    Returns:
        Tuple of (should_stop: bool, reason: str)
    """
    # Get starting equity for today (from config or estimate)
    starting_equity = config.get("_starting_equity_today", current_equity)
    
    # Check absolute floor
    if current_equity < MIN_EQUITY_FLOOR:
        return True, f"EQUITY FLOOR: ${current_equity:.2f} < ${MIN_EQUITY_FLOOR:.2f}"
    
    # Check daily loss percentage
    daily_pnl_pct = (current_equity - starting_equity) / starting_equity
    if daily_pnl_pct < -MAX_DAILY_LOSS_PCT:
        return True, f"MAX DAILY LOSS: {daily_pnl_pct*100:.1f}% < -{MAX_DAILY_LOSS_PCT*100:.0f}%"
    
    return False, ""


def run_silverlight_cycle(dry_run: bool = True):
    """
    Main trading cycle for Silver Light.
    
    1. Initialize position registry
    2. Check kill switch
    3. Fetch latest daily bars for TQQQ, SQQQ, SPY
    4. Attach indicators
    5. Generate signal
    6. Calculate target allocation
    7. Rebalance positions with confirmation
    """
    config = load_config()
    
    logging.info("=" * 60)
    logging.info("Starting Silver Light Trading Cycle (v2)")
    logging.info("=" * 60)
    logging.info(f"Dry Run: {dry_run}")
    
    # Initialize position registry for state persistence
    registry = PositionRegistry(bot_tag=BOT_TAG)
    
    # Check if market is open
    if not alpaca_market_open():
        logging.info("Market is closed. Skipping cycle.")
        log_json({"type": "SKIP", "reason": "MARKET_CLOSED"})
        return
    
    # Get account equity and check kill switch
    current_equity = get_account_equity()
    logging.info(f"Account Equity: ${current_equity:,.2f}")
    
    # Store starting equity for today if not set
    if "_starting_equity_today" not in config:
        config["_starting_equity_today"] = current_equity
    
    should_stop, stop_reason = check_kill_switch(config, current_equity)
    if should_stop:
        logging.error(f"KILL SWITCH TRIGGERED: {stop_reason}")
        log_json({"type": "KILL_SWITCH", "reason": stop_reason, "equity": current_equity})
        return
    
    # Get config values
    assets_cfg = config.get("assets", {})
    long_sym = assets_cfg.get("long", "TQQQ")
    short_sym = assets_cfg.get("short", "SQQQ")
    regime_sym = assets_cfg.get("regime_index", "SPY")
    vix_sym = assets_cfg.get("volatility_index", "VIXY")
    
    symbols = [long_sym, short_sym, regime_sym, vix_sym]
    
    # Fetch data (400 days for 200 SMA + buffer)
    logging.info(f"Fetching data for {symbols}...")
    bars_map, meta = fetch_latest_bars(
        symbols=symbols,
        timeframe="1Day",
        history_days=400,
        feed=config.get("execution", {}).get("feed", "iex"),
        rth_only=False,
        verbose=False
    )
    
    # Validate data
    if long_sym not in bars_map or bars_map[long_sym] is None or bars_map[long_sym].empty:
        logging.error(f"No data for {long_sym}")
        log_json({"type": "ERROR", "reason": f"NO_DATA_{long_sym}"})
        return
    
    if regime_sym not in bars_map or bars_map[regime_sym] is None or bars_map[regime_sym].empty:
        logging.error(f"No data for {regime_sym}")
        log_json({"type": "ERROR", "reason": f"NO_DATA_{regime_sym}"})
        return
    
    # Attach indicators
    tqqq_df = attach_indicators(bars_map[long_sym], config)
    spy_df = attach_indicators(bars_map[regime_sym], config)
    
    # Get VIX value (VIXY is an ETF proxy, scale threshold accordingly)
    # VIXY trades ~$10-50, so threshold of 25 is reasonable for VIXY
    vix_value = None
    if vix_sym in bars_map and bars_map[vix_sym] is not None and not bars_map[vix_sym].empty:
        vix_value = float(bars_map[vix_sym].iloc[-1]["close"])
        logging.info(f"VIXY Value: {vix_value:.2f}")
    
    # Generate signal
    signal, signal_meta = generate_signal(tqqq_df, spy_df, config, vix_value=vix_value)
    
    logging.info(f"Signal: {signal.value}")
    logging.info(f"Reason: {signal_meta.get('reason', 'N/A')}")
    log_json({"type": "SIGNAL", "signal": signal.value, **signal_meta})
    
    # Get current positions and sync with registry
    all_positions = get_positions()
    registry.sync_with_alpaca(all_positions)
    
    # Filter to only our positions
    my_positions = registry.filter_positions(all_positions)
    
    current_tqqq = 0
    current_sqqq = 0
    
    for p in my_positions:
        sym = p.get("symbol", "")
        qty = int(float(p.get("qty", 0)))
        if sym == long_sym:
            current_tqqq = qty
        elif sym == short_sym:
            current_sqqq = qty
    
    logging.info(f"Current Positions (registry): {long_sym}={current_tqqq}, {short_sym}={current_sqqq}")
    
    # Calculate target allocation
    target_alloc = get_target_allocation(signal, tqqq_df, config, current_equity)
    
    target_tqqq_dollars = target_alloc.get(long_sym, 0)
    target_sqqq_dollars = target_alloc.get(short_sym, 0)
    
    # Get current prices
    tqqq_price = float(tqqq_df.iloc[-1]["close"])
    sqqq_price = float(bars_map[short_sym].iloc[-1]["close"]) if short_sym in bars_map and bars_map[short_sym] is not None else 0
    
    target_tqqq_qty = int(target_tqqq_dollars / tqqq_price) if tqqq_price > 0 else 0
    target_sqqq_qty = int(target_sqqq_dollars / sqqq_price) if sqqq_price > 0 else 0
    
    logging.info(f"Target Allocation: {long_sym}=${target_tqqq_dollars:.2f} ({target_tqqq_qty} shares)")
    logging.info(f"Target Allocation: {short_sym}=${target_sqqq_dollars:.2f} ({target_sqqq_qty} shares)")
    
    # --- Rebalancing Logic with Confirmation ---
    
    # Sell TQQQ if needed
    if current_tqqq > target_tqqq_qty:
        sell_qty = current_tqqq - target_tqqq_qty
        if sell_qty > 0:
            logging.info(f"SELL {sell_qty} {long_sym}")
            if not dry_run:
                success, result = submit_market_order_with_confirmation(long_sym, sell_qty, "sell")
                log_json({"type": "ORDER", "side": "sell", "symbol": long_sym, "qty": sell_qty, "success": success, "result": result})
                if success:
                    registry.record_exit(long_sym, sell_qty, float(result.get("filled_avg_price", tqqq_price)))
            else:
                log_json({"type": "DRY_RUN_ORDER", "side": "sell", "symbol": long_sym, "qty": sell_qty})
    
    # Buy TQQQ if needed
    elif current_tqqq < target_tqqq_qty:
        buy_qty = target_tqqq_qty - current_tqqq
        if buy_qty > 0:
            logging.info(f"BUY {buy_qty} {long_sym}")
            if not dry_run:
                success, result = submit_market_order_with_confirmation(long_sym, buy_qty, "buy")
                log_json({"type": "ORDER", "side": "buy", "symbol": long_sym, "qty": buy_qty, "success": success, "result": result})
                if success:
                    registry.record_entry(long_sym, buy_qty, float(result.get("filled_avg_price", tqqq_price)))
            else:
                log_json({"type": "DRY_RUN_ORDER", "side": "buy", "symbol": long_sym, "qty": buy_qty})
    
    # Sell SQQQ if needed
    if current_sqqq > target_sqqq_qty:
        sell_qty = current_sqqq - target_sqqq_qty
        if sell_qty > 0:
            logging.info(f"SELL {sell_qty} {short_sym}")
            if not dry_run:
                success, result = submit_market_order_with_confirmation(short_sym, sell_qty, "sell")
                log_json({"type": "ORDER", "side": "sell", "symbol": short_sym, "qty": sell_qty, "success": success, "result": result})
                if success:
                    registry.record_exit(short_sym, sell_qty, float(result.get("filled_avg_price", sqqq_price)))
            else:
                log_json({"type": "DRY_RUN_ORDER", "side": "sell", "symbol": short_sym, "qty": sell_qty})
    
    # Buy SQQQ if needed
    elif current_sqqq < target_sqqq_qty:
        buy_qty = target_sqqq_qty - current_sqqq
        if buy_qty > 0:
            logging.info(f"BUY {buy_qty} {short_sym}")
            if not dry_run:
                success, result = submit_market_order_with_confirmation(short_sym, buy_qty, "buy")
                log_json({"type": "ORDER", "side": "buy", "symbol": short_sym, "qty": buy_qty, "success": success, "result": result})
                if success:
                    registry.record_entry(short_sym, buy_qty, float(result.get("filled_avg_price", sqqq_price)))
            else:
                log_json({"type": "DRY_RUN_ORDER", "side": "buy", "symbol": short_sym, "qty": buy_qty})
    
    # Save registry state
    registry.save()
    
    logging.info("=" * 60)
    logging.info("Silver Light Cycle Complete")
    logging.info("=" * 60)


def main():
    dry_run = os.getenv("DRY_RUN", "1") == "1"
    run_silverlight_cycle(dry_run=dry_run)


if __name__ == "__main__":
    main()
