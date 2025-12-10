#!/usr/bin/env python3
"""
Silver Light Live Trading Loop
==============================
Live/Paper trading loop for the TQQQ/SQQQ trend-following strategy.

Usage:
    DRY_RUN=1 python RubberBand/scripts/silverlight_live.py  # Paper trading
    DRY_RUN=0 python RubberBand/scripts/silverlight_live.py  # Live trading (DANGER!)

This script is COMPLETELY ISOLATED from Rubber Band bots.
Executes at 3:50 PM ET daily to capture the nearly-closed daily candle.
"""

import os
import sys
import json
import logging
import yaml
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
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


def get_account_equity() -> float:
    """Fetch current account equity from Alpaca."""
    import requests
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    headers = {
        "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID", ""),
        "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY", ""),
    }
    try:
        r = requests.get(f"{base_url}/v2/account", headers=headers, timeout=10)
        r.raise_for_status()
        return float(r.json().get("equity", 10000))
    except Exception as e:
        logging.error(f"Failed to get account equity: {e}")
        return 10000.0


def submit_market_order(symbol: str, qty: int, side: str) -> Dict[str, Any]:
    """Submit a simple market order."""
    import requests
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    headers = {
        "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID", ""),
        "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY", ""),
        "Content-Type": "application/json",
    }
    payload = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    try:
        r = requests.post(f"{base_url}/v2/orders", headers=headers, json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.error(f"Order failed: {e}")
        return {"error": str(e)}


def run_silverlight_cycle(dry_run: bool = True):
    """
    Main trading cycle for Silver Light.
    
    1. Fetch latest daily bars for TQQQ, SQQQ, SPY
    2. Attach indicators
    3. Generate signal
    4. Calculate target allocation
    5. Rebalance positions
    """
    config = load_config()
    
    logging.info("=" * 60)
    logging.info("Starting Silver Light Trading Cycle")
    logging.info("=" * 60)
    logging.info(f"Dry Run: {dry_run}")
    
    # Check if market is open
    if not alpaca_market_open():
        logging.info("Market is closed. Skipping cycle.")
        log_json({"type": "SKIP", "reason": "MARKET_CLOSED"})
        return
    
    # Get config values
    assets_cfg = config.get("assets", {})
    long_sym = assets_cfg.get("long", "TQQQ")
    short_sym = assets_cfg.get("short", "SQQQ")
    regime_sym = assets_cfg.get("regime_index", "SPY")
    
    symbols = [long_sym, short_sym, regime_sym]
    
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
    
    # Generate signal
    signal, meta = generate_signal(tqqq_df, spy_df, config)
    
    logging.info(f"Signal: {signal.value}")
    logging.info(f"Reason: {meta.get('reason', 'N/A')}")
    log_json({"type": "SIGNAL", "signal": signal.value, **meta})
    
    # Get current positions
    positions = get_positions()
    current_tqqq = 0
    current_sqqq = 0
    
    for p in positions:
        sym = p.get("symbol", "")
        qty = int(float(p.get("qty", 0)))
        if sym == long_sym:
            current_tqqq = qty
        elif sym == short_sym:
            current_sqqq = qty
    
    logging.info(f"Current Positions: {long_sym}={current_tqqq}, {short_sym}={current_sqqq}")
    
    # Get account equity and calculate target allocation
    equity = get_account_equity()
    target_alloc = get_target_allocation(signal, tqqq_df, config, equity)
    
    target_tqqq_dollars = target_alloc.get(long_sym, 0)
    target_sqqq_dollars = target_alloc.get(short_sym, 0)
    
    # Get current prices
    tqqq_price = float(tqqq_df.iloc[-1]["close"])
    sqqq_price = float(bars_map[short_sym].iloc[-1]["close"]) if short_sym in bars_map else 0
    
    target_tqqq_qty = int(target_tqqq_dollars / tqqq_price) if tqqq_price > 0 else 0
    target_sqqq_qty = int(target_sqqq_dollars / sqqq_price) if sqqq_price > 0 else 0
    
    logging.info(f"Target Allocation: {long_sym}=${target_tqqq_dollars:.2f} ({target_tqqq_qty} shares)")
    logging.info(f"Target Allocation: {short_sym}=${target_sqqq_dollars:.2f} ({target_sqqq_qty} shares)")
    
    # Sell TQQQ if needed
    if current_tqqq > target_tqqq_qty:
        sell_qty = current_tqqq - target_tqqq_qty
        if sell_qty > 0:
            logging.info(f"SELL {sell_qty} {long_sym}")
            if not dry_run:
                result = submit_market_order(long_sym, sell_qty, "sell")
                log_json({"type": "ORDER", "side": "sell", "symbol": long_sym, "qty": sell_qty, "result": result})
            else:
                log_json({"type": "DRY_RUN_ORDER", "side": "sell", "symbol": long_sym, "qty": sell_qty})
    
    # Buy TQQQ if needed
    elif current_tqqq < target_tqqq_qty:
        buy_qty = target_tqqq_qty - current_tqqq
        if buy_qty > 0:
            logging.info(f"BUY {buy_qty} {long_sym}")
            if not dry_run:
                result = submit_market_order(long_sym, buy_qty, "buy")
                log_json({"type": "ORDER", "side": "buy", "symbol": long_sym, "qty": buy_qty, "result": result})
            else:
                log_json({"type": "DRY_RUN_ORDER", "side": "buy", "symbol": long_sym, "qty": buy_qty})
    
    # Sell SQQQ if needed
    if current_sqqq > target_sqqq_qty:
        sell_qty = current_sqqq - target_sqqq_qty
        if sell_qty > 0:
            logging.info(f"SELL {sell_qty} {short_sym}")
            if not dry_run:
                result = submit_market_order(short_sym, sell_qty, "sell")
                log_json({"type": "ORDER", "side": "sell", "symbol": short_sym, "qty": sell_qty, "result": result})
            else:
                log_json({"type": "DRY_RUN_ORDER", "side": "sell", "symbol": short_sym, "qty": sell_qty})
    
    # Buy SQQQ if needed
    elif current_sqqq < target_sqqq_qty:
        buy_qty = target_sqqq_qty - current_sqqq
        if buy_qty > 0:
            logging.info(f"BUY {buy_qty} {short_sym}")
            if not dry_run:
                result = submit_market_order(short_sym, buy_qty, "buy")
                log_json({"type": "ORDER", "side": "buy", "symbol": short_sym, "qty": buy_qty, "result": result})
            else:
                log_json({"type": "DRY_RUN_ORDER", "side": "buy", "symbol": short_sym, "qty": buy_qty})
    
    logging.info("=" * 60)
    logging.info("Silver Light Cycle Complete")
    logging.info("=" * 60)


def main():
    dry_run = os.getenv("DRY_RUN", "1") == "1"
    run_silverlight_cycle(dry_run=dry_run)


if __name__ == "__main__":
    main()
