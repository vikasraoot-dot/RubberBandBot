#!/usr/bin/env python3
"""
Morning Safety Check: Verify all positions have protective orders.

This script runs before market open to ensure:
1. All STOCK positions have TP/SL orders (or places them if missing)
2. All OPTION positions are logged (bot-managed, no auto-orders)

Usage:
    python safety_check.py           # Live mode
    python safety_check.py --dry-run # Report only, no orders
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any
import requests

# ==============================================================================
# Configuration
# ==============================================================================

DRY_RUN = False

def _alpaca_creds() -> tuple:
    # Use 'or' to handle both missing AND empty env vars
    base = os.environ.get("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets"
    key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_KEY_ID") or ""
    secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY") or ""
    return base, key, secret

def _headers(key: str, secret: str) -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Content-Type": "application/json",
    }

# ==============================================================================
# API Functions
# ==============================================================================

def get_positions() -> List[Dict]:
    """Fetch all open positions."""
    base, key, secret = _alpaca_creds()
    resp = requests.get(f"{base}/v2/positions", headers=_headers(key, secret), timeout=30)
    if resp.status_code != 200:
        print(f"[ERROR] Failed to fetch positions: {resp.text}")
        return []
    return resp.json()


def get_open_orders() -> List[Dict]:
    """Fetch all open orders."""
    base, key, secret = _alpaca_creds()
    resp = requests.get(f"{base}/v2/orders?status=open", headers=_headers(key, secret), timeout=30)
    if resp.status_code != 200:
        print(f"[ERROR] Failed to fetch orders: {resp.text}")
        return []
    return resp.json()


def place_oco_order(symbol: str, qty: int, tp_price: float, sl_price: float) -> Dict:
    """Place a GTC OCO order (bracket exit)."""
    if DRY_RUN:
        print(f"[DRY RUN] Would place OCO for {symbol}: qty={qty}, TP=${tp_price:.2f}, SL=${sl_price:.2f}")
        return {"status": "dry_run"}
    
    base, key, secret = _alpaca_creds()
    
    order_data = {
        "symbol": symbol,
        "qty": str(qty),
        "side": "sell",
        "type": "limit",
        "time_in_force": "gtc",
        "order_class": "oco",
        "take_profit": {
            "limit_price": str(tp_price)
        },
        "stop_loss": {
            "stop_price": str(sl_price)
        }
    }
    
    resp = requests.post(f"{base}/v2/orders", headers=_headers(key, secret), json=order_data, timeout=30)
    
    if resp.status_code in (200, 201):
        print(f"[OCO] Placed for {symbol}: TP=${tp_price:.2f}, SL=${sl_price:.2f}")
        return resp.json()
    else:
        print(f"[ERROR] Failed to place OCO for {symbol}: {resp.text}")
        return {"error": resp.text}


def is_option_symbol(symbol: str) -> bool:
    """Check if symbol is an OCC option."""
    return len(symbol) > 10 and (symbol[-9] in ('C', 'P'))


# ==============================================================================
# Main Logic
# ==============================================================================

def run_safety_check():
    """Main safety check routine."""
    global DRY_RUN
    
    parser = argparse.ArgumentParser(description="Morning Safety Check")
    parser.add_argument("--dry-run", action="store_true", help="Report only, no orders")
    args = parser.parse_args()
    DRY_RUN = args.dry_run
    
    print("\n" + "=" * 60)
    print("  MORNING SAFETY CHECK")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {'DRY RUN' if DRY_RUN else 'LIVE'}")
    print("=" * 60 + "\n")
    
    # Fetch data
    positions = get_positions()
    orders = get_open_orders()
    
    if not positions:
        print("[INFO] No open positions found.")
        return
    
    print(f"[INFO] Found {len(positions)} open positions")
    print(f"[INFO] Found {len(orders)} open orders")
    
    # Build map of symbols with sell orders
    symbols_with_coverage = set()
    for order in orders:
        if order.get("side") == "sell":
            symbols_with_coverage.add(order.get("symbol"))
    
    # Check each position
    report = {
        "timestamp": datetime.now().isoformat(),
        "mode": "dry_run" if DRY_RUN else "live",
        "positions_checked": 0,
        "covered": [],
        "naked": [],
        "options": [],
        "orders_placed": [],
    }
    
    for pos in positions:
        symbol = pos.get("symbol", "")
        qty = int(float(pos.get("qty", 0)))
        entry = float(pos.get("avg_entry_price", 0))
        current = float(pos.get("current_price", 0))
        
        if qty == 0:
            continue
        
        report["positions_checked"] += 1
        
        if is_option_symbol(symbol):
            # Options are bot-managed
            print(f"[OPTION] {symbol}: qty={qty} - Bot Managed")
            report["options"].append({"symbol": symbol, "qty": qty})
            continue
        
        if symbol in symbols_with_coverage:
            # Position has sell order
            print(f"[COVERED] {symbol}: qty={qty} - Has exit orders")
            report["covered"].append({"symbol": symbol, "qty": qty})
            continue
        
        # NAKED position - needs protection
        print(f"[NAKED] {symbol}: qty={qty} @ ${entry:.2f} - NEEDS PROTECTION")
        report["naked"].append({"symbol": symbol, "qty": qty, "entry": entry})
        
        # Calculate TP/SL (simple 1.5R)
        atr_estimate = current * 0.02  # ~2% ATR estimate
        tp_price = round(entry + (1.5 * atr_estimate), 2)
        sl_price = round(entry - atr_estimate, 2)
        
        result = place_oco_order(symbol, qty, tp_price, sl_price)
        if result.get("status") != "dry_run" and not result.get("error"):
            report["orders_placed"].append({"symbol": symbol, "tp": tp_price, "sl": sl_price})
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Positions Checked: {report['positions_checked']}")
    print(f"  Covered (has TP/SL): {len(report['covered'])}")
    print(f"  Naked (need protection): {len(report['naked'])}")
    print(f"  Options (bot-managed): {len(report['options'])}")
    print(f"  Orders Placed: {len(report['orders_placed'])}")
    print("=" * 60 + "\n")
    
    # Save report
    with open("safety_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[INFO] Report saved to safety_report.json")


if __name__ == "__main__":
    run_safety_check()
