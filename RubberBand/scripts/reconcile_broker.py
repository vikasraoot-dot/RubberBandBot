#!/usr/bin/env python3
"""
Reconcile Broker: Validate and sync internal registries with Alpaca broker state.

This script:
1. Fetches all open positions from Alpaca
2. Fetches today's filled orders
3. Compares with internal bot registries
4. Identifies and logs discrepancies
5. Optionally restores missing registry entries from broker data

Usage:
    python RubberBand/scripts/reconcile_broker.py [--fix] [--verbose]
    
    --fix: Auto-fix registries to match broker state
    --verbose: Print detailed reconciliation info
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Set, Optional
from zoneinfo import ZoneInfo

# Ensure repo root is on path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.data import get_positions, get_daily_fills
from RubberBand.src.position_registry import (
    PositionRegistry, 
    BOT_TAGS, 
    parse_client_order_id,
)

ET = ZoneInfo("US/Eastern")

# Registry directory path (relative to CWD, which should be repo root)
REGISTRY_DIR = ".position_registry"


def _now_et() -> datetime:
    return datetime.now(ET)


def _log(msg: str, verbose: bool = True):
    """Structured logging."""
    if verbose:
        ts = _now_et().isoformat()
        print(f"[{ts}] {msg}", flush=True)


def parse_occ_symbol(occ_symbol: str) -> Dict[str, Any]:
    """
    Parse OCC option symbol to extract components.
    Format: UNDERLYING + YYMMDD + C/P + STRIKE (8 digits, implied decimal)
    Example: NVDA251226C00177500 -> {underlying: NVDA, expiration: 2025-12-26, type: C, strike: 177.50}
    """
    if len(occ_symbol) < 15:
        # Likely a stock symbol, not an option
        return {"underlying": occ_symbol, "is_option": False}
    
    # Find where numbers start
    underlying = ""
    for i, c in enumerate(occ_symbol):
        if c.isdigit():
            underlying = occ_symbol[:i]
            rest = occ_symbol[i:]
            break
    else:
        return {"underlying": occ_symbol, "is_option": False}
    
    if len(rest) < 15:
        return {"underlying": occ_symbol, "is_option": False}
    
    try:
        expiration = f"20{rest[:2]}-{rest[2:4]}-{rest[4:6]}"
        option_type = rest[6]
        strike = float(rest[7:]) / 1000
        return {
            "underlying": underlying,
            "expiration": expiration,
            "option_type": option_type,
            "strike": strike,
            "is_option": True,
        }
    except (ValueError, IndexError):
        return {"underlying": occ_symbol, "is_option": False}


def get_orders_for_week(
    base_url: Optional[str] = None,
    key: Optional[str] = None,
    secret: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch filled orders from the past 7 days.
    Uses pagination if needed.
    """
    import requests
    from RubberBand.src.data import _base_url_from_env, _alpaca_headers
    
    base = _base_url_from_env(base_url)
    headers = _alpaca_headers(key, secret)
    
    # Get orders from past 7 days
    after_date = (_now_et() - timedelta(days=7)).strftime("%Y-%m-%dT00:00:00Z")
    
    all_orders = []
    page_token = None
    max_pages = 10  # Safety limit
    
    for _ in range(max_pages):
        params = {
            "status": "filled",
            "after": after_date,
            "limit": 500,
            "direction": "desc",
        }
        if page_token:
            params["page_token"] = page_token
        
        try:
            resp = requests.get(
                f"{base}/v2/orders",
                headers=headers,
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            orders = resp.json()
            all_orders.extend(orders)
            
            # Check for next page
            page_token = resp.headers.get("next-page-token")
            if not page_token or len(orders) < 500:
                break
        except Exception as e:
            print(f"[reconcile] Error fetching orders: {e}")
            break
    
    return all_orders


def extract_bot_tag_from_order(order: Dict[str, Any]) -> Optional[str]:
    """Extract bot tag from order's client_order_id."""
    coid = order.get("client_order_id", "")
    parsed = parse_client_order_id(coid)
    return parsed.get("bot_tag")


def reconcile_positions(
    fix: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Main reconciliation logic.
    
    Returns:
        Dict with reconciliation results
    """
    results = {
        "timestamp": _now_et().isoformat(),
        "broker_positions": [],
        "registry_positions": {},
        "discrepancies": [],
        "fixes_applied": [],
    }
    
    # 1. Fetch broker state
    _log("Fetching positions from Alpaca...", verbose)
    try:
        broker_positions = get_positions()
    except Exception as e:
        _log(f"ERROR: Failed to fetch positions: {e}", verbose)
        results["error"] = str(e)
        return results
    
    results["broker_positions"] = [
        {"symbol": p.get("symbol"), "qty": p.get("qty"), "avg_entry_price": p.get("avg_entry_price")}
        for p in broker_positions
    ]
    _log(f"Found {len(broker_positions)} positions in broker", verbose)
    
    # 2. Fetch recent orders to identify bot ownership
    _log("Fetching recent orders...", verbose)
    orders = get_orders_for_week()
    _log(f"Found {len(orders)} filled orders in past 7 days", verbose)
    
    # Map symbol -> bot_tag based on order client_order_id
    symbol_to_bot: Dict[str, str] = {}
    for order in orders:
        sym = order.get("symbol", "")
        bot_tag = extract_bot_tag_from_order(order)
        if bot_tag and sym:
            symbol_to_bot[sym] = bot_tag
    
    # 3. Load all registries
    _log("Loading internal registries...", verbose)
    registries: Dict[str, PositionRegistry] = {}
    for tag in BOT_TAGS:
        try:
            reg = PositionRegistry(bot_tag=tag, registry_dir=REGISTRY_DIR)
            registries[tag] = reg
            results["registry_positions"][tag] = list(reg.positions.keys())
            _log(f"  {tag}: {len(reg.positions)} positions", verbose)
        except Exception as e:
            _log(f"  {tag}: Failed to load ({e})", verbose)
    
    # 4. Compare broker positions with registries
    _log("Comparing broker vs registries...", verbose)
    
    for pos in broker_positions:
        symbol = pos.get("symbol", "")
        qty = int(pos.get("qty", 0))
        
        # Determine which bot owns this position
        bot_tag = symbol_to_bot.get(symbol)
        
        if not bot_tag:
            # Try to find by checking all registries
            for tag, reg in registries.items():
                if symbol in reg.positions:
                    bot_tag = tag
                    break
        
        if not bot_tag:
            # Unknown position - not tracked by any bot
            results["discrepancies"].append({
                "type": "UNTRACKED",
                "symbol": symbol,
                "qty": qty,
                "message": f"Position {symbol} exists in broker but not tracked by any bot",
            })
            _log(f"  ⚠️ UNTRACKED: {symbol} (qty={qty})", verbose)
            continue
        
        # Check if registry has this position
        reg = registries.get(bot_tag)
        if reg and symbol not in reg.positions:
            results["discrepancies"].append({
                "type": "MISSING_FROM_REGISTRY",
                "bot_tag": bot_tag,
                "symbol": symbol,
                "qty": qty,
                "message": f"{symbol} in broker but missing from {bot_tag} registry",
            })
            _log(f"  ❌ MISSING: {symbol} not in {bot_tag} registry", verbose)
            
            if fix and reg:
                # Restore from order data
                matching_order = next(
                    (o for o in orders if o.get("symbol") == symbol and extract_bot_tag_from_order(o) == bot_tag),
                    None
                )
                if matching_order:
                    coid = matching_order.get("client_order_id", f"{bot_tag}_{symbol}_restored")
                    entry_price = float(matching_order.get("filled_avg_price", 0))
                    reg.positions[symbol] = {
                        "symbol": symbol,
                        "client_order_id": coid,
                        "qty": qty,
                        "entry_price": entry_price,
                        "entry_date": matching_order.get("filled_at", _now_et().isoformat()),
                        "status": "open",
                        "restored_from_broker": True,
                    }
                    reg.save()
                    results["fixes_applied"].append({
                        "action": "RESTORED",
                        "bot_tag": bot_tag,
                        "symbol": symbol,
                    })
                    _log(f"  ✅ FIXED: Restored {symbol} to {bot_tag} registry", verbose)
    
    # 5. Check for orphaned registry entries (in registry but not in broker)
    for tag, reg in registries.items():
        broker_symbols = {p.get("symbol") for p in broker_positions}
        for sym in list(reg.positions.keys()):
            if sym not in broker_symbols:
                results["discrepancies"].append({
                    "type": "ORPHANED_REGISTRY",
                    "bot_tag": tag,
                    "symbol": sym,
                    "message": f"{sym} in {tag} registry but not in broker",
                })
                _log(f"  👻 ORPHANED: {sym} in {tag} registry but not in broker", verbose)
                
                if fix:
                    # Remove orphaned entry
                    del reg.positions[sym]
                    reg.save()
                    results["fixes_applied"].append({
                        "action": "REMOVED_ORPHAN",
                        "bot_tag": tag,
                        "symbol": sym,
                    })
                    _log(f"  ✅ FIXED: Removed orphaned {sym} from {tag} registry", verbose)
    
    # Summary
    _log("=" * 50, verbose)
    _log(f"Reconciliation Complete:", verbose)
    _log(f"  Broker positions: {len(broker_positions)}", verbose)
    _log(f"  Discrepancies found: {len(results['discrepancies'])}", verbose)
    _log(f"  Fixes applied: {len(results['fixes_applied'])}", verbose)
    _log("=" * 50, verbose)
    
    return results


def restore_registry_from_broker(
    bot_tag: str,
    verbose: bool = True,
) -> bool:
    """
    Restore a single bot's registry entirely from broker data.
    Called on startup if registry is empty/missing.
    
    Args:
        bot_tag: Which bot's registry to restore
        
    Returns:
        True if restoration successful
    """
    _log(f"Restoring {bot_tag} registry from broker...", verbose)
    
    # Get broker state
    try:
        positions = get_positions()
        orders = get_orders_for_week()
    except Exception as e:
        _log(f"Failed to fetch broker data: {e}", verbose)
        return False
    
    # Find positions belonging to this bot
    reg = PositionRegistry(bot_tag=bot_tag, registry_dir=REGISTRY_DIR)
    restored_count = 0
    
    for order in orders:
        order_bot_tag = extract_bot_tag_from_order(order)
        if order_bot_tag != bot_tag:
            continue
        
        symbol = order.get("symbol", "")
        side = order.get("side", "")
        
        # Only process buy orders (entries)
        if side != "buy":
            continue
        
        # Check if position still exists in broker
        broker_pos = next((p for p in positions if p.get("symbol") == symbol), None)
        if not broker_pos:
            continue  # Position was closed
        
        # Check if already in registry
        if symbol in reg.positions:
            continue
        
        # Restore entry
        coid = order.get("client_order_id", f"{bot_tag}_{symbol}_restored")
        entry_price = float(order.get("filled_avg_price", 0))
        qty = int(broker_pos.get("qty", 0))
        
        reg.positions[symbol] = {
            "symbol": symbol,
            "client_order_id": coid,
            "qty": qty,
            "entry_price": entry_price,
            "entry_date": order.get("filled_at", _now_et().isoformat()),
            "status": "open",
            "restored_from_broker": True,
        }
        restored_count += 1
        _log(f"  Restored: {symbol} (qty={qty}, price={entry_price})", verbose)
    
    if restored_count > 0:
        reg.save()
        _log(f"Restored {restored_count} positions to {bot_tag} registry", verbose)
    else:
        _log(f"No positions to restore for {bot_tag}", verbose)
    
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile registries with Alpaca broker")
    parser.add_argument("--fix", action="store_true", help="Auto-fix discrepancies")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--restore", type=str, help="Restore specific bot registry (e.g., 15M_OPT)")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()
    
    if args.restore:
        if args.restore not in BOT_TAGS:
            print(f"Invalid bot tag: {args.restore}. Must be one of {BOT_TAGS}")
            return 1
        success = restore_registry_from_broker(args.restore, verbose=True)
        return 0 if success else 1
    
    results = reconcile_positions(fix=args.fix, verbose=args.verbose or True)
    
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output}")
    
    # Return non-zero if discrepancies found and not fixed
    if results.get("discrepancies") and not args.fix:
        return 2
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
