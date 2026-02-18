#!/usr/bin/env python3
"""
Persist Daily Results: Save consolidated daily trading results for historical analysis.

This script:
1. Reconciles registries with broker (source of truth)
2. Fetches PnL from Alpaca account
3. Fetches all trades for each bot
4. Saves consolidated report to results/daily/{YYYY-MM-DD}.json

Usage:
    python RubberBand/scripts/persist_daily_results.py [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Ensure repo root is on path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.data import get_positions, get_daily_fills
from RubberBand.src.position_registry import (
    BOT_TAGS,
    parse_client_order_id,
    ensure_all_registries_exist,
)
from RubberBand.scripts.reconcile_broker import (
    reconcile_positions,
    get_orders_for_week,
    extract_bot_tag_from_order,
)

ET = ZoneInfo("US/Eastern")

# Output directory for daily results
DAILY_RESULTS_DIR = "results/daily"


def _now_et() -> datetime:
    return datetime.now(ET)


def _log(msg: str):
    """Simple logging."""
    ts = _now_et().isoformat()
    print(f"[{ts}] {msg}", flush=True)


def _extract_order_symbol_side(order: Dict[str, Any]) -> tuple[str, str, bool]:
    """
    Extract symbol and side from an order, handling multi-leg (mleg) orders.

    For standard (stock/single-option) orders, returns the top-level fields.
    For mleg orders (spreads), the top-level ``symbol`` and ``side`` are
    empty/null — the real data lives in ``order["legs"]``.

    Strategy for mleg:
    - Use the **buy** leg's symbol as primary (the ATM long contract).
    - If no buy leg, fall back to the first filled leg.
    - Multi-leg spreads are mapped to ``side="buy"`` for P&L accounting
      (spreads are net-debit buy-to-open), with ``is_spread=True`` as
      separate metadata.

    Note:
        For mleg orders, the top-level ``filled_qty`` and ``filled_avg_price``
        may represent the net debit/credit rather than individual leg values.
        Callers needing per-leg detail should inspect ``order["legs"]``.

    Returns:
        (symbol, side, is_spread) tuple.  ``is_spread`` is True when the
        order has multiple filled legs.  Both strings are empty when nothing
        is extractable.
    """
    symbol = (order.get("symbol") or "").strip()
    side = (order.get("side") or "").strip()

    if symbol and side:
        return symbol, side, False

    # Multi-leg order: extract from legs array
    legs = order.get("legs") or []
    if not legs:
        return "", "", False

    filled = [lg for lg in legs if lg.get("status") == "filled"]
    if not filled:
        filled = legs  # fall back to all legs if none marked filled

    # Prefer the buy leg (long contract)
    buy_legs = [lg for lg in filled if lg.get("side") == "buy"]
    primary = buy_legs[0] if buy_legs else filled[0]

    symbol = (primary.get("symbol") or "").strip()
    is_spread = len(filled) > 1
    # Map spreads to "buy" for P&L accounting; downstream uses is_spread
    # to distinguish from single-leg buys.
    side = "buy" if is_spread else (primary.get("side") or "")
    return symbol, side, is_spread


def get_account_info(
    base_url: Optional[str] = None,
    key: Optional[str] = None,
    secret: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch account info from Alpaca."""
    import requests
    from RubberBand.src.data import _base_url_from_env, _alpaca_headers
    
    base = _base_url_from_env(base_url)
    headers = _alpaca_headers(key, secret)
    
    try:
        resp = requests.get(f"{base}/v2/account", headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error("Error fetching account: %s", e, exc_info=True)
        return {}


def calculate_bot_pnl(
    orders: List[Dict[str, Any]],
    positions: List[Dict[str, Any]],
    bot_tag: str,
    target_date: str,
) -> Dict[str, Any]:
    """
    Calculate PnL for a specific bot on a specific date.
    
    IMPORTANT: For bracket orders, the stop-loss/take-profit child orders
    don't have the bot tag in client_order_id. We match untagged sells
    to tagged buys for the same symbol on the same day.
    
    Args:
        orders: List of filled orders
        positions: Current open positions
        bot_tag: Bot identifier
        target_date: Date string YYYY-MM-DD
        
    Returns:
        Dict with realized_pnl, unrealized_pnl, trades, positions
    """
    result = {
        "bot_tag": bot_tag,
        "date": target_date,
        "realized_pnl": Decimal("0"),
        "unrealized_pnl": Decimal("0"),
        "total_pnl": Decimal("0"),
        "trades": [],
        "open_positions": [],
    }
    
    # Step 1: Find all BUY orders that belong to this bot on target date
    bot_buy_symbols = set()
    bot_orders = []

    for order in orders:
        filled_at = order.get("filled_at", "")
        if not filled_at.startswith(target_date):
            continue

        order_bot = extract_bot_tag_from_order(order)
        sym, side, _is_spread = _extract_order_symbol_side(order)
        if not sym:
            logger.warning(
                "Skipping order with empty symbol: id=%s filled_at=%s",
                order.get("id", "?"), order.get("filled_at", "?"),
            )
            continue

        if order_bot == bot_tag:
            bot_orders.append(order)
            if side == "buy":
                bot_buy_symbols.add(sym)

    # Step 2: Find SELL orders that match our buy symbols (even if untagged)
    # This catches stop-loss/take-profit orders from bracket orders
    for order in orders:
        filled_at = order.get("filled_at", "")
        if not filled_at.startswith(target_date):
            continue

        order_bot = extract_bot_tag_from_order(order)
        sym, side, _is_spread = _extract_order_symbol_side(order)
        if not sym:
            continue

        # Include untagged sells for symbols we bought today
        if side == "sell" and order_bot is None and sym in bot_buy_symbols:
            # Check if we already have this order
            order_id = order.get("id")
            if not any(o.get("id") == order_id for o in bot_orders):
                bot_orders.append(order)

    # Step 3: Group by symbol to calculate realized P&L
    symbol_trades: Dict[str, Dict[str, Any]] = {}
    for order in bot_orders:
        sym, side, is_spread = _extract_order_symbol_side(order)
        qty = Decimal(str(order.get("filled_qty", 0)))
        price = Decimal(str(order.get("filled_avg_price", 0)))

        if sym not in symbol_trades:
            symbol_trades[sym] = {
                "buy_qty": Decimal("0"), "buy_value": Decimal("0"),
                "sell_qty": Decimal("0"), "sell_value": Decimal("0"),
            }

        if side == "buy":
            symbol_trades[sym]["buy_qty"] += qty
            symbol_trades[sym]["buy_value"] += qty * price
        elif side == "sell":
            symbol_trades[sym]["sell_qty"] += qty
            symbol_trades[sym]["sell_value"] += qty * price

        result["trades"].append({
            "symbol": sym,
            "side": side,
            "is_spread": is_spread,
            "qty": str(qty),
            "price": str(price),
            "filled_at": order.get("filled_at"),
        })
    
    # Calculate realized P&L (sells - buys for closed portions)
    for sym, stats in symbol_trades.items():
        # Realized P&L = sell proceeds - buy cost for matched quantities
        matched_qty = min(stats["buy_qty"], stats["sell_qty"])
        if matched_qty > 0 and stats["buy_qty"] > 0 and stats["sell_qty"] > 0:
            avg_buy = stats["buy_value"] / stats["buy_qty"]
            avg_sell = stats["sell_value"] / stats["sell_qty"]
            result["realized_pnl"] += matched_qty * (avg_sell - avg_buy)
    
    # Find open positions for this bot
    for pos in positions:
        sym = pos.get("symbol", "")
        if sym in symbol_trades:
            # This position might belong to this bot
            open_qty = symbol_trades[sym]["buy_qty"] - symbol_trades[sym]["sell_qty"]
            if open_qty > 0:
                entry_price = symbol_trades[sym]["buy_value"] / symbol_trades[sym]["buy_qty"]
                current_price = Decimal(str(pos.get("current_price", 0)))
                unrealized = (current_price - entry_price) * open_qty
                result["unrealized_pnl"] += unrealized
                result["open_positions"].append({
                    "symbol": sym,
                    "qty": str(open_qty),
                    "entry_price": str(entry_price),
                    "current_price": str(current_price),
                    "unrealized_pnl": str(unrealized),
                })
    
    result["realized_pnl"] = str(result["realized_pnl"])
    result["unrealized_pnl"] = str(result["unrealized_pnl"])
    result["total_pnl"] = str(Decimal(result["realized_pnl"]) + Decimal(result["unrealized_pnl"]))
    return result


def persist_daily_results(target_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Main function to persist daily results.
    
    Args:
        target_date: Date string YYYY-MM-DD, defaults to today
        
    Returns:
        Consolidated daily results dict
    """
    if not target_date:
        target_date = _now_et().strftime("%Y-%m-%d")
    
    _log(f"Persisting daily results for {target_date}")
    
    # 1. Ensure registry files exist, then reconcile with broker
    _log("Step 1: Ensuring registries exist + broker reconciliation...")
    created = ensure_all_registries_exist()
    if created:
        _log(f"  Created missing registries: {created}")
    reconciliation = reconcile_positions(fix=True, verbose=False)
    
    # 2. Fetch broker data
    _log("Step 2: Fetching broker data...")
    try:
        positions = get_positions()
        orders = get_orders_for_week()
        account = get_account_info()
    except Exception as e:
        logger.error("Error fetching broker data: %s", e, exc_info=True)
        return {"error": str(e)}
    
    # 3. Calculate PnL for each bot
    _log("Step 3: Calculating PnL by bot...")
    bot_results = {}
    for bot_tag in BOT_TAGS:
        bot_pnl = calculate_bot_pnl(orders, positions, bot_tag, target_date)
        if bot_pnl["trades"] or bot_pnl["open_positions"]:
            bot_results[bot_tag] = bot_pnl
            _log(f"  {bot_tag}: Realized ${bot_pnl['realized_pnl']}, Unrealized ${bot_pnl['unrealized_pnl']}")

    # 4. Build consolidated report
    total_realized = sum((Decimal(r["realized_pnl"]) for r in bot_results.values()), Decimal("0"))
    total_unrealized = sum((Decimal(r["unrealized_pnl"]) for r in bot_results.values()), Decimal("0"))

    report = {
        "date": target_date,
        "generated_at": _now_et().isoformat(),
        "account": {
            "equity": str(Decimal(str(account.get("equity", 0)))),
            "cash": str(Decimal(str(account.get("cash", 0)))),
            "buying_power": str(Decimal(str(account.get("buying_power", 0)))),
        },
        "summary": {
            "total_realized_pnl": str(total_realized),
            "total_unrealized_pnl": str(total_unrealized),
            "total_pnl": str(total_realized + total_unrealized),
            "bots_active": list(bot_results.keys()),
        },
        "bots": bot_results,
        "reconciliation": {
            "discrepancies_found": len(reconciliation.get("discrepancies", [])),
            "fixes_applied": len(reconciliation.get("fixes_applied", [])),
        },
        "positions": [
            {
                "symbol": p.get("symbol"),
                "qty": p.get("qty"),
                "market_value": p.get("market_value"),
                "unrealized_pl": p.get("unrealized_pl"),
            }
            for p in positions
        ],
    }
    
    # 5. Save to file
    os.makedirs(DAILY_RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(DAILY_RESULTS_DIR, f"{target_date}.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    
    _log(f"Results saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"DAILY SUMMARY: {target_date}")
    print("=" * 60)
    print(f"Total Realized P&L:   ${total_realized:,.2f}")
    print(f"Total Unrealized P&L: ${total_unrealized:,.2f}")
    print(f"Total Day P&L:        ${total_realized + total_unrealized:,.2f}")
    print("-" * 60)
    for bot, data in bot_results.items():
        print(f"  {bot}: ${Decimal(data['total_pnl']):,.2f} ({len(data['trades'])} trades)")
    print("=" * 60 + "\n")

    # 6. Feed watchdog performance database (non-fatal if watchdog not set up)
    try:
        from RubberBand.src.watchdog.performance_db import PerformanceDB
        perf_db = PerformanceDB()
        watchdog_rows = perf_db.ingest_daily(target_date, results_dir=DAILY_RESULTS_DIR.rsplit("/", 1)[0])
        if watchdog_rows:
            _log(f"Watchdog ingested {len(watchdog_rows)} bot(s) for {target_date}")
    except Exception as e:
        logger.warning("Watchdog ingest skipped (non-fatal): %s", e, exc_info=True)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Persist daily trading results")
    parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format (default: today)")
    args = parser.parse_args()
    
    try:
        results = persist_daily_results(args.date)
        if "error" in results:
            return 1
        return 0
    except Exception as e:
        _log(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
