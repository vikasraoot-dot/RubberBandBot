#!/usr/bin/env python3
"""
PnL Dashboard: Flask API Server (V2)

Provides REST endpoints for the dashboard UI to fetch PnL data.
Fixed: Spread consolidation, portfolio value, better filtering.
"""

import os
import sys
import json
import re
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict

# Add parent directory to path to import reconcile_pnl
sys.path.insert(0, str(Path(__file__).parent.parent))
from reconcile_pnl import (
    fetch_broker_orders,
    fetch_bot_logs,
    fetch_broker_positions,
    analyze_orders,
    get_open_positions_by_category,
    is_option_symbol,
    CACHE_DIR,
    ORDERS_CACHE,
    ET,
    _alpaca_creds,
    _alpaca_headers,
)
import requests

app = Flask(__name__)

# ==============================================================================
# Helper Functions
# ==============================================================================

def parse_option_symbol(symbol):
    """
    Parse OCC option symbol into components.
    Example: AMD260116C00207500 -> (AMD, 2026-01-16, C, 207.50)
    """
    if not symbol or len(symbol) < 15:
        return None
    
    # Find where the date starts (6 digits after underlying)
    match = re.match(r'^([A-Z]+)(\d{6})([CP])(\d{8})$', symbol)
    if not match:
        return None
    
    underlying = match.group(1)
    date_str = match.group(2)
    call_put = match.group(3)
    strike_raw = match.group(4)
    strike = int(strike_raw) / 1000
    
    # Parse date
    exp_date = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
    
    return {
        "underlying": underlying,
        "expiration": exp_date,
        "type": "Call" if call_put == "C" else "Put",
        "strike": strike,
    }


def consolidate_spreads(option_positions):
    """
    Consolidate option legs into spreads where applicable.
    
    Spread detection: Same underlying, same expiration, opposite qty signs (long/short).
    """
    # Group by underlying + expiration
    groups = defaultdict(list)
    
    for pos in option_positions:
        parsed = parse_option_symbol(pos["symbol"])
        if parsed:
            key = (parsed["underlying"], parsed["expiration"])
            groups[key].append({**pos, **parsed})
        else:
            # Can't parse, show as-is
            groups[("UNKNOWN", pos["symbol"])].append(pos)
    
    consolidated = []
    
    for (underlying, exp), legs in groups.items():
        if len(legs) == 2:
            # Potential spread: Check for long/short pair
            leg1, leg2 = legs
            if leg1.get("qty", 0) * leg2.get("qty", 0) < 0:
                # Opposite signs - this is a spread
                long_leg = leg1 if leg1["qty"] > 0 else leg2
                short_leg = leg2 if leg1["qty"] > 0 else leg1
                
                spread_pnl = leg1["unrealized_pnl"] + leg2["unrealized_pnl"]
                spread_cost = abs(long_leg["avg_entry"]) - abs(short_leg["avg_entry"])
                spread_current = abs(long_leg["current_price"]) - abs(short_leg["current_price"])
                
                consolidated.append({
                    "type": "spread",
                    "underlying": underlying,
                    "expiration": exp,
                    "description": f"{underlying} {long_leg['strike']}/{short_leg['strike']}C {exp[5:]}",
                    "qty": abs(long_leg["qty"]),
                    "avg_entry": spread_cost,
                    "current_price": spread_current,
                    "unrealized_pnl": spread_pnl,
                    "legs": [leg1["symbol"], leg2["symbol"]],
                })
                continue
        
        # Single leg(s) or unmatched - show individually
        for leg in legs:
            consolidated.append({
                "type": "single",
                "underlying": leg.get("underlying", underlying),
                "expiration": leg.get("expiration", exp),
                "description": leg["symbol"],
                "qty": leg["qty"],
                "avg_entry": leg["avg_entry"],
                "current_price": leg["current_price"],
                "unrealized_pnl": leg["unrealized_pnl"],
            })
    
    return consolidated


def get_portfolio_value():
    """Fetch current portfolio equity from Alpaca."""
    base, key, secret = _alpaca_creds()
    if not key:
        return None
    
    try:
        resp = requests.get(f"{base}/v2/account", headers=_alpaca_headers(key, secret), timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "equity": float(data.get("equity", 0)),
                "buying_power": float(data.get("buying_power", 0)),
                "cash": float(data.get("cash", 0)),
                "portfolio_value": float(data.get("portfolio_value", 0)),
                "last_equity": float(data.get("last_equity", 0)),
                "day_change": float(data.get("equity", 0)) - float(data.get("last_equity", 0)),
            }
    except Exception as e:
        print(f"[ERROR] Failed to fetch account: {e}")
    return None


def count_filled_orders_by_bot(orders, bot_tag):
    """Count filled orders (entries) for a specific bot, not just realized trades."""
    count = 0
    for order in orders:
        if order.get("status") not in ("filled", "partially_filled"):
            continue
        cid = (order.get("client_order_id") or "").upper()
        
        if bot_tag == "15m_stock" and cid.startswith("15M_STK"):
            count += 1
        elif bot_tag == "15m_options" and cid.startswith("15M_OPT"):
            count += 1
        elif bot_tag == "weekly_stock" and cid.startswith("WK_STK"):
            count += 1
        elif bot_tag == "weekly_options" and cid.startswith("WK_OPT"):
            count += 1
    return count


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/api/portfolio")
def api_portfolio():
    """Get current portfolio value."""
    data = get_portfolio_value()
    if data:
        return jsonify(data)
    return jsonify({"error": "Failed to fetch portfolio"}), 500


@app.route("/api/pnl")
def api_pnl():
    """Get PnL summary for a specific date."""
    date = request.args.get("date", datetime.now(ET).strftime("%Y-%m-%d"))
    bot_filter = request.args.get("bot", "all")
    
    orders = fetch_broker_orders(date, force_refresh=False)
    positions = fetch_broker_positions()
    analysis = analyze_orders(orders, positions)
    categorized = get_open_positions_by_category(positions)
    
    # Consolidate option spreads
    consolidated_options = consolidate_spreads(categorized["options"])
    
    # Calculate totals based on filter
    if bot_filter == "all":
        total_realized = sum(a["realized_pnl"] for a in analysis.values())
        total_trades = sum(a["winners"] + a["losers"] for a in analysis.values())
    else:
        total_realized = analysis.get(bot_filter, {}).get("realized_pnl", 0)
        total_trades = analysis.get(bot_filter, {}).get("winners", 0) + analysis.get(bot_filter, {}).get("losers", 0)
    
    total_unrealized = sum(p["unrealized_pnl"] for p in categorized["stocks"])
    total_unrealized += sum(p["unrealized_pnl"] for p in categorized["options"])
    
    # Count entries (not just realized trades)
    entries_15m_stock = count_filled_orders_by_bot(orders, "15m_stock")
    entries_15m_options = count_filled_orders_by_bot(orders, "15m_options")
    entries_weekly_stock = count_filled_orders_by_bot(orders, "weekly_stock")
    entries_weekly_options = count_filled_orders_by_bot(orders, "weekly_options")
    
    return jsonify({
        "date": date,
        "bot_filter": bot_filter,
        "bots": {
            "15m_stock": {
                "entries": entries_15m_stock,
                "trades": analysis["15m_stock"]["winners"] + analysis["15m_stock"]["losers"],
                "winners": analysis["15m_stock"]["winners"],
                "losers": analysis["15m_stock"]["losers"],
                "realized_pnl": analysis["15m_stock"]["realized_pnl"],
            },
            "15m_options": {
                "entries": entries_15m_options,
                "trades": analysis["15m_options"]["winners"] + analysis["15m_options"]["losers"],
                "winners": analysis["15m_options"]["winners"],
                "losers": analysis["15m_options"]["losers"],
                "realized_pnl": analysis["15m_options"]["realized_pnl"],
                "open_count": len([o for o in consolidated_options if o["type"] == "spread"]) + 
                              len([o for o in consolidated_options if o["type"] == "single"]),
                "unrealized_pnl": sum(p["unrealized_pnl"] for p in categorized["options"]),
            },
            "weekly_stock": {
                "entries": entries_weekly_stock,
                "trades": analysis["weekly_stock"]["winners"] + analysis["weekly_stock"]["losers"],
                "winners": analysis["weekly_stock"]["winners"],
                "losers": analysis["weekly_stock"]["losers"],
                "realized_pnl": analysis["weekly_stock"]["realized_pnl"],
            },
            "weekly_options": {
                "entries": entries_weekly_options,
                "trades": analysis["weekly_options"]["winners"] + analysis["weekly_options"]["losers"],
                "winners": analysis["weekly_options"]["winners"],
                "losers": analysis["weekly_options"]["losers"],
                "realized_pnl": analysis["weekly_options"]["realized_pnl"],
            },
        },
        "totals": {
            "realized": total_realized,
            "unrealized": total_unrealized,
            "net": total_realized + total_unrealized,
            "trades": total_trades,
        },
        "open_positions": {
            "stocks": len(categorized["stocks"]),
            "options_legs": len(categorized["options"]),
            "options_consolidated": len(consolidated_options),
        }
    })


@app.route("/api/chart/pnl")
def api_chart_pnl():
    """Get PnL data for charting over a period."""
    period = request.args.get("period", "7d")
    bot = request.args.get("bot", "all")
    
    # Determine number of days
    if period == "7d":
        days = 10  # Look back 10 to get 7 trading days
    elif period == "30d":
        days = 45
    elif period == "ytd":
        start_of_year = datetime(datetime.now(ET).year, 1, 1, tzinfo=ET)
        days = (datetime.now(ET) - start_of_year).days + 1
    else:
        days = 10
    
    data = []
    current = datetime.now(ET).date()
    
    for i in range(days):
        check_date = current - timedelta(days=i)
        if check_date.weekday() >= 5:  # Skip weekends
            continue
        
        date_str = check_date.strftime("%Y-%m-%d")
        cache_file = ORDERS_CACHE / f"{date_str}.json"
        
        if cache_file.exists():
            orders = json.load(open(cache_file))
            analysis = analyze_orders(orders, [])
            
            if bot == "all":
                total = sum(a["realized_pnl"] for a in analysis.values())
            else:
                total = analysis.get(bot, {}).get("realized_pnl", 0)
            
            data.append({"date": date_str, "pnl": total})
    
    data.reverse()  # Chronological order
    return jsonify(data)


@app.route("/api/chart/trades")
def api_chart_trades():
    """Get trade count data for charting."""
    period = request.args.get("period", "7d")
    bot = request.args.get("bot", "all")
    
    days = 10 if period == "7d" else 45 if period == "30d" else 10
    data = []
    current = datetime.now(ET).date()
    
    for i in range(days):
        check_date = current - timedelta(days=i)
        if check_date.weekday() >= 5:
            continue
        
        date_str = check_date.strftime("%Y-%m-%d")
        cache_file = ORDERS_CACHE / f"{date_str}.json"
        
        if cache_file.exists():
            orders = json.load(open(cache_file))
            
            # Count ENTRIES (filled orders with bot CID), not just realized trades
            if bot == "all":
                count = sum(count_filled_orders_by_bot(orders, b) for b in 
                           ["15m_stock", "15m_options", "weekly_stock", "weekly_options"])
            else:
                count = count_filled_orders_by_bot(orders, bot)
            
            data.append({"date": date_str, "count": count})
    
    data.reverse()
    return jsonify(data)


@app.route("/api/trades")
def api_trades():
    """Get trade details for a specific date and bot filter."""
    date = request.args.get("date", datetime.now(ET).strftime("%Y-%m-%d"))
    bot_filter = request.args.get("bot", "all")
    
    cache_file = ORDERS_CACHE / f"{date}.json"
    if not cache_file.exists():
        return jsonify([])
    
    orders = json.load(open(cache_file))
    trades = []
    
    for order in orders:
        if order.get("status") not in ("filled", "partially_filled"):
            continue
        
        # Determine category from CID
        cid = (order.get("client_order_id") or "").upper()
        if cid.startswith("15M_STK"):
            category = "15m_stock"
        elif cid.startswith("15M_OPT"):
            category = "15m_options"
        elif cid.startswith("WK_STK"):
            category = "weekly_stock"
        elif cid.startswith("WK_OPT"):
            category = "weekly_options"
        else:
            category = "manual"
        
        # Apply filter
        if bot_filter != "all" and category != bot_filter:
            continue
        
        # Handle multi-leg orders (spreads, brackets)
        symbol = order.get("symbol", "")
        side = order.get("side", "")
        legs = order.get("legs", [])
        
        if not symbol and legs:
            # Multi-leg order: extract from first filled leg or describe the spread
            filled_legs = [l for l in legs if l.get("status") == "filled"]
            if filled_legs:
                # For options spreads, show the spread description
                if category == "15m_options":
                    leg_symbols = [l.get("symbol", "") for l in filled_legs]
                    symbol = " / ".join(leg_symbols[:2])
                    side = "spread"
                else:
                    # For bracket orders, use first leg
                    symbol = filled_legs[0].get("symbol", "")
                    side = filled_legs[0].get("side", "")
        
        qty = float(order.get("filled_qty", 0) or 0)
        price = float(order.get("filled_avg_price", 0) or 0)
        
        # Skip if still no data
        if not symbol:
            continue
        
        trades.append({
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "category": category,
            "time": order.get("filled_at", ""),
        })
    
    # Sort by time
    trades.sort(key=lambda x: x.get("time", ""))
    return jsonify(trades)


@app.route("/api/positions")
def api_positions():
    """Get current open positions with consolidated spreads."""
    positions = fetch_broker_positions()
    categorized = get_open_positions_by_category(positions)
    
    # Consolidate options into spreads
    consolidated_options = consolidate_spreads(categorized["options"])
    
    return jsonify({
        "stocks": categorized["stocks"],
        "options": consolidated_options,
    })


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  PnL Dashboard Server (V2)")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)
