#!/usr/bin/env python3
"""
PnL Dashboard: Flask API Server (V2)

Provides REST endpoints for the dashboard UI to fetch PnL data.
Fixed: Spread consolidation, portfolio value, better filtering.
"""

import os
# Set Alpaca credentials before any imports that use them
if not os.environ.get("APCA_API_KEY_ID"):
    os.environ["APCA_API_KEY_ID"] = "PK5D0P6J7VQ66PNCOY3E"
    os.environ["APCA_API_SECRET_KEY"] = "jGs61lhYDGYthVh0K45aJvRK8x7SnseGpNvDO3zg"

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
import subprocess

app = Flask(__name__)

# Repository root for git operations
REPO_ROOT = Path(__file__).parent.parent

# Track last sync time to avoid excessive pulls
_last_sync_time = None
_SYNC_COOLDOWN_SECONDS = 30  # Minimum seconds between syncs

def git_pull_sync():
    """
    Pull the latest changes from GitHub to sync auditor_state.json.
    Returns tuple of (success, message).
    """
    global _last_sync_time
    
    # Check cooldown
    now = datetime.now()
    if _last_sync_time and (now - _last_sync_time).total_seconds() < _SYNC_COOLDOWN_SECONDS:
        return True, "Sync skipped (cooldown)"
    
    try:
        # Ensure Git is in PATH
        env = os.environ.copy()
        git_paths = [
            r"C:\Program Files\Git\cmd",
            r"C:\Program Files\Git\bin",
            r"C:\Program Files\GitHub CLI",
        ]
        env["PATH"] = ";".join(git_paths) + ";" + env.get("PATH", "")
        
        # Run git pull with safe.directory config
        result = subprocess.run(
            ["git", "-c", "safe.directory=*", "pull", "origin", "main", "--ff-only"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        
        _last_sync_time = now
        
        if result.returncode == 0:
            # Check if files were updated
            if "Already up to date" in result.stdout:
                return True, "Already up to date"
            else:
                return True, f"Synced: {result.stdout.strip()}"
        else:
            return False, f"Git pull failed: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, "Git pull timed out"
    except FileNotFoundError:
        return False, "Git not found in PATH"
    except Exception as e:
        return False, f"Sync error: {str(e)}"


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
    exp_date_str = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
    
    return {
        "underlying": underlying,
        "expiration": exp_date_str,
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
    today = datetime.now(ET).date()
    
    for pos in option_positions:
        parsed = parse_option_symbol(pos["symbol"])
        if parsed:
            # Calculate DTE
            exp_date = datetime.strptime(parsed["expiration"], "%Y-%m-%d").date()
            dte = (exp_date - today).days
            
            # Enrich position with parsed data
            pos_data = {**pos, **parsed, "dte": dte}
            key = (parsed["underlying"], parsed["expiration"])
            groups[key].append(pos_data)
        else:
            # Can't parse, show as-is
            pos["dte"] = 0
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
                    "dte": long_leg["dte"],  # Same for both legs
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
                "dte": leg.get("dte", 0),
                "description": leg["symbol"],
                "qty": leg["qty"],
                "avg_entry": leg["avg_entry"],
                "current_price": leg["current_price"],
                "unrealized_pnl": leg["unrealized_pnl"],
            })
    
    # Sort by DTE (ascending) then Symbol
    consolidated.sort(key=lambda x: (x.get("dte", 999), x.get("underlying", "")))
    return consolidated

def get_portfolio_value():
    """Fetch current portfolio equity from Alpaca."""
    base, key, secret = _alpaca_creds()
    if not key:
        return {}
    
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
    return {}


def count_filled_orders_by_bot(orders, bot_tag):
    """Count filled orders (entries) for a specific bot."""
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


@app.route("/api/sync")
def api_sync():
    """Trigger a git pull to sync auditor_state.json with GitHub."""
    success, message = git_pull_sync()
    return jsonify({
        "success": success,
        "message": message,
        "timestamp": datetime.now(ET).isoformat(),
    })


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
        
        # Analyze if cache exists, otherwise assume 0
        if cache_file.exists():
            try:
                orders = json.load(open(cache_file))
                analysis = analyze_orders(orders, [])
                
                if bot == "all":
                    total = sum(a["realized_pnl"] for a in analysis.values())
                else:
                    total = analysis.get(bot, {}).get("realized_pnl", 0)
                
                data.append({"date": date_str, "pnl": total})
            except:
                data.append({"date": date_str, "pnl": 0})
        else:
             data.append({"date": date_str, "pnl": 0})
    
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
            try:
                orders = json.load(open(cache_file))
                
                # Count ENTRIES (filled orders with bot CID), not just realized trades
                if bot == "all":
                    count = sum(count_filled_orders_by_bot(orders, b) for b in 
                               ["15m_stock", "15m_options", "weekly_stock", "weekly_options"])
                else:
                    count = count_filled_orders_by_bot(orders, bot)
                
                data.append({"date": date_str, "count": count})
            except:
                data.append({"date": date_str, "count": 0})
        else:
             data.append({"date": date_str, "count": 0})
    
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
    
    try:
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
                        symbol = filled_legs[0].get("symbol", "")
                        side = filled_legs[0].get("side", "")
            
            qty = float(order.get("filled_qty", 0) or 0)
            price = float(order.get("filled_avg_price", 0) or 0)
            
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
    except Exception as e:
        print(f"Error loading trades: {e}")
        return jsonify([])

def get_alpaca_entry_dates(positions):
    """
    Fetch recent orders from Alpaca to find entry timestamps.
    Returns map: symbol -> filled_at (ISO string)
    """
    if not positions:
        return {}
        
    entry_dates = {}
    needed = {} # symbol -> side (buy/sell) needed for entry
    
    for pos in positions:
        qty = float(pos.get("qty", 0))
        if qty == 0: continue
        # If long (qty > 0), entry was a BUY. If short (qty < 0), entry was a SELL.
        needed[pos["symbol"]] = "buy" if qty > 0 else "sell"
        
    if not needed:
        return {}
        
    base, key, secret = _alpaca_creds()
    if not key: return {}
    
    try:
        # Fetch last 500 closed orders (descending = newest first)
        url = f"{base}/v2/orders?status=closed&limit=500&direction=desc"
        resp = requests.get(url, headers=_alpaca_headers(key, secret), timeout=10)
        
        if resp.status_code == 200:
            orders = resp.json()
            for order in orders:
                sym = order.get("symbol")
                side = order.get("side")
                
                # Check if this order is a relevant entry for one of our positions
                if sym in needed and needed[sym] == side:
                    if sym not in entry_dates:
                        # Found most recent entry!
                        entry_dates[sym] = order.get("filled_at")
                        
                # Optimization: Stop if we found everything
                if len(entry_dates) == len(needed):
                    break
    except Exception as e:
        print(f"[ERROR] Failed to fetch orders for dates: {e}")
        
    return entry_dates


@app.route("/api/positions")
def api_positions():
    """Get current open positions with consolidated spreads and bot filtering."""
    bot_filter = request.args.get("bot", "all")
    
    positions = fetch_broker_positions()
    categorized = get_open_positions_by_category(positions)
    
    # Consolidate options into spreads (and calculate DTE)
    consolidated_options = consolidate_spreads(categorized["options"])
    
    # Enrich with entry dates from Auditor
    auditor_state = load_auditor_state()
    entry_lookup = {}
    
    # 1. Load from Auditor (Base source)
    for pos in auditor_state.get("positions", {}).values():
        if pos.get("status") == "OPEN" and pos.get("entry_time"):
            sym = pos.get("symbol")
            entry_lookup[sym] = pos.get("entry_time")

    # 2. Load from Alpaca Orders (Override/Supplement)
    # This catches positions that are open in Broker but missing/closed in Auditor
    all_legs = categorized["stocks"] + categorized["options"]
    alpaca_dates = get_alpaca_entry_dates(all_legs)
    entry_lookup.update(alpaca_dates)
    
    # helper to format date
    def fmt_date(iso_str):
        if not iso_str: return "-"
        return iso_str[:16].replace("T", " ")
    
    # Enrich Stocks
    for stock in categorized["stocks"]:
        raw_date = entry_lookup.get(stock["symbol"])
        stock["entry_date"] = fmt_date(raw_date)
        
    # Enrich Options
    for opt in consolidated_options:
        # For single legs, direct lookup
        if opt["type"] == "single":
            raw_date = entry_lookup.get(opt["description"])
            # Fallback to underlying if needed
            if not raw_date:
                raw_date = entry_lookup.get(opt["description"]) # retry exact
            opt["entry_date"] = fmt_date(raw_date)
        
        # For spreads, try to find entry date of legs
        elif opt["type"] == "spread":
            # Just take the first leg's date if available
            legs = opt.get("legs", [])
            dates = [entry_lookup.get(leg) for leg in legs if leg in entry_lookup]
            opt["entry_date"] = fmt_date(dates[0]) if dates else "-"

    result = {
        "stocks": [],
        "options": []
    }
    
    # Apply rudimentary bot filtering based on asset class
    # Since we can't perfectly link live positions to bots without a DB,
    # we assume Stock Bots trade Stocks and Option Bots trade Options.
    
    show_stocks = bot_filter == "all" or "stock" in bot_filter
    show_options = bot_filter == "all" or "options" in bot_filter
    
    if show_stocks:
        result["stocks"] = categorized["stocks"]
        
    if show_options:
        result["options"] = consolidated_options
    
    return jsonify(result)


# ==============================================================================
# Auditor / Shadow Ledger Endpoints
# ==============================================================================

AUDITOR_STATE_FILE = Path(__file__).parent.parent / "auditor_state.json"


def load_auditor_state():
    """Load the auditor state JSON file."""
    if AUDITOR_STATE_FILE.exists():
        try:
            with open(AUDITOR_STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load auditor_state.json: {e}")
    return {"positions": {}, "closed_positions": []}


@app.route("/auditor")
def auditor_page():
    """Serve the Auditor / Shadow Ledger page."""
    return render_template("auditor.html")


@app.route("/api/auditor/positions")
def api_auditor_positions():
    """Get open shadow positions, optionally filtered by entry date."""
    state = load_auditor_state()
    
    # Date filters
    start_date = request.args.get("start")
    end_date = request.args.get("end")
    
    positions = []
    
    for pos_id, pos in state.get("positions", {}).items():
        if pos.get("status") != "OPEN":
            continue
        
        # Apply date filter on entry_time
        if start_date or end_date:
            entry_time = pos.get("entry_time", "")
            if entry_time:
                entry_date = entry_time[:10]  # Extract YYYY-MM-DD
                if start_date and entry_date < start_date:
                    continue
                if end_date and entry_date > end_date:
                    continue
        
        positions.append({
            "id": pos_id,
            "symbol": pos.get("symbol"),
            "bot_tag": pos.get("bot_tag"),
            "type": pos.get("type"),
            "entry_price": pos.get("entry_price", 0),
            "entry_time": pos.get("entry_time"),
            "qty": pos.get("qty", 100),
            "tp_price": pos.get("tp_price"),
            "sl_price": pos.get("sl_price"),
            "current_price": pos.get("current_price", pos.get("entry_price", 0)),
            "unrealized_pnl": pos.get("unrealized_pnl", 0),
            "reason": pos.get("reason"),
        })
    
    # Sort by unrealized PnL descending
    positions.sort(key=lambda x: x.get("unrealized_pnl", 0), reverse=True)
    return jsonify(positions)


@app.route("/api/auditor/closed")
def api_auditor_closed():
    """Get closed shadow trades with realized PnL, optionally filtered by exit date."""
    state = load_auditor_state()
    
    # Date filters
    start_date = request.args.get("start")
    end_date = request.args.get("end")
    limit = int(request.args.get("limit", 100))
    
    closed = []
    
    for pos in state.get("closed_positions", []):
        # Apply date filter on exit_time
        if start_date or end_date:
            exit_time = pos.get("exit_time", "")
            if exit_time:
                exit_date = exit_time[:10]  # Extract YYYY-MM-DD
                if start_date and exit_date < start_date:
                    continue
                if end_date and exit_date > end_date:
                    continue
        
        closed.append({
            "symbol": pos.get("symbol"),
            "bot_tag": pos.get("bot_tag"),
            "type": pos.get("type"),
            "entry_price": pos.get("entry_price", 0),
            "entry_time": pos.get("entry_time"),
            "exit_price": pos.get("exit_price", 0),
            "exit_time": pos.get("exit_time"),
            "exit_reason": pos.get("exit_reason"),
            "qty": pos.get("qty", 100),
            "realized_pnl": pos.get("realized_pnl", 0),
        })
    
    # Sort by exit time descending (most recent first)
    closed.sort(key=lambda x: x.get("exit_time", ""), reverse=True)
    return jsonify(closed[:limit])


@app.route("/api/auditor/summary")
def api_auditor_summary():
    """Get aggregate shadow trading statistics, optionally filtered by date."""
    state = load_auditor_state()
    
    # Date filters
    start_date = request.args.get("start")
    end_date = request.args.get("end")
    
    # Filter open positions by entry date
    all_open = state.get("positions", {}).values()
    open_positions = []
    for p in all_open:
        if p.get("status") != "OPEN":
            continue
        if start_date or end_date:
            entry_time = p.get("entry_time", "")
            if entry_time:
                entry_date = entry_time[:10]
                if start_date and entry_date < start_date:
                    continue
                if end_date and entry_date > end_date:
                    continue
        open_positions.append(p)
    
    total_unrealized = sum(p.get("unrealized_pnl", 0) for p in open_positions)
    
    # Filter closed positions by exit date
    all_closed = state.get("closed_positions", [])
    closed = []
    for p in all_closed:
        if start_date or end_date:
            exit_time = p.get("exit_time", "")
            if exit_time:
                exit_date = exit_time[:10]
                if start_date and exit_date < start_date:
                    continue
                if end_date and exit_date > end_date:
                    continue
        closed.append(p)
    
    total_realized = sum(p.get("realized_pnl", 0) for p in closed)
    
    winners = [p for p in closed if p.get("realized_pnl", 0) > 0]
    losers = [p for p in closed if p.get("realized_pnl", 0) < 0]
    
    win_rate = (len(winners) / len(closed) * 100) if closed else 0
    
    avg_win = sum(p.get("realized_pnl", 0) for p in winners) / len(winners) if winners else 0
    avg_loss = sum(p.get("realized_pnl", 0) for p in losers) / len(losers) if losers else 0
    
    # Positions by bot
    by_bot = {}
    for pos in open_positions:
        bot = pos.get("bot_tag", "unknown")
        if bot not in by_bot:
            by_bot[bot] = {"count": 0, "unrealized": 0}
        by_bot[bot]["count"] += 1
        by_bot[bot]["unrealized"] += pos.get("unrealized_pnl", 0)
    
    return jsonify({
        "open_count": len(open_positions),
        "closed_count": len(closed),
        "total_realized": total_realized,
        "total_unrealized": total_unrealized,
        "net_pnl": total_realized + total_unrealized,
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "by_bot": by_bot,
    })


# ==============================================================================
# Daily Report API
# ==============================================================================

import subprocess


def get_workflow_runs(limit=10):
    """Get recent GitHub workflow runs."""
    try:
        cmd = ["gh", "run", "list", "--limit", str(limit), "--json", "databaseId,displayTitle,conclusion,status,createdAt,workflowName"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(Path(__file__).parent.parent))
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception as e:
        print(f"[ERROR] Failed to get workflow runs: {e}")
    return []


@app.route("/api/daily-report")
def api_daily_report():
    """Generate a comprehensive daily report."""
    date = request.args.get("date", datetime.now(ET).strftime("%Y-%m-%d"))
    
    # Get portfolio value
    portfolio = get_portfolio_value() or {}
    
    # Get positions
    positions = fetch_broker_positions()
    categorized = get_open_positions_by_category(positions)
    
    # Get orders for the date
    orders = fetch_broker_orders(date, force_refresh=False)
    analysis = analyze_orders(orders, positions)
    
    # Get workflow runs
    runs = get_workflow_runs(15)
    workflow_summary = {
        "total": len(runs),
        "success": len([r for r in runs if r.get("conclusion") == "success"]),
        "failure": len([r for r in runs if r.get("conclusion") == "failure"]),
        "in_progress": len([r for r in runs if r.get("status") == "in_progress"]),
        "recent": runs[:5]
    }
    
    # Get auditor summary
    auditor_state = load_auditor_state()
    open_shadow = [p for p in auditor_state.get("positions", {}).values() if p.get("status") == "OPEN"]
    closed_shadow = auditor_state.get("closed_positions", [])
    
    # Calculate totals
    stock_unrealized = sum(p.get("unrealized_pnl", 0) for p in categorized["stocks"])
    options_unrealized = sum(p.get("unrealized_pnl", 0) for p in categorized["options"])
    total_realized = sum(a["realized_pnl"] for a in analysis.values())
    
    return jsonify({
        "date": date,
        "generated_at": datetime.now(ET).isoformat(),
        "account": {
            "equity": portfolio.get("equity", 0),
            "cash": portfolio.get("cash", 0),
            "buying_power": portfolio.get("buying_power", 0),
            "day_change": portfolio.get("day_change", 0),
        },
        "positions": {
            "stocks_count": len(categorized["stocks"]),
            "stocks_unrealized": stock_unrealized,
            "options_count": len(categorized["options"]),
            "options_unrealized": options_unrealized,
            "total_unrealized": stock_unrealized + options_unrealized,
        },
        "trading": {
            "realized_pnl": total_realized,
            "15m_stock": analysis["15m_stock"],
            "15m_options": analysis["15m_options"],
            "weekly_stock": analysis["weekly_stock"],
            "weekly_options": analysis["weekly_options"],
        },
        "workflows": workflow_summary,
        "auditor": {
            "open_positions": len(open_shadow),
            "total_unrealized": sum(p.get("unrealized_pnl", 0) for p in open_shadow),
            "closed_count": len(closed_shadow),
            "total_realized": sum(p.get("realized_pnl", 0) for p in closed_shadow),
        }
    })


@app.route("/api/daily-report/export")
def api_daily_report_export():
    """Export daily report as markdown."""
    from flask import Response
    
    date = request.args.get("date", datetime.now(ET).strftime("%Y-%m-%d"))
    
    # Get the report data
    report_resp = api_daily_report()
    report = json.loads(report_resp.data)
    
    # Generate markdown
    md = f"""# Daily Trading Report - {date}

## Account Summary
| Metric | Value |
|--------|-------|
| **Equity** | ${report['account']['equity']:,.2f} |
| **Day Change** | ${report['account']['day_change']:+,.2f} |
| **Cash** | ${report['account']['cash']:,.2f} |
| **Buying Power** | ${report['account']['buying_power']:,.2f} |

## Position Summary
| Category | Count | Unrealized PnL |
|----------|-------|----------------|
| Stocks | {report['positions']['stocks_count']} | ${report['positions']['stocks_unrealized']:+,.2f} |
| Options | {report['positions']['options_count']} | ${report['positions']['options_unrealized']:+,.2f} |
| **Total** | {report['positions']['stocks_count'] + report['positions']['options_count']} | **${report['positions']['total_unrealized']:+,.2f}** |

## Bot Performance
| Bot | Entries | Winners | Losers | Realized PnL |
|-----|---------|---------|--------|--------------|
| 15m Stock | {report['trading']['15m_stock']['winners'] + report['trading']['15m_stock']['losers']} | {report['trading']['15m_stock']['winners']} | {report['trading']['15m_stock']['losers']} | ${report['trading']['15m_stock']['realized_pnl']:+,.2f} |
| 15m Options | {report['trading']['15m_options']['winners'] + report['trading']['15m_options']['losers']} | {report['trading']['15m_options']['winners']} | {report['trading']['15m_options']['losers']} | ${report['trading']['15m_options']['realized_pnl']:+,.2f} |
| Weekly Stock | {report['trading']['weekly_stock']['winners'] + report['trading']['weekly_stock']['losers']} | {report['trading']['weekly_stock']['winners']} | {report['trading']['weekly_stock']['losers']} | ${report['trading']['weekly_stock']['realized_pnl']:+,.2f} |
| Weekly Options | {report['trading']['weekly_options']['winners'] + report['trading']['weekly_options']['losers']} | {report['trading']['weekly_options']['winners']} | {report['trading']['weekly_options']['losers']} | ${report['trading']['weekly_options']['realized_pnl']:+,.2f} |

## Workflow Status
- ✅ Success: {report['workflows']['success']}
- ❌ Failed: {report['workflows']['failure']}
- 🔄 In Progress: {report['workflows']['in_progress']}

## Shadow Ledger (Auditor)
| Metric | Value |
|--------|-------|
| Open Positions | {report['auditor']['open_positions']} |
| Unrealized PnL | ${report['auditor']['total_unrealized']:+,.2f} |
| Closed Trades | {report['auditor']['closed_count']} |
| Realized PnL | ${report['auditor']['total_realized']:+,.2f} |

---
*Generated at {report['generated_at']}*
"""
    
    return Response(
        md,
        mimetype="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=daily_report_{date}.md"}
    )


@app.route("/reports")
def reports_page():
    """Serve the Reports download page."""
    return render_template("reports.html")


@app.route("/api/daily-report/export-range")
def api_daily_report_export_range():
    """Export a date range report as markdown."""
    from flask import Response
    
    start = request.args.get("start")
    end = request.args.get("end")
    
    if not start or not end:
        return jsonify({"error": "Both start and end dates are required"}), 400
    
    # Parse dates
    try:
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
    
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    
    # Generate combined report
    md = f"""# Trading Report: {start} to {end}

Generated at: {datetime.now(ET).isoformat()}

"""
    
    # Aggregate metrics
    total_realized = 0
    total_trades = 0
    all_entries = {"15m_stock": 0, "15m_options": 0, "weekly_stock": 0, "weekly_options": 0}
    
    current = start_date
    daily_summaries = []
    
    while current <= end_date:
        # Skip weekends
        if current.weekday() < 5:
            date_str = current.strftime("%Y-%m-%d")
            orders = fetch_broker_orders(date_str, force_refresh=False)
            
            if orders:
                analysis = analyze_orders(orders, [])
                day_realized = sum(a["realized_pnl"] for a in analysis.values())
                day_trades = sum(a["winners"] + a["losers"] for a in analysis.values())
                
                total_realized += day_realized
                total_trades += day_trades
                
                for bot in all_entries.keys():
                    all_entries[bot] += analysis[bot]["winners"] + analysis[bot]["losers"]
                
                if day_trades > 0 or day_realized != 0:
                    daily_summaries.append({
                        "date": date_str,
                        "realized": day_realized,
                        "trades": day_trades,
                    })
        
        current += timedelta(days=1)
    
    # Summary section
    md += f"""## Summary

| Metric | Value |
|--------|-------|
| **Period** | {start} to {end} |
| **Trading Days** | {len(daily_summaries)} |
| **Total Trades** | {total_trades} |
| **Total Realized PnL** | **${total_realized:+,.2f}** |

## Bot Performance

| Bot | Trades |
|-----|--------|
| 15m Stock | {all_entries['15m_stock']} |
| 15m Options | {all_entries['15m_options']} |
| Weekly Stock | {all_entries['weekly_stock']} |
| Weekly Options | {all_entries['weekly_options']} |

## Daily Breakdown

| Date | Trades | Realized PnL |
|------|--------|--------------|
"""
    
    for day in daily_summaries:
        md += f"| {day['date']} | {day['trades']} | ${day['realized']:+,.2f} |\n"
    
    if not daily_summaries:
        md += "| *No trading activity found* | - | - |\n"
    
    md += f"\n---\n*Report generated by RubberBandBot Dashboard*\n"
    
    return Response(
        md,
        mimetype="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=report_{start}_to_{end}.md"}
    )


# ==============================================================================
# Watchdog Endpoints
# ==============================================================================

WATCHDOG_DIR = Path(__file__).parent.parent / "results" / "watchdog"


def _load_watchdog_json(filename: str) -> dict:
    """Load a watchdog JSON state file (fail-safe)."""
    path = WATCHDOG_DIR / filename
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load {filename}: {e}")
    return {}


@app.route("/api/watchdog/status")
def api_watchdog_status():
    """Return combined watchdog status: pause flags, profit locks, intraday health."""
    return jsonify({
        "intraday_health": _load_watchdog_json("intraday_health.json"),
        "pause_flags": _load_watchdog_json("bot_pause_flags.json"),
        "profit_locks": _load_watchdog_json("profit_locks.json"),
    })


@app.route("/api/watchdog/alerts")
def api_watchdog_alerts():
    """Return recent alerts from alerts.jsonl (newest first, max 200)."""
    alerts_path = WATCHDOG_DIR / "alerts.jsonl"
    alerts = []
    if alerts_path.exists():
        try:
            with open(alerts_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            alerts.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"[ERROR] Failed to read alerts.jsonl: {e}")

    # Newest first, limit to 200
    alerts.reverse()
    limit = int(request.args.get("limit", 200))
    return jsonify(alerts[:limit])


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  PnL Dashboard Server (V2)")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)


