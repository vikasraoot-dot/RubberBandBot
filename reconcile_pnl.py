#!/usr/bin/env python3
"""
Reconciliation Tool: Daily PnL Analysis & Trade Attribution

This script provides:
1. Trade attribution (15m Stock, 15m Options, Weekly Stock, Weekly Options)
2. Daily and multi-day PnL tracking
3. Open position tracking with entry cost

Usage:
    python reconcile_pnl.py              # Default: Today's performance
    python reconcile_pnl.py --refresh    # Force refresh data
    python reconcile_pnl.py --date 2026-01-05  # Specific date
    python reconcile_pnl.py --deep-clean # Rebuild cache

Author: RubberBandBot Assistant
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from zoneinfo import ZoneInfo

import requests
import pandas as pd

# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT_DIR = Path(__file__).parent.absolute()
CACHE_DIR = SCRIPT_DIR / "data" / "cache"
ORDERS_CACHE = CACHE_DIR / "orders"
LOGS_CACHE = CACHE_DIR / "bot_logs"
POSITIONS_CACHE = CACHE_DIR / "positions"

ET = ZoneInfo("US/Eastern")
UTC = timezone.utc

# Workflow names for each bot
WORKFLOWS = {
    "15m_stock": "rubberband-live-loop-am.yml",
    "15m_options": "rubberband-options-spreads.yml",
    "weekly_stock": "weekly-stock-live.yml",
    "weekly_options": "weekly-options-live.yml",
}

# ==============================================================================
# Alpaca API
# ==============================================================================

def _alpaca_creds() -> tuple:
    """Get Alpaca credentials from environment."""
    base = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    key = os.environ.get("APCA_API_KEY_ID", "")
    secret = os.environ.get("APCA_API_SECRET_KEY", "")
    if not key or not secret:
        # Try alternative names
        key = os.environ.get("ALPACA_KEY_ID", "")
        secret = os.environ.get("ALPACA_SECRET_KEY", "")
    return base, key, secret


def _alpaca_headers(key: str, secret: str) -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Content-Type": "application/json",
    }


def fetch_broker_orders(date: str, force_refresh: bool = False) -> List[Dict]:
    """
    Fetch all orders for a given date from Alpaca.
    
    Args:
        date: Date string in YYYY-MM-DD format
        force_refresh: If True, fetch even if cache exists
    
    Returns:
        List of order dictionaries
    """
    cache_file = ORDERS_CACHE / f"{date}.json"
    
    # Check cache first
    if cache_file.exists() and not force_refresh:
        print(f"[cache] Loading orders from {cache_file}")
        with open(cache_file, "r") as f:
            return json.load(f)
    
    # Fetch from API
    base, key, secret = _alpaca_creds()
    if not key:
        print("[ERROR] Alpaca credentials not found in environment")
        return []
    
    # Calculate date range (full day in UTC)
    start_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt = start_dt + timedelta(days=1)
    
    url = f"{base}/v2/orders"
    params = {
        "status": "all",
        "after": start_dt.isoformat(),
        "until": end_dt.isoformat(),
        "limit": 500,  # Max allowed
        "nested": "true",  # Include child orders for brackets
    }
    
    print(f"[fetch] Fetching orders for {date} from Alpaca...")
    try:
        resp = requests.get(url, headers=_alpaca_headers(key, secret), params=params, timeout=30)
        if resp.status_code != 200:
            print(f"[ERROR] Alpaca API returned {resp.status_code}: {resp.text}")
            return []
        
        orders = resp.json()
        print(f"[fetch] Retrieved {len(orders)} orders")
        
        # Save to cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(orders, f, indent=2)
        print(f"[cache] Saved to {cache_file}")
        
        return orders
    except Exception as e:
        print(f"[ERROR] Failed to fetch orders: {e}")
        return []


def fetch_broker_positions() -> List[Dict]:
    """
    Fetch current open positions from Alpaca.
    """
    base, key, secret = _alpaca_creds()
    if not key:
        print("[ERROR] Alpaca credentials not found in environment")
        return []
    
    url = f"{base}/v2/positions"
    
    try:
        resp = requests.get(url, headers=_alpaca_headers(key, secret), timeout=30)
        if resp.status_code != 200:
            print(f"[ERROR] Alpaca API returned {resp.status_code}: {resp.text}")
            return []
        return resp.json()
    except Exception as e:
        print(f"[ERROR] Failed to fetch positions: {e}")
        return []


# ==============================================================================
# GitHub Artifacts
# ==============================================================================

def fetch_bot_logs(date: str, force_refresh: bool = False) -> Dict[str, List[Dict]]:
    """
    Fetch bot logs for a given date from GitHub artifacts.
    
    Args:
        date: Date string in YYYY-MM-DD format
        force_refresh: If True, re-download even if cache exists
    
    Returns:
        Dict mapping bot_tag to list of log entries
    """
    date_cache = LOGS_CACHE / date
    
    # Check if we already have logs for all workflows
    if date_cache.exists() and not force_refresh:
        existing = list(date_cache.glob("*.jsonl")) + list(date_cache.glob("*.json"))
        if len(existing) >= len(WORKFLOWS):
            print(f"[cache] Loading logs from {date_cache}")
            return _load_cached_logs(date_cache)
    
    # Parse target date
    target_date = datetime.strptime(date, "%Y-%m-%d").date()
    
    date_cache.mkdir(parents=True, exist_ok=True)
    logs = {}
    
    for bot_tag, workflow_file in WORKFLOWS.items():
        print(f"[fetch] Checking runs for {bot_tag} on {date}...")
        
        # Get runs for this workflow
        runs = _get_workflow_runs(workflow_file, limit=10)
        
        # Filter to runs that started on target date (UTC)
        matching_runs = []
        for run in runs:
            started = run.get("startedAt", "")
            if started:
                run_date = datetime.fromisoformat(started.replace("Z", "+00:00")).date()
                if run_date == target_date:
                    matching_runs.append(run)
        
        if not matching_runs:
            print(f"[fetch] No runs found for {bot_tag} on {date}")
            continue
        
        print(f"[fetch] Found {len(matching_runs)} run(s) for {bot_tag}")
        
        # Download artifacts for each matching run
        for run in matching_runs:
            run_id = run.get("databaseId")
            if not run_id:
                continue
            
            artifact_dir = date_cache / f"{bot_tag}_{run_id}"
            if artifact_dir.exists() and not force_refresh:
                print(f"[cache] Artifact already exists: {artifact_dir}")
            else:
                print(f"[fetch] Downloading artifacts for run {run_id}...")
                _download_run_artifacts(run_id, artifact_dir)
        
        # Load logs from all downloaded artifacts for this bot
        logs[bot_tag] = _load_bot_logs_from_cache(date_cache, bot_tag)
    
    return logs


def _get_workflow_runs(workflow_file: str, limit: int = 10) -> List[Dict]:
    """Query GitHub for recent workflow runs."""
    cmd = [
        "gh", "run", "list",
        f"--workflow={workflow_file}",
        f"--limit={limit}",
        "--json", "databaseId,startedAt,conclusion,status"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"[WARN] gh run list failed: {result.stderr}")
            return []
        return json.loads(result.stdout)
    except Exception as e:
        print(f"[ERROR] Failed to query workflow runs: {e}")
        return []


def _download_run_artifacts(run_id: int, output_dir: Path) -> bool:
    """Download artifacts for a specific run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = ["gh", "run", "download", str(run_id), "--dir", str(output_dir)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"[WARN] gh run download failed: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download artifacts: {e}")
        return False


def _load_cached_logs(cache_dir: Path) -> Dict[str, List[Dict]]:
    """Load all cached logs from a date directory."""
    logs = {}
    
    for bot_tag in WORKFLOWS.keys():
        logs[bot_tag] = _load_bot_logs_from_cache(cache_dir, bot_tag)
    
    return logs


def _load_bot_logs_from_cache(cache_dir: Path, bot_tag: str) -> List[Dict]:
    """Load logs for a specific bot from cache."""
    entries = []
    
    # Find all directories matching this bot
    for subdir in cache_dir.iterdir():
        if not subdir.is_dir():
            continue
        if not subdir.name.startswith(bot_tag):
            continue
        
        # Look for JSONL files (trade logs)
        for jsonl_file in subdir.rglob("*.jsonl"):
            try:
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entries.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            except Exception as e:
                print(f"[WARN] Failed to read {jsonl_file}: {e}")
        
        # Look for JSON files (position registry, trade trackers)
        for json_file in subdir.rglob("*.json"):
            if "console" in json_file.name:
                continue  # Skip console logs
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        entries.extend(data)
                    elif isinstance(data, dict):
                        # Might be a position registry or tracker
                        if "positions" in data:
                            entries.append({"type": "REGISTRY", "data": data})
                        else:
                            entries.append(data)
            except Exception as e:
                print(f"[WARN] Failed to read {json_file}: {e}")
        
        # Parse console.log for embedded JSON lines
        for console_file in subdir.rglob("console.log"):
            entries.extend(_parse_console_log(console_file, bot_tag))
    
    return entries


def _parse_console_log(console_file: Path, bot_tag: str) -> List[Dict]:
    """
    Parse console.log to extract embedded JSON records.
    
    The console.log contains a mix of:
    - Plain text log lines
    - JSON objects on their own lines
    
    We extract JSON objects that represent trade events.
    """
    entries = []
    
    try:
        # Try UTF-16 first (Windows PowerShell default), then UTF-8
        try:
            with open(console_file, "r", encoding="utf-16") as f:
                content = f.read()
        except (UnicodeDecodeError, UnicodeError):
            with open(console_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            
            # Skip common non-JSON lines
            if line.startswith("[") and not line.startswith('{"'):
                # Could be [INFO], [loop], etc.
                continue
            
            # Try to parse as JSON
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    # Add bot_tag if not present
                    if "bot_tag" not in obj:
                        obj["_source"] = bot_tag
                    entries.append(obj)
                except json.JSONDecodeError:
                    pass
    
    except Exception as e:
        print(f"[WARN] Failed to parse console.log {console_file}: {e}")
    
    return entries


# ==============================================================================
# Attribution & Analysis
# ==============================================================================

def is_option_symbol(symbol: str) -> bool:
    """Check if a symbol is an OCC option symbol (e.g., AAPL260117C00150000)."""
    if not symbol:
        return False
    # OCC symbols are 21 characters: SYMBOL(6) + YYMMDD(6) + C/P(1) + STRIKE(8)
    # But Alpaca sometimes has shorter symbols too
    return len(symbol) > 10 and (symbol[-9] in ('C', 'P'))


def categorize_order(order: Dict) -> str:
    """
    Categorize an order by bot type based on client_order_id patterns.
    
    Actual Patterns (from orders):
    - 15M_STK_<SYM>_<TS>: 15m Stock Bot
    - 15M_OPT_...: 15m Options Bot 
    - WK_STK_...: Weekly Stock Bot
    - WK_OPT_...: Weekly Options Bot
    - RB_... (legacy): 15m Stock Bot
    """
    cid = order.get("client_order_id", "") or ""
    symbol = order.get("symbol", "")
    
    # Check client order ID patterns first
    cid_upper = cid.upper()
    
    # Primary patterns (current bot naming)
    if cid_upper.startswith("15M_STK"):
        return "15m_stock"
    elif cid_upper.startswith("15M_OPT"):
        return "15m_options"
    elif cid_upper.startswith("WK_STK"):
        return "weekly_stock"
    elif cid_upper.startswith("WK_OPT"):
        return "weekly_options"
    
    # Legacy patterns
    elif cid_upper.startswith("RB_"):
        return "15m_stock"
    elif cid_upper.startswith(("OPT_", "SPREAD_")):
        return "15m_options"
    elif cid_upper.startswith(("WEEKLY_", "WKO_", "WKOPT")):
        return "weekly_options"
    
    # Fallback: Check symbol type
    if is_option_symbol(symbol):
        return "options_unknown"
    else:
        return "stock_unknown"


def analyze_orders(orders: List[Dict], positions: List[Dict]) -> Dict[str, Any]:
    """
    Analyze orders and calculate PnL by bot category.
    
    Strategy:
    1. First pass: categorize all orders and group by symbol
    2. For symbols with orphan sells (sell CID unknown, but buy has known category),
       attribute the sell to the buy's category
    3. Calculate realized PnL for matched buy/sell pairs
    """
    results = {
        "15m_stock": {"trades": [], "filled_orders": [], "realized_pnl": 0.0, "winners": 0, "losers": 0},
        "15m_options": {"trades": [], "filled_orders": [], "realized_pnl": 0.0, "winners": 0, "losers": 0},
        "weekly_stock": {"trades": [], "filled_orders": [], "realized_pnl": 0.0, "winners": 0, "losers": 0},
        "weekly_options": {"trades": [], "filled_orders": [], "realized_pnl": 0.0, "winners": 0, "losers": 0},
        "unknown": {"trades": [], "filled_orders": [], "realized_pnl": 0.0, "winners": 0, "losers": 0},
    }
    
    # Group filled orders by symbol
    symbol_data = {}  # symbol -> {buys: [], sells: [], category: str}
    
    for order in orders:
        status = order.get("status", "")
        if status not in ("filled", "partially_filled"):
            continue
        
        symbol = order.get("symbol", "")
        side = order.get("side", "")
        qty = float(order.get("filled_qty", 0) or order.get("qty", 0))
        avg_price = float(order.get("filled_avg_price", 0) or 0)
        category = categorize_order(order)
        
        if symbol not in symbol_data:
            symbol_data[symbol] = {"buys": [], "sells": [], "buy_category": None, "sell_category": None}
        
        entry = {"qty": qty, "price": avg_price, "category": category, "cid": order.get("client_order_id", "")}
        
        if side == "buy":
            symbol_data[symbol]["buys"].append(entry)
            # Track the known category from buy orders
            if category not in ("stock_unknown", "options_unknown", "unknown"):
                symbol_data[symbol]["buy_category"] = category
        else:
            symbol_data[symbol]["sells"].append(entry)
            if category not in ("stock_unknown", "options_unknown", "unknown"):
                symbol_data[symbol]["sell_category"] = category
    
    # Calculate realized PnL per symbol
    for symbol, data in symbol_data.items():
        # Determine effective category: prefer buy category (entry bot), fallback to sell
        effective_category = data["buy_category"] or data["sell_category"]
        if not effective_category:
            effective_category = "unknown"
        if effective_category.endswith("_unknown"):
            effective_category = "unknown"
        
        total_buy_cost = sum(b["qty"] * b["price"] for b in data["buys"])
        total_buy_qty = sum(b["qty"] for b in data["buys"])
        total_sell_proceeds = sum(s["qty"] * s["price"] for s in data["sells"])
        total_sell_qty = sum(s["qty"] for s in data["sells"])
        
        # Record filled orders for display
        for b in data["buys"]:
            results[effective_category]["filled_orders"].append({
                "symbol": symbol, "side": "buy", "qty": b["qty"], "price": b["price"], "cid": b["cid"]
            })
        for s in data["sells"]:
            results[effective_category]["filled_orders"].append({
                "symbol": symbol, "side": "sell", "qty": s["qty"], "price": s["price"], "cid": s["cid"]
            })
        
        # Calculate matched pairs PnL
        matched_qty = min(total_buy_qty, total_sell_qty)
        if matched_qty > 0 and total_buy_qty > 0 and total_sell_qty > 0:
            avg_buy = total_buy_cost / total_buy_qty
            avg_sell = total_sell_proceeds / total_sell_qty
            
            # For options, multiply by 100
            multiplier = 100 if is_option_symbol(symbol) else 1
            pnl = (avg_sell - avg_buy) * matched_qty * multiplier
            
            results[effective_category]["realized_pnl"] += pnl
            results[effective_category]["trades"].append({
                "symbol": symbol,
                "qty": matched_qty,
                "pnl": pnl,
            })
            
            if pnl >= 0:
                results[effective_category]["winners"] += 1
            else:
                results[effective_category]["losers"] += 1
    
    return results


def get_open_positions_by_category(positions: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize open positions by likely bot source."""
    categorized = {
        "stocks": [],
        "options": [],
    }
    
    for pos in positions:
        symbol = pos.get("symbol", "")
        entry = {
            "symbol": symbol,
            "qty": float(pos.get("qty", 0)),
            "avg_entry": float(pos.get("avg_entry_price", 0)),
            "current_price": float(pos.get("current_price", 0)),
            "unrealized_pnl": float(pos.get("unrealized_pl", 0)),
            "unrealized_pnl_pct": float(pos.get("unrealized_plpc", 0) or 0) * 100,
        }
        
        if is_option_symbol(symbol):
            categorized["options"].append(entry)
        else:
            categorized["stocks"].append(entry)
    
    return categorized


# ==============================================================================
# Pretty Output
# ==============================================================================

def print_report(date: str, analysis: Dict, positions: Dict):
    """Print a nicely formatted performance report."""
    
    # Header
    print("\n" + "╔" + "═" * 66 + "╗")
    print("║" + f"  📊 DAILY PERFORMANCE: {date}".center(66) + "║")
    print("╚" + "═" * 66 + "╝")
    
    total_realized = 0.0
    total_unrealized = 0.0
    
    # 15m Stock Bot
    stk = analysis["15m_stock"]
    total_trades = stk["winners"] + stk["losers"]
    win_rate = (stk["winners"] / total_trades * 100) if total_trades > 0 else 0
    print("\n┌" + "─" * 66 + "┐")
    print("│  🏦 15-MINUTE STOCK BOT".ljust(67) + "│")
    print("├" + "─" * 66 + "┤")
    print(f"│  Trades: {total_trades:<8} Winners: {stk['winners']} ({win_rate:.0f}%)      Losers: {stk['losers']:<8}│")
    print(f"│  Realized PnL: {'${:+,.2f}'.format(stk['realized_pnl']):<50}│")
    if stk["trades"]:
        top = sorted(stk["trades"], key=lambda x: x["pnl"], reverse=True)
        winners = [f"{t['symbol']} ${t['pnl']:+.0f}" for t in top[:3] if t["pnl"] > 0]
        losers = [f"{t['symbol']} ${t['pnl']:+.0f}" for t in top[-3:] if t["pnl"] < 0]
        if winners:
            print(f"│  Top Winners: {', '.join(winners):<50}│")
        if losers:
            print(f"│  Top Losers: {', '.join(losers):<51}│")
    print("└" + "─" * 66 + "┘")
    total_realized += stk["realized_pnl"]
    
    # 15m Options Bot
    opt = analysis["15m_options"]
    total_opt_trades = opt["winners"] + opt["losers"]
    print("\n┌" + "─" * 66 + "┐")
    print("│  📈 15-MINUTE OPTIONS BOT".ljust(67) + "│")
    print("├" + "─" * 66 + "┤")
    print(f"│  Closed Trades: {total_opt_trades:<10} Realized PnL: {'${:+,.2f}'.format(opt['realized_pnl']):<25}│")
    open_opts = positions.get("options", [])
    if open_opts:
        unrealized = sum(p["unrealized_pnl"] for p in open_opts)
        print(f"│  Open Positions: {len(open_opts):<9} Unrealized: {'${:+,.2f}'.format(unrealized):<28}│")
        total_unrealized += unrealized
        print("│" + " " * 67 + "│")
        for p in open_opts[:5]:  # Show first 5
            sym_short = p["symbol"][:20]
            line = f"│    {sym_short:<20} | Entry: ${p['avg_entry']:<6.2f} | Now: ${p['current_price']:<6.2f} | ${p['unrealized_pnl']:+.0f}"
            print(line.ljust(67) + "│")
    print("└" + "─" * 66 + "┘")
    total_realized += opt["realized_pnl"]
    
    # Weekly Stock Bot
    wk_stk = analysis["weekly_stock"]
    total_wk_trades = wk_stk["winners"] + wk_stk["losers"]
    print("\n┌" + "─" * 66 + "┐")
    print("│  📅 WEEKLY STOCK BOT".ljust(67) + "│")
    print("├" + "─" * 66 + "┤")
    print(f"│  Trades: {total_wk_trades:<10} Realized PnL: {'${:+,.2f}'.format(wk_stk['realized_pnl']):<29}│")
    if wk_stk["filled_orders"]:
        entries = [o["symbol"] for o in wk_stk["filled_orders"] if o["side"] == "buy"][:5]
        exits = [o["symbol"] for o in wk_stk["filled_orders"] if o["side"] == "sell"][:5]
        if entries:
            print(f"│  Entries: {', '.join(set(entries)):<55}│")
        if exits:
            print(f"│  Exits: {', '.join(set(exits)):<57}│")
    print("└" + "─" * 66 + "┘")
    total_realized += wk_stk["realized_pnl"]
    
    # Weekly Options Bot
    wk_opt = analysis["weekly_options"]
    total_wko_trades = wk_opt["winners"] + wk_opt["losers"]
    print("\n┌" + "─" * 66 + "┐")
    print("│  📅 WEEKLY OPTIONS BOT".ljust(67) + "│")
    print("├" + "─" * 66 + "┤")
    print(f"│  Trades: {total_wko_trades:<10} Realized PnL: {'${:+,.2f}'.format(wk_opt['realized_pnl']):<29}│")
    print("└" + "─" * 66 + "┘")
    total_realized += wk_opt["realized_pnl"]
    
    # Stock positions summary
    stock_positions = positions.get("stocks", [])
    if stock_positions:
        stock_unrealized = sum(p["unrealized_pnl"] for p in stock_positions)
        total_unrealized += stock_unrealized
    
    # Totals
    print("\n╔" + "═" * 66 + "╗")
    print("║  💰 TOTALS".ljust(67) + "║")
    print("╠" + "═" * 66 + "╣")
    print(f"║  Realized Today:   {'${:+,.2f}'.format(total_realized):<45}║")
    print(f"║  Unrealized:       {'${:+,.2f}'.format(total_unrealized):<45}║")
    print(f"║  Net:              {'${:+,.2f}'.format(total_realized + total_unrealized):<45}║")
    print("╚" + "═" * 66 + "╝")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Reconciliation Tool: Daily PnL Analysis")
    parser.add_argument("--date", type=str, default=None, 
                        help="Date to analyze (YYYY-MM-DD). Default: today")
    parser.add_argument("--refresh", action="store_true",
                        help="Force refresh cached data")
    parser.add_argument("--deep-clean", action="store_true",
                        help="Clear all cache and rebuild")
    parser.add_argument("--backfill", type=int, default=0,
                        help="Backfill cache for N past trading days")
    args = parser.parse_args()
    
    # Backfill mode
    if args.backfill > 0:
        print(f"\n{'='*60}")
        print(f"  Backfilling cache for {args.backfill} trading days")
        print(f"{'='*60}\n")
        
        from datetime import date
        import holidays
        
        us_holidays = holidays.US(years=[2025, 2026])
        current = datetime.now(ET).date()
        days_fetched = 0
        days_checked = 0
        
        while days_fetched < args.backfill and days_checked < 30:
            days_checked += 1
            current = current - timedelta(days=1)
            
            # Skip weekends
            if current.weekday() >= 5:
                continue
            # Skip holidays
            if current in us_holidays:
                continue
            
            date_str = current.strftime("%Y-%m-%d")
            print(f"\n[backfill] Processing {date_str}...")
            
            fetch_broker_orders(date_str, force_refresh=False)
            fetch_bot_logs(date_str, force_refresh=False)
            days_fetched += 1
        
        print(f"\n[backfill] Complete! Cached {days_fetched} trading days.")
        return
    
    # Determine target date
    if args.date:
        target_date = args.date
    else:
        # Default to today (Eastern Time)
        target_date = datetime.now(ET).strftime("%Y-%m-%d")
    
    print(f"\n{'='*60}")
    print(f"  Reconciliation Tool - {target_date}")
    print(f"{'='*60}\n")
    
    # Deep clean if requested
    if args.deep_clean:
        print("[clean] Removing all cached data...")
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(parents=True)
        ORDERS_CACHE.mkdir(parents=True)
        LOGS_CACHE.mkdir(parents=True)
        POSITIONS_CACHE.mkdir(parents=True)
    
    # Fetch data
    force = args.refresh or args.deep_clean
    
    orders = fetch_broker_orders(target_date, force_refresh=force)
    print(f"[data] Orders loaded: {len(orders)}")
    
    logs = fetch_bot_logs(target_date, force_refresh=force)
    for bot, entries in logs.items():
        print(f"[data] Logs for {bot}: {len(entries)} entries")
    
    positions = fetch_broker_positions()
    print(f"[data] Current positions: {len(positions)}")
    
    # Analyze
    analysis = analyze_orders(orders, positions)
    categorized_positions = get_open_positions_by_category(positions)
    
    # Print report
    print_report(target_date, analysis, categorized_positions)


if __name__ == "__main__":
    main()

