import os
import json
import math
import time
import requests
import pandas as pd
import datetime as dt
from typing import List, Dict, Any, Optional, Tuple, Iterable

# ──────────────────────────────────────────────────────────────────────────────
# Global / Constants
# ──────────────────────────────────────────────────────────────────────────────
ISO_UTC = "%Y-%m-%dT%H:%M:%SZ"

def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def _iso_utc(ts: Optional[dt.datetime] = None) -> str:
    ts = ts or _now_utc()
    return ts.strftime(ISO_UTC)

def _minutes(td: dt.timedelta) -> int:
    return int(td.total_seconds() // 60)

def load_symbols_from_file(path: str) -> List[str]:
    """Load symbols from a text file (one per line), filtering comments and deduping."""
    if not os.path.exists(path):
        return []
    seen = set()
    tickers = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ticker = ln.strip().upper()
            # Skip empty lines, comments, and triple-quote markers
            if not ticker or ticker.startswith("#") or ticker.startswith("'''") or ticker.startswith('"""'):
                continue
            # Dedupe while preserving order
            if ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)
    return tickers

# ──────────────────────────────────────────────────────────────────────────────
# Creds / Alpaca helpers
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_key_secret(key: Optional[str], secret: Optional[str]) -> Tuple[str, str]:
    """
    Prefer explicit args; otherwise fall back to common env names.
    """
    k = (key or os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_KEY_ID") or "").strip()
    s = (secret or os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY") or "").strip()
    return k, s

def _alpaca_headers(key: Optional[str], secret: Optional[str]) -> Dict[str, str]:
    k, s = _resolve_key_secret(key, secret)
    return {
        "APCA-API-KEY-ID": k,
        "APCA-API-SECRET-KEY": s,
        "Content-Type": "application/json",
    }

def _base_url_from_env(base_url: Optional[str] = None) -> str:
    # Support all common envs; prefer APCA_API_BASE_URL if set.
    return (
        (base_url or os.getenv("APCA_API_BASE_URL") or os.getenv("APCA_BASE_URL") or os.getenv("ALPACA_BASE_URL")
         or "https://paper-api.alpaca.markets")
        .rstrip("/")
    )

def alpaca_market_open(base_url: Optional[str] = None, key: Optional[str] = None, secret: Optional[str] = None) -> bool:
    base = _base_url_from_env(base_url)
    try:
        r = requests.get(f"{base}/v2/clock", headers=_alpaca_headers(key, secret), timeout=10)
        if r.status_code == 401:
            return False
        r.raise_for_status()
        j = r.json() or {}
        return bool(j.get("is_open"))
    except Exception as e:
        print(f"[warn] alpaca_market_open error: {type(e).__name__}: {e}")
        return False

# Risk ops
def cancel_all_orders(base_url: Optional[str], key: Optional[str], secret: Optional[str]) -> Dict[str, Any]:
    base = _base_url_from_env(base_url)
    r = requests.delete(f"{base}/v2/orders", headers=_alpaca_headers(key, secret), timeout=15)
    if r.status_code not in (200, 204):
        r.raise_for_status()
    return {"ok": True}


def cancel_orders_for_bots(
    base_url: Optional[str],
    key: Optional[str],
    secret: Optional[str],
    bot_prefixes: List[str],
) -> Dict[str, Any]:
    """Cancel only orders belonging to specified bot tags, preserving all others.

    Uses nested order queries to trace bracket/OCO legs back to their parent
    order's client_order_id for accurate bot attribution.  This prevents the
    EOD flatten from destroying weekly-hold bracket legs or safety-check OCOs.

    Args:
        base_url: Alpaca API base URL (or None to use env).
        key: API key (or None to use env).
        secret: API secret (or None to use env).
        bot_prefixes: Bot tag prefixes whose orders should be cancelled
                      (e.g. ``["15M_STK", "15M_OPT"]``).

    Returns:
        Dict with ``cancelled``, ``preserved`` counts and ``errors`` list.
    """
    base = _base_url_from_env(base_url)
    H = _alpaca_headers(key, secret)

    # --- Step 1: map bracket/OCO leg IDs to intraday parents ---------------
    # Query recent orders (last 3 days) with nested=true so legs appear under
    # their parent.  This lets us attribute UUID-based leg IDs to the bot that
    # placed the parent order.  3-day window keeps us well under the 500 limit
    # while covering any intraday bracket that could still have open legs.
    lookback = (_now_utc() - dt.timedelta(days=3)).strftime("%Y-%m-%dT00:00:00Z")
    try:
        r = requests.get(
            f"{base}/v2/orders", headers=H,
            params={"status": "all", "limit": 500, "nested": "true", "after": lookback},
            timeout=30,
        )
        r.raise_for_status()
        all_orders_nested = r.json() or []
    except Exception as e:
        print(f"[eod] Failed to fetch nested orders for leg mapping: {e}")
        return {"cancelled": 0, "preserved": 0, "errors": [str(e)]}

    if len(all_orders_nested) >= 500:
        print(f"[eod] WARNING: nested order query hit 500-order limit — "
              "some leg mappings may be incomplete (unmapped legs are preserved)")

    intraday_order_ids: set = set()
    for order in all_orders_nested:
        coid = order.get("client_order_id", "")
        if any(coid.startswith(f"{p}_") for p in bot_prefixes):
            intraday_order_ids.add(order.get("id"))
            for leg in order.get("legs", []):
                intraday_order_ids.add(leg.get("id"))

    # --- Step 2: fetch open orders (flat) ----------------------------------
    try:
        r = requests.get(
            f"{base}/v2/orders", headers=H,
            params={"status": "open", "limit": 500},
            timeout=15,
        )
        r.raise_for_status()
        open_orders = r.json() or []
    except Exception as e:
        print(f"[eod] Failed to fetch open orders: {e}")
        return {"cancelled": 0, "preserved": 0, "errors": [str(e)]}

    # --- Step 3: cancel only intraday orders --------------------------------
    cancelled = 0
    preserved_orders: List[Dict[str, str]] = []
    errors: List[str] = []

    for order in open_orders:
        oid = order.get("id", "")
        coid = order.get("client_order_id", "")
        symbol = order.get("symbol", "")

        by_id = oid in intraday_order_ids
        by_prefix = any(coid.startswith(f"{p}_") for p in bot_prefixes)

        if by_id or by_prefix:
            result = cancel_order_by_id(base_url, key, secret, oid)
            if result.get("error"):
                # Single retry for transient failures
                time.sleep(0.3)
                result = cancel_order_by_id(base_url, key, secret, oid)
            if result.get("error"):
                errors.append(f"{symbol} ({oid}): {result['error']}")
                print(f"[eod] FAILED to cancel: {symbol} (coid={coid[:40]})")
            else:
                cancelled += 1
                print(f"[eod] Cancelled: {symbol} (coid={coid[:40]})")
        else:
            preserved_orders.append({"symbol": symbol, "client_order_id": coid})
            print(f"[eod] Preserved: {symbol} (coid={coid[:40]})")

    return {
        "cancelled": cancelled,
        "preserved": len(preserved_orders),
        "preserved_orders": preserved_orders,
        "errors": errors,
    }

def close_all_positions(base_url: Optional[str], key: Optional[str], secret: Optional[str]) -> Dict[str, Any]:
    base = _base_url_from_env(base_url)
    r = requests.delete(f"{base}/v2/positions", headers=_alpaca_headers(key, secret), timeout=20)
    if r.status_code not in (200, 204):
        r.raise_for_status()
    return {"ok": True}

def get_account_info_compat(base_url: Optional[str], key: Optional[str], secret: Optional[str]) -> Optional[Dict[str, Any]]:
    """Helper to fetch account info (equity, buying power)."""
    base = _base_url_from_env(base_url)
    try:
        r = requests.get(f"{base}/v2/account", headers=_alpaca_headers(key, secret), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[warn] Failed to get account info: {e}")
        return None

def close_position(base_url: Optional[str], key: Optional[str], secret: Optional[str], symbol: str) -> Dict[str, Any]:
    """Close a single position by symbol."""
    base = _base_url_from_env(base_url)
    r = requests.delete(f"{base}/v2/positions/{symbol}", headers=_alpaca_headers(key, secret), timeout=15)
    if r.status_code == 404:
        return {"ok": True, "message": "Position not found"}  # Already closed
    if r.status_code not in (200, 204):
        r.raise_for_status()
    return {"ok": True}

    return {"ok": True}

def get_account(base_url: Optional[str] = None, key: Optional[str] = None, secret: Optional[str] = None) -> Dict[str, Any]:
    """Get account details including equity and balance."""
    base = _base_url_from_env(base_url)
    try:
        r = requests.get(f"{base}/v2/account", headers=_alpaca_headers(key, secret), timeout=10)
        r.raise_for_status()
        return r.json() or {}
    except Exception as e:
        print(f"[warn] get_account error: {type(e).__name__}: {e}")
        return {}

def get_positions(base_url: Optional[str] = None, key: Optional[str] = None, secret: Optional[str] = None) -> List[Dict[str, Any]]:
    base = _base_url_from_env(base_url)
    try:
        r = requests.get(f"{base}/v2/positions", headers=_alpaca_headers(key, secret), timeout=12)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        arr = r.json() or []
        # Ensure list-of-dicts
        return arr if isinstance(arr, list) else []
    except Exception as e:
        print(f"[warn] get_positions error: {type(e).__name__}: {e}")
        return []

def get_daily_fills(
    base_url: Optional[str] = None, 
    key: Optional[str] = None, 
    secret: Optional[str] = None,
    bot_tag: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch all filled orders for the current UTC day.
    
    Args:
        base_url: Alpaca API base URL
        key: API key
        secret: API secret
        bot_tag: Optional bot tag prefix to filter by client_order_id (e.g., "15M_STK")
                 Also includes sells for symbols that have tagged buys (for bracket orders)
    """
    base = _base_url_from_env(base_url)
    # Start of today UTC
    today = _now_utc().strftime("%Y-%m-%d")
    params = {
        "status": "closed", # We want filled orders (which are 'closed')
        "limit": 500,
        "after": f"{today}T00:00:00Z"
    }
    try:
        r = requests.get(f"{base}/v2/orders", headers=_alpaca_headers(key, secret), params=params, timeout=15)
        r.raise_for_status()
        orders = r.json() or []
        # Filter for filled only
        fills = [o for o in orders if o.get("status") == "filled" and o.get("filled_qty") is not None]
        
        # Filter by bot_tag prefix if specified
        if bot_tag:
            # Step 1: Find fills with the bot_tag prefix (tagged orders)
            tagged_fills = [f for f in fills if (f.get("client_order_id") or "").startswith(f"{bot_tag}_")]
            
            # Step 2: Find symbols that have tagged BUY orders
            tagged_buy_symbols = set()
            for f in tagged_fills:
                if f.get("side") == "buy":
                    tagged_buy_symbols.add(f.get("symbol", ""))
            
            # Step 3: Include sells for those symbols (bracket order exits without bot tag)
            additional_sells = []
            for f in fills:
                if f.get("side") == "sell" and f.get("symbol", "") in tagged_buy_symbols:
                    coid = f.get("client_order_id") or ""
                    if not coid.startswith(f"{bot_tag}_"):  # Not already in tagged_fills
                        additional_sells.append(f)
            
            fills = tagged_fills + additional_sells
        
        return fills
    except Exception as e:
        print(f"[warn] Failed to fetch daily fills: {e}")
        return []

# ──────────────────────────────────────────────────────────────────────────────
# Kill Switch & Safety Functions
# ──────────────────────────────────────────────────────────────────────────────
class KillSwitchTriggered(Exception):
    """Raised when daily loss exceeds threshold."""
    pass

def get_order_by_id(
    base_url: Optional[str] = None, 
    key: Optional[str] = None, 
    secret: Optional[str] = None,
    order_id: str = ""
) -> Dict[str, Any]:
    """Get order details by Alpaca order ID."""
    base = _base_url_from_env(base_url)
    try:
        r = requests.get(f"{base}/v2/orders/{order_id}", headers=_alpaca_headers(key, secret), timeout=10)
        if r.status_code == 404:
            return {"error": "not_found"}
        r.raise_for_status()
        return r.json() or {}
    except Exception as e:
        print(f"[warn] Failed to get order {order_id}: {e}")
        return {"error": str(e)}

def cancel_order_by_id(
    base_url: Optional[str] = None,
    key: Optional[str] = None,
    secret: Optional[str] = None,
    order_id: str = ""
) -> Dict[str, Any]:
    """Cancel an order by ID."""
    base = _base_url_from_env(base_url)
    try:
        r = requests.delete(f"{base}/v2/orders/{order_id}", headers=_alpaca_headers(key, secret), timeout=10)
        if r.status_code == 404:
            return {"ok": True, "message": "already_canceled"}
        if r.status_code in (200, 204):
            return {"ok": True}
        r.raise_for_status()
        return {"ok": True}
    except Exception as e:
        print(f"[warn] Failed to cancel order {order_id}: {e}")
        return {"error": str(e)}

def get_latest_quote(
    base_url: Optional[str] = None,
    key: Optional[str] = None,
    secret: Optional[str] = None,
    symbol: str = ""
) -> Dict[str, Any]:
    """Get latest bid/ask quote for a symbol."""
    base = _base_url_from_env(base_url)
    # Use data API for quotes
    data_url = base.replace("api.alpaca.markets", "data.alpaca.markets")
    if "paper-api" in base:
        data_url = "https://data.alpaca.markets"
    
    try:
        r = requests.get(
            f"{data_url}/v2/stocks/{symbol}/quotes/latest",
            headers=_alpaca_headers(key, secret),
            timeout=10
        )
        r.raise_for_status()
        quote = r.json().get("quote", {})
        return {
            "bid": float(quote.get("bp", 0)),
            "ask": float(quote.get("ap", 0)),
            "bid_size": int(quote.get("bs", 0)),
            "ask_size": int(quote.get("as", 0)),
        }
    except Exception as e:
        print(f"[warn] Failed to get quote for {symbol}: {e}")
        return {"bid": 0, "ask": 0, "error": str(e)}

def order_exists_today(
    base_url: Optional[str] = None,
    key: Optional[str] = None,
    secret: Optional[str] = None,
    client_order_id: str = ""
) -> bool:
    """Check if an order with this client_order_id already exists today."""
    base = _base_url_from_env(base_url)
    today = _now_utc().strftime("%Y-%m-%d")
    params = {
        "status": "all",
        "limit": 500,
        "after": f"{today}T00:00:00Z"
    }
    try:
        r = requests.get(f"{base}/v2/orders", headers=_alpaca_headers(key, secret), params=params, timeout=15)
        r.raise_for_status()
        orders = r.json() or []
        return any(o.get("client_order_id") == client_order_id for o in orders)
    except Exception as e:
        print(f"[warn] Failed to check order existence: {e}")
        return False  # Assume doesn't exist on error

def calculate_realized_pnl(fills: List[Dict[str, Any]]) -> float:
    """
    Calculate realized P&L from a list of fills.
    
    CRITICAL: Only counts P&L for MATCHED buy/sell pairs (closed positions).
    Unmatched buys (open positions) are NOT counted as losses.
    """
    pnl = 0.0
    # Group by symbol
    symbol_fills: Dict[str, List[Dict[str, Any]]] = {}
    for f in fills:
        sym = f.get("symbol", "")
        if sym:
            symbol_fills.setdefault(sym, []).append(f)
    
    for sym, sym_fills in symbol_fills.items():
        buys = [f for f in sym_fills if f.get("side") == "buy"]
        sells = [f for f in sym_fills if f.get("side") == "sell"]
        
        # Calculate total quantities
        total_buy_qty = sum(int(f.get("filled_qty", 0)) for f in buys)
        total_sell_qty = sum(int(f.get("filled_qty", 0)) for f in sells)
        
        # Only calculate P&L for MATCHED portion (closed positions)
        # If we bought 100 shares and sold 0, realized P&L is $0 (not a loss!)
        matched_qty = min(total_buy_qty, total_sell_qty)
        
        if matched_qty <= 0:
            # No matched trades = no realized P&L for this symbol
            continue
        
        # Calculate weighted average prices
        total_buy_cost = sum(
            float(f.get("filled_avg_price", 0)) * int(f.get("filled_qty", 0))
            for f in buys
        )
        total_sell_proceeds = sum(
            float(f.get("filled_avg_price", 0)) * int(f.get("filled_qty", 0))
            for f in sells
        )
        
        avg_buy_price = total_buy_cost / total_buy_qty if total_buy_qty > 0 else 0
        avg_sell_price = total_sell_proceeds / total_sell_qty if total_sell_qty > 0 else 0
        
        # Realized P&L = (sell_price - buy_price) * matched_qty
        symbol_pnl = (avg_sell_price - avg_buy_price) * matched_qty
        pnl += symbol_pnl
    
    return pnl

def get_daily_invested_capital(
    base_url: Optional[str] = None,
    key: Optional[str] = None,
    secret: Optional[str] = None,
    bot_tag: Optional[str] = None
) -> float:
    """Get total capital invested today by this bot (sum of buy orders)."""
    fills = get_daily_fills(base_url, key, secret, bot_tag)
    buys = [f for f in fills if f.get("side") == "buy"]
    return sum(
        float(f.get("filled_avg_price", 0)) * int(f.get("filled_qty", 0))
        for f in buys
    )


class CapitalLimitExceeded(Exception):
    """Raised when aggregate capital usage exceeds the configured limit."""
    pass


def get_total_invested_capital(
    base_url: Optional[str] = None,
    key: Optional[str] = None,
    secret: Optional[str] = None,
) -> float:
    """
    Get TOTAL capital invested across ALL positions (cost basis sum).
    This is the aggregate amount of money deployed, regardless of which bot placed it.
    """
    try:
        positions = get_positions(base_url, key, secret)
        total = sum(float(p.get("cost_basis", 0)) for p in positions)
        return total
    except Exception as e:
        print(f"[CAPITAL CHECK] Error fetching positions: {e}")
        return 0.0


def check_capital_limit(
    base_url: Optional[str] = None,
    key: Optional[str] = None,
    secret: Optional[str] = None,
    proposed_trade_value: float = 0.0,
    max_capital: float = 100000.0,
    bot_tag: str = "",
) -> None:
    """
    Check if adding this trade would exceed maximum capital limit.
    
    Args:
        proposed_trade_value: Estimated cost of the proposed trade
        max_capital: Maximum total capital to deploy (default $100K)
        bot_tag: Bot identifier for logging
    
    Raises:
        CapitalLimitExceeded: If adding this trade would exceed limit
    """
    current_invested = get_total_invested_capital(base_url, key, secret)
    projected_total = current_invested + proposed_trade_value
    
    utilization_pct = (current_invested / max_capital) * 100 if max_capital > 0 else 0
    
    print(f"[CAPITAL CHECK] {bot_tag}: Current=${current_invested:,.0f} + Trade=${proposed_trade_value:,.0f} = ${projected_total:,.0f} (Max=${max_capital:,.0f}, {utilization_pct:.1f}% utilized)")
    
    if projected_total > max_capital:
        raise CapitalLimitExceeded(
            f"Capital limit exceeded: ${projected_total:,.0f} > ${max_capital:,.0f} "
            f"(current=${current_invested:,.0f}, trade=${proposed_trade_value:,.0f})"
        )


def check_kill_switch(
    base_url: Optional[str] = None,
    key: Optional[str] = None,
    secret: Optional[str] = None,
    bot_tag: str = "",
    max_loss_pct: float = 25.0
) -> bool:
    """
    Check if day's TOTAL P&L (Realized + Unrealized) exceeds max loss %.
    
    CRITICAL FIX (Dec 15, 2025): 
    1. Uses FULL position cost_basis (not just today's buys) as invested capital.
    2. This prevents false >100% loss when holding prior-day positions.
    3. Handles missing API data gracefully (skips check if price unavailable).
    """
    # Get all positions first
    try:
        positions = get_positions(base_url, key, secret)
    except Exception as e:
        print(f"[KILL SWITCH] Error fetching positions: {e}")
        return False  # Fail safe: don't trigger if API down
    
    # Get today's fills for this bot
    fills = get_daily_fills(base_url, key, secret, bot_tag)
    today_symbols = {f.get("symbol") for f in fills if f.get("side") == "buy"}
    
    # --- SAFETY NET: Account-Wide Equity Check ---
    # This catches "invisible losses" from overnight positions that were closed today
    # but not captured by 'today_symbols' or daily fill logic.
    try:
        acct = get_account(base_url, key, secret)
        if acct:
            equity = float(acct.get("equity", 0))
            last_equity = float(acct.get("last_equity", 0))
            
            # Daily PnL at account level
            account_daily_pnl = equity - last_equity
            
            # If we have invested capital (from either method)
            # We use a conservative estimate: if account drops > 25% of its value? 
            # Or > 25% of "invested"? 
            # Let's align with the existing logic: Loss > 25% of *Invested*.
            # But "Invested" is tricky for account-wide.
            # Let's fallback to: if Account Daily Loss > 5% of TOTAL ACCOUNT VALUE (hard stop).
            # This is a "Circuit Breaker".
            
            account_loss_pct = (account_daily_pnl / last_equity) * 100 if last_equity > 0 else 0
            if account_loss_pct < -5.0: # Hard 5% account daily stop
                 print(f"[KILL SWITCH] 🛑 ACCOUNT CRASH PROTECTION ({bot_tag}): Daily Loss {account_loss_pct:.2f}% < -5.0%")
                 return True

            # If we rely on the specific bot tag logic below, we might miss the overnight ones.
            # So we also check if account_daily_pnl is overwhelmingly negative compared to "invested".
    except Exception as e:
        print(f"[KILL SWITCH] ⚠️ Error fetching account equity: {e}")

    if not today_symbols:
        return False  # No activity today for this bot
    
    # 1. Realized P&L from closed trades today
    realized_pnl = calculate_realized_pnl(fills)
    
    # 2. Calculate invested capital and unrealized P&L from FULL positions
    # that we touched today (not just today's portion)
    invested = 0.0
    unrealized_pnl = 0.0
    
    for p in positions:
        sym = p.get("symbol")
        if sym not in today_symbols:
            continue
            
        qty = float(p.get("qty", 0))
        current = float(p.get("current_price", 0))
        
        # Use FULL cost_basis from position (includes all historical buys)
        cost_basis = float(p.get("cost_basis", 0))
        
        # SAFETY CHECK: If current price is 0 or None (API error), SKIP this position
        if current <= 0:
            print(f"[KILL SWITCH] ⚠️ Missing price for {sym}, skipping PnL check to avoid false trigger.")
            continue
            
        # --- FLASH CRASH PROTECTION ---
        # If price shows > 90% drop, verify with live quote provided avg_entry valid
        avg_entry = cost_basis / qty if qty else 0
        if avg_entry > 0 and current < (avg_entry * 0.1): 
             # Suspicious 90% drop. Check Real-Time Quote.
             try:
                 q = get_latest_quote(base_url, key, secret, sym)
                 bid = float(q.get("bid", 0))
                 ask = float(q.get("ask", 0))
                 
                 # If we have a valid bid that is HIGHER than current, use it
                 if bid > current:
                     print(f"[KILL SWITCH] ⚠️ {sym} Flash Crash Detected? Current={current}, Bid={bid}. Using Bid.")
                     current = bid
                 elif ask > 0 and current < (ask * 0.1):
                     # Current is < 10% of Ask? (e.g. Price=0.01, Ask=10.00)
                     # Likely bad data. Use Ask/2 or skip.
                     print(f"[KILL SWITCH] ⚠️ {sym} Bad Data? Current={current}, Ask={ask}. Using Ask/2.")
                     current = ask / 2
             except Exception as rx:
                 print(f"[KILL SWITCH] Error verifying quote for {sym}: {rx}")
        # -----------------------------
        
        # Add full cost basis to invested (not just today's portion)
        invested += cost_basis
        
        # Unrealized P&L for full position
        market_value = current * qty
        unrealized_pnl += (market_value - cost_basis)

    if invested <= 0:
        return False  # No capital at risk
    
    total_pnl = realized_pnl + unrealized_pnl
    
    # Debug logging for troubleshooting
    print(f"[KILL SWITCH DEBUG] {bot_tag}: realized_pnl=${realized_pnl:.2f}, unrealized_pnl=${unrealized_pnl:.2f}, total=${total_pnl:.2f}, invested=${invested:.2f}")
    
    # PnL Check
    loss_pct = (total_pnl / invested) * 100
    
    if loss_pct < -max_loss_pct:
        print(f"[KILL SWITCH] {bot_tag} exceeded {max_loss_pct}% loss: {loss_pct:.1f}% (P&L: ${total_pnl:.2f}, Invested: ${invested:.2f})")
        return True
    
    return False


def api_call_with_retry(
    func,
    *args,
    max_retries: int = 3,
    base_wait: float = 2.0,
    **kwargs
):
    """
    Wrapper with exponential backoff for API calls.
    Handles HTTP 429 (rate limit) and transient errors.
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except requests.HTTPError as e:
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 429:
                    wait = base_wait * (2 ** attempt)
                    print(f"[API] Rate limited (429), waiting {wait:.1f}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    last_exception = e
                    continue
            raise
        except requests.Timeout as e:
            wait = base_wait * (2 ** attempt)
            print(f"[API] Timeout, waiting {wait:.1f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
            last_exception = e
            continue
        except Exception as e:
            # Log other exceptions but don't retry
            print(f"[API] Error: {type(e).__name__}: {e}")
            raise
    
    # Max retries exceeded
    if last_exception:
        raise last_exception
    raise Exception(f"Max retries ({max_retries}) exceeded")

# ──────────────────────────────────────────────────────────────────────────────
# Multi-symbol bars (robust shape + pagination)
# ──────────────────────────────────────────────────────────────────────────────
def _chunked(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def _bars_json_to_map(j: Dict[str, Any]) -> Tuple[Dict[str, List[Dict[str, Any]]], str]:
    """
    Normalize Alpaca bars JSON into {symbol: [bar, ...]} and return (map, shape_tag).

    Alpaca /v2/stocks/bars can return:
      A) {"bars": {"AAPL":[{...}], "TSLA":[{...}]}, "next_page_token": "..."}  # dict keyed by symbol
      B) {"bars": [{...,"S":"AAPL"}, {...,"S":"TSLA"}, ...], "next_page_token": "..."}  # flat list
         (Sometimes uses "symbol" instead of "S")
    """
    bars = j.get("bars")
    out: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(bars, dict):
        # shape A
        for s, recs in bars.items():
            if not s:
                continue
            out.setdefault(s.upper(), []).extend(recs or [])
        return out, "dict"
    elif isinstance(bars, list):
        # shape B
        sym_key = None
        for b in bars[:3]:
            if "S" in b:
                sym_key = "S"; break
            if "symbol" in b:
                sym_key = "symbol"; break
        if sym_key is None:
            sym_key = "S"
        for b in bars:
            s = (b.get(sym_key) or b.get("S") or b.get("symbol") or "").upper()
            if s:
                out.setdefault(s, []).append(b)
        return out, "list"
    else:
        return {}, "empty"

def _build_params(symbols: List[str], timeframe: str, start_iso: str, end_iso: str, feed: str, limit: int, page_token: Optional[str]) -> Dict[str, Any]:
    p = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "start": start_iso,
        "end": end_iso,
        "limit": int(limit),
        "feed": feed,
        # "adjustment": "raw",
    }
    if page_token:
        p["page_token"] = page_token
    return p

def _build_df_from_bars(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert list of Alpaca bar dicts into a DataFrame with UTC index.
    Accepts both single-symbol endpoint bar shape and multi list shape.
    """
    if not records:
        return pd.DataFrame()
    times, o, h, l, c, v = [], [], [], [], [], []
    for b in records:
        t = b.get("t") or b.get("time")
        try:
            ts = pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo else pd.Timestamp(t, tz="UTC")
        except Exception:
            ts = pd.Timestamp(str(t), tz="UTC")
        times.append(ts)
        o.append(float(b.get("o") or b.get("open") or 0.0))
        h.append(float(b.get("h") or b.get("high") or 0.0))
        l.append(float(b.get("l") or b.get("low") or 0.0))
        c.append(float(b.get("c") or b.get("close") or 0.0))
        v.append(float(b.get("v") or b.get("volume") or 0.0))
    df = pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c, "volume": v},
        index=pd.DatetimeIndex(times, tz="UTC"),
    )
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

def filter_rth(df: pd.DataFrame, tz_name: str = "US/Eastern", start_hm: str = "09:30", end_hm: str = "15:55") -> pd.DataFrame:
    if df.empty:
        return df
    local = df.tz_convert(tz_name)
    sh, sm = map(int, start_hm.split(":"))
    eh, em = map(int, end_hm.split(":"))
    mask = (local.index.hour*60 + local.index.minute >= sh*60 + sm) & \
           (local.index.hour*60 + local.index.minute <= eh*60 + em)
    return df.loc[mask].copy()

def drop_unclosed_last_bar(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df.empty:
        return df
    tfm = (timeframe or "15Min").lower()
    step_min = 1
    if "min" in tfm:
        step_min = int(tfm.replace("min", "").replace("m", ""))
    elif tfm in ("1h", "60min", "60m"):
        step_min = 60
    last_ts = df.index[-1]
    if (_now_utc() - last_ts) < dt.timedelta(minutes=step_min):
        return df.iloc[:-1].copy()
    return df

def fetch_latest_bars(
    symbols: List[str],
    timeframe: str = "15Min",
    history_days: int = 30,
    feed: str = "iex",
    rth_only: bool = True,
    tz_name: str = "US/Eastern",
    rth_start: str = "09:30",
    rth_end: str = "15:55",
    allowed_windows: Optional[List[Dict[str, str]]] = None,  # currently unused here
    bar_limit: int = 10000,
    key: Optional[str] = None,
    secret: Optional[str] = None,
    dollar_vol_window: int = 20,
    dollar_vol_min_periods: int = 7,
    verbose: bool = True,
    end: Optional[Any] = None,  # Accepts datetime, date, or ISO string
    incomplete: bool = False, # If True, enables returning the incomplete (forming) last bar
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Robust multi-symbol fetch with pagination & dual-shape handling. Emits rich diagnostics.
    Returns: (bars_map, meta) where meta={"http_errors":[...], "stale_symbols":[...]}
    """
    bars_map: Dict[str, pd.DataFrame] = {}
    http_errors: List[Dict[str, Any]] = []
    syms_with_data: List[str] = []
    syms_empty: List[str] = []
    stale_syms: List[str] = []

    if not symbols:
        return bars_map, {"http_errors": [], "stale_symbols": []}

    # Universe log (kept to match your current behavior)
    if verbose:
        print(json.dumps({
            "type": "UNIVERSE",
            "loaded": len(symbols),
            "sample": symbols[:10],
            "when": _iso_utc()
        }, separators=(",", ":"), ensure_ascii=False), flush=True)

    base_data_url = "https://data.alpaca.markets/v2/stocks/bars"
    # Resolve creds from args or env (fixes 401 when args are not provided)
    H = _alpaca_headers(key, secret)
    if not H.get("APCA-API-KEY-ID") or not H.get("APCA-API-SECRET-KEY"):
        http_errors.append({
            "chunk": 0,
            "error": "missing_api_keys",
            "symbols": symbols[:min(10, len(symbols))],
            "hint": "Set APCA_API_KEY_ID / APCA_API_SECRET_KEY (or pass key/secret to fetch_latest_bars).",
        })
        # still continue; requests will 401, but we won’t crash

    if end is None:
        end_dt = _now_utc().replace(microsecond=0)
    else:
        # Normalize end to utc datetime
        if isinstance(end, str):
            try:
                # Try simple date first YYYY-MM-DD
                end_dt = dt.datetime.strptime(end, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=dt.timezone.utc)
            except ValueError:
                 try:
                     end_dt = dt.datetime.fromisoformat(end.replace("Z", "+00:00"))
                 except ValueError:
                     # Fallback to pandas
                     end_dt = pd.to_datetime(end).to_pydatetime()
                     if end_dt.tzinfo is None:
                         end_dt = end_dt.replace(tzinfo=dt.timezone.utc)
        elif isinstance(end, (dt.date, dt.datetime)):
             if isinstance(end, dt.date) and not isinstance(end, dt.datetime):
                  end_dt = dt.datetime.combine(end, dt.time(23, 59, 59, tzinfo=dt.timezone.utc))
             else:
                  end_dt = end
                  if end_dt.tzinfo is None:
                      end_dt = end_dt.replace(tzinfo=dt.timezone.utc)
        else:
             end_dt = _now_utc()

    start = end_dt - dt.timedelta(days=max(1, int(history_days)))
    
    # Format for Alpaca (RFC3339)
    start_iso = start.strftime(ISO_UTC)
    end_iso = end_dt.strftime(ISO_UTC)

    chunks = list(_chunked([s.upper() for s in symbols], 25))
    if verbose:
        print(json.dumps({
            "type": "BARS_FETCH_START",
            "requested": len(symbols),
            "chunks": len(chunks),
            "timeframe": timeframe,
            "feed": feed,
            "start": start_iso,
            "end": end_iso,
            "when": _iso_utc()
        }, separators=(",", ":"), ensure_ascii=False), flush=True)

    for idx, chunk in enumerate(chunks, start=1):
        collected: Dict[str, List[Dict[str, Any]]] = {}
        page_token = None
        pages = 0
        shape_seen = None
        MAX_PAGES = 20

        while True:
            pages += 1
            params = _build_params(chunk, timeframe, start_iso, end_iso, feed, bar_limit, page_token)
            try:
                r = requests.get(base_data_url, headers=H, params=params, timeout=20)
                if r.status_code in (401, 403):
                    http_errors.append({
                        "chunk": idx, "code": r.status_code,
                        "msg": "Unauthorized" if r.status_code == 401 else "Forbidden (data entitlement)",
                        "symbols": chunk
                    })
                    break
                r.raise_for_status()
                j = r.json() or {}
            except Exception as e:
                http_errors.append({"chunk": idx, "error": str(e), "symbols": chunk})
                break

            per_sym, shape = _bars_json_to_map(j)
            shape_seen = shape_seen or shape
            for s, recs in per_sym.items():
                collected.setdefault(s, []).extend(recs or [])

            page_token = j.get("next_page_token") or None
            if not page_token or pages >= MAX_PAGES:
                break

        if verbose:
            print(json.dumps({
                "type": "BARS_FETCH_CHUNK_COLLECTED",
                "chunk_index": idx,
                "symbols_in_chunk": len(chunk),
                "pages": pages,
                "source_shape": shape_seen or "none",
                "collected_symbols": len(collected)
            }, separators=(",", ":"), ensure_ascii=False), flush=True)

        for s in chunk:
            recs = collected.get(s) or []
            df = _build_df_from_bars(recs)
            if df.empty:
                syms_empty.append(s)
                continue

            # Skip RTH filter for daily/weekly timeframes (they don't have intraday timestamps)
            is_daily_or_longer = timeframe.lower() in ("1day", "1d", "day", "1week", "1w", "week")
            if rth_only and not is_daily_or_longer:
                df = filter_rth(df, tz_name=tz_name, start_hm=rth_start, end_hm=rth_end)
            
            # Drop unclosed last bar UNLESS specifically requested for auditing (incomplete=True)
            if not incomplete:
                df = drop_unclosed_last_bar(df, timeframe)

            if df.empty:
                syms_empty.append(s)
                continue

            dv = (df["close"] * df["volume"]).rolling(
                window=int(dollar_vol_window),
                min_periods=int(dollar_vol_min_periods)
            ).mean()
            df["dollar_vol_avg"] = dv.ffill().fillna(0.0)

            last_ts = df.index[-1]
            if (end_dt - last_ts) > dt.timedelta(days=7):
                stale_syms.append(s)

            bars_map[s] = df
            syms_with_data.append(s)

    if verbose:
        print(json.dumps({
            "type": "BARS_FETCH_SUMMARY",
            "requested": len(symbols),
            "with_data": len(syms_with_data),
            "empty": len(syms_empty),
            "stale": len(stale_syms),
            "sample_with_data": syms_with_data[:4],
            "sample_empty": syms_empty[:10],
            "when": _iso_utc()
        }, separators=(",", ":"), ensure_ascii=False), flush=True)

    return bars_map, {
        "http_errors": http_errors,
        "stale_symbols": stale_syms
    }

def _round_to_tick(px: float, tick: float = 0.01) -> float:
    """Round price to tick size using Decimal for precision."""
    from decimal import Decimal, ROUND_DOWN
    if px <= 0:
        return 0.0
    d_px = Decimal(str(px))
    d_tick = Decimal(str(tick))
    return float((d_px / d_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * d_tick)

def submit_bracket_order(
    base_url: Optional[str], key: Optional[str], secret: Optional[str],
    symbol: str, qty: int, side: str = "buy",
    limit_price: Optional[float] = None,
    take_profit_price: float = 0.0,
    stop_loss_price: float = 0.0,
    tif: str = "gtc",  # GTC so TP/SL exit orders persist overnight
    client_order_id: Optional[str] = None,
    limit_buffer_pct: float = 1.0,  # Buffer for limit orders when no price specified
    verify_fill: bool = False,      # Whether to verify fill status
    verify_timeout: int = 5,        # Seconds to wait for fill verification
) -> Dict[str, Any]:
    """
    Places a limit + OCO style bracket order.
    
    SAFETY: Always uses limit orders to prevent slippage.
    If no limit_price specified, uses current quote + buffer.
    
    Args:
        limit_buffer_pct: Buffer above ask (buy) or below bid (sell) for limit price
        verify_fill: If True, wait and verify order filled before returning
        verify_timeout: Seconds to wait for fill verification
    """
    base = _base_url_from_env(base_url)
    H = _alpaca_headers(key, secret)

    side = (side or "buy").lower()
    tick = 0.01
    
    # SAFETY: Always use limit orders, never market orders
    if limit_price is None:
        # Get current quote and apply buffer
        quote = get_latest_quote(base_url, key, secret, symbol)
        if quote.get("error") or (side == "buy" and quote.get("ask", 0) == 0) or (side != "buy" and quote.get("bid", 0) == 0):
            # Fallback: Use TP/SL as reference if need quote but it's unavailable
            print(f"[order] {symbol}: No valid quote for {side} (ask={quote.get('ask')}, bid={quote.get('bid')}), using TP/SL reference")
            if side == "buy":
                limit_price = take_profit_price * 0.98 if take_profit_price > 0 else stop_loss_price * 1.05
            else:
                limit_price = stop_loss_price * 0.95 if stop_loss_price > 0 else take_profit_price * 1.02
        else:
            if side == "buy":
                # Buy: limit slightly above ask to ensure fill
                limit_price = quote["ask"] * (1 + limit_buffer_pct / 100)
            else:
                # Sell: limit slightly below bid to ensure fill
                limit_price = quote["bid"] * (1 - limit_buffer_pct / 100)
        
        limit_price = _round_to_tick(limit_price, tick)
        print(f"[order] {symbol}: Using limit price ${limit_price:.2f} ({side}, buffer={limit_buffer_pct}%)")
    
    payload: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "qty": int(qty),
        "side": side,
        "time_in_force": tif,
        "type": "limit",
        "limit_price": float(limit_price),
    }
    
    if client_order_id:
        payload["client_order_id"] = client_order_id

    # Attach take-profit / stop-loss
    base_hint = float(limit_price)
    tp = _round_to_tick(float(take_profit_price), tick)
    sl = _round_to_tick(float(stop_loss_price), tick)

    # SAFETY FIX: Handle logic based on SIDE
    if side == "buy":
         min_tp = _round_to_tick(base_hint + tick, tick)
         if tp < min_tp:
             tp = min_tp
    else: # Sell (Short)
         # For shorts, TP must be BELOW entry (limit_price)
         max_tp = _round_to_tick(base_hint - tick, tick)
         # If TP is HIGHER than (limit - tick), it's invalid for a short profit take
         # (Or if it's 0, which means no TP)
         if tp > max_tp and tp > 0:
             tp = max_tp


    payload["order_class"] = "bracket"
    payload["take_profit"] = {"limit_price": tp}
    payload["stop_loss"] = {"stop_price": sl}

    try:
        r = requests.post(f"{base}/v2/orders", headers=H, json=payload, timeout=20)
        result = r.json() if r.content else {}
    except requests.Timeout:
        # Timeout - order may have been received
        print(f"[order] {symbol}: Timeout submitting order - checking status...")
        if client_order_id:
            # Try to find order by client_order_id
            if order_exists_today(base_url, key, secret, client_order_id):
                return {"id": None, "status": "timeout_but_may_exist", "client_order_id": client_order_id}
        return {"error": "timeout", "needs_verification": True}
    except Exception as e:
        print(f"[order] {symbol}: Error submitting order: {e}")
        return {"error": str(e)}
    
    # Check for API error
    if r.status_code >= 400 or result.get("code"):
        return result  # Return error response for caller to handle
    
    # Verify fill if requested
    if verify_fill and result.get("id"):
        order_id = result["id"]
        for i in range(verify_timeout):
            time.sleep(1)
            order = get_order_by_id(base_url, key, secret, order_id)
            status = order.get("status")
            
            if status == "filled":
                print(f"[order] {symbol}: Fill verified (took {i+1}s)")
                return order
            elif status in ("canceled", "expired", "rejected"):
                print(f"[order] {symbol}: Order {status}")
                return order
        
        # Still not filled - return with pending status
        print(f"[order] {symbol}: Order still pending after {verify_timeout}s")
        return {"id": order_id, "status": "pending", "needs_monitoring": True, **result}
    
    return result