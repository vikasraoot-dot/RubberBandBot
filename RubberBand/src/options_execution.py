"""
Options Execution Module: Submit and manage option orders via Alpaca.
"""
from __future__ import annotations

import os
import json
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from zoneinfo import ZoneInfo

ET = ZoneInfo("US/Eastern")

# ──────────────────────────────────────────────────────────────────────────────
# Credentials
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_creds() -> Tuple[str, str, str]:
    """Resolve Alpaca credentials from environment."""
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_KEY_ID") or ""
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY") or ""
    base = os.getenv("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets"
    return base.rstrip("/"), key.strip(), secret.strip()


def _headers(key: str, secret: str) -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Content-Type": "application/json",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Order Submission
# ──────────────────────────────────────────────────────────────────────────────
def submit_option_order(
    option_symbol: str,
    qty: int = 1,
    side: str = "buy",
    order_type: str = "limit",
    limit_price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Submit an option order.
    
    Args:
        option_symbol: OCC option symbol (e.g., "NVDA241204C00140000")
        qty: Number of contracts
        side: "buy" or "sell"
        order_type: "market" or "limit"
        limit_price: Required if order_type is "limit"
    
    Returns:
        Order response dict
    """
    base, key, secret = _resolve_creds()
    
    payload = {
        "symbol": option_symbol,
        "qty": int(qty),
        "side": side.lower(),
        "type": order_type.lower(),
        "time_in_force": "day",
    }
    
    if order_type.lower() == "limit":
        if limit_price is None:
            raise ValueError("limit_price required for limit orders")
        payload["limit_price"] = round(float(limit_price), 2)
    
    url = f"{base}/v2/orders"
    
    try:
        resp = requests.post(url, headers=_headers(key, secret), json=payload, timeout=15)
        result = resp.json()
        
        if resp.status_code >= 400:
            print(f"[options] Order error: {result}")
            return {"error": True, "message": result.get("message", str(result))}
        
        print(f"[options] Order submitted: {option_symbol} {side} {qty} @ {limit_price}")
        return result
    except Exception as e:
        print(f"[options] Order exception: {e}")
        return {"error": True, "message": str(e)}


def submit_spread_order(
    long_symbol: str,
    short_symbol: str,
    qty: int = 1,
    max_debit: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Submit a bull call spread order as a multi-leg order.
    
    Uses Alpaca's mleg order class to submit both legs together,
    ensuring the spread is treated as a defined-risk position.
    
    Args:
        long_symbol: OCC symbol for long leg (ATM call)
        short_symbol: OCC symbol for short leg (OTM call)
        qty: Number of spreads
        max_debit: Max net debit per spread (used as limit price)
    
    Returns:
        Result dict with order info or error
    """
    from RubberBand.src.options_data import get_option_quote
    
    base, key, secret = _resolve_creds()
    
    # Get quotes for both legs to determine limit price
    long_quote = get_option_quote(long_symbol)
    short_quote = get_option_quote(short_symbol)
    
    if not long_quote or not short_quote:
        return {"error": True, "message": "Cannot get quotes for spread legs"}
    
    # Calculate net debit: buy long at ask, sell short at bid
    long_ask = long_quote.get("ask", 0)
    short_bid = short_quote.get("bid", 0)
    net_debit = long_ask - short_bid
    
    if net_debit <= 0:
        return {"error": True, "message": f"Invalid spread pricing: debit={net_debit}"}
    
    if max_debit and net_debit > max_debit:
        return {"error": True, "message": f"Spread debit {net_debit} exceeds max {max_debit}"}
    
    # Use the calculated debit as limit price (positive = debit)
    limit_price = round(net_debit, 2)
    
    # Build multi-leg order request
    # Alpaca mleg order: order_class="mleg", legs array with each leg
    order_payload = {
        "symbol": long_symbol,  # Primary symbol (required by API, use long leg)
        "qty": str(qty),
        "side": "buy",  # Overall direction for debit spread
        "type": "limit",
        "time_in_force": "day",
        "limit_price": str(limit_price),  # Positive = debit to pay
        "order_class": "mleg",  # Multi-leg order
        "legs": [
            {
                "symbol": long_symbol,
                "side": "buy",
                "ratio_qty": "1",
                "position_intent": "buy_to_open",
            },
            {
                "symbol": short_symbol,
                "side": "sell",
                "ratio_qty": "1",
                "position_intent": "sell_to_open",
            },
        ],
    }
    
    url = f"{base}/v2/orders"
    
    try:
        print(f"[options] Submitting spread: {long_symbol} (buy) / {short_symbol} (sell) @ ${limit_price} debit")
        resp = requests.post(
            url,
            headers=_headers(key, secret),
            json=order_payload,
            timeout=15,
        )
        result = resp.json()
        
        if resp.status_code >= 400:
            error_msg = result.get("message", str(result))
            print(f"[options] Spread order error: {result}")
            return {"error": True, "message": error_msg}
        
        order_id = result.get("id", "")
        print(f"[options] Spread order submitted: {order_id}")
        return {
            "error": False,
            "order_id": order_id,
            "long_symbol": long_symbol,
            "short_symbol": short_symbol,
            "qty": qty,
            "limit_price": limit_price,
            "status": result.get("status", ""),
        }
    except Exception as e:
        print(f"[options] Spread order exception: {e}")
        return {"error": True, "message": str(e)}


def close_option_position(
    option_symbol: str,
    qty: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Close an option position (sell to close).
    
    Args:
        option_symbol: OCC option symbol
        qty: Contracts to close (None = close all)
    
    Returns:
        Order response dict
    """
    base, key, secret = _resolve_creds()
    
    # Close position via DELETE endpoint
    url = f"{base}/v2/positions/{option_symbol}"
    params = {}
    if qty is not None:
        params["qty"] = str(int(qty))
    
    try:
        resp = requests.delete(url, headers=_headers(key, secret), params=params, timeout=15)
        
        if resp.status_code == 404:
            print(f"[options] No position found for {option_symbol}")
            return {"error": True, "message": "Position not found"}
        
        result = resp.json()
        print(f"[options] Position closed: {option_symbol}")
        return result
    except Exception as e:
        print(f"[options] Close exception: {e}")
        return {"error": True, "message": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# Position Monitoring
# ──────────────────────────────────────────────────────────────────────────────
def get_option_positions() -> List[Dict[str, Any]]:
    """Get all open option positions."""
    base, key, secret = _resolve_creds()
    
    url = f"{base}/v2/positions"
    
    try:
        resp = requests.get(url, headers=_headers(key, secret), timeout=10)
        resp.raise_for_status()
        positions = resp.json() or []
        
        # Filter for options (symbol length > 10 typically indicates option)
        option_positions = [
            p for p in positions 
            if len(p.get("symbol", "")) > 10  # OCC symbols are long
        ]
        return option_positions
    except Exception as e:
        print(f"[options] Error fetching positions: {e}")
        return []


def get_position_pnl(position: Dict[str, Any]) -> Tuple[float, float]:
    """
    Calculate current P&L for a position.
    
    Returns:
        (pnl_dollars, pnl_percent)
    """
    try:
        cost_basis = float(position.get("cost_basis", 0))
        market_value = float(position.get("market_value", 0))
        
        pnl = market_value - cost_basis
        pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0.0
        
        return pnl, pnl_pct
    except (ValueError, TypeError):
        return 0.0, 0.0


def check_exit_conditions(
    position: Dict[str, Any],
    tp_pct: float = 30.0,
    sl_pct: float = -50.0,
) -> Tuple[bool, str]:
    """
    Check if position should be exited based on P&L thresholds.
    
    Args:
        position: Position dict from Alpaca
        tp_pct: Take profit percentage (e.g., 30 = +30%)
        sl_pct: Stop loss percentage (e.g., -50 = -50%)
    
    Returns:
        (should_exit, reason)
    """
    pnl, pnl_pct = get_position_pnl(position)
    
    if pnl_pct >= tp_pct:
        return True, f"TP ({pnl_pct:.1f}% >= {tp_pct}%)"
    
    if pnl_pct <= sl_pct:
        return True, f"SL ({pnl_pct:.1f}% <= {sl_pct}%)"
    
    # Time-based exit: 3:00 PM ET cutoff
    now_et = datetime.now(ET)
    cutoff = now_et.replace(hour=15, minute=0, second=0, microsecond=0)
    if now_et >= cutoff:
        return True, f"TIME (past 3:00 PM ET)"
    
    return False, ""


# ──────────────────────────────────────────────────────────────────────────────
# Flatten All Options
# ──────────────────────────────────────────────────────────────────────────────
def flatten_all_option_positions() -> List[Dict[str, Any]]:
    """Close all open option positions."""
    positions = get_option_positions()
    results = []
    
    for pos in positions:
        symbol = pos.get("symbol", "")
        if symbol:
            result = close_option_position(symbol)
            results.append({"symbol": symbol, "result": result})
    
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Trade Tracking
# ──────────────────────────────────────────────────────────────────────────────
class OptionsTradeTracker:
    """Track option trades for logging and P&L calculation."""
    
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
    
    def record_entry(
        self,
        underlying: str,
        option_symbol: str,
        qty: int,
        premium: float,
        strike: float,
        expiration: str,
    ):
        """Record a new option entry."""
        self.trades.append({
            "underlying": underlying,
            "option_symbol": option_symbol,
            "qty": qty,
            "entry_premium": premium,
            "strike": strike,
            "expiration": expiration,
            "entry_time": datetime.now(ET).isoformat(),
            "exit_premium": None,
            "exit_time": None,
            "exit_reason": None,
            "pnl": None,
        })
    
    def record_exit(
        self,
        option_symbol: str,
        exit_premium: float,
        reason: str,
    ):
        """Record an exit for an existing trade."""
        for trade in self.trades:
            if trade["option_symbol"] == option_symbol and trade["exit_time"] is None:
                trade["exit_premium"] = exit_premium
                trade["exit_time"] = datetime.now(ET).isoformat()
                trade["exit_reason"] = reason
                # P&L = (exit - entry) * qty * 100 (each contract = 100 shares)
                trade["pnl"] = (exit_premium - trade["entry_premium"]) * trade["qty"] * 100
                break
    
    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Get trades that haven't been closed."""
        return [t for t in self.trades if t["exit_time"] is None]
    
    def get_closed_trades(self) -> List[Dict[str, Any]]:
        """Get trades that have been closed."""
        return [t for t in self.trades if t["exit_time"] is not None]
    
    def get_total_pnl(self) -> float:
        """Get total P&L of all closed trades."""
        return sum(t.get("pnl", 0) or 0 for t in self.get_closed_trades())
    
    def to_json(self, path: str):
        """Save trades to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.trades, f, indent=2)
