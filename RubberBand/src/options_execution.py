"""
Options Execution Module: Submit and manage option orders via Alpaca.
"""
from __future__ import annotations

import os
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from zoneinfo import ZoneInfo

from RubberBand.src.alpaca_creds import resolve_credentials, get_headers

ET = ZoneInfo("US/Eastern")


# Alias for backward compatibility within this module
def _resolve_creds() -> Tuple[str, str, str]:
    """Resolve Alpaca credentials from environment."""
    key, secret, base = resolve_credentials()
    return base, key, secret


def _headers(key: str, secret: str) -> Dict[str, str]:
    """Build Alpaca API headers."""
    return get_headers(key, secret)


# ──────────────────────────────────────────────────────────────────────────────
# Mleg Fill Verification
# ──────────────────────────────────────────────────────────────────────────────
def _verify_mleg_fill(
    base: str, key: str, secret: str,
    order_id: str, long_symbol: str, short_symbol: str,
    timeout: int = 15,
) -> str:
    """
    Wait for mleg order to fill and verify both legs are in positions.
    Returns: 'filled', 'partial_fill', 'pending', or 'failed'.

    IMPORTANT: When Alpaca confirms "filled" on an mleg order, BOTH legs are filled.
    The positions API may have settlement latency, so we retry position checks
    several times before declaring a partial fill to avoid false positives.
    """
    for i in range(timeout):
        time.sleep(1)
        order = _get_order_by_id(base, key, secret, order_id)
        status = order.get("status", "")

        if status == "filled":
            # Broker confirmed both legs filled. Verify positions with retries
            # to account for settlement/propagation latency in the positions API.
            _POS_CHECK_RETRIES = 5
            for attempt in range(_POS_CHECK_RETRIES):
                try:
                    pos_resp = requests.get(f"{base}/v2/positions", headers=_headers(key, secret), timeout=10)
                    if pos_resp.status_code == 200:
                        positions = {p["symbol"]: p for p in pos_resp.json()}
                        has_long = long_symbol in positions
                        has_short = short_symbol in positions
                        if has_long and has_short:
                            print(f"[options] Mleg fill verified: both legs confirmed ({i+1}s, check {attempt+1})", flush=True)
                            return "filled"
                        else:
                            # Positions API may lag — retry before declaring partial fill
                            print(f"[options] Position check {attempt+1}/{_POS_CHECK_RETRIES}: "
                                  f"long={has_long} short={has_short} (retrying...)", flush=True)
                except Exception as e:
                    print(f"[options] Position check error (attempt {attempt+1}): {e}", flush=True)

                if attempt < _POS_CHECK_RETRIES - 1:
                    time.sleep(2)  # Wait 2s between position check retries

            # Exhausted retries. Broker says filled so trust that — do NOT close legs.
            print(f"[options] Position check retries exhausted but broker confirmed filled. Trusting broker.", flush=True)
            return "filled"

        elif status in ("canceled", "expired", "rejected"):
            print(f"[options] Mleg order {status}: {order_id}", flush=True)
            return "failed"

    print(f"[options] Mleg order still pending after {timeout}s: {order_id}", flush=True)
    return "pending"


def _close_naked_legs(base: str, key: str, secret: str, long_symbol: str, short_symbol: str) -> None:
    """Emergency close any naked option legs from a partial fill."""
    try:
        pos_resp = requests.get(f"{base}/v2/positions", headers=_headers(key, secret), timeout=10)
        if pos_resp.status_code != 200:
            return
        positions = {p["symbol"]: p for p in pos_resp.json()}

        for sym in [long_symbol, short_symbol]:
            if sym in positions:
                qty = abs(int(positions[sym].get("qty", 0)))
                if qty > 0:
                    side = "sell" if int(positions[sym].get("qty", 0)) > 0 else "buy"
                    print(f"[options] Emergency closing naked leg: {sym} {side} {qty}", flush=True)
                    submit_option_order(sym, qty=qty, side=side, order_type="market")
    except Exception as e:
        print(f"[options] CRITICAL: Failed to close naked legs: {e}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Order Submission
# ──────────────────────────────────────────────────────────────────────────────
def _get_order_by_id(base: str, key: str, secret: str, order_id: str) -> Dict[str, Any]:
    """Get order details by Alpaca order ID."""
    try:
        r = requests.get(f"{base}/v2/orders/{order_id}", headers=_headers(key, secret), timeout=10)
        if r.status_code == 404:
            return {"error": "not_found"}
        r.raise_for_status()
        return r.json() or {}
    except Exception as e:
        print(f"[options] Failed to get order {order_id}: {e}")
        return {"error": str(e)}


def submit_option_order(
    option_symbol: str,
    qty: int = 1,
    side: str = "buy",
    order_type: str = "limit",
    limit_price: Optional[float] = None,
    client_order_id: Optional[str] = None,
    verify_fill: bool = False,
    verify_timeout: int = 5,
) -> Dict[str, Any]:
    """
    Submit an option order.
    
    Args:
        option_symbol: OCC option symbol (e.g., "NVDA241204C00140000")
        qty: Number of contracts
        side: "buy" or "sell"
        order_type: "market" or "limit"
        limit_price: Required if order_type is "limit"
        client_order_id: Optional unique ID for order attribution (e.g., "WK_OPT_NVDA...")
        verify_fill: If True, wait and verify order fills before returning
        verify_timeout: Seconds to wait for fill verification
    
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
    
    if client_order_id:
        payload["client_order_id"] = client_order_id
    
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
        
        # Verify fill if requested
        if verify_fill and result.get("id"):
            order_id = result["id"]
            for i in range(verify_timeout):
                time.sleep(1)
                order = _get_order_by_id(base, key, secret, order_id)
                status = order.get("status")
                
                if status == "filled":
                    print(f"[options] Fill verified: {option_symbol} (took {i+1}s)")
                    return order
                elif status in ("canceled", "expired", "rejected"):
                    print(f"[options] Order {status}: {option_symbol}")
                    return order
            
            # Still not filled
            print(f"[options] Order pending after {verify_timeout}s: {option_symbol}")
            return {"id": order_id, "status": "pending", "needs_monitoring": True, **result}
        
        return result
    except requests.Timeout:
        print(f"[options] Order timeout: {option_symbol}")
        return {"error": True, "message": "timeout", "needs_verification": True}
    except Exception as e:
        print(f"[options] Order exception: {e}")
        return {"error": True, "message": str(e)}


def submit_spread_order(
    long_symbol: str,
    short_symbol: str,
    qty: int = 1,
    max_debit: Optional[float] = None,
    client_order_id: Optional[str] = None,
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
        client_order_id: Optional unique ID for order attribution (e.g., "15M_OPT_NVDA...")
    
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
    # Note: mleg orders do NOT use top-level "symbol" field
    order_payload = {
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
    
    if client_order_id:
        order_payload["client_order_id"] = client_order_id
    
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

        # MLEG FILL VERIFICATION: Wait for fill and verify both legs exist as positions.
        # When broker confirms "filled", both legs are filled — we trust the broker
        # and only use position checks to confirm (with retries for settlement latency).
        if order_id:
            verified_status = _verify_mleg_fill(base, key, secret, order_id, long_symbol, short_symbol, timeout=15)
            if verified_status == "failed":
                return {"error": True, "message": f"Mleg order {verified_status}"}
            elif verified_status == "pending":
                print(f"[options] WARNING: Mleg order still pending after timeout — may fill later", flush=True)

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


def close_spread(
    long_symbol: str,
    short_symbol: str,
    qty: int = 1,
) -> Dict[str, Any]:
    """
    Close a bull call spread by closing both legs together as a multi-leg order.
    
    IMPORTANT: This function will NEVER close legs individually to prevent orphaned legs.
    If the mleg order fails, it returns an error for the caller to handle/retry.
    
    For a bull call spread:
    - Sell to close the long leg (the one we bought)
    - Buy to close the short leg (the one we sold)
    
    Args:
        long_symbol: OCC symbol for long leg (the call we're long)
        short_symbol: OCC symbol for short leg (the call we're short)
        qty: Number of spreads to close
    
    Returns:
        Result dict with order info or error
    """
    from RubberBand.src.options_data import get_option_quote
    
    base, key, secret = _resolve_creds()
    
    # Get quotes for both legs to determine credit received
    long_quote = get_option_quote(long_symbol)
    short_quote = get_option_quote(short_symbol)
    
    if not long_quote or not short_quote:
        # DO NOT fall back to individual closes - return error for retry
        print(f"[options] Cannot get quotes for spread legs - will retry later")
        print(f"[options]   Long quote: {long_quote}")
        print(f"[options]   Short quote: {short_quote}")
        return {
            "error": True, 
            "message": "Cannot get quotes for spread legs",
            "retry": True,  # Signal that this is retryable
        }
    
    # For closing: sell long at bid, buy short at ask
    long_bid = long_quote.get("bid", 0)
    short_ask = short_quote.get("ask", 0)
    net_credit = long_bid - short_ask  # Positive = we receive credit, Negative = we pay debit

    # Determine order side based on whether closing produces credit or debit
    if net_credit >= 0:
        # Profitable close: we receive credit
        order_side = "sell"
        limit_price = round(net_credit, 2)
    else:
        # Losing close: we pay debit to close
        order_side = "buy"
        limit_price = round(abs(net_credit), 2)

    # Log the spread close attempt with full details
    close_type = "credit" if net_credit >= 0 else "debit"
    print(f"[options] Attempting spread close ({close_type}):")
    print(f"[options]   Long: {long_symbol} @ bid={long_bid}")
    print(f"[options]   Short: {short_symbol} @ ask={short_ask}")
    print(f"[options]   Net: ${net_credit:.2f}, side={order_side}, limit_price: ${limit_price:.2f}")

    # Build multi-leg close order
    # Note: mleg orders do NOT use top-level "symbol" field
    order_payload = {
        "qty": str(qty),
        "side": order_side,
        "type": "limit",
        "time_in_force": "day",
        "limit_price": str(limit_price),
        "order_class": "mleg",
        "legs": [
            {
                "symbol": long_symbol,
                "side": "sell",  # Sell to close the long
                "ratio_qty": "1",
                "position_intent": "sell_to_close",
            },
            {
                "symbol": short_symbol,
                "side": "buy",  # Buy to close the short
                "ratio_qty": "1",
                "position_intent": "buy_to_close",
            },
        ],
    }
    
    url = f"{base}/v2/orders"
    
    try:
        print(f"[options] Closing spread: {long_symbol} (sell) / {short_symbol} (buy) @ ${limit_price} credit")
        resp = requests.post(
            url,
            headers=_headers(key, secret),
            json=order_payload,
            timeout=15,
        )
        result = resp.json()
        
        if resp.status_code >= 400:
            error_msg = result.get("message", str(result))
            print(f"[options] Spread close ERROR: {result}")
            print(f"[options] NOT falling back to individual closes - will retry next cycle")
            # DO NOT fall back to individual closes - return error for retry
            return {
                "error": True,
                "message": f"mleg close failed: {error_msg}",
                "status_code": resp.status_code,
                "retry": True,  # Signal that this is retryable
            }
        
        order_id = result.get("id", "")
        order_status = result.get("status", "")
        print(f"[options] Spread close order submitted: {order_id} (status={order_status})")

        # Brief fill verification: poll up to 3 times (1s apart) to confirm fill
        _TERMINAL_STATUSES = ("canceled", "expired", "rejected", "suspended")
        if order_id and order_status not in ("filled", "partially_filled"):
            for poll in range(3):
                time.sleep(1)
                order_check = _get_order_by_id(base, key, secret, order_id)
                if order_check.get("error"):
                    continue
                order_status = order_check.get("status", "")
                if order_status in ("filled", "partially_filled"):
                    print(f"[options] Spread close FILLED (poll #{poll + 1})")
                    break
                if order_status in _TERMINAL_STATUSES:
                    print(f"[options] Spread close order TERMINAL: {order_status}")
                    break
            else:
                # Not filled after 3 seconds - cancel the pending order to prevent
                # duplicate closes when the next cycle submits a new order
                print(f"[options] WARNING: Close order {order_id} not filled after 3s (status={order_status})")
                try:
                    cancel_resp = requests.delete(
                        f"{base}/v2/orders/{order_id}",
                        headers=_headers(key, secret),
                        timeout=5,
                    )
                    if cancel_resp.status_code < 300:
                        print(f"[options] Canceled pending close order {order_id}")
                    else:
                        print(f"[options] Cancel attempt returned {cancel_resp.status_code} (may have filled)")
                except Exception as cancel_err:
                    print(f"[options] Cancel attempt failed: {cancel_err}")
                return {
                    "error": True,
                    "message": f"Order submitted but not filled (status={order_status})",
                    "order_id": order_id,
                    "retry": True,
                }

            # Handle terminal statuses (order died without filling)
            if order_status in _TERMINAL_STATUSES:
                return {
                    "error": True,
                    "message": f"Close order {order_status}",
                    "order_id": order_id,
                    "retry": True,
                }

        return {
            "error": False,
            "order_id": order_id,
            "long_symbol": long_symbol,
            "short_symbol": short_symbol,
            "qty": qty,
            "credit": limit_price,
            "status": order_status,
        }
    except Exception as e:
        print(f"[options] Spread close exception: {e}")
        print(f"[options] NOT falling back to individual closes - will retry next cycle")
        # DO NOT fall back to individual closes - return error for retry
        return {
            "error": True,
            "message": str(e),
            "retry": True,  # Signal that this is retryable
        }


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


def flatten_bot_spreads(bot_tag: str, registry_dir: str = ".position_registry", max_dte_to_close: int = 1) -> List[Dict[str, Any]]:
    """
    Close option spreads belonging to a specific bot that are expiring soon.
    
    This function reads the bot's position registry, checks DTE for each position,
    and only closes positions with DTE <= max_dte_to_close. Positions with more
    time remaining are kept open.
    
    Args:
        bot_tag: Bot tag to filter positions (e.g., "15M_OPT")
        registry_dir: Directory containing position registry files
        max_dte_to_close: Maximum DTE to close (default=1 = expiring tomorrow)
        
    Returns:
        List of close results for each position closed
    """
    import json
    import os
    import re
    from datetime import datetime, date
    from zoneinfo import ZoneInfo
    
    ET = ZoneInfo("US/Eastern")
    today = datetime.now(ET).date()
    
    registry_path = os.path.join(registry_dir, f"{bot_tag}_positions.json")
    results = []
    positions_to_keep = {}
    
    # Load registry
    if not os.path.exists(registry_path):
        print(f"[flatten] No registry found at {registry_path}")
        return results
    
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[flatten] Error loading registry: {e}")
        return results
    
    positions = data.get("positions", {})
    if not positions:
        print(f"[flatten] No positions found for {bot_tag}")
        return results
    
    print(f"[flatten] Found {len(positions)} positions for {bot_tag}")
    
    def parse_expiry_from_occ(symbol: str) -> date:
        """Parse expiry date from OCC symbol like AAPL251212C00150000"""
        # Format: SYMBOL + YYMMDD + C/P + STRIKE
        match = re.search(r'(\d{6})[CP]', symbol)
        if match:
            date_str = match.group(1)  # YYMMDD
            year = 2000 + int(date_str[:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            return date(year, month, day)
        return today  # Fallback to today if can't parse
    
    for symbol, pos_data in positions.items():
        long_symbol = pos_data.get("symbol", symbol)
        short_symbol = pos_data.get("short_symbol")
        qty = int(pos_data.get("qty", 1))
        
        # Parse expiry and calculate DTE
        expiry = parse_expiry_from_occ(long_symbol)
        dte = (expiry - today).days
        
        print(f"[flatten] {long_symbol}: Expiry={expiry}, DTE={dte}")
        
        # Only close if DTE <= max_dte_to_close
        if dte > max_dte_to_close:
            print(f"[flatten] Keeping {long_symbol} (DTE={dte} > {max_dte_to_close})")
            positions_to_keep[symbol] = pos_data
            continue
        
        try:
            if short_symbol:
                # This is a spread - close both legs together
                print(f"[flatten] Closing spread (DTE={dte}): {long_symbol} / {short_symbol}")
                result = close_spread(long_symbol, short_symbol, qty)
            else:
                # Single option position - close individually
                print(f"[flatten] Closing single position (DTE={dte}): {long_symbol}")
                result = close_option_position(long_symbol, qty)
            
            results.append({
                "symbol": long_symbol,
                "short_symbol": short_symbol,
                "qty": qty,
                "dte": dte,
                "result": result,
            })
        except Exception as e:
            print(f"[flatten] Error closing {long_symbol}: {e}")
            results.append({
                "symbol": long_symbol,
                "short_symbol": short_symbol,
                "error": str(e),
            })
    
    # Update registry - keep positions that weren't closed
    try:
        data["positions"] = positions_to_keep
        data["updated_at"] = datetime.now(ET).isoformat()
        # Track closed positions
        closed_list = data.get("closed_positions", [])
        for r in results:
            if not r.get("error"):
                closed_list.append({
                    "symbol": r["symbol"],
                    "short_symbol": r.get("short_symbol"),
                    "closed_at": datetime.now(ET).isoformat(),
                })
        data["closed_positions"] = closed_list[-100:]  # Keep last 100
        
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        
        kept = len(positions_to_keep)
        closed = len([r for r in results if not r.get("error")])
        print(f"[flatten] Registry updated: {closed} closed, {kept} kept (DTE > {max_dte_to_close})")
    except Exception as e:
        print(f"[flatten] Error updating registry: {e}")
    
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
