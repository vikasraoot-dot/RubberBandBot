"""
Options Data Module: Fetch 0DTE/1DTE option contracts and quotes from Alpaca.
"""
from __future__ import annotations

import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from zoneinfo import ZoneInfo

from RubberBand.src.alpaca_creds import resolve_credentials, get_headers

ET = ZoneInfo("US/Eastern")
UTC = ZoneInfo("UTC")


# Alias for backward compatibility within this module
def _resolve_creds() -> Tuple[str, str, str]:
    """Resolve Alpaca credentials from environment."""
    key, secret, base = resolve_credentials()
    return base, key, secret


def _headers(key: str, secret: str) -> Dict[str, str]:
    """Build Alpaca API headers."""
    return get_headers(key, secret)


# ──────────────────────────────────────────────────────────────────────────────
# Contract Fetching
# ──────────────────────────────────────────────────────────────────────────────
def get_0dte_expiration() -> str:
    """Get today's date in YYYY-MM-DD format (0DTE expiration)."""
    now_et = datetime.now(ET)
    return now_et.strftime("%Y-%m-%d")


def get_ndte_expiration(dte: int = 1) -> str:
    """
    Get expiration date N trading days from now.
    
    Args:
        dte: Days to expiration (1, 2, 3, etc.)
    
    Returns:
        YYYY-MM-DD format date string
    """
    now_et = datetime.now(ET)
    target = now_et
    days_added = 0
    
    while days_added < dte:
        target += timedelta(days=1)
        # Skip weekends
        if target.weekday() < 5:  # Monday=0 through Friday=4
            days_added += 1
    
    return target.strftime("%Y-%m-%d")


def get_1dte_expiration() -> str:
    """Get tomorrow's date in YYYY-MM-DD format (1DTE expiration)."""
    return get_ndte_expiration(1)


def fetch_option_contracts(
    underlying: str,
    expiration_date: Optional[str] = None,
    option_type: str = "call",
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Fetch option contracts for a given underlying symbol.
    
    Args:
        underlying: Stock symbol (e.g., "NVDA")
        expiration_date: YYYY-MM-DD format, defaults to 0DTE
        option_type: "call" or "put"
        limit: Max contracts to return
    
    Returns:
        List of contract dicts
    """
    base, key, secret = _resolve_creds()
    
    if not expiration_date:
        expiration_date = get_0dte_expiration()
    
    # Use Alpaca's trading API for options contracts
    url = f"{base}/v2/options/contracts"
    params = {
        "underlying_symbols": underlying.upper(),
        "expiration_date": expiration_date,
        "type": option_type.lower(),
        "limit": limit,
    }
    
    try:
        resp = requests.get(url, headers=_headers(key, secret), params=params, timeout=15)
        if resp.status_code == 404:
            print(f"[options] No contracts found for {underlying} expiring {expiration_date}")
            return []
        resp.raise_for_status()
        data = resp.json()
        return data.get("option_contracts", [])
    except Exception as e:
        print(f"[options] Error fetching contracts: {e}")
        return []


def get_underlying_price(symbol: str) -> Optional[float]:
    """Get current price of underlying stock."""
    base, key, secret = _resolve_creds()
    
    # Use last quote from data API
    data_url = "https://data.alpaca.markets/v2/stocks/quotes/latest"
    params = {"symbols": symbol.upper()}
    
    try:
        resp = requests.get(data_url, headers=_headers(key, secret), params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        quotes = data.get("quotes", {})
        quote = quotes.get(symbol.upper(), {})
        # Use midpoint of bid/ask
        bid = float(quote.get("bp", 0))
        ask = float(quote.get("ap", 0))
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return None
    except Exception as e:
        print(f"[options] Error fetching price for {symbol}: {e}")
        return None


def select_atm_contract(
    underlying: str,
    expiration_date: Optional[str] = None,
    option_type: str = "call",
) -> Optional[Dict[str, Any]]:
    """
    Select the At-The-Money (ATM) option contract.
    
    ATM = strike closest to current underlying price.
    
    Returns:
        Contract dict or None
    """
    # Get current price
    price = get_underlying_price(underlying)
    if price is None:
        print(f"[options] Cannot get price for {underlying}, cannot select ATM")
        return None
    
    # Fetch contracts
    contracts = fetch_option_contracts(underlying, expiration_date, option_type)
    if not contracts:
        return None
    
    # Find closest strike to current price
    best = None
    best_diff = float("inf")
    
    for c in contracts:
        if c.get("status") != "active" or not c.get("tradable"):
            continue
        try:
            strike = float(c.get("strike_price", 0))
            diff = abs(strike - price)
            if diff < best_diff:
                best_diff = diff
                best = c
        except (ValueError, TypeError):
            continue
    
    if best:
        print(f"[options] Selected ATM: {best.get('symbol')} strike={best.get('strike_price')} (underlying={price:.2f})")
    
    return best


def select_itm_contract(
    underlying: str,
    expiration_date: Optional[str] = None,
    option_type: str = "call",
    target_delta: float = 0.65,
) -> Optional[Dict[str, Any]]:
    """
    Select an In-The-Money (ITM) option contract.
    
    For calls: ITM means strike BELOW current price.
    Target Delta ~0.65 means strike ~3% below current price.
    
    Args:
        underlying: Stock symbol
        expiration_date: YYYY-MM-DD format
        option_type: "call" or "put"
        target_delta: Target delta (0.65 = ~3% ITM for calls)
    
    Returns:
        Contract dict or None
    """
    # Get current price
    price = get_underlying_price(underlying)
    if price is None:
        print(f"[options] Cannot get price for {underlying}, cannot select ITM")
        return None
    
    # Calculate target strike based on delta
    # For Delta 0.65 call, strike should be ~3% below current price
    # Formula: strike = price * (1 - (delta - 0.5) * 0.2)
    itm_factor = 1 - (target_delta - 0.5) * 0.2
    target_strike = price * itm_factor
    
    # Fetch contracts
    contracts = fetch_option_contracts(underlying, expiration_date, option_type)
    if not contracts:
        return None
    
    # Find strike closest to our ITM target
    best = None
    best_diff = float("inf")
    
    for c in contracts:
        if c.get("status") != "active" or not c.get("tradable"):
            continue
        try:
            strike = float(c.get("strike_price", 0))
            
            # For calls, we want ITM = strike BELOW current price
            if option_type.lower() == "call" and strike > price:
                continue  # Skip OTM strikes
            # For puts, we want ITM = strike ABOVE current price  
            if option_type.lower() == "put" and strike < price:
                continue  # Skip OTM strikes
                
            diff = abs(strike - target_strike)
            if diff < best_diff:
                best_diff = diff
                best = c
        except (ValueError, TypeError):
            continue
    
    if best:
        selected_strike = float(best.get("strike_price", 0))
        itm_pct = abs(price - selected_strike) / price * 100
        print(f"[options] Selected ITM: {best.get('symbol')} strike={selected_strike} ({itm_pct:.1f}% ITM, underlying={price:.2f})")
    else:
        print(f"[options] No ITM contract found for {underlying}, falling back to ATM")
        return select_atm_contract(underlying, expiration_date, option_type)
    
    return best


def get_option_quote(option_symbol: str) -> Optional[Dict[str, float]]:
    """
    Get real-time quote for an option contract.
    
    Returns:
        {"bid": float, "ask": float, "mid": float} or None
    """
    base, key, secret = _resolve_creds()
    
    # Use options quotes endpoint
    data_url = f"https://data.alpaca.markets/v1beta1/options/quotes/latest"
    params = {"symbols": option_symbol}
    
    try:
        resp = requests.get(data_url, headers=_headers(key, secret), params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        quotes = data.get("quotes", {})
        quote = quotes.get(option_symbol, {})
        
        bid = float(quote.get("bp", 0))
        ask = float(quote.get("ap", 0))
        
        if bid > 0 and ask > 0:
            return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2}
        return None
    except Exception as e:
        print(f"[options] Error fetching quote for {option_symbol}: {e}")
        return None


def get_option_snapshot(option_symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get snapshot (quote + greeks) for an option contract from Alpaca.
    
    Returns:
        Dict with bid, ask, mid, iv, delta, theta, gamma, vega or None
    """
    base, key, secret = _resolve_creds()
    
    # Use multi-symbol endpoint (single-symbol endpoint is unreliable for options)
    data_url = f"https://data.alpaca.markets/v1beta1/options/snapshots"
    params = {"symbols": option_symbol}
    
    try:
        resp = requests.get(data_url, headers=_headers(key, secret), params=params, timeout=10)
        if resp.status_code == 404:
            print(f"[options] No snapshot for {option_symbol}")
            return None
        resp.raise_for_status()
        data = resp.json()
        
        # Extract snapshot for specific symbol
        snapshot = data.get("snapshots", {}).get(option_symbol, {})
        if not snapshot:
             return None

        # Extract quote
        quote = snapshot.get("latestQuote", {})
        bid = float(quote.get("bp", 0))
        ask = float(quote.get("ap", 0))
        
        # Extract greeks
        greeks = snapshot.get("greeks", {})
        
        return {
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2 if bid > 0 and ask > 0 else 0,
            "iv": float(greeks.get("implied_volatility", 0)),
            "delta": float(greeks.get("delta", 0)),
            "theta": float(greeks.get("theta", 0)),
            "gamma": float(greeks.get("gamma", 0)),
            "vega": float(greeks.get("vega", 0)),
        }
    except Exception as e:
        print(f"[options] Error fetching snapshot for {option_symbol}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────
def is_options_trading_allowed() -> bool:
    """Check if we're within options trading hours (before 3:00 PM ET cutoff)."""
    now_et = datetime.now(ET)
    cutoff = now_et.replace(hour=15, minute=45, second=0, microsecond=0)
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    
    # Must be after market open and before 3:45 PM
    return market_open <= now_et < cutoff


def format_option_symbol(
    underlying: str,
    expiration: str,  # YYYY-MM-DD
    option_type: str,  # "call" or "put"
    strike: float,
) -> str:
    """
    Format an OCC option symbol.
    
    Format: SYMBOL + YYMMDD + C/P + 00000000 (strike * 1000, 8 digits)
    Example: NVDA241204C00140000 = NVDA Dec 4, 2024 $140 Call
    """
    # Parse date
    dt = datetime.strptime(expiration, "%Y-%m-%d")
    date_str = dt.strftime("%y%m%d")
    
    # Option type
    type_char = "C" if option_type.lower() == "call" else "P"
    
    # Strike (8 digits, strike * 1000)
    strike_int = int(strike * 1000)
    strike_str = f"{strike_int:08d}"
    
    return f"{underlying.upper()}{date_str}{type_char}{strike_str}"


def select_spread_contracts(
    underlying: str,
    dte: int = 3,
    spread_width_atr: float = 1.5,
    atr: float = None,
    min_dte: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Select contracts for a bull call spread.
    
    Args:
        underlying: Stock symbol
        dte: Target days to expiration (1-3 recommended)
        spread_width_atr: OTM strike = ATM + this * ATR (default 1.5)
        atr: Average True Range for volatility-based spread width
        min_dte: Minimum DTE required (skips expirations below this)
    
    Returns:
        Dict with 'long' (ATM) and 'short' (OTM) contracts, or None
    """
    # Get current price first
    price = get_underlying_price(underlying)
    if price is None:
        print(f"[spreads] Cannot get price for {underlying}")
        return None
    
    # Try current week first (dte ± 2), then next week (dte + 7 ± 2) if no match
    # This ensures we find a valid expiration even mid-week when DTE < min_dte
    current_week = [dte, dte - 1, dte + 1, dte - 2, dte + 2]
    next_week = [dte + 7, dte + 5, dte + 6, dte + 8, dte + 9]  # Next Friday
    dte_attempts = current_week + next_week
    dte_attempts = [d for d in dte_attempts if d >= 0]  # No negative DTE
    
    # If min_dte specified, filter out attempts below threshold
    if min_dte is not None:
        dte_attempts = [d for d in dte_attempts if d >= min_dte]
        if not dte_attempts:
            print(f"[spreads] No valid DTE options for {underlying} with min_dte={min_dte}")
            return None
    
    contracts = None
    used_expiration = None
    
    for try_dte in dte_attempts:
        expiration = get_ndte_expiration(try_dte)
        contracts = fetch_option_contracts(underlying, expiration, "call", limit=200)
        if contracts:
            used_expiration = expiration
            if try_dte != dte:
                print(f"[spreads] Using {try_dte} DTE ({expiration}) instead of {dte} DTE for {underlying}")
            break
    
    if not contracts:
        print(f"[spreads] No contracts for {underlying} in DTE range {min(dte_attempts)}-{max(dte_attempts)}")
        return None
    
    # Filter active, tradable contracts
    active = [c for c in contracts if c.get("status") == "active" and c.get("tradable")]
    if not active:
        return None
    
    # Sort by strike
    active.sort(key=lambda c: float(c.get("strike_price", 0)))
    
    # Find ATM (long leg) - closest to current price
    atm_contract = None
    atm_diff = float("inf")
    for c in active:
        strike = float(c.get("strike_price", 0))
        diff = abs(strike - price)
        if diff < atm_diff:
            atm_diff = diff
            atm_contract = c
    
    if not atm_contract:
        return None
    
    atm_strike = float(atm_contract.get("strike_price", 0))
    
    # Find OTM (short leg) - strike above ATM by spread width
    # ATR-based: target = atm_strike + (atr * spread_width_atr)
    # Fallback to 2% of price if ATR not provided
    if atr is not None and atr > 0:
        target_otm = atm_strike + (atr * spread_width_atr)
    else:
        # Fallback: use 2% of price if no ATR provided
        target_otm = atm_strike + (price * 0.02)
        print(f"[spreads] WARNING: No ATR for {underlying}, using 2% fallback")
    otm_contract = None
    otm_diff = float("inf")
    
    for c in active:
        strike = float(c.get("strike_price", 0))
        if strike <= atm_strike:
            continue  # Must be above ATM
        diff = abs(strike - target_otm)
        if diff < otm_diff:
            otm_diff = diff
            otm_contract = c
    
    if not otm_contract:
        # Fallback: just use next strike above ATM
        for c in active:
            strike = float(c.get("strike_price", 0))
            if strike > atm_strike:
                otm_contract = c
                break
    
    if not otm_contract:
        print(f"[spreads] No OTM strike found for {underlying}")
        return None
    
    otm_strike = float(otm_contract.get("strike_price", 0))
    spread_width = otm_strike - atm_strike
    
    # Validate spread width is positive
    if spread_width <= 0:
        print(f"[spreads] Invalid spread width for {underlying}: ATM={atm_strike}, OTM={otm_strike}")
        return None
    
    print(f"[spreads] {underlying}: Long {atm_strike} / Short {otm_strike} (width=${spread_width:.2f}, exp={used_expiration})")
    
    return {
        "underlying": underlying,
        "expiration": used_expiration,
        "dte": dte,
        "underlying_price": price,
        "long": atm_contract,  # ATM call (buy)
        "short": otm_contract,  # OTM call (sell)
        "atm_strike": atm_strike,
        "otm_strike": otm_strike,
        "spread_width": spread_width,
    }

