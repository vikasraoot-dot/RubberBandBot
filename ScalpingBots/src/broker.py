"""
Alpaca broker API helpers for ScalpingBots.
Self-contained module — no dependency on RubberBand.
"""
import os
import datetime as dt
from typing import Dict, List, Optional, Any, Tuple

import requests


# ── Credentials ──────────────────────────────────────────────────────────────

def _resolve_key_secret(key: Optional[str], secret: Optional[str]) -> Tuple[str, str]:
    k = (key or os.getenv("APCA_API_KEY_ID") or "").strip()
    s = (secret or os.getenv("APCA_API_SECRET_KEY") or "").strip()
    return k, s


def _alpaca_headers(key: Optional[str] = None, secret: Optional[str] = None) -> Dict[str, str]:
    k, s = _resolve_key_secret(key, secret)
    return {
        "APCA-API-KEY-ID": k,
        "APCA-API-SECRET-KEY": s,
        "Content-Type": "application/json",
    }


def _base_url_from_env(base_url: Optional[str] = None) -> str:
    return (
        base_url
        or os.getenv("APCA_API_BASE_URL")
        or "https://paper-api.alpaca.markets"
    ).rstrip("/")


# ── Account / Market ─────────────────────────────────────────────────────────

def alpaca_market_open(
    base_url: Optional[str] = None,
    key: Optional[str] = None,
    secret: Optional[str] = None,
) -> bool:
    base = _base_url_from_env(base_url)
    try:
        r = requests.get(
            f"{base}/v2/clock",
            headers=_alpaca_headers(key, secret),
            timeout=10,
        )
        if r.status_code == 401:
            return False
        r.raise_for_status()
        return bool((r.json() or {}).get("is_open"))
    except Exception as e:
        print(f"[warn] alpaca_market_open error: {type(e).__name__}: {e}")
        return False


def get_account(
    base_url: Optional[str] = None,
    key: Optional[str] = None,
    secret: Optional[str] = None,
) -> Dict[str, Any]:
    base = _base_url_from_env(base_url)
    try:
        r = requests.get(
            f"{base}/v2/account",
            headers=_alpaca_headers(key, secret),
            timeout=10,
        )
        r.raise_for_status()
        return r.json() or {}
    except Exception as e:
        print(f"[warn] get_account error: {type(e).__name__}: {e}")
        return {}


# ── Positions ────────────────────────────────────────────────────────────────

def get_positions(
    base_url: Optional[str] = None,
    key: Optional[str] = None,
    secret: Optional[str] = None,
) -> List[Dict[str, Any]]:
    base = _base_url_from_env(base_url)
    try:
        r = requests.get(
            f"{base}/v2/positions",
            headers=_alpaca_headers(key, secret),
            timeout=12,
        )
        if r.status_code == 404:
            return []
        r.raise_for_status()
        arr = r.json() or []
        return arr if isinstance(arr, list) else []
    except Exception as e:
        print(f"[warn] get_positions error: {type(e).__name__}: {e}")
        return []


def close_position(
    base_url: Optional[str] = None,
    key: Optional[str] = None,
    secret: Optional[str] = None,
    symbol: str = "",
) -> Dict[str, Any]:
    """Close a position via Alpaca DELETE /v2/positions/{symbol}.

    Returns:
        Dict with 'ok' bool, plus 'order_id' and other order fields
        when the broker returns a closing order in the response body.
    """
    base = _base_url_from_env(base_url)
    r = requests.delete(
        f"{base}/v2/positions/{symbol}",
        headers=_alpaca_headers(key, secret),
        timeout=15,
    )
    if r.status_code == 404:
        return {"ok": True, "message": "Position not found"}
    if r.status_code not in (200, 204):
        r.raise_for_status()

    # Parse response body — Alpaca returns the closing order object on 200
    result: Dict[str, Any] = {"ok": True}
    if r.content:
        try:
            body = r.json()
            if isinstance(body, dict):
                order_id = body.get("id", "")
                if order_id:
                    result["order_id"] = order_id
                result["status"] = body.get("status", "")
                filled_price = body.get("filled_avg_price")
                if filled_price is not None:
                    result["filled_avg_price"] = filled_price
                result["filled_qty"] = body.get("filled_qty")
                result["symbol"] = body.get("symbol", symbol)
        except (ValueError, KeyError) as e:
            print(f"[warn] close_position: failed to parse response body for {symbol}: {e}")
    return result


# ── Quotes ───────────────────────────────────────────────────────────────────

def get_latest_quote(
    base_url: Optional[str] = None,
    key: Optional[str] = None,
    secret: Optional[str] = None,
    symbol: str = "",
    feed: str = "iex",
) -> Dict[str, Any]:
    """Get latest bid/ask quote for a stock symbol.

    Args:
        feed: Data feed — "iex" (free, potentially stale) or "sip" (paid, real-time).
    """
    data_url = "https://data.alpaca.markets"
    try:
        r = requests.get(
            f"{data_url}/v2/stocks/{symbol}/quotes/latest",
            headers=_alpaca_headers(key, secret),
            params={"feed": feed},
            timeout=10,
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


# ── Orders / Fills ───────────────────────────────────────────────────────────

def get_daily_fills(
    base_url: Optional[str] = None,
    key: Optional[str] = None,
    secret: Optional[str] = None,
    bot_tag: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch all filled orders for the current UTC day."""
    base = _base_url_from_env(base_url)
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    params = {
        "status": "closed",
        "limit": 500,
        "after": f"{today}T00:00:00Z",
    }
    try:
        r = requests.get(
            f"{base}/v2/orders",
            headers=_alpaca_headers(key, secret),
            params=params,
            timeout=15,
        )
        r.raise_for_status()
        orders = r.json() or []
        fills = [
            o for o in orders
            if o.get("status") == "filled" and o.get("filled_qty") is not None
        ]

        if bot_tag:
            tagged = [
                f for f in fills
                if (f.get("client_order_id") or "").startswith(f"{bot_tag}_")
            ]
            tagged_buy_syms = {
                f.get("symbol", "")
                for f in tagged
                if f.get("side") == "buy"
            }
            extra_sells = [
                f for f in fills
                if f.get("side") == "sell"
                and f.get("symbol", "") in tagged_buy_syms
                and not (f.get("client_order_id") or "").startswith(f"{bot_tag}_")
            ]
            fills = tagged + extra_sells

        return fills
    except Exception as e:
        print(f"[warn] Failed to fetch daily fills: {e}")
        return []


def calculate_realized_pnl(fills: List[Dict[str, Any]]) -> float:
    """
    Realized P&L from matched buy/sell pairs only.
    Unmatched buys (open positions) are NOT counted as losses.
    """
    pnl = 0.0
    symbol_fills: Dict[str, List[Dict[str, Any]]] = {}
    for f in fills:
        sym = f.get("symbol", "")
        if sym:
            symbol_fills.setdefault(sym, []).append(f)

    for sym, sym_fills in symbol_fills.items():
        buys = [f for f in sym_fills if f.get("side") == "buy"]
        sells = [f for f in sym_fills if f.get("side") == "sell"]

        total_buy_qty = sum(int(f.get("filled_qty", 0)) for f in buys)
        total_sell_qty = sum(int(f.get("filled_qty", 0)) for f in sells)
        matched_qty = min(total_buy_qty, total_sell_qty)

        if matched_qty <= 0:
            continue

        total_buy_cost = sum(
            float(f.get("filled_avg_price", 0)) * int(f.get("filled_qty", 0))
            for f in buys
        )
        total_sell_proceeds = sum(
            float(f.get("filled_avg_price", 0)) * int(f.get("filled_qty", 0))
            for f in sells
        )

        avg_buy = total_buy_cost / total_buy_qty if total_buy_qty else 0
        avg_sell = total_sell_proceeds / total_sell_qty if total_sell_qty else 0

        pnl += (avg_sell - avg_buy) * matched_qty

    return pnl
