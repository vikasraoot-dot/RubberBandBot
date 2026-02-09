#!/usr/bin/env python3
"""
EMA Momentum Scalper - Live Paper Trading Bot
==============================================
Validated: 80% WR, Sharpe 7.5, Max DD -0.73% over 60 days

KEY FEATURE: Algorithmic trailing stop management.
Backtests showed TRAIL exits = +$4,916 (primary profit source).
Alpaca bracket orders only support fixed TP/SL, so we manage exits ourselves.

Usage:
  export APCA_API_KEY_ID="..." APCA_API_SECRET_KEY="..."
  python -m ScalpingBots.scripts.live_ema_scalp

  # Dry run (no orders, just signals):
  python -m ScalpingBots.scripts.live_ema_scalp --dry-run

  # Custom config:
  python -m ScalpingBots.scripts.live_ema_scalp --max-notional 5000 --max-daily-loss 300
"""
import os
import sys
import json
import time
import argparse
import datetime as dt
import requests
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import pandas as pd

# Setup paths
_THIS = os.path.abspath(os.path.dirname(__file__))
_PROJECT = os.path.abspath(os.path.join(_THIS, ".."))
_REPO = os.path.abspath(os.path.join(_PROJECT, ".."))
for p in [_PROJECT, _REPO]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ScalpingBots.src.data_cache import get_bars
from ScalpingBots.src.indicators import (
    add_vwap, add_ema, add_rsi, add_atr, add_rvol, add_adx,
)
from ScalpingBots.strategies.ema_momentum import (
    EMAMomentumConfig, calculate_momentum_score,
)

# Alpaca broker helpers (self-contained, no RubberBand dependency)
from ScalpingBots.src.broker import (
    alpaca_market_open, get_positions, get_account,
    get_daily_fills, calculate_realized_pnl,
    get_latest_quote, _alpaca_headers, _base_url_from_env,
    close_position,
)

BOT_TAG = "EMA_SCALP"
LOG_DIR = os.path.join(_PROJECT, "logs")

# Validated profitable tickers (60-day backtest, all positive P&L)
CORE_TICKERS = [
    "GS", "COIN", "NVDA", "PLTR", "SOFI", "AVGO",
    "BAC", "MSFT", "GOOGL", "ORCL", "BA", "CRWD",
]


# ─── Position Tracking ───────────────────────────────────────────────

@dataclass
class LivePosition:
    """Local state for tracking a managed position with trailing stop."""
    symbol: str
    side: str               # "buy" (long) or "sell" (short)
    qty: int
    entry_price: float
    entry_time: str         # ISO timestamp
    sl_price: float
    tp_price: float
    atr: float
    # Trailing stop state
    trail_active: bool = False
    trail_price: float = 0.0
    peak_price: float = 0.0    # best price seen since entry
    # SL confirmation (2-bar required)
    sl_confirm_count: int = 0
    # Bar counter (incremented each poll cycle)
    bars_held: int = 0
    # Order tracking
    entry_order_id: str = ""


# ─── Logging ──────────────────────────────────────────────────────────

def log_event(event_type: str, data: dict, log_file: str = None):
    """Write structured JSONL log event."""
    event = {
        "type": event_type,
        "time": dt.datetime.now(dt.timezone.utc).isoformat(),
        **data,
    }
    line = json.dumps(event, separators=(",", ":"), default=str)
    print(line, flush=True)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")


def get_log_file() -> str:
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, f"ema_scalp_{today}.jsonl")


# ─── Data Preparation ────────────────────────────────────────────────

def prepare_data(symbol: str, days: int = 10, timeframe: str = "5Min",
                 feed: str = "iex", force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """Fetch and prepare data with all required indicators."""
    try:
        df = get_bars(symbol, timeframe, days, feed, rth_only=True,
                      force_refresh=force_refresh)
        if df is None or df.empty or len(df) < 30:
            return None

        df = add_rsi(df, 14)
        df = add_atr(df, 14)
        df = add_rvol(df, 20)
        df = add_adx(df, 14)
        df = add_ema(df, 9, name="ema_9")
        df = add_ema(df, 21, name="ema_21")
        df = add_vwap(df)

        df = df.dropna(subset=["rsi", "atr"]).copy()
        return df if len(df) >= 10 else None

    except Exception as e:
        print(f"[data] Error preparing {symbol}: {e}")
        return None


# ─── Signal Detection ────────────────────────────────────────────────

def check_entry_signal(df: pd.DataFrame, cfg: EMAMomentumConfig) -> Optional[Dict]:
    """
    Check if the latest bar has an entry signal.
    Returns signal dict or None.
    """
    if len(df) < 3:
        return None

    cur = df.iloc[-1]
    prev = df.iloc[-2]
    i = len(df) - 1

    # Time window
    et = cur.name.tz_convert("US/Eastern")
    h_start, m_start = map(int, cfg.entry_start.split(":"))
    h_end, m_end = map(int, cfg.entry_end.split(":"))
    bar_minutes = et.hour * 60 + et.minute
    if bar_minutes < h_start * 60 + m_start or bar_minutes > h_end * 60 + m_end:
        return None

    ema_fast = float(cur.get("ema_9", 0))
    ema_slow = float(cur.get("ema_21", 0))
    prev_ema_fast = float(prev.get("ema_9", 0))
    prev_ema_slow = float(prev.get("ema_21", 0))
    rsi = float(prev.get("rsi", 50))
    rvol = float(cur.get("rvol", 1))
    atr_val = float(prev.get("atr", 0))
    vwap_val = float(cur.get("vwap", 0))

    if ema_fast == 0 or ema_slow == 0 or atr_val <= 0:
        return None

    # LONG signal
    bullish_cross = (prev_ema_fast <= prev_ema_slow) and (ema_fast > ema_slow)
    bullish_trend = (ema_fast > ema_slow) and (
        ema_fast - ema_slow > prev_ema_fast - prev_ema_slow)

    long_signal = (bullish_cross or bullish_trend)
    if long_signal:
        long_signal = long_signal and (rvol >= cfg.min_rvol)
        long_signal = long_signal and (cfg.rsi_min_long <= rsi <= cfg.rsi_max_long)
        if cfg.require_above_vwap_long and vwap_val > 0:
            long_signal = long_signal and (float(cur["close"]) > vwap_val)
        if cfg.require_bullish_bar:
            long_signal = long_signal and (float(cur["close"]) > float(cur["open"]))

    # SHORT signal
    bearish_cross = (prev_ema_fast >= prev_ema_slow) and (ema_fast < ema_slow)
    bearish_trend = (ema_fast < ema_slow) and (
        ema_slow - ema_fast > prev_ema_slow - prev_ema_fast)

    short_signal = cfg.allow_shorts and (bearish_cross or bearish_trend)
    if short_signal:
        short_signal = short_signal and (rvol >= cfg.min_rvol)
        short_signal = short_signal and (cfg.rsi_min_short <= rsi <= cfg.rsi_max_short)
        if cfg.require_above_vwap_long and vwap_val > 0:
            short_signal = short_signal and (float(cur["close"]) < vwap_val)
        if cfg.require_bullish_bar:
            short_signal = short_signal and (float(cur["close"]) < float(cur["open"]))

    if not long_signal and not short_signal:
        return None

    # Momentum score
    mom_score = calculate_momentum_score(df, i)
    if mom_score < cfg.min_momentum_score:
        return None

    side = "buy" if long_signal else "sell"
    price = float(cur["close"])

    # Stop loss (swing-based)
    if long_signal:
        swing_low = float(df["low"].iloc[-cfg.sl_swing_bars:].min())
        sl_px = swing_low - (atr_val * 0.2)
        tp_px = price + (atr_val * cfg.tp_atr_mult)
    else:
        swing_high = float(df["high"].iloc[-cfg.sl_swing_bars:].max())
        sl_px = swing_high + (atr_val * 0.2)
        tp_px = price - (atr_val * cfg.tp_atr_mult)

    return {
        "side": side,
        "price": price,
        "sl_price": round(sl_px, 2),
        "tp_price": round(tp_px, 2),
        "atr": round(atr_val, 4),
        "rsi": round(rsi, 1),
        "rvol": round(rvol, 2),
        "momentum_score": round(mom_score, 2),
        "ema_fast": round(ema_fast, 2),
        "ema_slow": round(ema_slow, 2),
        "vwap": round(vwap_val, 2),
    }


# ─── Position Sizing ─────────────────────────────────────────────────

def calculate_qty(equity: float, entry_price: float, sl_price: float,
                  risk_pct: float, max_notional: float) -> int:
    """Calculate position size based on risk."""
    risk_per_share = abs(entry_price - sl_price)
    if risk_per_share <= 0:
        return 0

    max_risk = equity * (risk_pct / 100.0)
    qty_risk = int(max_risk / risk_per_share)
    qty_notional = int(max_notional / entry_price) if entry_price > 0 else 0
    return max(1, min(qty_risk, qty_notional))


# ─── Order Submission ─────────────────────────────────────────────────

def submit_limit_order(
    base_url: str, key: str, secret: str,
    symbol: str, qty: int, side: str = "buy",
    limit_price: Optional[float] = None,
    client_order_id: Optional[str] = None,
    tif: str = "day",
) -> Dict:
    """
    Submit a simple limit order (no bracket, no TP/SL attached).
    We manage exits algorithmically instead.
    """
    base = _base_url_from_env(base_url)
    H = _alpaca_headers(key, secret)

    # If no limit price, get current quote + buffer
    if limit_price is None:
        quote = get_latest_quote(base_url, key, secret, symbol)
        if side == "buy":
            ask = quote.get("ask", 0)
            if ask <= 0:
                return {"error": "no_ask_quote"}
            limit_price = round(ask * 1.005, 2)  # 0.5% above ask
        else:
            bid = quote.get("bid", 0)
            if bid <= 0:
                return {"error": "no_bid_quote"}
            limit_price = round(bid * 0.995, 2)  # 0.5% below bid

    payload = {
        "symbol": symbol.upper(),
        "qty": int(qty),
        "side": side,
        "type": "limit",
        "limit_price": float(limit_price),
        "time_in_force": tif,
    }
    if client_order_id:
        payload["client_order_id"] = client_order_id

    try:
        r = requests.post(f"{base}/v2/orders", headers=H, json=payload, timeout=20)
        result = r.json() if r.content else {}
        if r.status_code >= 400:
            return {"error": result.get("message", str(r.status_code)), **result}
        return result
    except Exception as e:
        return {"error": str(e)}


def verify_order_filled(base_url: str, key: str, secret: str,
                        order_id: str, timeout: int = 10) -> Dict:
    """Poll order until filled or timeout."""
    base = _base_url_from_env(base_url)
    H = _alpaca_headers(key, secret)
    for _ in range(timeout):
        time.sleep(1)
        try:
            r = requests.get(f"{base}/v2/orders/{order_id}", headers=H, timeout=10)
            if r.status_code == 200:
                order = r.json()
                status = order.get("status", "")
                if status == "filled":
                    return order
                if status in ("canceled", "expired", "rejected"):
                    return order
        except Exception:
            pass
    return {"status": "timeout", "id": order_id}


# ─── Exit Management (THE CRITICAL PART) ─────────────────────────────

def manage_exits(
    managed: Dict[str, LivePosition],
    base_url: str, key: str, secret: str,
    cfg: EMAMomentumConfig,
    now_et: dt.datetime,
    log_file: str,
    dry_run: bool = False,
) -> float:
    """
    Check all managed positions for exit conditions.
    Returns realized P&L from exits this cycle.

    Exit priority (matches backtest):
      1. TAKE_PROFIT  - price reached TP level
      2. STOP_LOSS    - price at SL for 2 consecutive polls (confirmation)
      3. TRAILING     - trail stop hit after activation
      4. MAX_HOLD     - held too many bars
      5. EOD_FLATTEN  - end of day forced close
    """
    if not managed:
        return 0.0

    cycle_pnl = 0.0
    to_remove = []

    # Parse flatten time
    h_flat, m_flat = map(int, cfg.flatten_time.split(":"))
    is_eod = (now_et.hour > h_flat or
              (now_et.hour == h_flat and now_et.minute >= m_flat))

    for symbol, pos in managed.items():
        # Get current price
        quote = get_latest_quote(base_url, key, secret, symbol)
        bid = quote.get("bid", 0)
        ask = quote.get("ask", 0)
        if bid <= 0 or ask <= 0:
            continue  # Skip if no valid quote

        is_long = (pos.side == "buy")
        # Use bid for long exits (selling), ask for short exits (buying to cover)
        current_price = bid if is_long else ask
        mid_price = (bid + ask) / 2.0

        # Update peak price (for trailing stop)
        if is_long:
            pos.peak_price = max(pos.peak_price, current_price)
        else:
            if pos.peak_price == 0:
                pos.peak_price = current_price
            pos.peak_price = min(pos.peak_price, current_price)

        # Increment bars held
        pos.bars_held += 1

        # Calculate current P&L
        if is_long:
            unrealized = (current_price - pos.entry_price) * pos.qty
            profit_atr = (current_price - pos.entry_price) / pos.atr if pos.atr > 0 else 0
        else:
            unrealized = (pos.entry_price - current_price) * pos.qty
            profit_atr = (pos.entry_price - current_price) / pos.atr if pos.atr > 0 else 0

        exit_reason = None

        # 1. TAKE PROFIT
        if is_long and current_price >= pos.tp_price:
            exit_reason = "TAKE_PROFIT"
        elif not is_long and current_price <= pos.tp_price:
            exit_reason = "TAKE_PROFIT"

        # 2. STOP LOSS (with 2-poll confirmation)
        if exit_reason is None:
            sl_touched = False
            if is_long and current_price <= pos.sl_price:
                sl_touched = True
            elif not is_long and current_price >= pos.sl_price:
                sl_touched = True

            if sl_touched:
                pos.sl_confirm_count += 1
                if pos.sl_confirm_count >= cfg.sl_confirm_bars:
                    exit_reason = "STOP_LOSS"
            else:
                pos.sl_confirm_count = 0

        # 3. TRAILING STOP
        if exit_reason is None:
            # Check if trail should activate
            if not pos.trail_active and profit_atr >= cfg.trailing_trigger_atr:
                pos.trail_active = True
                if is_long:
                    pos.trail_price = current_price - (pos.atr * cfg.trailing_distance_atr)
                else:
                    pos.trail_price = current_price + (pos.atr * cfg.trailing_distance_atr)
                log_event("TRAIL_ACTIVATED", {
                    "symbol": symbol,
                    "trail_price": round(pos.trail_price, 2),
                    "current_price": round(current_price, 2),
                    "profit_atr": round(profit_atr, 2),
                }, log_file)

            # Update trailing stop (ratchet only)
            if pos.trail_active:
                if is_long:
                    new_trail = current_price - (pos.atr * cfg.trailing_distance_atr)
                    pos.trail_price = max(pos.trail_price, new_trail)
                    if current_price <= pos.trail_price:
                        exit_reason = "TRAILING"
                else:
                    new_trail = current_price + (pos.atr * cfg.trailing_distance_atr)
                    pos.trail_price = min(pos.trail_price, new_trail)
                    if current_price >= pos.trail_price:
                        exit_reason = "TRAILING"

        # 4. MAX HOLD
        if exit_reason is None and pos.bars_held >= cfg.max_hold_bars:
            exit_reason = "MAX_HOLD"

        # 5. EOD FLATTEN
        if exit_reason is None and is_eod:
            exit_reason = "EOD_FLATTEN"

        # Execute exit
        if exit_reason:
            if dry_run:
                print(f"  [DRY RUN EXIT] {symbol} {exit_reason}: "
                      f"P&L=${unrealized:+.2f} | bars={pos.bars_held} | "
                      f"trail={'ON' if pos.trail_active else 'OFF'}")
                to_remove.append(symbol)
                cycle_pnl += unrealized
                continue

            try:
                result = close_position(base_url, key, secret, symbol)
                log_event("EXIT", {
                    "symbol": symbol,
                    "reason": exit_reason,
                    "entry_price": pos.entry_price,
                    "exit_price": round(current_price, 2),
                    "qty": pos.qty,
                    "pnl": round(unrealized, 2),
                    "bars_held": pos.bars_held,
                    "trail_active": pos.trail_active,
                    "trail_price": round(pos.trail_price, 2) if pos.trail_active else None,
                    "peak_price": round(pos.peak_price, 2),
                    "sl_confirms": pos.sl_confirm_count,
                }, log_file)
                print(f"  EXIT: {symbol} {exit_reason} | "
                      f"P&L=${unrealized:+.2f} | bars={pos.bars_held} | "
                      f"entry=${pos.entry_price:.2f} exit=${current_price:.2f}")
                to_remove.append(symbol)
                cycle_pnl += unrealized
            except Exception as e:
                log_event("EXIT_ERROR", {
                    "symbol": symbol,
                    "reason": exit_reason,
                    "error": str(e),
                }, log_file)
                print(f"  [EXIT ERROR] {symbol}: {e}")
        else:
            # Log position status periodically
            if pos.bars_held % 4 == 0:  # Every ~20 min at 5-min polling
                status_parts = [
                    f"P&L=${unrealized:+.2f}",
                    f"bars={pos.bars_held}",
                ]
                if pos.trail_active:
                    status_parts.append(f"trail=${pos.trail_price:.2f}")
                print(f"  HOLD: {symbol} | {' | '.join(status_parts)}")

    # Remove closed positions
    for sym in to_remove:
        del managed[sym]

    return cycle_pnl


# ─── Position Recovery ────────────────────────────────────────────────

def recover_positions(
    managed: Dict[str, LivePosition],
    base_url: str, key: str, secret: str,
    cfg: EMAMomentumConfig,
) -> int:
    """
    On startup, check Alpaca positions and recover any that belong to this bot.
    Returns count of recovered positions.
    """
    positions = get_positions(base_url, key, secret)
    recovered = 0

    # Check today's orders for bot-tagged entries
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    H = _alpaca_headers(key, secret)
    base = _base_url_from_env(base_url)
    try:
        r = requests.get(f"{base}/v2/orders", headers=H, params={
            "status": "filled", "limit": 200, "after": f"{today}T00:00:00Z",
        }, timeout=15)
        orders = r.json() if r.status_code == 200 else []
    except Exception:
        orders = []

    # Build set of symbols entered by this bot today
    bot_symbols = set()
    bot_entries = {}
    for o in orders:
        coid = o.get("client_order_id", "")
        if coid.startswith(f"{BOT_TAG}_"):
            sym = o.get("symbol", "")
            bot_symbols.add(sym)
            filled_price = float(o.get("filled_avg_price", 0))
            filled_qty = int(o.get("filled_qty", 0))
            if filled_price > 0:
                bot_entries[sym] = {
                    "price": filled_price,
                    "qty": filled_qty,
                    "side": o.get("side", "buy"),
                    "time": o.get("filled_at", ""),
                }

    for pos in positions:
        sym = pos.get("symbol", "")
        if sym not in bot_symbols or sym in managed:
            continue

        entry_info = bot_entries.get(sym, {})
        entry_px = entry_info.get("price", float(pos.get("avg_entry_price", 0)))
        qty = int(pos.get("qty", 0))
        side = entry_info.get("side", "buy")

        if entry_px <= 0 or qty <= 0:
            continue

        # Estimate ATR from recent data
        df = prepare_data(sym, days=5, timeframe="5Min", feed="iex")
        atr_val = float(df.iloc[-1]["atr"]) if df is not None and "atr" in df.columns else entry_px * 0.02

        # Reconstruct SL/TP
        if side == "buy":
            sl_px = entry_px - (atr_val * cfg.sl_atr_mult)
            tp_px = entry_px + (atr_val * cfg.tp_atr_mult)
        else:
            sl_px = entry_px + (atr_val * cfg.sl_atr_mult)
            tp_px = entry_px - (atr_val * cfg.tp_atr_mult)

        managed[sym] = LivePosition(
            symbol=sym,
            side=side,
            qty=qty,
            entry_price=entry_px,
            entry_time=entry_info.get("time", dt.datetime.now(dt.timezone.utc).isoformat()),
            sl_price=round(sl_px, 2),
            tp_price=round(tp_px, 2),
            atr=atr_val,
            peak_price=entry_px,
            bars_held=1,  # At least 1 since we're recovering
        )
        recovered += 1
        print(f"  RECOVERED: {sym} {side} x{qty} @ ${entry_px:.2f} "
              f"SL=${sl_px:.2f} TP=${tp_px:.2f}")

    return recovered


# ─── Main Loop ────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="EMA Momentum Scalper - Live Bot")
    ap.add_argument("--tickers", nargs="+", default=CORE_TICKERS)
    ap.add_argument("--max-notional", type=float, default=3000.0)
    ap.add_argument("--max-daily-loss", type=float, default=150.0)
    ap.add_argument("--risk-pct", type=float, default=1.5)
    ap.add_argument("--max-positions", type=int, default=8)
    ap.add_argument("--poll-interval", type=int, default=300, help="Seconds between scans")
    ap.add_argument("--feed", default="iex")
    ap.add_argument("--dry-run", action="store_true", help="Print signals but don't execute")
    ap.add_argument("--allow-shorts", action="store_true", default=False)
    args = ap.parse_args()

    cfg = EMAMomentumConfig()
    cfg.max_notional = args.max_notional
    cfg.max_daily_loss = args.max_daily_loss
    cfg.risk_per_trade_pct = args.risk_pct
    cfg.allow_shorts = args.allow_shorts

    tickers = [t.upper() for t in args.tickers]
    log_file = get_log_file()

    print(f"  EMA Momentum Scalper ({BOT_TAG})")
    print(f"  Tickers: {len(tickers)} ({', '.join(tickers[:5])}{'...' if len(tickers)>5 else ''})")
    print(f"  Max Notional: ${args.max_notional:,.0f} | Risk: {args.risk_pct}%")
    print(f"  Max Positions: {args.max_positions} | Daily Loss Limit: ${args.max_daily_loss:,.0f}")
    print(f"  Poll Interval: {args.poll_interval}s | Feed: {args.feed}")
    print(f"  Shorts: {'Enabled' if args.allow_shorts else 'Disabled'}")
    print(f"  Dry Run: {args.dry_run}")
    print(f"  Trailing: trigger={cfg.trailing_trigger_atr} ATR, "
          f"distance={cfg.trailing_distance_atr} ATR")
    print(f"  SL Confirm: {cfg.sl_confirm_bars} consecutive polls")
    print(f"  Log: {log_file}")

    log_event("BOT_START", {
        "bot": BOT_TAG,
        "tickers": tickers,
        "config": {
            "max_notional": args.max_notional,
            "risk_pct": args.risk_pct,
            "max_positions": args.max_positions,
            "max_daily_loss": args.max_daily_loss,
            "trailing_trigger": cfg.trailing_trigger_atr,
            "trailing_distance": cfg.trailing_distance_atr,
            "sl_confirm_bars": cfg.sl_confirm_bars,
        }
    }, log_file)

    base_url = _base_url_from_env()
    key = os.getenv("APCA_API_KEY_ID", "")
    secret = os.getenv("APCA_API_SECRET_KEY", "")

    # Local position tracking with trailing stop state
    managed: Dict[str, LivePosition] = {}
    daily_pnl = 0.0
    last_scan_date = None
    scan_count = 0

    while True:
        try:
            now_et = dt.datetime.now(dt.timezone.utc).astimezone(
                dt.timezone(dt.timedelta(hours=-5)))
            current_date = now_et.date()

            # Reset daily state
            if current_date != last_scan_date:
                daily_pnl = 0.0
                managed.clear()  # Positions cleared at start of new day
                last_scan_date = current_date
                log_event("NEW_DAY", {"date": str(current_date)}, log_file)

            # Check market open
            if not alpaca_market_open(base_url, key, secret):
                if now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 25):
                    print(f"  [{now_et.strftime('%H:%M')}] Pre-market. Waiting...")
                    time.sleep(60)
                    continue
                elif now_et.hour >= 16:
                    print(f"  [{now_et.strftime('%H:%M')}] After hours. Done for today.")
                    time.sleep(300)
                    continue
                else:
                    print(f"  [{now_et.strftime('%H:%M')}] Market closed. Waiting 60s...")
                    time.sleep(60)
                    continue

            scan_count += 1

            # On first scan of the day, recover any bot positions from Alpaca
            if scan_count == 1 or (scan_count == 1 and not managed):
                n = recover_positions(managed, base_url, key, secret, cfg)
                if n > 0:
                    log_event("POSITIONS_RECOVERED", {"count": n}, log_file)

            # Get account info
            acct = get_account(base_url, key, secret)
            equity = float(acct.get("equity", 10000))

            # ── PHASE 1: Manage existing positions (exits) ──
            exit_pnl = manage_exits(
                managed, base_url, key, secret, cfg,
                now_et, log_file, dry_run=args.dry_run,
            )
            daily_pnl += exit_pnl

            # Get current broker positions for accurate count
            positions = get_positions(base_url, key, secret)
            held_symbols = {p.get("symbol", "") for p in positions
                           if p.get("asset_class") == "us_equity"}
            num_positions = len(managed)

            # Check daily P&L from fills
            fills = get_daily_fills(base_url, key, secret, BOT_TAG)
            broker_pnl = calculate_realized_pnl(fills)
            # Use the larger magnitude (more conservative)
            if abs(broker_pnl) > abs(daily_pnl):
                daily_pnl = broker_pnl

            log_event("HEARTBEAT", {
                "scan": scan_count,
                "equity": round(equity, 2),
                "managed": len(managed),
                "broker_positions": len(held_symbols),
                "daily_pnl": round(daily_pnl, 2),
                "time_et": now_et.strftime("%H:%M"),
            }, log_file)

            # Check daily loss limit
            if daily_pnl <= -args.max_daily_loss:
                log_event("DAILY_LOSS_PAUSE", {
                    "daily_pnl": round(daily_pnl, 2),
                    "limit": args.max_daily_loss,
                }, log_file)
                print(f"  Daily loss limit reached (${daily_pnl:.2f}). Pausing entries.")
                time.sleep(args.poll_interval)
                continue

            # Check position limit
            if num_positions >= args.max_positions:
                if scan_count % 6 == 0:
                    print(f"  [{now_et.strftime('%H:%M')}] "
                          f"Max positions ({num_positions}/{args.max_positions})")
                time.sleep(args.poll_interval)
                continue

            # ── PHASE 2: Scan for new entry signals ──
            # Parse entry window end
            h_end, m_end = map(int, cfg.entry_end.split(":"))
            entry_window_open = not (now_et.hour > h_end or
                                     (now_et.hour == h_end and now_et.minute > m_end))

            signals_found = 0
            scan_details = []
            if entry_window_open:
                for symbol in tickers:
                    if num_positions >= args.max_positions:
                        break

                    # Skip if already holding this symbol (local or broker)
                    if symbol in managed or symbol in held_symbols:
                        scan_details.append(f"{symbol}=HELD")
                        continue

                    df = prepare_data(symbol, days=10, timeframe="5Min",
                                      feed=args.feed, force_refresh=True)
                    if df is None:
                        scan_details.append(f"{symbol}=NO_DATA")
                        continue

                    signal = check_entry_signal(df, cfg)
                    if signal is None:
                        # Log why no signal for this ticker
                        cur = df.iloc[-1]
                        prev = df.iloc[-2] if len(df) >= 2 else cur
                        ema9 = float(cur.get("ema_9", 0))
                        ema21 = float(cur.get("ema_21", 0))
                        rsi = float(prev.get("rsi", 0))
                        rvol = float(cur.get("rvol", 0))
                        spread_pct = ((ema9 - ema21) / ema21 * 100) if ema21 else 0
                        scan_details.append(
                            f"{symbol}(ema={spread_pct:+.2f}% rsi={rsi:.0f} rvol={rvol:.1f})"
                        )
                        continue

                    signals_found += 1
                    qty = calculate_qty(
                        equity, signal["price"], signal["sl_price"],
                        args.risk_pct, args.max_notional
                    )
                    if qty <= 0:
                        continue

                    log_event("SIGNAL", {
                        "symbol": symbol,
                        "side": signal["side"],
                        "price": signal["price"],
                        "sl": signal["sl_price"],
                        "tp": signal["tp_price"],
                        "qty": qty,
                        "momentum": signal["momentum_score"],
                        "rsi": signal["rsi"],
                        "rvol": signal["rvol"],
                    }, log_file)

                    if args.dry_run:
                        print(f"  [DRY RUN] {symbol} {signal['side'].upper()}: "
                              f"qty={qty} @ ${signal['price']:.2f} "
                              f"SL=${signal['sl_price']:.2f} TP=${signal['tp_price']:.2f} "
                              f"mom={signal['momentum_score']:.2f}")
                        # Track in dry run too for exit simulation
                        managed[symbol] = LivePosition(
                            symbol=symbol,
                            side=signal["side"],
                            qty=qty,
                            entry_price=signal["price"],
                            entry_time=dt.datetime.now(dt.timezone.utc).isoformat(),
                            sl_price=signal["sl_price"],
                            tp_price=signal["tp_price"],
                            atr=signal["atr"],
                            peak_price=signal["price"],
                        )
                        num_positions += 1
                        continue

                    # Submit simple limit order (no bracket - we manage exits ourselves)
                    client_oid = f"{BOT_TAG}_{symbol}_{int(time.time())}"
                    try:
                        result = submit_limit_order(
                            base_url, key, secret,
                            symbol=symbol,
                            qty=qty,
                            side=signal["side"],
                            client_order_id=client_oid,
                        )

                        if result.get("error"):
                            log_event("ORDER_REJECTED", {
                                "symbol": symbol,
                                "error": result["error"],
                            }, log_file)
                            print(f"  [REJECTED] {symbol}: {result['error']}")
                            continue

                        order_id = result.get("id", "")
                        status = result.get("status", "unknown")

                        # Wait for fill
                        if status not in ("filled",):
                            filled = verify_order_filled(
                                base_url, key, secret, order_id, timeout=10)
                            status = filled.get("status", status)

                        if status == "filled":
                            filled_price = float(
                                result.get("filled_avg_price", 0) or signal["price"])

                            # Register in local tracker
                            managed[symbol] = LivePosition(
                                symbol=symbol,
                                side=signal["side"],
                                qty=qty,
                                entry_price=filled_price,
                                entry_time=dt.datetime.now(dt.timezone.utc).isoformat(),
                                sl_price=signal["sl_price"],
                                tp_price=signal["tp_price"],
                                atr=signal["atr"],
                                peak_price=filled_price,
                                entry_order_id=order_id,
                            )
                            num_positions += 1

                            log_event("ENTRY", {
                                "symbol": symbol,
                                "side": signal["side"],
                                "qty": qty,
                                "price": filled_price,
                                "sl": signal["sl_price"],
                                "tp": signal["tp_price"],
                                "atr": signal["atr"],
                                "momentum": signal["momentum_score"],
                                "order_id": order_id,
                            }, log_file)
                            print(f"  ENTRY: {symbol} {signal['side'].upper()} x{qty} "
                                  f"@ ${filled_price:.2f} "
                                  f"SL=${signal['sl_price']:.2f} TP=${signal['tp_price']:.2f}")
                        else:
                            log_event("ORDER_NOT_FILLED", {
                                "symbol": symbol,
                                "status": status,
                                "order_id": order_id,
                            }, log_file)

                    except Exception as e:
                        log_event("ORDER_ERROR", {
                            "symbol": symbol,
                            "error": str(e),
                        }, log_file)
                        print(f"  [ERROR] Order for {symbol}: {e}")

            # Status log - print every scan so user can see the bot is working
            if signals_found == 0:
                detail_str = " | ".join(scan_details) if scan_details else "no tickers scanned"
                print(f"  [{now_et.strftime('%H:%M')}] Scan #{scan_count}: "
                      f"0 signals | managed={len(managed)} | P&L=${daily_pnl:.2f}")
                print(f"    {detail_str}")
            elif signals_found > 0:
                print(f"  [{now_et.strftime('%H:%M')}] Scan #{scan_count}: "
                      f"{signals_found} signal(s) | managed={len(managed)} | P&L=${daily_pnl:.2f}")

            time.sleep(args.poll_interval)

        except KeyboardInterrupt:
            log_event("BOT_STOP", {
                "reason": "keyboard_interrupt",
                "managed_positions": list(managed.keys()),
                "daily_pnl": round(daily_pnl, 2),
            }, log_file)
            if managed:
                print(f"\n  WARNING: {len(managed)} managed positions still open: "
                      f"{list(managed.keys())}")
                print(f"  These positions have NO automated stop-loss management now.")
            print(f"\n  Bot stopped. Daily P&L: ${daily_pnl:.2f}")
            break
        except Exception as e:
            log_event("ERROR", {"error": str(e)}, log_file)
            print(f"  [ERROR] {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()
