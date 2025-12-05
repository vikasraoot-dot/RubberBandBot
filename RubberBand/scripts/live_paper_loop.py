#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live loop (paper): Rubber Band (Mean Reversion) Strategy.
- Scans on a 5m cadence.
- Writes compact JSONL results via TradeLogger.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pandas as pd

# Local imports
from RubberBand.src.data import (
    load_symbols_from_file,
    fetch_latest_bars,
    alpaca_market_open,
    submit_bracket_order,
    get_positions,
    get_daily_fills,
)
from RubberBand.strategy import attach_verifiers
from RubberBand.src.trade_logger import TradeLogger
from RubberBand.src.ticker_health import TickerHealthManager


# -------- helpers --------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _utc_stamp(fmt: str = "%Y%m%d") -> str:
    # used for log filename (e.g., live_20251007.jsonl)
    return datetime.now(timezone.utc).strftime(fmt)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tickers", required=True)
    p.add_argument("--dry-run", type=int, default=0)
    p.add_argument("--force-run", type=int, default=0)
    return p.parse_args()


def _load_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _session_label(now_et: datetime, rth_start: str, rth_end: str) -> str:
    # AM = from open to 12:30, PM = 12:30 -> close (just a label for logs)
    hhmm = now_et.strftime("%H:%M")
    return "AM" if hhmm < "12:30" else "PM"


def _in_entry_window(now_et: datetime, windows: List[dict]) -> bool:
    if not windows:
        return True
    hhmm = now_et.strftime("%H:%M")
    for w in windows:
        if w["start"] <= hhmm <= w["end"]:
            return True
    return False


def _cap_qty_by_notional(qty: int, entry: float, max_notional: float | None) -> int:
    if max_notional is None or max_notional <= 0:
        return qty
    if entry <= 0:
        return 0
    return min(qty, int(max_notional // entry))


def _pretty_timeframe(tf: str) -> str:
    # normalize like "15m" -> "15Min"
    tf = tf.strip()
    if tf.endswith("m"):
        return f"{tf[:-1]}Min"
    return tf


def _broker_creds(cfg: dict) -> tuple[str, str, str]:
    """Pull broker creds from config or environment, with safe defaults."""
    broker = cfg.get("broker", {}) or {}
    base_url = broker.get("base_url") or os.getenv("ALPACA_BASE_URL") or os.getenv("APCA_API_BASE_URL") or ""
    key = broker.get("key") or os.getenv("ALPACA_KEY_ID") or os.getenv("APCA_API_KEY_ID") or ""
    secret = broker.get("secret") or os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY") or ""
    return base_url, key, secret


def _force_paper_base_url(base_url: str, now_iso: str) -> str:
    """Return paper trading URL, emitting a SAFETY log if we override a live URL."""
    paper = "https://paper-api.alpaca.markets"
    if not base_url:
        # No URL configured; default to paper.
        print(json.dumps({
            "type": "SAFETY",
            "action": "default_base_url_to_paper",
            "now": paper,
            "when": now_iso
        }), flush=True)
        return paper

    if "paper-api.alpaca.markets" in base_url:
        return paper

    # Looks like a live endpoint (api.alpaca.markets or something else) -> override.
    print(json.dumps({
        "type": "SAFETY",
        "action": "override_base_url_to_paper",
        "was": base_url,
        "now": paper,
        "when": now_iso
    }), flush=True)
    return paper


def _alpaca_market_open_compat(base_url: str, key: str, secret: str) -> bool:
    """Call alpaca_market_open with credentials; fall back to 0-arg signature if present."""
    try:
        return bool(alpaca_market_open(base_url, key, secret))
    except TypeError:
        return bool(alpaca_market_open())


def _get_positions_compat(base_url: str, key: str, secret: str):
    """Call get_positions with credentials; fall back to 0-arg signature if present."""
    try:
        return get_positions(base_url, key, secret)
    except TypeError:
        return get_positions()


# -------- main --------
def main() -> int:
    args = _parse_args()
    cfg = _load_config(args.config)

    # Reference timeframe
    intervals = cfg.get("intervals") or ["15m"]
    timeframe = _pretty_timeframe(intervals[0])  # "15Min"
    
    # Resolve history_days: prefer 'periods' map, fallback to top-level 'history_days'
    periods_map = cfg.get("periods", {})
    raw_period = periods_map.get(intervals[0])
    if raw_period and raw_period.endswith("d"):
        history_days = int(raw_period.replace("d", ""))
    else:
        history_days = int(cfg.get("history_days", 30))

    # Session clock
    import pytz

    tz = pytz.timezone(cfg.get("timezone", "US/Eastern"))
    rth_start = cfg.get("rth_start", "09:30")
    rth_end = cfg.get("rth_end", "15:55")
    now_utc = _now_utc()
    now_iso = now_utc.isoformat()
    now_et = now_utc.astimezone(tz)
    session = _session_label(now_et, rth_start, rth_end)

    # Logger
    results_dir = cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f"live_{_utc_stamp('%Y%m%d')}.jsonl")
    log = TradeLogger(log_path)

    # Health Manager
    health_file = os.path.join(results_dir, "ticker_health.json")
    health_mgr = TickerHealthManager(health_file, cfg.get("resilience", {}))

    # Broker creds (for market/positions/order calls)
    base_url_raw, key, secret = _broker_creds(cfg)
    base_url = _force_paper_base_url(base_url_raw, now_iso)

    # Ensure SDKs/helpers that read env use **paper** and correct keys
    os.environ["APCA_API_BASE_URL"] = base_url
    if key:
        os.environ["APCA_API_KEY_ID"] = key
    if secret:
        os.environ["APCA_API_SECRET_KEY"] = secret

    # Universe
    symbols = load_symbols_from_file(args.tickers)
    print(json.dumps({"type": "UNIVERSE", "loaded": len(symbols), "sample": symbols[:10], "when": now_iso}), flush=True)

    # Entry windows?
    windows = cfg.get("entry_windows", [])
    if not _in_entry_window(now_et, windows):
        print(json.dumps({"type": "HEARTBEAT", "session": session, "market_open": True, "ts": now_iso}))
        return 0

    # Market open check (paper)
    if not _alpaca_market_open_compat(base_url, key, secret):
        print(json.dumps({"type": "HEARTBEAT", "session": session, "market_open": False, "ts": now_iso}))
        return 0
    print(json.dumps({"type": "HEARTBEAT", "session": session, "market_open": True, "ts": now_iso}))

    # Fetch bars (Intraday)
    feed = cfg.get("feed", "iex")
    print(
        json.dumps(
            {
                "type": "BARS_FETCH_START",
                "requested": len(symbols),
                "chunks": 2,
                "timeframe": timeframe,
                "feed": feed,
                "start": (now_utc - timedelta(days=history_days)).isoformat(),
                "end": now_iso,
                "when": now_iso,
            }
        ),
        flush=True,
    )

    res = None
    try:
        res = fetch_latest_bars(symbols, timeframe, history_days, feed)
    except Exception as e:
        print(json.dumps({"type": "BARS_FETCH_ERROR", "reason": "exception", "message": str(e)}), flush=True)
        return 1

    if not isinstance(res, tuple) or len(res) != 2:
        print(json.dumps({"type": "BARS_FETCH_ERROR", "reason": "no_result"}), flush=True)
        return 1

    bars_map, bars_meta = res

    # Fetch Daily Bars for Trend Filter (SMA 200)
    trend_cfg = cfg.get("trend_filter", {})
    daily_map = {}
    if trend_cfg.get("enabled", False):
        sma_period = int(trend_cfg.get("sma_period", 200))
        daily_days = max(history_days, int(sma_period * 2.5))
        try:
            daily_map, _ = fetch_latest_bars(
                symbols, 
                "1Day", 
                daily_days, 
                feed, 
                rth_only=False, # Daily bars have 00:00 timestamp
                verbose=False
            )
        except Exception as e:
            print(f"[warn] Failed to fetch daily bars for trend filter: {e}", flush=True)

    # Summary
    with_data = sorted([s for s, d in bars_map.items() if isinstance(d, pd.DataFrame) and not d.empty])
    empty = sorted(list(set(symbols) - set(with_data)))
    stale = sorted(list((bars_meta or {}).get("stale_symbols", [])))
    http_errors = (bars_meta or {}).get("http_errors", [])

    print(
        json.dumps(
            {
                "type": "BARS_FETCH_SUMMARY",
                "requested": len(symbols),
                "with_data": len(with_data),
                "empty": len(empty),
                "stale": len(stale),
                "sample_with_data": with_data[:4],
                "sample_empty": empty[:10],
                "when": now_iso,
            }
        ),
        flush=True,
    )

    # Current positions (for "already in position" gate)
    positions_raw = _get_positions_compat(base_url, key, secret) or []
    positions = {p["symbol"]: p for p in positions_raw}

    # Daily cooldown: Get all tickers traded today (prevents re-entry after TP/SL)
    try:
        daily_fills = get_daily_fills(base_url, key, secret) or []
        traded_today = set(f.get("symbol") for f in daily_fills if f.get("symbol"))
    except Exception as e:
        print(f"[warn] Could not fetch daily fills for cooldown: {e}", flush=True)
        traded_today = set()

    # Risk knobs
    brackets = cfg.get("brackets", {}) or {}
    sl_mult = float(brackets.get("atr_mult_sl", 2.5))
    tp_r = float(brackets.get("take_profit_r", 1.5))
    allow_shorts = cfg.get("allow_shorts", False)

    # Size guard
    base_qty = int(cfg.get("qty", 1))
    max_shares = int(cfg.get("max_shares_per_trade", base_qty))
    max_notional = cfg.get("max_notional_per_trade", None)
    try:
        max_notional = float(max_notional) if max_notional is not None else None
        if max_notional is not None and max_notional <= 0:
            max_notional = None
    except (ValueError, TypeError):
        max_notional = None

    # Iterate symbols
    for sym in symbols:
        df = bars_map.get(sym)
        if df is None or df.empty or len(df) < 20: # Need enough data for Keltner(20)
            continue

        # Resilience Check
        is_paused, reason = health_mgr.is_paused(sym, now=now_utc)
        if is_paused:
            # Log only once per session or if verbose?
            # For now, just skip silently or with a minimal print if debug needed
            # print(f"[skip] {sym} is PAUSED: {reason}", flush=True)
            continue

        df = attach_verifiers(df, cfg)
        last = df.iloc[-1]
        close = float(last["close"])

        # Trend Filter Check
        is_bull_trend = False
        is_bear_trend = False
        is_strong_bull = False
        
        if trend_cfg.get("enabled", False):
            df_daily = daily_map.get(sym, pd.DataFrame())
            if not df_daily.empty and len(df_daily) >= sma_period:
                # Calculate SMAs
                sma_series = df_daily["close"].rolling(window=sma_period).mean()
                
                # Secondary SMA (e.g. 22)
                sma_period_2 = int(trend_cfg.get("secondary_sma_period", 22))
                sma_series_2 = df_daily["close"].rolling(window=sma_period_2).mean()
                
                # FIX: Ensure we use the last CLOSED day's SMA
                # If the last bar is "today", we must use the previous bar's SMA
                last_bar_date = df_daily.index[-1].date()
                current_date = now_et.date()
                
                if last_bar_date == current_date:
                    # Last bar is today (partial), use yesterday's SMA
                    if len(sma_series) > 1:
                        trend_sma = sma_series.iloc[-2]
                        trend_sma_2 = sma_series_2.iloc[-2] if len(sma_series_2) > 1 else float('nan')
                    else:
                        trend_sma = float('nan')
                        trend_sma_2 = float('nan')
                else:
                    # Last bar is yesterday (closed), use it
                    trend_sma = sma_series.iloc[-1]
                    trend_sma_2 = sma_series_2.iloc[-1]
                
                if not pd.isna(trend_sma):
                    if close > trend_sma:
                        is_bull_trend = True
                        # Check Secondary SMA for Strength
                        if not pd.isna(trend_sma_2) and close > trend_sma_2:
                            is_strong_bull = True
                    else:
                        is_bear_trend = True
            else:
                # Fallback if no daily data but filter enabled?
                # For safety, maybe skip? Or default to Bull?
                # Backtest logic: skip if filter enabled and no SMA
                continue
        else:
            # Filter disabled -> assume Bull (allow Longs)
            is_bull_trend = True

        # Extract Signal
        long_signal = bool(last["long_signal"])
        rsi = float(last["rsi"]) if not pd.isna(last["rsi"]) else None
        kc_lower = float(last["kc_lower"]) if not pd.isna(last["kc_lower"]) else None
        kc_upper = float(last["kc_upper"]) if "kc_upper" in last and not pd.isna(last["kc_upper"]) else None
        
        # Short Signal Logic
        short_signal = False
        if allow_shorts and is_bear_trend:
            if kc_upper and close > kc_upper and rsi is not None and rsi > 70:
                short_signal = True

        # Filter Long Signal by Trend
        if long_signal and not is_bull_trend:
            long_signal = False

        # Entry price reference (Close of the signal bar)
        entry = close
        
        # Log Signal
        # Build entry reason from signal components
        entry_reasons = []
        if long_signal:
            if rsi is not None and rsi < 30:
                entry_reasons.append(f"RSI_oversold({rsi:.1f})")
            if kc_lower and close <= kc_lower:
                entry_reasons.append("Lower_KC_touch")
            if is_strong_bull:
                entry_reasons.append("Strong_bull_trend")
            elif is_bull_trend:
                entry_reasons.append("Bull_trend")
        elif short_signal:
            if rsi is not None and rsi > 70:
                entry_reasons.append(f"RSI_overbought({rsi:.1f})")
            if kc_upper and close >= kc_upper:
                entry_reasons.append("Upper_KC_touch")
            entry_reasons.append("Bear_trend")
        
        entry_reason = " + ".join(entry_reasons) if entry_reasons else "RubberBand_signal"
        
        sig_row = {
            "symbol": sym,
            "session": session,
            "cid": f"RB_{sym}_{now_utc.strftime('%Y%m%d_%H%M%S')}",
            "tf": timeframe,
            "long_signal": 1 if long_signal else 0,
            "short_signal": 1 if short_signal else 0,
            "ref_bar_ts": str(df.index[-1]),
            "last_close": close,
            "trend": "BULL" if is_bull_trend else ("BEAR" if is_bear_trend else "NONE"),
            "entry_reason": entry_reason,
        }
        
        # Gating: Check if already in position
        if sym in positions:
            try:
                log.gate(
                    symbol=sym, session=session, cid=sig_row["cid"],
                    decision="BLOCK", reasons=["already in position"]
                )
            except Exception:
                pass
            continue

        # Gating: Daily Cooldown - prevent re-entry on tickers traded today
        if sym in traded_today:
            try:
                log.gate(
                    symbol=sym, session=session, cid=sig_row["cid"],
                    decision="BLOCK", reasons=["traded_today"]
                )
            except Exception:
                pass
            continue

        # Gating: Check Signal
        if not long_signal and not short_signal:
            # Optional: Log heartbeat for no signal? No, too verbose.
            continue

        # Determine Side
        side = "buy" if long_signal else "sell" # Alpaca 'sell' = short open if no position

        # Log Signal Event
        try:
            log.signal(**sig_row)
        except Exception:
            pass

        # ATR Calculation (Use pre-calculated from attach_verifiers)
        atr_val = float(last.get("atr", 0.0))
        
        if side == "buy":
            stop_price = round(entry - sl_mult * atr_val, 2)
            take_profit = round(entry + tp_r * atr_val, 2)
            if not (stop_price < entry < take_profit):
                 print(f"[order] skip {sym}: bad TP/SL (entry={entry}, sl={stop_price}, tp={take_profit})", flush=True)
                 continue
        else: # Short
            stop_price = round(entry + sl_mult * atr_val, 2)
            take_profit = round(entry - tp_r * atr_val, 2)
            if not (take_profit < entry < stop_price):
                 print(f"[order] skip {sym}: bad TP/SL (entry={entry}, sl={stop_price}, tp={take_profit})", flush=True)
                 continue

        # Size
        base_qty = int(cfg.get("qty", 1))
        max_shares = int(cfg.get("max_shares_per_trade", base_qty))
        qty = max(1, min(base_qty, max_shares))
        
        # Dual SMA Sizing Logic
        # Strong Bull (Price > 120 & > 22) -> Full Size
        # Weak Bull (Price > 120 & < 22) -> 1/3 Size
        effective_notional = max_notional
        if not is_strong_bull and max_notional is not None:
             effective_notional = max_notional / 3.0
        
        qty = _cap_qty_by_notional(qty, entry, effective_notional)
        if qty < 1:
            print(f"[order] skip {sym}: qty<1 after notional cap", flush=True)
            continue

        # Log the planned order
        try:
            log.entry_submit(
                symbol=sym,
                session=session,
                cid=sig_row["cid"],
                qty=qty,
                side=side,
                entry_price=round(entry, 2),
                stop_loss_price=stop_price,
                take_profit_price=take_profit,
                atr=round(atr_val, 4),
                method="bracket",
                tif="day",
                dry_run=bool(args.dry_run),
                entry_reason=entry_reason,
                rsi=rsi,
            )
        except Exception:
            pass

        if args.dry_run:
            print(
                f"[order] DRY-RUN: would submit BRACKET {sym} {side.upper()} qty={qty} entry≈{entry:.2f} "
                f"sl={stop_price:.2f} tp={take_profit:.2f} (ATR={atr_val:.3f})",
                flush=True,
            )
            try:
                log.entry_ack(
                    symbol=sym,
                    session=session,
                    cid=sig_row["cid"],
                    order_id=None,
                    client_order_id=None,
                    broker_resp=None,
                    dry_run=True,
                )
            except Exception:
                pass
        else:
            try:
                resp = submit_bracket_order(
                    base_url,
                    key,
                    secret,
                    symbol=sym,
                    qty=qty,
                    side=side,
                    limit_price=None,  # market entry
                    take_profit_price=take_profit,
                    stop_loss_price=stop_price,
                    tif="day",
                )
                print(f"[order] BRACKET submitted for {sym}: {json.dumps(resp)[:300]}", flush=True)
                try:
                    oid = (resp.get("id") if isinstance(resp, dict) else None)
                    coid = (resp.get("client_order_id") if isinstance(resp, dict) else None)
                    log.entry_ack(
                        symbol=sym,
                        session=session,
                        cid=sig_row["cid"],
                        order_id=oid,
                        client_order_id=coid,
                        broker_resp=resp,
                        dry_run=False,
                    )
                except Exception:
                    pass
            except Exception as e:
                print(f"[order] ERROR submitting bracket for {sym}: {e}", flush=True)
                try:
                    log.entry_reject(
                        symbol=sym,
                        session=session,
                        cid=sig_row["cid"],
                        reason=str(e),
                    )
                except Exception:
                    pass

    # --- Session Summary ---
    print("\n=== Session Summary ===", flush=True)
    try:
        # Fetch fills (get_daily_fills already imported at top of file)
        fills = get_daily_fills(base_url, key, secret)
        
        if not fills:
            print("No trades filled today.", flush=True)
        else:
            # Aggregate per symbol
            stats = {}
            for f in fills:
                sym = f.get("symbol")
                side = f.get("side")
                qty = float(f.get("filled_qty", 0))
                px = float(f.get("filled_avg_price", 0))
                
                if sym not in stats:
                    stats[sym] = {"buy_qty": 0, "buy_val": 0.0, "sell_qty": 0, "sell_val": 0.0}
                
                if side == "buy":
                    stats[sym]["buy_qty"] += qty
                    stats[sym]["buy_val"] += (qty * px)
                elif side == "sell":
                    stats[sym]["sell_qty"] += qty
                    stats[sym]["sell_val"] += (qty * px)

            # Print Table
            header = f"{'Ticker':<8} {'Bought':<8} {'Avg Ent':<10} {'Basis':<12} {'Sold':<8} {'Avg Ex':<10} {'Day PnL':<10}"
            print("-" * len(header), flush=True)
            print(header, flush=True)
            print("-" * len(header), flush=True)

            total_pnl = 0.0
            total_vol = 0.0

            for sym in sorted(stats.keys()):
                s = stats[sym]
                b_qty = s["buy_qty"]
                b_val = s["buy_val"]
                s_qty = s["sell_qty"]
                s_val = s["sell_val"]

                avg_ent = (b_val / b_qty) if b_qty > 0 else 0.0
                avg_ex = (s_val / s_qty) if s_qty > 0 else 0.0
                
                # Calculate PnL only on matched intraday quantity
                matched_qty = min(b_qty, s_qty)
                if matched_qty > 0:
                    realized_pnl = (avg_ex - avg_ent) * matched_qty
                else:
                    realized_pnl = 0.0
                
                total_pnl += realized_pnl
                total_vol += (b_val + s_val)

                pnl_str = f"{realized_pnl:,.2f}" if matched_qty > 0 else "-"

                pnl_str = f"{realized_pnl:,.2f}" if matched_qty > 0 else "-"

                # Update Health Manager with Realized PnL
                if matched_qty > 0:
                    # We use a synthetic trade ID based on date/symbol for now, 
                    # or just let the manager handle deduping by timestamp if we had it.
                    # Since this runs once per day/session end, we can just push it.
                    # Ideally we'd have exact trade IDs. 
                    # For now, we pass a unique-ish ID for the day's aggregate.
                    trade_id = f"{sym}_{now_iso[:10]}"
                    health_mgr.update_trade(sym, realized_pnl, trade_id, now=now_utc)

                print(f"{sym:<8} {int(b_qty):<8} {avg_ent:<10.2f} {b_val:<12.2f} {int(s_qty):<8} {avg_ex:<10.2f} {pnl_str:<10}", flush=True)

            print("-" * len(header), flush=True)
            print(f"TOTAL Day PnL: ${total_pnl:,.2f} | TOTAL VOL: ${total_vol:,.2f}", flush=True)
            print("(Note: Day PnL calculated on matched intraday buy/sell quantity)", flush=True)
            print("=== End Summary ===", flush=True)
            
            # Emit structured EOD_SUMMARY to JSONL
            try:
                log.eod_summary(total_pnl=total_pnl, total_vol=total_vol)
            except Exception:
                pass

    except Exception as e:
        print(f"[warn] Failed to generate summary: {e}", flush=True)

    try:
        log.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
