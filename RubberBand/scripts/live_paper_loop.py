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
)
from RubberBand.strategy import attach_verifiers
from RubberBand.src.trade_logger import TradeLogger


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
    if not max_notional:
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

    # Fetch bars
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

    # Risk knobs
    brackets = cfg.get("brackets", {}) or {}

    # Size guard
    base_qty = int(cfg.get("qty", 1))
    max_shares = int(cfg.get("max_shares_per_trade", base_qty))
    max_notional = cfg.get("max_notional_per_trade", None)
    max_notional = float(max_notional) if max_notional not in (None, "", "0") else None

    # Iterate symbols
    for sym in symbols:
        df = bars_map.get(sym)
        if df is None or df.empty or len(df) < 20: # Need enough data for Keltner(20)
            continue

        df = attach_verifiers(df, cfg)
        last = df.iloc[-1]

        # Extract Signal
        long_signal = bool(last["long_signal"])
        close = float(last["close"])
        rsi = float(last["rsi"]) if not pd.isna(last["rsi"]) else None
        kc_lower = float(last["kc_lower"]) if not pd.isna(last["kc_lower"]) else None
        
        # Log Signal
        sig_row = {
            "symbol": sym,
            "session": session,
            "cid": f"RB_{sym}_{now_utc.strftime('%Y%m%d_%H%M%S')}",
            "tf": timeframe,
            "long_signal": 1 if long_signal else 0,
            "ref_bar_ts": str(df.index[-1]),
            "last_close": close,
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

        # Gating: Check Signal
        if not long_signal:
            # Optional: Log heartbeat for no signal? No, too verbose.
            continue

        # Log Signal Event
        try:
            log.signal(**sig_row)
        except Exception:
            pass

        # ATR Calculation
        atr_len = int(cfg.get("atr_length", 14))
        tr = pd.concat(
            [
                (df["high"] - df["low"]),
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"] - df["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(atr_len, min_periods=atr_len).mean()
        atr_val = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0

        # Bracket Calculation (Match Backtest: TP = Entry + TP_R * ATR)
        entry = close # Use last close as proxy for entry
        sl_mult = float(brackets.get("atr_mult_sl", 1.5))
        tp_r = float(brackets.get("take_profit_r", 2.0))
        
        stop_price = round(entry - sl_mult * atr_val, 2)
        take_profit = round(entry + tp_r * atr_val, 2) # Direct ATR multiple
        
        if not (stop_price < entry < take_profit):
            print(f"[order] skip {sym}: bad TP/SL (entry={entry}, sl={stop_price}, tp={take_profit})", flush=True)
            continue

        # Size
        base_qty = int(cfg.get("qty", 1))
        max_shares = int(cfg.get("max_shares_per_trade", base_qty))
        qty = max(1, min(base_qty, max_shares))
        qty = _cap_qty_by_notional(qty, entry, max_notional)
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
                side="buy",
                entry_price=round(entry, 2),
                stop_loss_price=stop_price,
                take_profit_price=take_profit,
                atr=round(atr_val, 4),
                method="bracket",
                tif="day",
                dry_run=bool(args.dry_run),
            )
        except Exception:
            pass

        if args.dry_run:
            print(
                f"[order] DRY-RUN: would submit BRACKET {sym} qty={qty} entry≈{entry:.2f} "
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
                    side="buy",
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
                        reason="BROKER_ERROR",
                    )
                except Exception:
                    pass

    # --- Session Summary ---
    print("\n=== Session Summary ===", flush=True)
    try:
        from RubberBand.src.data import get_daily_fills
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
            # Columns: Ticker | Bought | Avg Ent | Basis | Sold | Avg Ex | Day PnL
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

                # Format PnL: show value if matched > 0, else "-"
                pnl_str = f"{realized_pnl:,.2f}" if matched_qty > 0 else "-"

                print(f"{sym:<8} {int(b_qty):<8} {avg_ent:<10.2f} {b_val:<12.2f} {int(s_qty):<8} {avg_ex:<10.2f} {pnl_str:<10}", flush=True)

            print("-" * len(header), flush=True)
            print(f"TOTAL Day PnL: ${total_pnl:,.2f} | TOTAL VOL: ${total_vol:,.2f}", flush=True)
            print("(Note: Day PnL calculated on matched intraday buy/sell quantity)", flush=True)
            print("=== End Summary ===", flush=True)

    except Exception as e:
        print(f"[warn] Failed to generate summary: {e}", flush=True)

    try:
        log.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())