#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import argparse
import pandas as pd
from collections import Counter

pd.set_option('future.no_silent_downcasting', True)

# Global counter
FILTER_REJECTS = Counter()

# --- Ensure repo root on sys.path ---
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.utils import load_config, read_tickers
from RubberBand.src.data import fetch_latest_bars
from RubberBand.strategy import attach_verifiers

# ------------------------------------------------------------------
# Data loading helper
# ------------------------------------------------------------------
def load_bars_for_symbol(symbol: str, cfg: dict, days: int,
                         timeframe_override=None, limit_override=None, rth_only_override=True) -> pd.DataFrame:
    timeframe = timeframe_override or cfg.get("timeframe", "15Min")
    feed = cfg.get("feed", "iex")
    tz_name = cfg.get("timezone", "US/Eastern")
    rth_start = cfg.get("rth_start", "09:30")
    rth_end = cfg.get("rth_end", "15:55")

    key = os.getenv("APCA_API_KEY_ID", "")
    secret = os.getenv("APCA_API_SECRET_KEY", "")
    history_days = int(days)
    bar_limit = int(limit_override) if (limit_override is not None) else int(cfg.get("bar_limit", 10000))

    bars_map, _ = fetch_latest_bars(
        symbols=[symbol],
        timeframe=timeframe,
        history_days=history_days,
        feed=feed,
        rth_only=rth_only_override,
        tz_name=tz_name,
        rth_start=rth_start,
        rth_end=rth_end,
        bar_limit=bar_limit,
        key=key,
        secret=secret,
    )
    
    df = bars_map.get(symbol, pd.DataFrame())
    if df.empty:
        return df

    # Keep approx last N days
    bars_per_day_15m = 390 // 15
    approx_rows = int(days) * bars_per_day_15m
    return df.tail(approx_rows)

# ------------------------------------------------------------------
# Backtest simulator
# ------------------------------------------------------------------
def simulate_mean_reversion(df: pd.DataFrame, cfg: dict, start_cash=10_000.0, risk_pct: float = 0.01):
    """
    Mean Reversion Backtest:
    - Entry: Price < Lower Keltner & RSI < 30 (long_signal)
    - Exit: Price > Middle Keltner (Mean) OR Bracket SL/TP
    """
    if df is None or df.empty or len(df) < 30:
        return dict(trades=0, gross=0.0, net=0.0, win_rate=0.0, ret_pct=0.0, equity=start_cash)

    df = attach_verifiers(df, cfg).copy()
    
    # DEBUG: Print last 5 rows with indicators
    print(df[["close", "kc_lower", "rsi", "long_signal"]].tail())

    # Bracket params
    bcfg = cfg.get("brackets", {})
    atr_mult_sl = float(bcfg.get("atr_mult_sl", 1.5))
    take_profit_r = float(bcfg.get("take_profit_r", 2.0))
    use_brackets = (atr_mult_sl is not None)

    max_notional = float(cfg.get("max_notional_per_trade", 0))

    equity = float(start_cash)
    in_pos = False
    qty = 0
    entry_px = 0.0
    entry_ts = None

    wins = 0
    losses = 0
    gross = 0.0
    detailed_trades = []

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        cur = df.iloc[i]
        open_px = float(cur["open"])

        if not in_pos:
            # Entry Signal
            if prev.get("long_signal", False):
                # Risk Sizing
                atr_val = float(prev.get("atr", 0.0))
                if atr_val <= 0:
                    print(f"REJECT: ATR={atr_val}")
                    continue
                
                stop_dist = max(0.01, atr_mult_sl * atr_val)
                risk_dollars = equity * float(risk_pct)
                raw_qty = int(risk_dollars // stop_dist) if stop_dist > 0 else 0
                
                cap_qty = int((max_notional // open_px)) if max_notional > 0 else raw_qty
                qty = max(1, min(raw_qty, cap_qty))
                
                if qty <= 0:
                    continue

                entry_px = open_px
                in_pos = True
                entry_ts = cur.name
                
                entry_state = {
                    "rsi": float(prev.get("rsi", 0)),
                    "kc_lower": float(prev.get("kc_lower", 0)),
                    "atr": atr_val,
                }

        else:
            # Exit Logic
            atr_val = float(prev.get("atr", 0.0))
            
            # 1. Brackets
            stop_px = None
            tp_px = None
            if use_brackets and atr_val > 0:
                stop_px = max(0.01, entry_px - atr_mult_sl * atr_val)
                tp_px = entry_px + take_profit_r * atr_val

            hit_sl = (stop_px is not None) and (cur["low"] <= stop_px)
            hit_tp = (tp_px is not None) and (cur["high"] >= tp_px)

            # 2. Mean Reversion Exits
            # Configurable: exit_at_mean (default True), exit_at_upper (optional)
            exit_at_mean = bool(cfg.get("exit_at_mean", True))
            exit_at_upper = bool(cfg.get("exit_at_upper", False))
            
            kc_mid = float(cur.get("kc_middle", 0.0))
            kc_upper = float(cur.get("kc_upper", 0.0))
            
            hit_mean = exit_at_mean and (cur["high"] >= kc_mid)
            hit_upper = exit_at_upper and (cur["high"] >= kc_upper)

            exit_px = 0.0
            reason = ""
            
            if hit_sl:
                exit_px = stop_px
                reason = "SL"
            elif hit_tp:
                exit_px = tp_px
                reason = "TP"
            elif hit_upper:
                exit_px = max(open_px, kc_upper)
                if exit_px > cur["high"]: exit_px = cur["high"]
                reason = "UPPER"
            elif hit_mean:
                exit_px = max(open_px, kc_mid) # Conservative: exit at mean or open
                if exit_px > cur["high"]: exit_px = cur["high"] # Cap at high
                reason = "MEAN"
            
            if reason:
                pnl = (exit_px - entry_px) * qty
                equity += pnl
                gross += pnl
                if pnl > 0: wins += 1
                else: losses += 1
                
                in_pos = False
                qty = 0
                
                detailed_trades.append({
                    "symbol": "UNKNOWN",
                    "entry_time": entry_ts,
                    "exit_time": cur.name,
                    "entry_price": entry_px,
                    "exit_price": exit_px,
                    "pnl": pnl,
                    "result": "WIN" if pnl > 0 else "LOSS",
                    "reason": reason,
                    **entry_state
                })

    trades = wins + losses
    net = gross
    ret_pct = (equity / start_cash - 1.0) * 100.0 if start_cash else 0.0
    wr = (100.0 * wins / max(trades, 1))
    return dict(
        trades=trades,
        gross=round(gross, 2),
        net=round(net, 2),
        win_rate=round(wr, 1),
        ret_pct=round(ret_pct, 2),
        equity=round(equity, 2),
        detailed_trades=detailed_trades
    )

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="RubberBand/config.yaml")
    ap.add_argument("--tickers", default="RubberBand/tickers.txt")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--cash", type=float, default=10_000)
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--timeframe", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rth-only", dest="rth_only", action="store_true")
    ap.add_argument("--no-rth-only", dest="rth_only", action="store_false")
    ap.set_defaults(rth_only=True)
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = read_tickers(args.tickers)

    def _load(sym: str) -> pd.DataFrame:
        return load_bars_for_symbol(
            sym, cfg, args.days,
            timeframe_override=args.timeframe,
            limit_override=args.limit,
            rth_only_override=args.rth_only,
        )

    rows = []
    for sym in symbols:
        try:
            df = _load(sym)
        except Exception as e:
            print(f"[{sym}] data error: {e}")
            continue

        if df.empty:
            continue

        print(f"[{sym}] bars={len(df)}")
        res = simulate_mean_reversion(df, cfg, start_cash=args.cash, risk_pct=args.risk)
        rows.append({"symbol": sym, **res})

    if not rows:
        print("No results.")
        return

    out = pd.DataFrame(rows).sort_values("net", ascending=False)
    print("\n=== Backtest Summary (last {} days) ===".format(args.days))
    print(out.to_string(index=False))

    total_trades = int(out["trades"].sum())
    total_net = round(float(out["net"].sum()), 2)
    avg_winrate = round((out["trades"] * out["win_rate"] / 100.0).sum() / max(total_trades, 1) * 100.0, 1)
    
    print("\nTOTAL trades={} net={} win_rate={}%".format(total_trades, total_net, avg_winrate))

if __name__ == "__main__":
    main()
