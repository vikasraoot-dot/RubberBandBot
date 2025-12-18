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
from RubberBand.src.ticker_health import TickerHealthManager

# ------------------------------------------------------------------
# Data loading helper
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Data loading helper
# ------------------------------------------------------------------
def load_bars_for_symbol(symbol: str, cfg: dict, days: int,
                         timeframe_override=None, limit_override=None, rth_only_override=True, verbose=True) -> pd.DataFrame:
    # 1. Fetch Intraday Bars (15m)
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
        verbose=verbose
    )
    
    df = bars_map.get(symbol, pd.DataFrame())
    if df.empty:
        return df

    # 2. Fetch Daily Bars for Trend Filter (SMA 200)
    # We need more history for SMA 200 (at least 200 days + buffer)
    trend_cfg = cfg.get("trend_filter", {})
    if trend_cfg.get("enabled", False):
        sma_period = int(trend_cfg.get("sma_period", 120))
        sma_period_2 = int(trend_cfg.get("secondary_sma_period", 22))
        
        # Need enough calendar days for trading days SMA
        # 200 trading days ~= 290 calendar days. Use 2x buffer to be safe.
        max_sma = max(sma_period, sma_period_2)
        daily_days = max(history_days, int(max_sma * 2.5)) 
        
        daily_map, _ = fetch_latest_bars(
            symbols=[symbol],
            timeframe="1Day",
            history_days=daily_days,
            feed=feed,
            rth_only=False, # Daily bars have 00:00 timestamp, RTH filter kills them
            tz_name=tz_name,
            key=key,
            secret=secret,
            verbose=False # Reduce noise
        )
        
        df_daily = daily_map.get(symbol, pd.DataFrame())
        if not df_daily.empty:
            # Calculate SMAs
            df_daily["trend_sma"] = df_daily["close"].rolling(window=sma_period).mean()
            df_daily["trend_sma_2"] = df_daily["close"].rolling(window=sma_period_2).mean()
            
            # Shift by 1 to avoid lookahead bias in backtest
            df_daily["trend_sma"] = df_daily["trend_sma"].shift(1)
            df_daily["trend_sma_2"] = df_daily["trend_sma_2"].shift(1)
            
            # Merge into Intraday DF
            df["date_only"] = df.index.date
            df_daily["date_only"] = df_daily.index.date
            
            # Create mapping dicts
            sma_map = df_daily.set_index("date_only")["trend_sma"].to_dict()
            sma_map_2 = df_daily.set_index("date_only")["trend_sma_2"].to_dict()
            
            # DEBUG
            # print(f"=== DAILY BARS: {len(df_daily)} (Requested: {daily_days}) ===")
            
            df["trend_sma"] = df["date_only"].map(sma_map)
            df["trend_sma_2"] = df["date_only"].map(sma_map_2)
            df.drop(columns=["date_only"], inplace=True)
            
            # Check if we have NaNs
            if df["trend_sma"].isna().all():
                print("WARNING: All Trend SMA values are NaN!")
        else:
            df["trend_sma"] = float('nan')
            df["trend_sma_2"] = float('nan')
    else:
        df["trend_sma"] = float('nan')
        df["trend_sma_2"] = float('nan')

    # Keep approx last N days
    bars_per_day_15m = 390 // 15
    approx_rows = int(days) * bars_per_day_15m
    return df.tail(approx_rows)

# ------------------------------------------------------------------
# Backtest simulator
# ------------------------------------------------------------------
def simulate_mean_reversion(df: pd.DataFrame, cfg: dict, health_mgr: TickerHealthManager, sym: str, start_cash=10_000.0, risk_pct: float = 0.01, verbose: bool = False):
    """
    Mean Reversion Backtest:
    - Entry: Price < Lower Keltner & RSI < 30 (long_signal)
    - Exit: Price > Middle Keltner (Mean) OR Bracket SL/TP
    
    Args:
        verbose: If True, print each trade's entry/exit details
    """
    if df is None or df.empty or len(df) < 30:
        return dict(trades=0, gross=0.0, net=0.0, win_rate=0.0, ret_pct=0.0, equity=start_cash, detailed_trades=[])

    df = attach_verifiers(df, cfg).copy()
    
    # DEBUG: Print last 5 rows with indicators
    if cfg.get("verbose", True):
        print(df[["close", "kc_lower", "rsi", "long_signal", "trend_sma"]].tail())

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
    side = "LONG" # Default

    wins = 0
    losses = 0
    gross = 0.0
    detailed_trades = []

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        cur = df.iloc[i]
        open_px = float(cur["open"])

        if not in_pos:
            # Resilience Check
            is_paused, reason = health_mgr.is_paused(sym, now=cur.name)
            if is_paused:
                continue

            # Slope Filter (Panic Buyer Logic)
            # Use same logic as backtest_spreads: (KC[i-1] - KC[i-4]) / 3
            # We want to buy ONLY if slope is steep enough (Panic).
            # We skip if slope is too flat (Drift).
            # E.g. Thresh -0.20. Slope -0.14 is > -0.20 -> SKIP.
            
            # Check 1: 3-bar slope (immediate crash, 45m)
            slope_threshold = cfg.get("slope_threshold")
            if slope_threshold is not None:
                 # Check sufficient history (need i >= 4)
                 if i >= 4 and "kc_middle" in df.columns:
                      current_slope_3 = (df["kc_middle"].iloc[i-1] - df["kc_middle"].iloc[i-4]) / 3
                      if current_slope_3 > float(slope_threshold):
                           continue
            
            # Check 2: 10-bar slope (sustained crash, 2.5h)
            slope_threshold_10 = cfg.get("slope_threshold_10")
            if slope_threshold_10 is not None:
                 if i >= 11 and "kc_middle" in df.columns:
                      current_slope_10 = (df["kc_middle"].iloc[i-1] - df["kc_middle"].iloc[i-11]) / 10
                      if current_slope_10 > float(slope_threshold_10):
                           continue

            # Trend Filter Check
            trend_sma = prev.get("trend_sma", float('nan'))
            trend_sma_2 = prev.get("trend_sma_2", float('nan'))
            
            is_bull_trend = False
            is_bear_trend = False
            is_strong_bull = False
            
            if not pd.isna(trend_sma):
                if prev["close"] > trend_sma:
                    is_bull_trend = True
                    # Check Secondary SMA for Strength
                    if not pd.isna(trend_sma_2) and prev["close"] > trend_sma_2:
                        is_strong_bull = True
                else:
                    is_bear_trend = True
            else:
                # If no SMA (not enough history), default to Bull or Neutral?
                # For safety, skip if filter enabled and no SMA
                if cfg.get("trend_filter", {}).get("enabled", False):
                    continue
                is_bull_trend = True # Fallback

            # LONG Entry Signal (Only in Bull Trend)
            if is_bull_trend and prev.get("long_signal", False):
                # Risk Sizing
                atr_val = float(prev.get("atr", 0.0))
                if atr_val <= 0:
                    continue
                
                # Sizing
                base_qty = int(cfg.get("qty", 10000))
                max_shares = int(cfg.get("max_shares_per_trade", 10000))
                qty = max(1, min(base_qty, max_shares))
                
                # Dual SMA Sizing Logic
                # Strong Bull (Price > 120 & > 22) -> Full Size
                # Weak Bull (Price > 120 & < 22) -> 1/3 Size
                effective_notional = max_notional
                if not is_strong_bull and max_notional > 0:
                     effective_notional = max_notional / 3.0
                
                if effective_notional > 0 and open_px > 0:
                    notional_cap_qty = int(effective_notional // open_px)
                    qty = min(qty, notional_cap_qty)
                
                if qty <= 0:
                    continue

                entry_px = open_px
                in_pos = True
                side = "LONG"
                entry_ts = cur.name
                
                entry_state = {
                    "rsi": float(prev.get("rsi", 0)),
                    "atr": atr_val,
                    "sl_px": entry_px - (atr_val * atr_mult_sl) if use_brackets else 0.0,
                    "tp_px": entry_px + (atr_val * take_profit_r) if use_brackets else 0.0
                }
                continue

            # SHORT Entry Signal (Only in Bear Trend)
            short_signal = False
            if is_bear_trend and cfg.get("allow_shorts", False):
                # Short Signal: Close > Upper Band AND RSI > 70
                # Assuming 'kc_upper' is in df
                if prev["close"] > prev.get("kc_upper", float('inf')) and prev.get("rsi", 0) > 70:
                    short_signal = True
            
            if short_signal:
                 # Risk Sizing
                atr_val = float(prev.get("atr", 0.0))
                if atr_val <= 0:
                    continue
                
                # Sizing
                base_qty = int(cfg.get("qty", 10000))
                max_shares = int(cfg.get("max_shares_per_trade", 10000))
                qty = max(1, min(base_qty, max_shares))
                if max_notional > 0 and open_px > 0:
                    notional_cap_qty = int(max_notional // open_px)
                    qty = min(qty, notional_cap_qty)
                
                if qty <= 0:
                    continue

                entry_px = open_px
                in_pos = True
                side = "SHORT"
                entry_ts = cur.name
                
                entry_state = {
                    "rsi": float(prev.get("rsi", 0)),
                    "atr": atr_val,
                    "sl_px": entry_px + (atr_val * atr_mult_sl) if use_brackets else 0.0,
                    "tp_px": entry_px - (atr_val * take_profit_r) if use_brackets else 0.0
                }
                continue

        else:
            # In Position - Check Exit
            # ------------------------
            exit_signal = False
            reason = ""
            
            # 1. Bracket Exit (SL/TP)
            if use_brackets:
                if side == "LONG":
                    if cur["low"] <= entry_state["sl_px"]:
                        exit_signal = True
                        reason = "SL"
                        exit_px = entry_state["sl_px"] # Assume fill at SL
                    elif cur["high"] >= entry_state["tp_px"]:
                        exit_signal = True
                        reason = "TP"
                        exit_px = entry_state["tp_px"]
                elif side == "SHORT":
                    if cur["high"] >= entry_state["sl_px"]:
                        exit_signal = True
                        reason = "SL"
                        exit_px = entry_state["sl_px"]
                    elif cur["low"] <= entry_state["tp_px"]:
                        exit_signal = True
                        reason = "TP"
                        exit_px = entry_state["tp_px"]

            # 2. Technical Exit (Mean Reversion)
            if not exit_signal:
                if side == "LONG":
                    if cur["close"] > cur["kc_middle"]:
                        exit_signal = True
                        reason = "MeanRev"
                        exit_px = cur["close"]
                elif side == "SHORT":
                    if cur["close"] < cur["kc_middle"]:
                        exit_signal = True
                        reason = "MeanRev"
                        exit_px = cur["close"]

            # 3. Time-Based Exits
            flatten_eod = cfg.get("_flatten_eod", True)
            max_hold_days = int(cfg.get("_max_hold_days", 0))
            
            is_time_exit = False
            exit_reason_time = ""

            if flatten_eod:
                if i < len(df) - 1:
                    next_bar = df.iloc[i+1]
                    if next_bar.name.day != cur.name.day:
                        is_time_exit = True
                        exit_reason_time = "EOD"
                else:
                    is_time_exit = True
                    exit_reason_time = "EOD"
            elif max_hold_days > 0 and entry_ts:
                held_delta = cur.name - entry_ts
                if held_delta.days >= max_hold_days:
                    is_time_exit = True
                    exit_reason_time = f"HOLD_{max_hold_days}D"

            if is_time_exit and not exit_signal:
                exit_signal = True
                exit_px = float(cur["close"])
                reason = exit_reason_time

            if exit_signal:
                # Calculate PnL
                if side == "LONG":
                    pnl = (exit_px - entry_px) * qty
                else:
                    pnl = (entry_px - exit_px) * qty
                
                gross += pnl
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                
                # Calculate hold duration
                hold_duration = cur.name - entry_ts
                try:
                    hold_bars = i - df.index.get_loc(entry_ts)
                except (KeyError, ValueError):
                    hold_bars = 0  # Fallback for edge case
                hold_days = hold_duration.total_seconds() / 86400
                
                detailed_trades.append({
                    "symbol": sym,
                    "entry_time": entry_ts,
                    "exit_time": cur.name,
                    "side": side,
                    "entry_price": entry_px,
                    "exit_price": exit_px,
                    "qty": qty,
                    "pnl": pnl,
                    "exit_reason": reason,
                    "hold_duration_bars": hold_bars,
                    "hold_duration_days": round(hold_days, 2),
                    "entry_rsi": entry_state.get("rsi", 0),
                    "entry_atr": entry_state.get("atr", 0),
                    "sl_px": entry_state.get("sl_px", 0),
                    "tp_px": entry_state.get("tp_px", 0)
                })
                
                # Verbose logging
                if verbose:
                    pnl_sign = "+" if pnl >= 0 else ""
                    entry_ts_str = str(entry_ts)[:16] if entry_ts else "N/A"
                    exit_ts_str = str(cur.name)[:16]
                    print(f"  [{sym}] {side} ENTRY: {entry_ts_str} | RSI={entry_state.get('rsi', 0):.1f} | Price=${entry_px:.2f}")
                    print(f"         EXIT:  {exit_ts_str} | Reason={reason} | P&L={pnl_sign}${pnl:.2f}")
                
                in_pos = False
                qty = 0
                
                # Update Health Manager
                trade_id = f"{sym}_{cur.name}"
                health_mgr.update_trade(sym, pnl, trade_id, now=cur.name)

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
    ap.add_argument("--days", default="30", help="Comma-separated list of days to backtest (e.g. 30,90,120)")
    ap.add_argument("--cash", type=float, default=10_000)
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--timeframe", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rth-only", dest="rth_only", action="store_true")
    ap.add_argument("--no-rth-only", dest="rth_only", action="store_false")
    ap.add_argument("--flatten-eod", dest="flatten_eod", action="store_true")
    ap.add_argument("--no-flatten-eod", dest="flatten_eod", action="store_false")
    ap.add_argument("--max-hold-days", type=int, default=0, help="Max days to hold (0=infinite/until signal)")
    ap.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    ap.add_argument("--verbose", "-v", action="store_true", help="Show detailed entry/exit for each trade")
    ap.add_argument("--slope-threshold", type=float, default=None, help="Require 3-bar slope to be steeper than this (e.g. -0.20) to enter")
    ap.add_argument("--slope-threshold-10", type=float, default=None, help="Require 10-bar slope to be steeper than this (e.g. -0.15) to enter")
    ap.add_argument("--adx-max", type=float, default=0, help="Skip entries when ADX > this value (0=disabled)")
    
    # Optimization Parameters
    ap.add_argument("--rsi-entry", type=float, default=None, help="Override RSI Entry Threshold (e.g. 25, 30)")
    ap.add_argument("--tp-r", type=float, default=None, help="Override Take Profit R-Multiple (e.g. 2.0)")
    ap.add_argument("--sl-atr", type=float, default=None, help="Override Stop Loss ATR Multiplier (e.g. 1.5)")
    
    ap.set_defaults(rth_only=True, flatten_eod=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    # Override config with CLI args if needed
    cfg["_flatten_eod"] = args.flatten_eod
    cfg["_max_hold_days"] = args.max_hold_days
    if args.slope_threshold is not None:
        cfg["slope_threshold"] = args.slope_threshold
    if getattr(args, 'slope_threshold_10', None) is not None:
        cfg["slope_threshold_10"] = args.slope_threshold_10

    # Inject ADX/RSI overrides into 'filters' section
    if "filters" not in cfg: cfg["filters"] = {}
    if args.adx_max > 0:
        cfg["filters"]["adx_threshold"] = args.adx_max
    if args.rsi_entry is not None:
        cfg["filters"]["rsi_oversold"] = args.rsi_entry
        
    # Inject Bracket overrides
    if "brackets" not in cfg: cfg["brackets"] = {}
    if args.tp_r is not None:
        cfg["brackets"]["take_profit_r"] = args.tp_r
    if args.sl_atr is not None:
        cfg["brackets"]["atr_mult_sl"] = args.sl_atr

    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = read_tickers(args.tickers)

    # Parse days list
    try:
        days_list = [int(d.strip()) for d in str(args.days).split(",") if d.strip()]
    except ValueError:
        print("Error: --days must be a comma-separated list of integers")
        return
    
    max_days = max(days_list)

    # Initialize Health Manager for Backtest
    # Use a temp file to avoid messing with live data
    health_file = "backtest_health.json"
    if os.path.exists(health_file):
        os.remove(health_file)
    
    health_mgr = TickerHealthManager(health_file, cfg.get("resilience", {}))

    def _load(sym: str) -> pd.DataFrame:
        # Load max history once
        # Pass verbose=False if quiet mode is on
        # Note: We need to update load_bars_for_symbol signature or just call fetch_latest_bars directly?
        # load_bars_for_symbol calls fetch_latest_bars. Let's update load_bars_for_symbol first.
        # Wait, load_bars_for_symbol is in this file. I need to update it too.
        # Fetch extra calendar days to ensure we cover the requested trading days (RTH)
        # 1.6 multiplier covers weekends and holidays (approx 252 trading days / 365 days = 0.69)
        fetch_days = int(max_days * 1.6)
        return load_bars_for_symbol(
            sym, cfg, fetch_days,
            timeframe_override=args.timeframe,
            limit_override=args.limit,
            rth_only_override=args.rth_only,
            verbose=not args.quiet
        )

    rows = []
    for sym in symbols:
        try:
            df_full = _load(sym)
        except Exception as e:
            print(f"[{sym}] data error: {e}")
            continue

        if df_full.empty:
            continue

        if not args.quiet:
            print(f"[{sym}] loaded bars={len(df_full)} (max_days={max_days})")
        
        # Run for each requested timeframe
        for d in days_list:
            # Slice approx rows for this window
            bars_per_day_15m = 390 // 15
            approx_rows = int(d) * bars_per_day_15m
            
            if len(df_full) > approx_rows:
                df_run = df_full.tail(approx_rows).copy()
            else:
                df_run = df_full.copy()
            
            # Inject symbol for logging
            cfg["_symbol"] = sym
            cfg["verbose"] = not args.quiet # Pass quiet state to simulator
            res = simulate_mean_reversion(df_run, cfg, health_mgr, sym, start_cash=args.cash, risk_pct=args.risk, verbose=args.verbose)
            rows.append({"symbol": sym, "days": d, **res})

    if not rows:
        print("No results.")
        return

    out = pd.DataFrame(rows).sort_values(["days", "net"], ascending=[True, False])
    if not args.quiet:
        print("\n=== Backtest Summary (days={}) ===".format(days_list))
        print(out.to_string(index=False))

    total_trades = int(out["trades"].sum())
    total_net = round(float(out["net"].sum()), 2)
    avg_winrate = round((out["trades"] * out["win_rate"] / 100.0).sum() / max(total_trades, 1) * 100.0, 1)
    
    print("\nTOTAL trades={} net={} win_rate={}%".format(total_trades, total_net, avg_winrate))

    # Save summary to JSON for analysis
    summary_list = []
    for i in out.index:
        summary_list.append({
            "symbol": out.loc[i, "symbol"],
            "days": int(out.loc[i, "days"]),
            "trades": int(out.loc[i, "trades"]),
            "net": float(out.loc[i, "net"]),
            "win_rate": float(out.loc[i, "win_rate"])
        })
    import json
    with open("backtest_summary.json", "w") as f:
        json.dump(summary_list, f, indent=2)

    # Daily Stats Analysis
    all_trades = []
    for row in rows:
        for t in row["detailed_trades"]:
            all_trades.append(t)
    
    if all_trades:
        df_trades = pd.DataFrame(all_trades)
        df_trades["exit_time"] = pd.to_datetime(df_trades["exit_time"])
        df_trades["entry_time"] = pd.to_datetime(df_trades["entry_time"])
        df_trades["date"] = df_trades["exit_time"].dt.date
        
        # Calculate Max Capital Usage per Day
        events = []
        for t in all_trades:
            # Entry event: +Capital
            # Use 'qty' if available, else approximate
            qty = t.get("qty", 0)
            if qty == 0 and t["entry_price"] > 0:
                 qty = int(2000 / t["entry_price"]) # Fallback approximation
            
            cost = t["entry_price"] * qty
            events.append({"time": t["entry_time"], "change": cost, "type": "entry"})
            # Exit event: -Capital
            events.append({"time": t["exit_time"], "change": -cost, "type": "exit"})
            
        df_events = pd.DataFrame(events).sort_values("time")
        df_events["current_capital"] = df_events["change"].cumsum()
        df_events["date"] = df_events["time"].dt.date
        
        # Max capital per day
        daily_max_cap = df_events.groupby("date")["current_capital"].max()
        
        # Group by date for stats
        daily = df_trades.groupby("date").agg(
            wins=('pnl', lambda x: (x > 0).sum()),
            losses=('pnl', lambda x: (x <= 0).sum()),
            net_pnl=('pnl', 'sum'),
            trades=('pnl', 'count')
        ).sort_index()
        
        # Join max capital
        daily = daily.join(daily_max_cap.rename("max_capital"))
        
        
        if not args.quiet:
            print("\n=== Daily Win/Loss Stats (Last 10 Days) ===")
            print(daily.tail(10))
        daily.to_csv("daily_stats.csv")
        if not args.quiet:
            print("\nSaved daily breakdown to daily_stats.csv")
        
        # Save detailed trades for loss analysis
        required_cols = [
            "symbol", "date", "entry_time", "exit_time", "side", 
            "entry_price", "exit_price", "qty", "pnl", "exit_reason",
            "hold_duration_bars", "hold_duration_days",
            "entry_rsi", "entry_atr", "sl_px", "tp_px"
        ]
        
        # Check for missing columns
        missing_cols = set(required_cols) - set(df_trades.columns)
        if missing_cols:
            if not args.quiet:
                print(f"WARNING: Missing columns in trades: {missing_cols}")
            # Use only available columns
            available_cols = [c for c in required_cols if c in df_trades.columns]
            df_trades_export = df_trades[available_cols].copy()
        else:
            df_trades_export = df_trades[required_cols].copy()
        
        df_trades_export.to_csv("detailed_trades.csv", index=False)
        
        # Save only losing trades for focused analysis
        df_losses = df_trades_export[df_trades_export["pnl"] <= 0].copy()
        if not df_losses.empty:
            df_losses.to_csv("loss_analysis.csv", index=False)
            if not args.quiet:
                print(f"Saved {len(df_trades_export)} trades to detailed_trades.csv")
                print(f"Saved {len(df_losses)} losing trades to loss_analysis.csv")
        else:
            if not args.quiet:
                print(f"Saved {len(df_trades_export)} trades to detailed_trades.csv")
                print("No losing trades! All trades were profitable.")

if __name__ == "__main__":
    main()
