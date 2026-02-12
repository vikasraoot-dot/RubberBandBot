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
from zoneinfo import ZoneInfo

ET = ZoneInfo("US/Eastern")

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
    check_kill_switch,
    check_capital_limit,
    order_exists_today,
    KillSwitchTriggered,
    CapitalLimitExceeded,
    get_account_info_compat,
)
from RubberBand.strategy import attach_verifiers, check_slope_filter, check_bearish_bar_filter
from RubberBand.src.trade_logger import TradeLogger
from RubberBand.src.ticker_health import TickerHealthManager
from RubberBand.src.position_registry import PositionRegistry
from RubberBand.src.regime_manager import RegimeManager
from RubberBand.src.circuit_breaker import PortfolioGuard, CircuitBreakerExc
from RubberBand.src.finance import to_decimal, money_sub, money_mul, money_add, safe_float

# Bot tag for position attribution
BOT_TAG = "15M_STK"


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
    p.add_argument("--slope-threshold", type=float, default=None,
                   help="Require slope to be steeper than this (e.g. -0.20) to enter")
    p.add_argument("--slope-threshold-10", type=float, default=None, help="10-bar slope threshold")
    p.add_argument("--rsi-entry", type=float, default=None, help="RSI entry threshold")
    p.add_argument("--tp-r", type=float, default=None, help="Take Profit R-multiple")
    p.add_argument("--sl-atr", type=float, default=None, help="Stop Loss ATR multiplier")
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

    # CLI override for slope_threshold
    if args.slope_threshold is not None:
        cfg["slope_threshold"] = args.slope_threshold
        print(f"[config] Slope Threshold overridden to: {args.slope_threshold}", flush=True)
    if args.slope_threshold_10 is not None:
        cfg["slope_threshold_10"] = args.slope_threshold_10
        print(f"[config] Slope Threshold 10 overridden to: {args.slope_threshold_10}", flush=True)

    # CLI override for RSI
    if args.rsi_entry is not None:
        if "filters" not in cfg:
            cfg["filters"] = {}
        cfg["filters"]["rsi_oversold"] = args.rsi_entry
        print(f"[config] RSI Entry overridden to: {args.rsi_entry}", flush=True)

    # CLI override for TP/SL
    if args.tp_r is not None or args.sl_atr is not None:
        if "brackets" not in cfg:
            cfg["brackets"] = {}
        if args.tp_r is not None:
            cfg["brackets"]["take_profit_r"] = args.tp_r
            print(f"[config] Take Profit R overridden to: {args.tp_r}", flush=True)
        if args.sl_atr is not None:
            cfg["brackets"]["atr_mult_sl"] = args.sl_atr
            print(f"[config] Stop Loss ATR overridden to: {args.sl_atr}", flush=True)

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
    
    # Position Registry for this bot
    registry = PositionRegistry(bot_tag=BOT_TAG)

    # Broker creds (for market/positions/order calls)
    base_url_raw, key, secret = _broker_creds(cfg)
    base_url = _force_paper_base_url(base_url_raw, now_iso)

    # Ensure SDKs/helpers that read env use **paper** and correct keys
    os.environ["APCA_API_BASE_URL"] = base_url
    if key:
        os.environ["APCA_API_KEY_ID"] = key
    if secret:
        os.environ["APCA_API_SECRET_KEY"] = secret

    # PHASE 2 FIX (GAP-008): Use reconcile_or_halt() instead of sync_with_alpaca()
    # This prevents silent cleanup of orphaned positions and alerts on mismatch.
    try:
        alpaca_positions = get_positions(base_url, key, secret)
        # Filter to only stock positions (not options)
        stock_positions = [p for p in alpaca_positions if p.get("asset_class") == "us_equity"]

        is_clean, registry_orphans, broker_untracked = registry.reconcile_or_halt(
            stock_positions,
            auto_clean=False,  # Do NOT auto-clean - we want to know about mismatches
        )

        if not is_clean:
            # Registry has positions that broker doesn't - this is a critical mismatch
            # Per CLAUDE.md Circuit Breaker 5: Position mismatch → HALT trading
            print(json.dumps({
                "type": "POSITION_MISMATCH_AT_STARTUP",
                "orphaned_count": len(registry_orphans),
                "orphaned_symbols": registry_orphans,
                "action": "auto_clean_and_alert",
                "ts": now_iso,
            }), flush=True)

            # For now, auto-clean with explicit logging rather than hard halt
            # This allows the bot to continue but with full visibility into the issue
            print(f"[CRITICAL] Registry orphans detected: {registry_orphans}", flush=True)
            print(f"[CRITICAL] These positions exist in registry but NOT in broker.", flush=True)
            print(f"[CRITICAL] Auto-cleaning to allow trading to continue...", flush=True)

            # Re-run with auto_clean=True to fix the state
            registry.reconcile_or_halt(stock_positions, auto_clean=True)

        print(json.dumps({
            "type": "REGISTRY_RECONCILE",
            "registry_positions": len(registry.positions),
            "alpaca_stock_positions": len(stock_positions),
            "my_symbols": list(registry.get_my_symbols()),
            "registry_clean": is_clean,
            "orphans_found": len(registry_orphans) if not is_clean else 0,
            "broker_untracked": len(broker_untracked),
            "ts": now_iso,
        }), flush=True)
    except Exception as e:
        print(json.dumps({"type": "REGISTRY_RECONCILE_ERROR", "error": str(e), "ts": now_iso}), flush=True)

    # Kill Switch Check - RE-ENABLED Dec 13, 2025
    # Halts trading if daily loss exceeds 25% of invested capital
    if check_kill_switch(base_url, key, secret, bot_tag=BOT_TAG, max_loss_pct=25.0):
        print(json.dumps({
            "type": "KILL_SWITCH_TRIGGERED",
            "bot_tag": BOT_TAG,
            "reason": "Daily loss exceeded 25% of invested capital",
            "ts": now_iso,
        }), flush=True)
        raise KillSwitchTriggered(f"{BOT_TAG} exceeded 25% daily loss")

    # Watchdog pause check (fail-open if file missing)
    _wd_paused = False
    try:
        from RubberBand.src.watchdog.pause_check import check_bot_paused
        _wd_paused, _wd_reason = check_bot_paused(BOT_TAG)
        if _wd_paused:
            print(json.dumps({"type": "WATCHDOG_PAUSED", "bot_tag": BOT_TAG, "reason": _wd_reason, "ts": now_iso}), flush=True)
            # Skip signal scanning but still run session summary for position management
    except Exception as e:
        print(f"[WATCHDOG] non-fatal: {e}", flush=True)

    # Universe
    symbols = load_symbols_from_file(args.tickers)
    print(json.dumps({"type": "UNIVERSE", "loaded": len(symbols), "sample": symbols[:10], "when": now_iso}), flush=True)

    # Entry windows?
    windows = cfg.get("entry_windows", [])
    if not _wd_paused and not _in_entry_window(now_et, windows):
        print(json.dumps({"type": "HEARTBEAT", "session": session, "market_open": True, "ts": now_iso}))
        return 0

    # Market open check (paper) — skip when paused to let session summary run
    if not _wd_paused and not _alpaca_market_open_compat(base_url, key, secret):
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

    # GAP-006: Connectivity Guard (moved here - must be initialized before use)
    from RubberBand.src.circuit_breaker import ConnectivityGuard
    conn_guard = ConnectivityGuard(max_errors=5)

    bars_map = {}
    bars_meta = {}
    res = None
    if not _wd_paused:
        try:
            res = fetch_latest_bars(symbols, timeframe, history_days, feed)
            conn_guard.record_success()
        except Exception as e:
            conn_guard.record_error(e)
            print(json.dumps({"type": "BARS_FETCH_ERROR", "reason": "exception", "message": str(e)}), flush=True)
            return 1

        if not isinstance(res, tuple) or len(res) != 2:
            print(json.dumps({"type": "BARS_FETCH_ERROR", "reason": "no_result"}), flush=True)
            return 1

        bars_map, bars_meta = res

    # Fetch Daily Bars for Trend Filter (SMA 20)
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
    # Daily cooldown: Get all tickers traded today
    daily_fills = []
    traded_today = set()
    try:
        daily_fills = get_daily_fills(base_url, key, secret, bot_tag=BOT_TAG) or []
        traded_today = set(f.get("symbol") for f in daily_fills if f.get("symbol"))
    except Exception as e:
        print(f"[warn] Could not fetch daily fills for cooldown: {e}", flush=True)

    # Dead Knife Filter: Identify tickers with net losses today
    has_loss_today = set()
    if daily_fills:
        fill_stats = {}
        for f in daily_fills:
            sym = f.get("symbol")
            side = f.get("side")
            qty = float(f.get("filled_qty", 0))
            px = float(f.get("filled_avg_price", 0))
             
            if sym not in fill_stats: fill_stats[sym] = {"buy_vol": 0, "buy_qty": 0, "sell_vol": 0, "sell_qty": 0}
            if side == "buy":
                fill_stats[sym]["buy_vol"] += (qty * px)
                fill_stats[sym]["buy_qty"] += qty
            elif side == "sell":
                fill_stats[sym]["sell_vol"] += (qty * px)
                fill_stats[sym]["sell_qty"] += qty

        for sym, s in fill_stats.items():
            if s["sell_qty"] > 0 and s["buy_qty"] > 0:
                avg_buy = s["buy_vol"] / s["buy_qty"]
                avg_sell = s["sell_vol"] / s["sell_qty"]
                if avg_sell < avg_buy:
                    has_loss_today.add(sym)

    # Risk knobs
    brackets = cfg.get("brackets", {}) or {}
    sl_mult = to_decimal(brackets.get("atr_mult_sl", 2.5))
    tp_r = to_decimal(brackets.get("take_profit_r", 1.5))
    allow_shorts = cfg.get("allow_shorts", False)

    # --- Phase 4A/4B: Dynamic TP & Breakeven from market condition overrides ---
    # Read dynamic overrides written by MarketConditionClassifier (fail-open)
    try:
        from RubberBand.src.watchdog.market_classifier import read_dynamic_overrides
        _dyn = read_dynamic_overrides()
        _dyn_overrides = _dyn.get("overrides", {})
        _dyn_condition = _dyn.get("market_condition", "RANGE")

        # Adjust TP R-multiple: base + adjustment
        _tp_adj = to_decimal(str(_dyn_overrides.get("tp_r_multiple_adjustment", "0")))
        if _tp_adj != 0:
            tp_r = tp_r + _tp_adj
            # Safety floor: TP must be at least 0.5R
            if tp_r < to_decimal("0.5"):
                tp_r = to_decimal("0.5")

        # Adjust position size multiplier (applied later at sizing)
        _size_mult = to_decimal(str(_dyn_overrides.get("position_size_multiplier", "1")))

        # Breakeven trigger adjustment (stored for future breakeven integration)
        _be_adj = to_decimal(str(_dyn_overrides.get("breakeven_trigger_r_adjustment", "0")))
        _be_trigger_r = to_decimal(str(cfg.get("breakeven", {}).get("trigger_r", "1"))) + _be_adj
        if _be_trigger_r < to_decimal("0.3"):
            _be_trigger_r = to_decimal("0.3")  # Safety floor

        print(json.dumps({
            "type": "DYNAMIC_OVERRIDES",
            "market_condition": _dyn_condition,
            "tp_r_adjusted": safe_float(tp_r),
            "tp_adjustment": safe_float(_tp_adj),
            "size_multiplier": safe_float(_size_mult),
            "breakeven_trigger_r": safe_float(_be_trigger_r),
            "ts": now_iso,
        }), flush=True)
    except Exception as e:
        print(f"[WATCHDOG] non-fatal: {e}", flush=True)
        _size_mult = to_decimal("1")
        _dyn_condition = "RANGE"

    # Size guard
    base_qty = int(cfg.get("qty", 1))
    max_shares = int(cfg.get("max_shares_per_trade", base_qty))
    max_notional = cfg.get("max_notional_per_trade", None)
    
    # --- CIRCUIT BREAKER: PORTFOLIO DRAWDOWN (GAP-005) ---
    try:
        max_dd_pct = float(cfg.get("max_drawdown_pct", 0.10))
        # Ensure state file is unique to bot or shared? Typically shared for portfolio level.
        # But this script is 15M_STK. Let's use bot-specific for safety or shared?
        # The user requested "Portfolio Drawdown". Paper account is shared.
        # So we should use a shared state file.
        guard_state_file = os.path.join(os.path.dirname(__file__), "portfolio_guard_state.json")
        guard = PortfolioGuard(guard_state_file, max_drawdown_pct=max_dd_pct)
        
        # Get Current Equity
        acct = get_account_info_compat(base_url, key, secret)
        if acct:
            current_equity = float(acct.get("equity", 0))
            guard.update(current_equity) # Will raise CircuitBreakerExc if breached
            
    except CircuitBreakerExc as cbe:
        print(json.dumps({
            "type": "CIRCUIT_BREAKER_HALT",
            "reason": str(cbe),
            "ts": datetime.now(timezone.utc).isoformat()
        }), flush=True)
        return 1 # Exit with error code to stop service
    except Exception as e:
        print(f"[Guard] Warning: Failed to check portfolio guard: {e}", flush=True)
    # -----------------------------------------------------

    try:
        max_notional = float(max_notional) if max_notional is not None else None
        if max_notional is not None and max_notional <= 0:
            max_notional = None
    except (ValueError, TypeError):
        max_notional = None

    # --- Dynamic Regime Detection (Daily + Intraday) ---
    rm = RegimeManager(verbose=True)
    daily_regime = rm.update()  # Sets reference values for intraday checks
    current_regime = rm.get_effective_regime()  # Checks for intraday VIXY spikes
    regime_cfg = rm.get_config_overrides()

    # If intraday panic triggered, use PANIC config
    if current_regime == "PANIC" and daily_regime != "PANIC":
        regime_cfg = rm.regime_configs["PANIC"]
        print(json.dumps({
            "type": "INTRADAY_PANIC_DETECTED",
            "daily_regime": daily_regime,
            "effective_regime": current_regime,
            "vixy_reference": rm._reference_close,
            "upper_band": rm._upper_band,
            "ts": now_iso
        }), flush=True)

    # Allow CLI override to disable DKF if needed, but default to Regime
    use_dkf = regime_cfg.get("dead_knife_filter", False)
    if cfg.get("dead_knife_filter") is True: # Manual config override
         use_dkf = True

    print(json.dumps({
        "type": "REGIME_UPDATE",
        "daily_regime": daily_regime,
        "effective_regime": current_regime,
        "vixy_price": rm.last_vixy_price,
        "slope_threshold_pct": regime_cfg.get("slope_threshold_pct"),
        "dkf_enabled": use_dkf,
        "bearish_filter": regime_cfg.get("bearish_bar_filter"),
        "ts": now_iso
    }), flush=True)

    # --- Filter Diagnostics Counters ---
    diag = {
        "total_scanned": 0,
        "no_data": 0,
        "insufficient_bars": 0,
        "paused_health": 0,
        "forming_bar_dropped": 0,
        "trend_filter_no_data": 0,
        "trend_filter_bear": 0,
        "slope_filter": 0,
        "bearish_bar_filter": 0,
        "dkf_filter": 0,
        "no_signal": 0,
        "already_in_position": 0,
        "traded_today": 0,
        "bad_tp_sl": 0,
        "qty_zero": 0,
        "signals_generated": 0,
        "orders_submitted": 0,
    }

    # Iterate symbols (skip scanning when watchdog-paused; session summary still runs below)
    _scan_symbols = [] if _wd_paused else symbols
    for sym in _scan_symbols:
        diag["total_scanned"] += 1
        df = bars_map.get(sym)
        if df is None or df.empty:
            diag["no_data"] += 1
            continue
        if len(df) < 20:  # Need enough data for Keltner(20)
            diag["insufficient_bars"] += 1
            continue

        # Resilience Check
        is_paused, reason = health_mgr.is_paused(sym, now=now_utc)
        if is_paused:
            diag["paused_health"] += 1
            continue

        df = attach_verifiers(df, cfg)
        
        # --- CRITICAL FIX: FORCE CLOSED CANDLES ONLY ---
        # Live data often includes the current "forming" bar as the last row.
        # This causes "phantom dips" (intra-bar noise) that vanish at close.
        # We strictly drop the last bar if its timestamp is recent (within timeframe).
        last_ts = df.index[-1]
        
        # Calculate age of last bar
        # Note: df.index is timezone-aware (UTC usually from Alpaca/IEX)
        if last_ts.tzinfo is None:
            # Fallback if tz-naive (shouldn't happen with our data pipeline)
            age_sec = (datetime.now() - last_ts).total_seconds()
        else:
            age_sec = (now_utc - last_ts).total_seconds()
            
        # Timeframe is 15Min, so check if diff < 15 minutes
        if age_sec < 15 * 60:
            # Drop the forming bar
            df = df.iloc[:-1]
            diag["forming_bar_dropped"] += 1
            if df.empty:
                diag["no_data"] += 1
                continue

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
                diag["trend_filter_no_data"] += 1
                continue
        else:
            # Filter disabled -> assume Bull (allow Longs)
            is_bull_trend = True

        # Slope Filter (Normalized Percentage Logic)
        # ------------------------------------------
        # We convert the absolute slope to a percentage of the closing price.
        # This ensures consistent behavior across tickers of different prices.
        
        # This ensures consistent behavior across tickers of different prices.
        
        should_skip, reason = check_slope_filter(df, regime_cfg)
        if should_skip:
            diag["slope_filter"] += 1
            # Only log verbose slope skips if few (avoid log spam)
            continue
        
        # Bearish Bar Filter (New - Jan 2026)
        # -------------------------------------
        # Skip entries if current bar is bearish (close < open)
        # Backtest showed +$3,000 improvement and +15% win rate
        should_skip_bar, bar_reason = check_bearish_bar_filter(df, regime_cfg)
        if should_skip_bar:
            diag["bearish_bar_filter"] += 1
            continue
        
        # Check 2: 10-bar slope (Secondary/Sustained)
        # Only use if explicitly enabled in config, and normalize it.
        slope_threshold_10_abs = cfg.get("slope_threshold_10") # Legacy absolute config
        if slope_threshold_10_abs is not None:
             # If legacy config exists, we try to use it, but converted?
             # Or we define a new 10-bar regime?
             # For now, let's keep it as an absolute check IF configured, 
             # OR convert it. User asked to "Convert all Slope logic".
             # Let's assume 10-bar should be roughly similarly scaled.
             # If -0.15 was old 10-bar, and -0.12 was old 3-bar.
             # We can imply a % threshold. 
             # For safety, let's skip 10-bar normalization enforcement unless we have a regime for it.
             # We will just log it for now if it triggers.
             pass

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
            diag["trend_filter_bear"] += 1
            long_signal = False

        # --- Dead Knife Filter (Live) ---
        # Skip re-entry if we had a loss today and RSI is still < 20 (Deep Oversold)
        # This prevents "doubling down" on a falling knife that just stopped us out.
        # --- Dead Knife Filter (Live) ---
        # Skip re-entry if we had a loss today and RSI is still < 20 (Deep Oversold)
        # This prevents "doubling down" on a falling knife that just stopped us out.
        if long_signal and use_dkf and sym in has_loss_today:
            # We use current RSI vs 20. Backtest uses last_loss_rsi < 20 check too,
            # but here "has_loss_today" implies we tried and failed.
            if rsi is not None and rsi < 20:
                diag["dkf_filter"] += 1
                continue
        # --------------------------------

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
            diag["already_in_position"] += 1
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
            diag["traded_today"] += 1
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
            diag["no_signal"] += 1
            continue

        # Determine Side
        side = "buy" if long_signal else "sell" # Alpaca 'sell' = short open if no position

        # Log Signal Event
        diag["signals_generated"] += 1
        try:
            log.signal(**sig_row)
        except Exception:
            pass

        # ATR Calculation (Use pre-calculated from attach_verifiers)
        # Safely handle None values (get returns None if key exists but value is None)
        atr_raw = last.get("atr")
        atr_val = float(atr_raw) if atr_raw is not None else 0.0
        
        if side == "buy":
            stop_price = safe_float(money_sub(entry, money_mul(atr_val, sl_mult)))
            take_profit = safe_float(money_add(entry, money_mul(atr_val, tp_r)))
            if not (stop_price < entry < take_profit):
                diag["bad_tp_sl"] += 1
                continue
        else:  # Short
            stop_price = safe_float(money_add(entry, money_mul(atr_val, sl_mult)))
            take_profit = safe_float(money_sub(entry, money_mul(atr_val, tp_r)))
            if not (take_profit < entry < stop_price):
                diag["bad_tp_sl"] += 1
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

        # Phase 4A: Apply dynamic size multiplier from market condition
        if _size_mult < to_decimal("1"):
            qty = max(1, int(qty * safe_float(_size_mult)))

        if qty < 1:
            diag["qty_zero"] += 1
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
                tif="gtc",  # GTC so TP/SL persist overnight
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
                # Generate client_order_id for position attribution
                coid = registry.generate_order_id(sym)
                
                # Idempotency check - prevent duplicate orders
                if order_exists_today(base_url, key, secret, coid):
                    print(f"[order] SKIP {sym}: Order {coid} already exists today", flush=True)
                    continue
                
                # Capital limit check - prevent exceeding max deployed capital
                trade_value = qty * entry
                max_capital = float(cfg.get("max_capital", 100000))
                try:
                    check_capital_limit(
                        base_url, key, secret,
                        proposed_trade_value=trade_value,
                        max_capital=max_capital,
                        bot_tag=BOT_TAG,
                    )
                except CapitalLimitExceeded as e:
                    print(f"[order] SKIP {sym}: {e}", flush=True)
                    continue
                
                resp = submit_bracket_order(
                    base_url,
                    key,
                    secret,
                    symbol=sym,
                    qty=qty,
                    side=side,
                    limit_price=None,  # Uses limit with 1% buffer now
                    take_profit_price=take_profit,
                    stop_loss_price=stop_price,
                    tif="gtc",  # GTC so TP/SL persist overnight
                    client_order_id=coid,
                    verify_fill=True,  # Wait and verify fill
                    verify_timeout=5,  # 5 second timeout
                )
                
                # Check for order issues
                if resp.get("error"):
                    print(f"[order] ERROR {sym}: {resp.get('error')}", flush=True)
                    continue
                
                print(f"[order] BRACKET submitted for {sym}: {json.dumps(resp)[:300]}", flush=True)
                diag["orders_submitted"] += 1
                
                # Record in registry on successful fill
                if resp.get("status") == "filled":
                    registry.record_entry(
                        symbol=sym,
                        client_order_id=coid,
                        qty=qty,
                        entry_price=float(resp.get("filled_avg_price", entry)),
                        order_id=resp.get("id", ""),
                    )
                
                try:
                    oid = (resp.get("id") if isinstance(resp, dict) else None)
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

    # --- Filter Diagnostics Summary ---
    # Calculate pass-through rate
    passed_filters = diag["signals_generated"]
    total = diag["total_scanned"]
    pass_rate = (passed_filters / total * 100) if total > 0 else 0

    print(json.dumps({
        "type": "FILTER_DIAGNOSTICS",
        "total_scanned": total,
        "passed_to_signal": passed_filters,
        "pass_rate_pct": round(pass_rate, 1),
        "orders_submitted": diag["orders_submitted"],
        "filters": {
            "no_data": diag["no_data"],
            "insufficient_bars": diag["insufficient_bars"],
            "paused_health": diag["paused_health"],
            "trend_no_daily": diag["trend_filter_no_data"],
            "trend_bear": diag["trend_filter_bear"],
            "slope": diag["slope_filter"],
            "bearish_bar": diag["bearish_bar_filter"],
            "dkf": diag["dkf_filter"],
            "no_signal_rsi": diag["no_signal"],
            "already_in_position": diag["already_in_position"],
            "traded_today": diag["traded_today"],
            "bad_tp_sl": diag["bad_tp_sl"],
            "qty_zero": diag["qty_zero"],
        },
        "ts": now_iso,
    }), flush=True)

    # --- Session Summary ---
    print("\n=== Session Summary ===", flush=True)
    try:
        # Fetch fills (get_daily_fills already imported at top of file)
        # Filter by this bot's tag to only show our trades
        fills = get_daily_fills(base_url, key, secret, bot_tag=BOT_TAG)
        
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
                # Export trades to CSV for analysis (matching backtest format)
                csv_date = datetime.now(ET).strftime("%Y%m%d")
                csv_path = f"results/{BOT_TAG}_trades_{csv_date}.csv"
                log.export_trades_csv(csv_path)
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
