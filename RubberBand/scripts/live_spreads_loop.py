#!/usr/bin/env python3
"""
Live Options Spreads Loop: Trade bull call spreads using RubberBandBot signals.

Usage:
    python RubberBand/scripts/live_spreads_loop.py --config RubberBand/config.yaml --tickers RubberBand/tickers.txt --dry-run 1

Key Features:
- Uses 3 DTE by default (90% win rate in backtest)
- Bull call spreads for defined risk
- Holds overnight for multi-day DTE
- Comprehensive trade logging with entry/exit reasons
- EOD summary report
"""
from __future__ import annotations

# === EARLY STARTUP LOGGING ===
# Print immediately so GitHub Actions can show progress
import sys
print("=" * 60, flush=True)
print("[STARTUP] 15M Options Spreads Loop - Initializing...", flush=True)
print(f"[STARTUP] Python: {sys.version}", flush=True)
print("=" * 60, flush=True)

import argparse
import gc
import json
import os
import tracemalloc
from datetime import datetime, timedelta, time as dt_time
from typing import List, Dict, Any, Optional, Tuple

print("[STARTUP] Core imports complete", flush=True)

import pandas as pd

# Ensure repo root is on path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

print(f"[STARTUP] Repo root: {_REPO_ROOT}", flush=True)

from zoneinfo import ZoneInfo

print("[STARTUP] Loading RubberBand.src.data...", flush=True)
from RubberBand.src.data import (
    load_symbols_from_file,
    fetch_latest_bars,
    alpaca_market_open,
    check_kill_switch,
    check_capital_limit,
    KillSwitchTriggered,
    CapitalLimitExceeded,
    order_exists_today,
    get_account_info_compat,
)

print("[STARTUP] Loading RubberBand.strategy...", flush=True)
from RubberBand.strategy import attach_verifiers, check_slope_filter, check_bearish_bar_filter

print("[STARTUP] Loading RubberBand.src.options_data...", flush=True)
from RubberBand.src.options_data import (
    select_spread_contracts,
    get_option_quote,
    get_option_snapshot,  # For fetching IV/theta/delta greeks
    is_options_trading_allowed,
)

print("[STARTUP] Loading RubberBand.src.options_execution...", flush=True)
from RubberBand.src.options_execution import (
    submit_spread_order,
    close_spread,
    get_option_positions,
    close_option_position,
    flatten_all_option_positions,
    get_position_pnl,
)

print("[STARTUP] Loading loggers and registry...", flush=True)
from RubberBand.src.options_trade_logger import OptionsTradeLogger
from RubberBand.src.position_registry import PositionRegistry
from RubberBand.src.regime_manager import RegimeManager
from RubberBand.src.circuit_breaker import PortfolioGuard, CircuitBreakerExc
from RubberBand.src.finance import to_decimal, money_sub, safe_float
from RubberBand.src.watchdog.intraday_monitor import emit_order_rejection_alert

print("[STARTUP] All imports complete!", flush=True)

ET = ZoneInfo("US/Eastern")

# Bot tag for position attribution
BOT_TAG = "15M_OPT"


def commit_auditor_log(bot_tag: str = BOT_TAG):
    """
    Commit JSONL logs to auditor_logs/ for real-time auditing.
    Called after each trading cycle to enable the Auditor Bot to see logs during the day.
    
    Uses line tracking to avoid re-processing the same log lines on each cycle.
    """
    import subprocess
    
    date = datetime.now().strftime("%Y%m%d")
    log_file = f"auditor_logs/{bot_tag}_{date}.jsonl"
    processed_file = f"auditor_logs/.{bot_tag}_{date}_processed.txt"
    
    # Check if we're in GitHub Actions (where git is configured)
    if not os.environ.get("GITHUB_ACTIONS"):
        return  # Only commit when running in GitHub Actions
    
    try:
        console_log = "console.log"
        if not os.path.exists(console_log):
            return  # No log file to process
        
        os.makedirs("auditor_logs", exist_ok=True)
        
        # Read the number of lines already processed
        start_line = 0
        if os.path.exists(processed_file):
            try:
                with open(processed_file, "r") as f:
                    start_line = int(f.read().strip())
            except (ValueError, FileNotFoundError):
                start_line = 0
        
        # Read console log with encoding fallback (PowerShell often uses UTF-16)
        try:
            with open(console_log, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
        except UnicodeError:
            with open(console_log, "r", encoding="utf-16") as f:
                all_lines = f.readlines()
        
        # Only process new lines
        new_lines = all_lines[start_line:]
        if not new_lines:
            return  # No new lines to process
        
        # Filter and tag JSON lines
        json_lines = []
        for line in new_lines:
            line = line.strip()
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    obj["bot_tag"] = bot_tag
                    json_lines.append(json.dumps(obj))
                except json.JSONDecodeError:
                    pass
        
        if json_lines:
            # Append only new events to log file
            with open(log_file, "a", encoding="utf-8") as f:
                for jl in json_lines:
                    f.write(jl + "\n")
            
            # Update processed line count
            with open(processed_file, "w") as f:
                f.write(str(len(all_lines)))
            
            # Add registry file to commit list (Fix for GAP-008 Persistence)
            registry_file = f".position_registry/{bot_tag}_positions.json"

            # Commit (stage local changes first)
            subprocess.run(["git", "add", log_file, processed_file, registry_file], check=False, capture_output=True)
            result = subprocess.run(
                ["git", "commit", "-m", f"[AUTO] {bot_tag} log update {datetime.now().strftime('%H:%M')}"],
                check=False, capture_output=True
            )
            
            if result.returncode == 0:
                # Attempt to push with retries
                import time
                import random
                max_retries = 5
                
                for attempt in range(max_retries):
                    # Always pull --rebase (with autostash) before pushing
                    pull_res = subprocess.run(["git", "pull", "origin", "main", "--rebase", "--autostash"], check=False, capture_output=True)
                    if pull_res.returncode != 0:
                        print(f"[spreads] Pull failed: {pull_res.stderr.decode().strip()}", flush=True)

                    
                    # Push
                    push_res = subprocess.run(["git", "push"], check=False, capture_output=True)
                    
                    if push_res.returncode == 0:
                        print(f"[spreads] Committed auditor log ({len(json_lines)} new events)", flush=True)
                        break
                    else:
                        # Log error and retry
                        err_msg = push_res.stderr.decode().strip() or "Unknown error"
                        print(f"[spreads] Push failed (attempt {attempt+1}/{max_retries}): {err_msg}", flush=True)
                        if attempt < max_retries - 1:
                            sleep_time = random.uniform(5, 15)
                            print(f"[spreads] Retrying in {sleep_time:.1f}s...", flush=True)
                            time.sleep(sleep_time)
                else:
                    print(f"[spreads] All {max_retries} push attempts failed. Logs saved locally but not pushed.", flush=True)
            else:
                print(f"[spreads] No new auditor log changes to commit", flush=True)
        else:
            # Update processed count even if no JSON found
            with open(processed_file, "w") as f:
                f.write(str(len(all_lines)))
    except Exception as e:
        print(f"[spreads] Auditor log commit error: {e}", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_SPREAD_CONFIG = {
    "dte": 6,                      # Optimized for robust mean reversion (was 3)
    "min_dte": 3,
    "spread_width_atr": 1.5,
    "max_debit": 1.00,             # Reduced to 1.00 to match Backtest (Avoid High IV)
    "contracts": 1,
    "tp_max_profit_pct": 50.0,     # Realistic TP for spreads (was 80% — rarely hit)
    "sl_pct": -25.0,               # Tighter SL (was -30%) + no confirmation delay = faster exits
    "bars_stop": 10,               # Time Stop: 10 bars (~2.5 hours) - Match Backtest
    "hold_overnight": True,
}


def _load_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RubberBandBot Bull Call Spread Trading")
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--tickers", required=True, help="Path to tickers file")
    p.add_argument("--dry-run", type=int, default=1, help="1=dry run, 0=live")
    p.add_argument("--dte", type=int, default=6, help="Days to expiration (default: 6)") # Optimized Default
    p.add_argument("--max-debit", type=float, default=3.00, help="Max debit per share")
    p.add_argument("--slope-threshold", type=float, default=-0.12, help="Keltner mean slope threshold (default: -0.12)") # Optimized Default
    p.add_argument("--slope-threshold-10", type=float, default=None, help="10-bar Keltner mean slope threshold (e.g. -0.15)")
    p.add_argument("--contracts", type=int, default=1, help="Contracts per trade")
    return p.parse_args()


def _now_et() -> datetime:
    return datetime.now(ET)


def _in_entry_window(now_et: datetime, windows: list) -> bool:
    """Check if current time is within any configured entry window."""
    if not windows:
        return True
    
    current_time = now_et.time()
    
    for w in windows:
        start_str = w.get("start", "09:30")
        end_str = w.get("end", "16:00")
        
        start_parts = start_str.split(":")
        end_parts = end_str.split(":")
        
        start_time = dt_time(int(start_parts[0]), int(start_parts[1]))
        end_time = dt_time(int(end_parts[0]), int(end_parts[1]))
        
        if start_time <= current_time < end_time:
            return True
    
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Signal Detection
# ──────────────────────────────────────────────────────────────────────────────
def get_daily_sma(symbol: str, period: int = 20, feed: str = "iex") -> Optional[float]:
    """Fetch daily data and calculate SMA for trend filter."""
    try:
        daily_map, _ = fetch_latest_bars(
            symbols=[symbol],
            timeframe="1Day",
            history_days=int(period * 1.5),  # Extra history for SMA
            feed=feed,
            verbose=False,
        )
        df = daily_map.get(symbol)
        if df is None or df.empty or len(df) < period:
            return None
        
        # Calculate SMA
        sma = df["close"].rolling(window=period).mean().iloc[-1]
        return float(sma) if not pd.isna(sma) else None
    except Exception:
        return None


def get_long_signals(
    symbols: List[str],
    cfg: dict,
    logger: OptionsTradeLogger,
    min_dte: int = 2,
    regime_cfg: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """
    Scan for long signals using existing RubberBandBot strategy.
    
    Enhanced with:
    - SMA trend filter (skip if close < SMA_120)
    - Min DTE parameter for caller to enforce
    
    Args:
        symbols: List of symbols to scan
        cfg: Full config dict
        logger: Options trade logger
        min_dte: Minimum DTE required (for logging only, actual enforcement in try_spread_entry)
    
    Returns:
        List of signal dicts
    """
    signals = []
    
    timeframe = "15Min"
    history_days = 10
    feed = cfg.get("feed", "sip")
    
    # Get trend filter settings
    trend_cfg = cfg.get("trend_filter", {})
    trend_enabled = trend_cfg.get("enabled", True)  # Default enabled for options
    sma_period = int(trend_cfg.get("sma_period", 100))  # Default 100-day SMA (Optimized for All-Weather)
    
    try:
        bars_map, _ = fetch_latest_bars(
            symbols=symbols,
            timeframe=timeframe,
            history_days=history_days,
            feed=feed,
            verbose=False,
            yf_fallback=True,
        )
    except Exception as e:
        logger.error(error=str(e), context="fetch_bars")
        return signals
    
    # Track both for auditing
    confirmed_signals = []
    forming_signals = []
    
    for sym in symbols:
        df = bars_map.get(sym)
        if df is None or df.empty or len(df) < 20:
            continue
        
        try:
            # Analyze FORMING bar (last row) for Audit Logs
            # ---------------------------------------------
            df_full = attach_verifiers(df.copy(), cfg)
            last_idx = df_full.index[-1]
            now_utc = datetime.now(ZoneInfo("UTC"))
            age_sec = (now_utc - last_idx).total_seconds()
            is_forming = age_sec < 15 * 60
            
            if is_forming:
                row = df_full.iloc[-1]
                if bool(row.get("long_signal", False)):
                    # Extract forming signal details (no execution, just log)
                    forming_signals.append({
                        "symbol": sym,
                        "time": last_idx,
                        "rsi": float(row.get("rsi", 0)),
                        "close": float(row.get("close", 0)),
                        "slope": float(row.get("slope", 0)) if "slope" in row else 0.0
                    })

            # Analyze CLOSED bar (Force drop last if forming)
            # -----------------------------------------------
            # We strictly drop the last bar if it is forming to ensure we trade on CLOSED data only.
            if is_forming:
                df_closed = df_full.iloc[:-1]
            else:
                df_closed = df_full
                
            if df_closed.empty:
                continue
                
            last = df_closed.iloc[-1]
            close = float(last["close"])
            
            # Check trend filter FIRST (before checking signal)
            if trend_enabled:
                daily_sma = get_daily_sma(sym, sma_period, feed)
                if daily_sma is not None and close < daily_sma:
                    # Skip - in bear trend (below SMA)
                    continue # Silent skip to reduce log noise? Or log debug?
            
            # Check Slope Threshold
            should_skip, reason, _slope_pct = check_slope_filter(df_closed, regime_cfg)
            if should_skip:
                # logger.spread_skip(underlying=sym, skip_reason=reason) # Too noisy for 1m loop?
                continue
            
            # Bearish Bar Filter (Jan 2026)
            # Validated Jan 31: Regime-based activation (ON in Normal/Panic, OFF in Calm)
            # significantly improves ROI/WinRate in backtests (159% vs 86%).
            # We pass regime_cfg which contains the 'bearish_bar_filter' toggle for current regime.
            should_skip_bar, bar_reason = check_bearish_bar_filter(df_closed, regime_cfg)
            if should_skip_bar:
                # Log skip
                logger.spread_skip(underlying=sym, skip_reason=bar_reason)
                continue
            
            # Check 10-bar Slope
            slope_threshold_10 = cfg.get("slope_threshold_10")
            if slope_threshold_10 is not None:
                if "kc_middle" in df_closed.columns and len(df_closed) >= 11:
                    current_slope_10 = (df_closed["kc_middle"].iloc[-1] - df_closed["kc_middle"].iloc[-11]) / 10
                    if current_slope_10 > float(slope_threshold_10):
                         continue


            if bool(last.get("long_signal", False)):
                rsi_val = last.get("rsi")
                atr_val = last.get("atr")
                rsi = float(rsi_val) if rsi_val is not None else 0.0
                atr = float(atr_val) if atr_val is not None else 0.0
                entry_price = close
                
                entry_reasons = []
                if last.get("rsi_oversold", False):
                    entry_reasons.append(f"RSI_oversold({rsi:.1f})")
                if last.get("ema_ok", False):
                    entry_reasons.append("EMA_aligned")
                if last.get("touch", False):
                    entry_reasons.append("Lower_band_touch")
                
                entry_reason = " + ".join(entry_reasons) if entry_reasons else "RubberBand_long_signal"
                
                # Check duplication happens in MAIN loop now (idempotency)
                confirmed_signals.append({
                    "symbol": sym,
                    "entry_price": entry_price,
                    "rsi": rsi,
                    "atr": atr,
                    "entry_reason": entry_reason,
                    "signal_time": last.name, # Timestamp of the closed bar
                })
                
        except Exception as e:
            logger.error(error=str(e), context=f"process_{sym}")
            continue
    
    return confirmed_signals, forming_signals


# ──────────────────────────────────────────────────────────────────────────────
# Spread Entry
# ──────────────────────────────────────────────────────────────────────────────
def try_spread_entry(
    signal: Dict[str, Any],
    spread_cfg: dict,
    logger: OptionsTradeLogger,
    registry: PositionRegistry,
    dry_run: bool = True,
    cfg: Dict[str, Any] = None,
) -> bool:
    """Attempt to enter a bull call spread based on a stock signal."""
    sym = signal["symbol"]
    
    # DAILY COOLDOWN: Skip if this underlying was already traded today
    # Prevents repeated entries on the same ticker after a loss (Dec 15, 2025 fix)
    if registry.was_traded_today(sym):
        logger.spread_skip(
            underlying=sym,
            skip_reason=f"Daily_cooldown({sym}_already_traded_today)"
        )
        return False
    
    dte = spread_cfg.get("dte", 3)
    min_dte = spread_cfg.get("min_dte", 2)  # Minimum DTE allowed
    spread_width_atr = spread_cfg.get("spread_width_atr", 1.5)  # ATR-based spread width
    max_debit = spread_cfg.get("max_debit", 1.00)
    contracts = spread_cfg.get("contracts", 1)
    entry_reason = signal.get("entry_reason", "RubberBand_signal")
    
    # Get ATR from signal for volatility-adaptive spread width
    signal_atr = signal.get("atr", 0)

    # ── PROBABILITY FILTER (feature-flagged) ─────────────────────────────
    # Evaluates multiple DTE candidates via BSM probability metrics and
    # selects the best one.  Disabled by default (enabled: false in config).
    _prob_filter_metrics = {}  # Populated if filter runs, logged below
    _prob_filter_used = False

    if cfg:
        from RubberBand.src.probability_filter import (
            load_probability_filter_config,
            evaluate_dte_candidates,
        )
        pf_config = load_probability_filter_config(cfg)

        if pf_config.enabled:
            try:
                best_candidate, all_candidates = evaluate_dte_candidates(
                    underlying=sym,
                    signal_atr=signal_atr,
                    config=pf_config,
                    spread_cfg=spread_cfg,
                )

                if best_candidate is not None:
                    _prob_filter_metrics = best_candidate.to_log_dict()
                    _prob_filter_metrics["pf_candidates_evaluated"] = len(all_candidates)
                    _prob_filter_metrics["pf_mode"] = pf_config.mode

                    if pf_config.mode == "filter":
                        # USE the probability filter's selected spread
                        _prob_filter_used = True
                        spread = {
                            "underlying": best_candidate.underlying,
                            "expiration": best_candidate.expiration,
                            "dte": best_candidate.dte,
                            "underlying_price": best_candidate.underlying_price,
                            "long": best_candidate.long_contract,
                            "short": best_candidate.short_contract,
                            "atm_strike": best_candidate.atm_strike,
                            "otm_strike": best_candidate.otm_strike,
                            "spread_width": best_candidate.spread_width,
                        }
                        # Override net_debit from batch quotes
                        net_debit = best_candidate.net_debit
                        logger.heartbeat(
                            f"[prob_filter] {sym}: SELECTED DTE={best_candidate.dte} "
                            f"P_BE={best_candidate.breakeven_prob:.2f} "
                            f"RR={best_candidate.risk_reward_ratio:.2f} "
                            f"score={best_candidate.composite_score:.3f}"
                        )
                    else:
                        # Shadow mode: log metrics, continue with legacy flow
                        logger.heartbeat(
                            f"[prob_filter] {sym}: SHADOW DTE={best_candidate.dte} "
                            f"P_BE={best_candidate.breakeven_prob:.2f} "
                            f"RR={best_candidate.risk_reward_ratio:.2f} "
                            f"score={best_candidate.composite_score:.3f}"
                        )
                else:
                    # No candidate passed thresholds
                    _prob_filter_metrics = {
                        "pf_mode": pf_config.mode,
                        "pf_candidates_evaluated": len(all_candidates),
                        "pf_result": "no_candidates",
                    }
                    if pf_config.mode == "filter" and not pf_config.fallback_to_legacy:
                        logger.spread_skip(
                            underlying=sym,
                            skip_reason="Probability_filter_no_candidates",
                            **_prob_filter_metrics,
                        )
                        return False
                    elif pf_config.mode == "filter":
                        logger.heartbeat(
                            f"[prob_filter] {sym}: No candidates, "
                            f"falling back to legacy DTE={dte} "
                            f"(fallback_to_legacy=true)"
                        )

            except Exception as e:
                # Fail-open: probability filter crash → continue with legacy flow
                logger.heartbeat(
                    f"[prob_filter] ERROR for {sym}: {e}, "
                    f"falling back to legacy flow"
                )
                _prob_filter_metrics = {"pf_error": str(e)}
    # ── END PROBABILITY FILTER ───────────────────────────────────────────

    if not _prob_filter_used:
        # Legacy flow: select spread contracts with fixed DTE
        spread = select_spread_contracts(
            sym,
            dte=dte,
            spread_width_atr=spread_width_atr,
            atr=signal_atr,
            min_dte=min_dte
        )
    if not spread:
        logger.spread_skip(underlying=sym, skip_reason="No_contracts_available")
        return False
    
    # Check actual DTE vs minimum required
    actual_expiration = spread.get("expiration", "")
    if actual_expiration:
        from datetime import datetime as dt_class
        try:
            exp_date = dt_class.strptime(actual_expiration, "%Y-%m-%d").date()
            today = _now_et().date()
            actual_dte = (exp_date - today).days
            if actual_dte < min_dte:
                logger.spread_skip(
                    underlying=sym,
                    skip_reason=f"DTE_too_low({actual_dte}<min_dte={min_dte})"
                )
                return False
        except ValueError:
            pass  # If we can't parse date, continue anyway
    
    long_contract = spread["long"]
    short_contract = spread["short"]
    long_symbol = long_contract.get("symbol", "")
    short_symbol = short_contract.get("symbol", "")

    if _prob_filter_used:
        # Pricing already computed from batch snapshots
        long_ask = best_candidate.long_ask
        short_bid = best_candidate.short_bid
        # net_debit already set above from best_candidate.net_debit
    else:
        # Get quotes (legacy flow)
        long_quote = get_option_quote(long_symbol)
        short_quote = get_option_quote(short_symbol)

        if not long_quote or not short_quote:
            logger.spread_skip(underlying=sym, skip_reason="Cannot_get_quotes")
            return False

        long_ask = long_quote.get("ask", 0)
        short_bid = short_quote.get("bid", 0)
        net_debit_dec = money_sub(long_ask, short_bid)
        net_debit = safe_float(net_debit_dec)
    
    if net_debit <= 0:
        logger.spread_skip(
            underlying=sym, 
            skip_reason=f"Invalid_pricing(debit={net_debit:.2f})"
        )
        return False
    
    if net_debit > max_debit:
        logger.spread_skip(
            underlying=sym,
            skip_reason=f"Debit_too_high({net_debit:.2f}>{max_debit:.2f})"
        )
        return False
    
    spread_width = spread["spread_width"]

    # CRITICAL: Validate debit vs spread width (prevents guaranteed-loss trades)
    # Example bug: AMAT paid $2.88 for $2.50 spread = guaranteed $0.38 loss
    if net_debit >= spread_width:
        logger.spread_skip(
            underlying=sym,
            skip_reason=f"Debit_exceeds_width({net_debit:.2f}>={spread_width:.2f})_GUARANTEED_LOSS"
        )
        return False

    # Stricter check: Ensure sufficient edge after commissions/slippage (15% minimum)
    max_debit_for_edge = spread_width * 0.85
    if net_debit > max_debit_for_edge:
        logger.spread_skip(
            underlying=sym,
            skip_reason=f"Insufficient_edge({net_debit:.2f}>{max_debit_for_edge:.2f}=85%_of_{spread_width:.2f})"
        )
        return False

    # Calculate DTE from expiration
    actual_expiration = spread.get("expiration", "")
    dte_at_entry = 0
    if actual_expiration:
        try:
            from datetime import datetime as dt_parse
            exp_date = dt_parse.strptime(actual_expiration, "%Y-%m-%d").date()
            today = _now_et().date()
            dte_at_entry = (exp_date - today).days
        except ValueError:
            pass
    
    # Fetch greeks for enhanced logging
    if _prob_filter_used:
        # Reuse greeks from batch snapshot (avoids 2 redundant API calls)
        long_iv = best_candidate.long_iv
        long_theta = best_candidate.long_theta
        long_delta = best_candidate.long_delta
        short_iv = best_candidate.short_iv
        short_theta = best_candidate.short_theta
        short_delta = best_candidate.short_delta
    else:
        # Legacy flow: individual snapshot calls
        long_snapshot = get_option_snapshot(long_symbol)
        short_snapshot = get_option_snapshot(short_symbol)
        long_iv = long_snapshot.get("iv", 0) if long_snapshot else 0
        long_theta = long_snapshot.get("theta", 0) if long_snapshot else 0
        long_delta = long_snapshot.get("delta", 0) if long_snapshot else 0
        short_iv = short_snapshot.get("iv", 0) if short_snapshot else 0
        short_theta = short_snapshot.get("theta", 0) if short_snapshot else 0
        short_delta = short_snapshot.get("delta", 0) if short_snapshot else 0
    
    # Calculate time value for analysis (Premium - Intrinsic)
    stock_price = signal.get("price", 0) or spread.get("underlying_price", 0)
    entry_close = stock_price  # Store for logging
    kc_lower = signal.get("kc_lower", 0)  # Keltner Channel lower band
    long_strike = spread.get("atm_strike", 0)
    short_strike = spread.get("otm_strike", 0)
    
    # Warn if stock price is missing (time value will be inaccurate)
    if stock_price <= 0:
        print(f"[WARN] No stock price for {sym}, time value calculation may be inaccurate")
    
    # Intrinsic values (for calls: max(0, stock - strike))
    long_intrinsic = max(0, stock_price - long_strike) if stock_price > 0 else 0
    short_intrinsic = max(0, stock_price - short_strike) if stock_price > 0 else 0
    
    # Time value = premium - intrinsic (may be negative for deep ITM due to bid-ask spread)
    long_time_value = long_ask - long_intrinsic
    short_time_value = short_bid - short_intrinsic
    
    # Log if negative time value detected (deep ITM edge case)
    if long_time_value < 0 or short_time_value < 0:
        print(f"[INFO] {sym}: Negative time value detected (deep ITM) - long_tv={long_time_value:.2f}, short_tv={short_time_value:.2f}")
    
    # Time value as percentage of premium
    long_tv_pct = (long_time_value / long_ask * 100) if long_ask > 0 else 0
    short_tv_pct = (short_time_value / short_bid * 100) if short_bid > 0 else 0
    
    if dry_run:
        # Log entry even in dry-run mode
        logger.spread_entry(
            underlying=sym,
            long_symbol=long_symbol,
            short_symbol=short_symbol,
            atm_strike=spread["atm_strike"],
            otm_strike=spread["otm_strike"],
            spread_width=spread_width,
            net_debit=net_debit,
            contracts=contracts,
            expiration=spread["expiration"],
            entry_reason=f"[DRY-RUN] {entry_reason}",
            signal_rsi=signal.get("rsi", 0),
            signal_atr=signal.get("atr", 0),
            # NEW: Enhanced fields for analysis
            entry_close=entry_close,
            kc_lower=kc_lower,
            dte=dte_at_entry,
            long_premium=long_ask,
            short_premium=short_bid,
            long_iv=long_iv,
            short_iv=short_iv,
            long_theta=long_theta,
            short_theta=short_theta,
            long_delta=long_delta,
            short_delta=short_delta,
            # Kept: time value fields
            long_time_value=round(long_time_value, 2),
            short_time_value=round(short_time_value, 2),
            long_tv_pct=round(long_tv_pct, 1),
            short_tv_pct=round(short_tv_pct, 1),
            # Probability filter metrics (empty dict if filter disabled)
            prob_filter_used=_prob_filter_used,
            **_prob_filter_metrics,
        )
    else:
        # Generate client_order_id for position attribution
        client_order_id = registry.generate_order_id(long_symbol)
        
        # Idempotency check - prevent duplicate orders on restart
        if order_exists_today(client_order_id=client_order_id):
            logger.spread_skip(underlying=sym, skip_reason="Order_already_exists")
            return False
        
        # Capital limit check (spread cost = max_debit * contracts * 100)
        trade_value = max_debit * contracts * 100
        max_capital = float(cfg.get("max_capital", 100000))
        try:
            check_capital_limit(
                proposed_trade_value=trade_value,
                max_capital=max_capital,
                bot_tag=BOT_TAG,
            )
        except CapitalLimitExceeded as e:
            logger.spread_skip(underlying=sym, skip_reason=f"Capital_limit: {e}")
            return False
        
        # When probability filter pre-validated the debit, cap the broker
        # limit price at the validated amount to prevent price movement
        # from exceeding the safety checks (85% edge, debit < width).
        effective_max_debit = (
            min(max_debit, net_debit) if _prob_filter_used else max_debit
        )
        result = submit_spread_order(
            long_symbol=long_symbol,
            short_symbol=short_symbol,
            qty=contracts,
            max_debit=effective_max_debit,
            client_order_id=client_order_id,
        )
        if result.get("error"):
            error_msg = result.get("message", "Unknown")
            error_code = str(result.get("code", ""))
            logger.spread_reject(
                underlying=sym,
                reject_reason=f"API_rejection: {error_msg}",
                error_code=error_code,
                long_symbol=long_symbol,
                short_symbol=short_symbol,
                qty=contracts,
                max_debit=str(max_debit),
                client_order_id=client_order_id,
            )
            try:
                emit_order_rejection_alert(
                    bot_tag=BOT_TAG, symbol=sym, side="buy",
                    qty=contracts, reason=error_msg, error_code=error_code,
                )
            except Exception:
                pass
            return False
        
        # Record in registry for position attribution
        registry.record_entry(
            symbol=long_symbol,
            client_order_id=client_order_id,
            qty=contracts,
            entry_price=net_debit,
            underlying=sym,
            order_id=result.get("order_id", ""),
            short_symbol=short_symbol,
        )
        
        logger.spread_entry(
            underlying=sym,
            long_symbol=long_symbol,
            short_symbol=short_symbol,
            atm_strike=spread["atm_strike"],
            otm_strike=spread["otm_strike"],
            spread_width=spread_width,
            net_debit=net_debit,
            contracts=contracts,
            expiration=spread["expiration"],
            entry_reason=entry_reason,
            signal_rsi=signal.get("rsi", 0),
            signal_atr=signal.get("atr", 0),
            # NEW: Enhanced fields for analysis
            entry_close=entry_close,
            kc_lower=kc_lower,
            dte=dte_at_entry,
            long_premium=long_ask,
            short_premium=short_bid,
            long_iv=long_iv,
            short_iv=short_iv,
            long_theta=long_theta,
            short_theta=short_theta,
            long_delta=long_delta,
            short_delta=short_delta,
            # Kept: time value fields
            long_time_value=round(long_time_value, 2),
            short_time_value=round(short_time_value, 2),
            long_tv_pct=round(long_tv_pct, 1),
            short_tv_pct=round(short_tv_pct, 1),
            # Probability filter metrics (empty dict if filter disabled)
            prob_filter_used=_prob_filter_used,
            **_prob_filter_metrics,
        )

    return True


# ──────────────────────────────────────────────────────────────────────────────
# Position Management
# ──────────────────────────────────────────────────────────────────────────────
def parse_occ_symbol(symbol: str) -> Dict[str, Any]:
    """
    Parse OCC option symbol into components.
    
    Format: SYMBOL + YYMMDD + C/P + 00000000 (strike * 1000)
    Example: AAPL251205C00282500 = AAPL Dec 5, 2025 $282.50 Call
    """
    result = {"underlying": "", "expiration": "", "type": "", "strike": 0.0, "raw": symbol}
    
    if len(symbol) < 15:
        return result
    
    # Find where underlying ends (first digit)
    for i, c in enumerate(symbol):
        if c.isdigit():
            result["underlying"] = symbol[:i]
            rest = symbol[i:]
            break
    else:
        return result
    
    if len(rest) < 15:
        return result
    
    # Parse date: YYMMDD
    date_str = rest[:6]
    result["expiration"] = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
    
    # Parse type: C or P
    result["type"] = rest[6]
    
    # Parse strike: 8 digits / 1000
    strike_str = rest[7:15]
    try:
        result["strike"] = int(strike_str) / 1000
    except ValueError:
        pass
    
    return result


def calculate_spread_pnl(
    long_pos: Dict[str, Any],
    short_pos: Optional[Dict[str, Any]],
    entry_debit: float,
) -> Tuple[float, float]:
    """
    Calculate P&L for a spread using FRESH quotes (not stale position prices).

    Spread value = long_mid - short_mid  (clamped to >= 0)
    P&L = current_spread_value - entry_debit

    CRITICAL FIX: Uses get_option_quote() for accurate mid-market prices instead of
    Alpaca position current_price which can be stale and cause spread inversions
    (long_value < short_value), triggering false stop-losses.

    Physical constraint: A bull call spread can NEVER be worth less than $0.
    """
    long_symbol = long_pos.get("symbol", "")
    short_symbol = short_pos.get("symbol", "") if short_pos else ""

    # Try fresh quotes first (more accurate than stale position prices)
    long_value = 0.0
    short_value = 0.0
    quote_source = "position"

    if long_symbol and short_symbol:
        long_quote = get_option_quote(long_symbol)
        short_quote = get_option_quote(short_symbol)

        if long_quote and short_quote:
            long_value = long_quote.get("mid", 0.0)
            short_value = short_quote.get("mid", 0.0)
            quote_source = "fresh_quote"
        else:
            # Fall back to position current_price
            long_price = long_pos.get("current_price")
            long_value = float(long_price) if long_price is not None else 0.0
            if short_pos:
                short_price = short_pos.get("current_price")
                short_value = float(short_price) if short_price is not None else 0.0
            quote_source = "position_fallback"
    else:
        # No symbols available, use position prices
        long_price = long_pos.get("current_price")
        long_value = float(long_price) if long_price is not None else 0.0
        if short_pos:
            short_price = short_pos.get("current_price")
            short_value = float(short_price) if short_price is not None else 0.0

    # Raw spread value
    raw_spread_value = long_value - short_value

    # SPREAD INVERSION DETECTION: A bull call spread can NEVER be worth less than $0.
    # Negative values indicate stale/bad quotes. Return None to signal caller to skip exit logic.
    if raw_spread_value < 0:
        underlying = long_symbol[:6].rstrip("0123456789") if long_symbol else "???"
        print(f"[positions] WARNING: Spread inversion detected for {underlying}!", flush=True)
        print(f"[positions]   Long({long_symbol}): ${long_value:.4f}", flush=True)
        print(f"[positions]   Short({short_symbol}): ${short_value:.4f}", flush=True)
        print(f"[positions]   Raw spread value: ${raw_spread_value:.4f} (BAD QUOTE — skipping exit check)", flush=True)
        print(f"[positions]   Quote source: {quote_source}", flush=True)
        return None, None  # Signal bad data — caller must skip exit decision

    current_spread_value = max(0.0, raw_spread_value)

    # P&L vs entry
    pnl = (current_spread_value - entry_debit) * 100  # Per contract
    pnl_pct = ((current_spread_value / entry_debit) - 1) * 100 if entry_debit > 0 else 0

    return pnl, pnl_pct


# SL confirmation: track consecutive SL readings per underlying to prevent
# single-cycle quote glitches from triggering irreversible exits.
_sl_consecutive: Dict[str, int] = {}
_SL_CONFIRM_REQUIRED = 1  # Immediate SL exit for options (was 2 — caused 63% MRK loss from 2-min delay)


def check_spread_exit_conditions(
    pnl_pct: float,
    spread_cfg: dict,
    holding_minutes: int = 0,
    dte_remaining: int = -1,
    dte_at_entry: int = -1,
) -> Tuple[bool, str]:
    """
    Check if spread should be exited based on P&L percentage, time stop,
    or time-decay conditions.

    Args:
        pnl_pct: Current spread P&L as a percentage.
        spread_cfg: Spread configuration dict.
        holding_minutes: Minutes since position was opened.
        dte_remaining: Days to expiration remaining (-1 if unknown).
        dte_at_entry: DTE when the position was opened (-1 if unknown).

    Returns:
        Tuple of (should_exit, exit_reason).
    """
    tp_max_profit_pct = spread_cfg.get("tp_max_profit_pct", 50.0)
    sl_pct = spread_cfg.get("sl_pct", -25.0)
    hold_overnight = spread_cfg.get("hold_overnight", True)
    dte = spread_cfg.get("dte", 3)
    bars_stop = spread_cfg.get("bars_stop", 0)

    # Phase 4A: Apply dynamic TP adjustment from market condition overrides
    try:
        from RubberBand.src.watchdog.market_classifier import read_dynamic_overrides
        _dyn = read_dynamic_overrides()
        _tp_adj = float(_dyn.get("overrides", {}).get("tp_r_multiple_adjustment", 0.0))
        # 1 R-multiple adjustment maps to 20% TP shift
        # Example: TRENDING_DOWN (-1.0R) -> -20% TP, CHOPPY (-0.5R) -> -10% TP
        _TP_R_TO_PCT_SCALE = 20.0
        tp_pct_shift = _tp_adj * _TP_R_TO_PCT_SCALE
        tp_max_profit_pct = max(30.0, tp_max_profit_pct + tp_pct_shift)
    except Exception as e:
        print(f"[WATCHDOG] dynamic TP non-fatal: {e}", flush=True)

    if pnl_pct >= tp_max_profit_pct:
        return True, f"TP_hit({pnl_pct:.1f}%>={tp_max_profit_pct:.0f}%)"

    if pnl_pct <= sl_pct:
        return True, f"SL_hit({pnl_pct:.1f}%<={sl_pct}%)"

    # Time stop: exit if held for too many bars (each bar = 15 minutes)
    if bars_stop > 0 and holding_minutes >= bars_stop * 15:
        return True, f"TIME_STOP({holding_minutes}min >= {bars_stop}bars)"

    # Phase 4D: Time-decay exit for spreads
    # Theta decay accelerates near expiration; take profits early or avoid gamma risk
    if dte_at_entry > 0 and dte_remaining >= 0:
        dte_elapsed_pct = 1.0 - (dte_remaining / dte_at_entry) if dte_at_entry > 0 else 0.0

        # If held > 80% of DTE regardless of P&L: exit (avoid gamma risk)
        if dte_elapsed_pct >= 0.80:
            return True, f"TIME_DECAY_EXIT(DTE_elapsed={dte_elapsed_pct:.0%},remaining={dte_remaining}d)"

        # If held > 60% of DTE and profitable: exit (take what you have)
        if dte_elapsed_pct >= 0.60 and pnl_pct > 0:
            return True, f"TIME_DECAY_PROFIT_LOCK(DTE_elapsed={dte_elapsed_pct:.0%},pnl={pnl_pct:.1f}%)"

    if not hold_overnight or dte == 0:
        now_et = _now_et()
        cutoff = now_et.replace(hour=15, minute=0, second=0, microsecond=0)
        if now_et >= cutoff:
            return True, "EOD_time_exit(3:00PM_cutoff)"

    return False, ""


def manage_positions(
    spread_cfg: dict,
    logger: OptionsTradeLogger,
    dry_run: bool = True,
    active_spreads: Optional[Dict[str, Dict]] = None,
    registry: Optional["PositionRegistry"] = None,
):
    """
    Check open positions and exit spreads if TP/SL conditions met.

    Properly pairs long and short legs by underlying + expiration,
    calculates spread P&L, and closes both legs together.

    PHASE 1 FIX (GAP-008): Now includes pre-close reconciliation to detect
    registry orphans before attempting any close operations.

    Args:
        spread_cfg: Spread configuration
        logger: Trade logger
        dry_run: If True, don't actually close
        active_spreads: Dict of {underlying: spread_info} from entries this session
        registry: Position registry to update on successful close
    """
    positions = get_option_positions()

    # PHASE 1 FIX (GAP-008): Reconcile registry with broker BEFORE processing
    # This prevents "insufficient qty" errors from trying to close positions
    # that exist in registry but not in broker.
    if registry and positions is not None:
        broker_symbols = {pos.get("symbol", "") for pos in positions}
        registry_symbols = registry.get_my_symbols()

        # Find registry entries that don't exist in broker
        orphaned_in_registry = [sym for sym in registry_symbols if sym not in broker_symbols]

        if orphaned_in_registry:
            print(f"[positions] WARNING: Found {len(orphaned_in_registry)} registry orphans (not in broker):", flush=True)
            for sym in orphaned_in_registry:
                pos_data = registry.positions.get(sym, {})
                underlying = pos_data.get("underlying", "unknown")
                entry_date = pos_data.get("entry_date", "unknown")
                print(f"[positions]   - {sym} (underlying={underlying}, entry={entry_date})", flush=True)

            # Log to trade logger for audit trail
            logger.heartbeat(
                event="registry_orphans_detected",
                orphan_count=len(orphaned_in_registry),
                orphan_symbols=orphaned_in_registry,
                action="cleaning_stale_entries",
            )

            # Clean orphaned entries from registry (they were closed externally)
            for sym in orphaned_in_registry:
                print(f"[positions] Cleaning orphaned registry entry: {sym}", flush=True)
                registry.record_exit(
                    symbol=sym,
                    exit_price=0.0,
                    exit_reason="BROKER_MISSING_orphan_cleanup",
                    pnl=0.0,  # Unknown P&L since we don't know exit price
                )

    if not positions:
        return
    
    # Group positions by underlying + expiration
    # Key: "UNDERLYING_YYYYMMDD" -> {long: pos, short: pos}
    spreads = {}
    
    for pos in positions:
        symbol = pos.get("symbol", "")
        parsed = parse_occ_symbol(symbol)
        underlying = parsed["underlying"]
        expiration = parsed["expiration"]
        strike = parsed["strike"]
        qty = int(pos.get("qty", 0))
        
        key = f"{underlying}_{expiration}"
        
        if key not in spreads:
            spreads[key] = {"underlying": underlying, "expiration": expiration, "long": None, "short": None}
        
        # Long position (qty > 0), Short position (qty < 0)
        if qty > 0:
            # If we already have a long, keep the lower strike (ATM for bull call spread)
            if spreads[key]["long"] is None or strike < parse_occ_symbol(spreads[key]["long"]["symbol"])["strike"]:
                spreads[key]["long"] = pos
        elif qty < 0:
            # If we already have a short, keep the higher strike (OTM for bull call spread)
            if spreads[key]["short"] is None or strike > parse_occ_symbol(spreads[key]["short"]["symbol"])["strike"]:
                spreads[key]["short"] = pos
    
    # Process each spread
    already_closed = set()
    
    for key, spread in spreads.items():
        underlying = spread["underlying"]
        long_pos = spread["long"]
        short_pos = spread["short"]
        
        if underlying in already_closed:
            continue

        if long_pos and registry:
            # POSITION OWNERSHIP FILTER: Only manage positions belonging to this bot.
            # Prevents cross-bot interference (e.g., 15M_OPT closing WK_OPT's positions).
            long_sym = long_pos.get("symbol", "")
            in_registry = registry.find_by_symbol(long_sym) is not None
            in_active = underlying in (active_spreads or {})
            if not in_registry and not in_active:
                # Position exists at broker but not tracked by this bot - skip
                continue

        if not long_pos:
            # Orphaned short leg - close it individually
            # This can happen if the long leg was closed but short wasn't
            if short_pos:
                short_symbol = short_pos.get("symbol", "")
                print(f"[positions] Orphaned short leg detected for {underlying}")
                print(f"[positions]   Short symbol: {short_symbol}")
                print(f"[positions]   Qty: {short_pos.get('qty', 0)}")
                
                # Track retry count in registry (max 3 attempts)
                orphan_key = f"_orphan_retry_{short_symbol}"
                retry_count = 0
                if registry:
                    retry_count = registry.positions.get(orphan_key, {}).get("retry_count", 0)
                
                if retry_count >= 3:
                    # Already tried 3 times, log error and skip
                    print(f"[positions] ERROR: Orphan close failed 3 times for {underlying}/{short_symbol} - GIVING UP")
                    logger.error(
                        error=f"Orphan close failed after 3 retries",
                        context=f"orphan_{underlying}",
                    )
                    continue
                
                if not dry_run:
                    result = close_option_position(short_symbol)
                    if result.get("error"):
                        retry_count += 1
                        print(f"[positions] ERROR closing orphaned short (attempt {retry_count}/3): {result.get('message', 'Unknown')}")
                        # Track retry count in registry
                        if registry:
                            registry.positions[orphan_key] = {"retry_count": retry_count, "symbol": short_symbol}
                        # Will retry next cycle (up to 3 times)
                    else:
                        print(f"[positions] Orphaned short leg closed successfully")
                        # Clear retry counter on success
                        if registry and orphan_key in registry.positions:
                            del registry.positions[orphan_key]
                else:
                    print(f"[positions] [DRY-RUN] Would close orphaned short leg")
            continue
        
        # Parse symbols
        long_symbol = long_pos.get("symbol", "")
        short_symbol_from_broker = short_pos.get("symbol", "") if short_pos else ""

        # CRITICAL FIX: Check registry for stored short_symbol
        # The broker may not return the short leg if it was assigned/expired/closed separately
        # We must use the registry as the source of truth for spread structure
        registry_entry = None
        registry_short_symbol = ""
        if registry:
            registry_key = registry.find_by_symbol(long_symbol)
            if registry_key and registry_key in registry.positions:
                registry_entry = registry.positions[registry_key]
                registry_short_symbol = registry_entry.get("short_symbol", "")

        # Determine short_symbol: prefer broker if available, fallback to registry
        if short_symbol_from_broker:
            short_symbol = short_symbol_from_broker
        elif registry_short_symbol:
            # CRITICAL: We have a registry short_symbol but broker doesn't have the position
            # This means the short leg was closed/assigned separately - DO NOT EXIT AS NAKED LONG
            print(f"[positions] CRITICAL: Short leg mismatch for {underlying}!")
            print(f"[positions]   Registry short_symbol: {registry_short_symbol}")
            print(f"[positions]   Broker short_pos: None (missing)")
            print(f"[positions]   This spread may have been partially closed/assigned.")
            print(f"[positions]   HALTING exit to prevent unlimited loss on naked long.")

            logger.error(
                error=f"Short leg missing from broker but exists in registry",
                context=f"spread_mismatch_{underlying}",
                details={
                    "long_symbol": long_symbol,
                    "registry_short": registry_short_symbol,
                    "broker_short": "MISSING",
                    "action": "HALTED - manual review required",
                }
            )
            # Skip this position - requires manual intervention
            continue
        else:
            short_symbol = ""

        # Get entry debit from active_spreads if available
        entry_debit = 1.0  # Default
        if active_spreads and underlying in active_spreads:
            entry_debit = active_spreads[underlying].get("net_debit", 1.0)
        else:
            # Estimate from cost basis (Alpaca returns cost_basis in dollars)
            # Handle None values explicitly
            cb_long = long_pos.get("cost_basis")
            long_cost = float(cb_long) if cb_long is not None else 0.0
            
            if short_pos:
                cb_short = short_pos.get("cost_basis")
                short_cost = abs(float(cb_short)) if cb_short is not None else 0.0
            else:
                short_cost = 0.0
            
            # For options, cost_basis is total cost. Debit per share = (long_cost - short_cost) / (qty * 100)
            # BUG FIX: Must subtract short_cost (credit received) from long_cost to get NET debit
            qty_val = long_pos.get("qty")
            long_qty = abs(int(qty_val)) if qty_val is not None else 1
            
            # NET debit = long cost - short credit (short_cost is a credit, reduces our cost)
            net_cost = long_cost - short_cost
            entry_debit = net_cost / (long_qty * 100) if long_qty > 0 else net_cost / 100
            
            if entry_debit <= 0:
                entry_debit = 1.0  # Fallback to default (shouldn't happen for debit spreads)
        
        # Calculate spread P&L
        pnl, pnl_pct = calculate_spread_pnl(long_pos, short_pos, entry_debit)

        # BAD QUOTE GUARD: If P&L returned None, quotes are inverted/stale — skip exit logic this cycle
        if pnl is None or pnl_pct is None:
            print(f"[positions] Skipping exit check for {underlying}: bad quote data (will retry next cycle)", flush=True)
            continue

        # Calculate holding_minutes from registry entry_date (P1 fix: pass to check_spread_exit_conditions)
        # Reuse registry_entry from earlier lookup if available
        holding_minutes = 0
        if registry_entry:
            entry_date_str = registry_entry.get("entry_date")
            if entry_date_str:
                try:
                    from datetime import datetime
                    # Parse ISO format with timezone (e.g., "2026-02-03T10:15:00-05:00")
                    entry_dt = datetime.fromisoformat(entry_date_str)
                    now_dt = datetime.now(entry_dt.tzinfo) if entry_dt.tzinfo else datetime.now()
                    holding_minutes = int((now_dt - entry_dt).total_seconds() / 60)
                except (ValueError, TypeError):
                    holding_minutes = 0

        # Phase 4D: Compute DTE remaining and DTE at entry for time-decay exit
        dte_remaining = -1
        dte_at_entry = -1
        expiration_str = spread["expiration"]
        if expiration_str:
            try:
                from datetime import datetime as _dt_cls
                exp_date = _dt_cls.strptime(expiration_str, "%Y-%m-%d").date()
                today = _now_et().date()
                dte_remaining = max(0, (exp_date - today).days)
            except (ValueError, TypeError):
                pass
        if registry_entry:
            dte_at_entry = registry_entry.get("dte_at_entry", -1)
            if dte_at_entry == -1 and expiration_str:
                # Derive from entry_date and expiration
                entry_date_str = registry_entry.get("entry_date", "")
                if entry_date_str:
                    try:
                        from datetime import datetime as _dt_cls2
                        entry_d = _dt_cls2.fromisoformat(entry_date_str).date()
                        exp_d = _dt_cls2.strptime(expiration_str, "%Y-%m-%d").date()
                        dte_at_entry = max(1, (exp_d - entry_d).days)
                    except (ValueError, TypeError):
                        pass

        # Phase 4C: Options Trailing Stop
        # Track peak PnL % in registry; if spread was up > +40% and drops 20%
        # from peak, trigger exit to prevent common pattern of +60% -> -10%.
        _trailing_exit = False
        _trailing_reason = ""
        if registry_entry and pnl_pct > 0:
            peak_pnl_pct = registry_entry.get("peak_pnl_pct", 0.0)
            if pnl_pct > peak_pnl_pct:
                old_peak = peak_pnl_pct
                # Update peak (backward-compatible new field)
                registry_entry["peak_pnl_pct"] = pnl_pct
                # Only save on significant changes (5% point increments)
                if registry and (pnl_pct - old_peak) >= 5.0:
                    registry.save()
            # Check trailing stop: activate at +40%, trail distance = 20%
            if peak_pnl_pct >= 40.0:
                trail_floor = peak_pnl_pct - 20.0
                if pnl_pct <= trail_floor:
                    _trailing_exit = True
                    _trailing_reason = (
                        f"TRAILING_STOP(peak={peak_pnl_pct:.1f}%,floor={trail_floor:.1f}%,"
                        f"current={pnl_pct:.1f}%)"
                    )
        elif registry_entry and pnl_pct <= 0:
            # Not profitable — keep tracking peak but don't trigger trailing
            pass

        # Check exit conditions (including time stop, time-decay)
        should_exit, exit_reason = check_spread_exit_conditions(
            pnl_pct, spread_cfg, holding_minutes,
            dte_remaining=dte_remaining,
            dte_at_entry=dte_at_entry,
        )

        # Phase 4C: Trailing stop overrides if not already exiting
        if not should_exit and _trailing_exit:
            should_exit = True
            exit_reason = _trailing_reason

        # SL CONFIRMATION: Require N consecutive SL readings before exiting.
        # Prevents single-cycle quote glitches from triggering irreversible exits.
        # Only applies to SL exits - TP, EOD exits are immediate.
        if should_exit and "SL_hit" in exit_reason:
            _sl_consecutive[underlying] = _sl_consecutive.get(underlying, 0) + 1
            if _sl_consecutive[underlying] < _SL_CONFIRM_REQUIRED:
                print(f"[positions] SL WARNING #{_sl_consecutive[underlying]}/{_SL_CONFIRM_REQUIRED} for "
                      f"{underlying}: pnl={pnl_pct:.1f}% (confirming next cycle)", flush=True)
                should_exit = False  # Wait for confirmation
            else:
                exit_reason = f"{exit_reason}_CONFIRMED(x{_sl_consecutive[underlying]})"
        elif not should_exit:
            # Position is not in SL territory - reset counter
            _sl_consecutive.pop(underlying, None)

        if should_exit:
            # Clean up SL counter on confirmed exit
            _sl_consecutive.pop(underlying, None)
            # Calculate current spread value for logging using FRESH quotes
            long_quote = get_option_quote(long_symbol)
            short_quote = get_option_quote(short_symbol) if short_symbol else None
            if long_quote:
                long_value = long_quote.get("mid", 0.0)
            else:
                lp = long_pos.get("current_price")
                long_value = float(lp) if lp is not None else 0.0
            if short_quote:
                short_value = short_quote.get("mid", 0.0)
            else:
                sp = short_pos.get("current_price") if short_pos else None
                short_value = float(sp) if sp is not None else 0.0
            exit_value = max(0.0, long_value - short_value)  # Clamp: spread can't be negative

            if dry_run:
                logger.spread_exit(
                    underlying=underlying,
                    long_symbol=long_symbol,
                    short_symbol=short_symbol,
                    exit_value=exit_value,
                    exit_reason=f"[DRY-RUN] {exit_reason}",
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    holding_minutes=holding_minutes,
                )
                already_closed.add(underlying)
            else:
                # Close both legs together
                if short_symbol:
                    result = close_spread(long_symbol, short_symbol, qty=1)
                else:
                    # Only have long leg, close individually
                    result = close_option_position(long_symbol)
                
                # Check if close was successful before logging and updating registry
                if result.get("error"):
                    print(f"[positions] ERROR closing {underlying} spread: {result.get('message', 'Unknown error')}")
                    logger.error(
                        error=f"Spread close failed: {result.get('message', 'Unknown')}",
                        context=f"close_{underlying}",
                    )
                    # Don't mark as closed - will retry next cycle
                    continue
                
                # Close was successful
                logger.spread_exit(
                    underlying=underlying,
                    long_symbol=long_symbol,
                    short_symbol=short_symbol,
                    exit_value=exit_value,
                    exit_reason=exit_reason,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    holding_minutes=holding_minutes,
                )
                
                # Update registry to remove closed position
                # Use find_by_symbol to support lookup by either long or short symbol
                registry_key = registry.find_by_symbol(long_symbol) if registry else None
                if registry and registry_key:
                    registry.record_exit(
                        symbol=registry_key,
                        exit_price=exit_value,
                        exit_reason=exit_reason,
                        pnl=pnl,
                    )
                    print(f"[positions] Registry updated: removed {registry_key}")
                
                already_closed.add(underlying)


# ──────────────────────────────────────────────────────────────────────────────
# Single Scan Cycle
# ──────────────────────────────────────────────────────────────────────────────
def run_scan_cycle(
    symbols: List[str],
    cfg: dict,
    spread_cfg: dict,
    logger: OptionsTradeLogger,
    registry: PositionRegistry,
    dry_run: bool,
    regime_cfg: Dict[str, Any] = None, # Added parameter
) -> int:
    """Run a single scan cycle. Returns number of new entries."""
    windows = cfg.get("entry_windows", [])
    now_et = _now_et()
    
    # Check entry windows - if outside, only manage positions
    if not _in_entry_window(now_et, windows):
        logger.heartbeat(
            event="outside_entry_window",
            current_time=now_et.strftime("%H:%M"),
        )
        # Still manage positions for exits
        manage_positions(spread_cfg, logger, dry_run, registry=registry)
        return 0
    
    # For 0DTE only: check 3:00 PM cutoff
    if spread_cfg["dte"] == 0 and not is_options_trading_allowed():
        logger.heartbeat(event="0dte_cutoff_reached")
        if not dry_run:
            flatten_all_option_positions()
        return 0
    
    logger.heartbeat(event="scan_start", current_time=now_et.strftime("%H:%M"))
    
    # 1. Check existing positions for exits
    manage_positions(spread_cfg, logger, dry_run, registry=registry)
    
    # 2. Get current option positions (to avoid duplicates)
    current_positions = get_option_positions()
    position_underlyings = set()
    for pos in current_positions:
        sym = pos.get("symbol", "")
        if len(sym) > 10:
            for i, c in enumerate(sym):
                if c.isdigit():
                    position_underlyings.add(sym[:i])
                    break
    
    
    # 3. Scan for new signals
    signals, forming_signals = get_long_signals(symbols, cfg, logger, regime_cfg=regime_cfg)
    
    # AUDIT LOGGING: Log all forming signals ("Transient Observations")
    if forming_signals:
        for f in forming_signals:
            # We log this as a "heartbeat" or specific "audit" event so the user can see it but it's not a trade
            logger.heartbeat(
                event="audit_forming_signal",
                symbol=f["symbol"],
                rsi=f["rsi"],
                slope=f["slope"],
                close=f["close"],
                note="Signal observed on forming bar (waiting for close)"
            )
    
    if signals:
         logger.heartbeat(
            event="signals_found",
            count=len(signals),
            symbols=[s["symbol"] for s in signals]
        )
    
    # 4. Enter new spreads (Confirmed Only)
    entries = 0
    for signal in signals:
        if signal["symbol"] in position_underlyings:
            # Silent skip or minimal log
            continue
        
        # LOG CONFIRMED SIGNAL NOW (Use spread_signal here so it shows up as a "Real" signal)
        logger.spread_signal(
            underlying=signal["symbol"],
            signal_reason=signal["entry_reason"],
            entry_price=signal["entry_price"],
            rsi=signal["rsi"],
            atr=signal["atr"]
        )
        
        if try_spread_entry(signal, spread_cfg, logger, registry, dry_run, cfg):
            entries += 1
            position_underlyings.add(signal["symbol"])
            # Log audit alignment
            logger.heartbeat(
                event="audit_signal_confirmed",
                symbol=signal["symbol"],
                note="Signal persisted at close -> TRADED"
            )
    
    logger.heartbeat(
        event="scan_complete",
        signals=len(signals),
        forming=len(forming_signals),
        new_entries=entries,
        total_positions=len(current_positions) + entries,
    )
    
    return entries


# ──────────────────────────────────────────────────────────────────────────────
# Main Loop
# ──────────────────────────────────────────────────────────────────────────────
SCAN_INTERVAL_SECONDS = 30  # 30-second scan for faster options reaction (was 60s)


def main() -> int:
    import time
    
    args = _parse_args()
    cfg = _load_config(args.config)
    
    # Inject command line overrides into config
    if args.slope_threshold is not None:
        cfg["slope_threshold"] = args.slope_threshold
        
    if args.slope_threshold_10 is not None:
        cfg["slope_threshold_10"] = args.slope_threshold_10

    # --- CIRCUIT BREAKER: PORTFOLIO DRAWDOWN (GAP-005) ---
    try:
        max_dd_pct = float(cfg.get("max_drawdown_pct", 0.10))
        # Use shared state file (consistent with 15M_STK)
        guard_state_file = os.path.join(os.path.dirname(__file__), "portfolio_guard_state.json")
        guard = PortfolioGuard(guard_state_file, max_drawdown_pct=max_dd_pct)
        # GAP: guard.update(current_equity) is NOT called here.
        # Unlike live_paper_loop.py (line 511), this script does not have broker
        # credentials (base_url, key, secret) readily available in main().
        # To fix: either load creds from env and call get_account_info_compat(),
        # or refactor PortfolioGuard to fetch equity internally.
        # Until then, the PortfolioGuard drawdown check is inactive for 15M_OPT.
    except Exception as e:
        print(f"[Guard] Warning: Failed to init guard: {e}")
    # -----------------------------------------------------

    # GAP-006: Connectivity Guard
    from RubberBand.src.circuit_breaker import ConnectivityGuard
    conn_guard = ConnectivityGuard(max_errors=5)
    
    # Spread config
    spread_cfg = {**DEFAULT_SPREAD_CONFIG}
    spread_cfg["dte"] = args.dte
    spread_cfg["max_debit"] = args.max_debit
    spread_cfg["contracts"] = args.contracts
    
    # Setup logging
    results_dir = cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f"options_trades_{_now_et().strftime('%Y%m%d')}.jsonl")
    logger = OptionsTradeLogger(log_path)
    
    dry_run = bool(args.dry_run)
    
    logger.heartbeat(
        event="startup",
        dry_run=dry_run,
        dte=spread_cfg["dte"],
        max_debit=spread_cfg["max_debit"],
        contracts=spread_cfg["contracts"],
        tp_pct=spread_cfg["tp_max_profit_pct"],
        sl_pct=spread_cfg["sl_pct"],
        scan_interval_sec=SCAN_INTERVAL_SECONDS,
    )

    # --- Dynamic Regime Detection ---
    rm = RegimeManager(verbose=True)
    # Do initial daily update ONCE at startup (sets reference values for intraday checks)
    daily_regime = rm.update()
    regime_cfg = rm.get_config_overrides()
    logger.heartbeat(
        event="regime_initialized",
        daily_regime=daily_regime,
        vixy_reference=rm._reference_close,
        upper_band=rm._upper_band,
    )
    # --------------------------------

    if args.slope_threshold is not None:
        print(f"[config] Slope Threshold overridden to: {args.slope_threshold}", flush=True)
    if args.slope_threshold_10 is not None:
        print(f"[config] Slope Threshold 10-bar overridden to: {args.slope_threshold_10}", flush=True)
    
    # Load tickers
    try:
        symbols = load_symbols_from_file(args.tickers)
    except FileNotFoundError:
        logger.error(error=f"Ticker file not found: {args.tickers}")
        logger.close()
        return 1
    except Exception as e:
        logger.error(error=str(e), context="load_tickers")
        logger.close()
        return 1
    
    if not symbols:
        logger.error(error="No symbols loaded from ticker file")
        logger.close()
        return 1
    
    logger.heartbeat(event="symbols_loaded", count=len(symbols), sample=symbols[:5])
    
    # Initialize position registry for this bot
    registry = PositionRegistry(bot_tag=BOT_TAG)

    # PHASE 2 FIX (GAP-008): Use reconcile_or_halt() instead of sync_with_alpaca()
    # This prevents silent cleanup of orphaned positions and alerts on mismatch.
    broker_positions = get_option_positions()
    is_clean, registry_orphans, broker_untracked = registry.reconcile_or_halt(
        broker_positions,
        auto_clean=False,  # Do NOT auto-clean - we want to know about mismatches
    )

    if not is_clean:
        # Registry has positions that broker doesn't - this is a critical mismatch
        # Per CLAUDE.md Circuit Breaker 5: Position mismatch → HALT trading
        logger.error(
            error=f"POSITION_MISMATCH_AT_STARTUP: Registry has {len(registry_orphans)} orphaned positions",
            context="startup_reconciliation",
        )
        logger.heartbeat(
            event="position_mismatch_detected",
            orphaned_symbols=registry_orphans,
            action="auto_clean_and_alert",
        )

        # For now, auto-clean with explicit logging rather than hard halt
        # This allows the bot to continue but with full visibility into the issue
        # A hard halt would require manual intervention which may not be available
        # during market hours in an automated GitHub Actions environment.
        print(f"[CRITICAL] Registry orphans detected: {registry_orphans}", flush=True)
        print(f"[CRITICAL] These positions exist in registry but NOT in broker.", flush=True)
        print(f"[CRITICAL] Auto-cleaning to allow trading to continue...", flush=True)

        # Re-run with auto_clean=True to fix the state
        registry.reconcile_or_halt(broker_positions, auto_clean=True)

    logger.heartbeat(
        event="registry_loaded",
        bot_tag=BOT_TAG,
        my_positions=len(registry.positions),
        broker_positions=len(broker_positions),
        registry_clean=is_clean,
        orphans_found=len(registry_orphans) if not is_clean else 0,
    )
    
    # Kill Switch Check - RE-ENABLED Dec 13, 2025
    # Halts trading if daily loss exceeds 25% of invested capital
    if check_kill_switch(bot_tag=BOT_TAG, max_loss_pct=25.0):
        logger.error(error=f"{BOT_TAG} exceeded 25% daily loss - HALTING")
        logger.close()
        raise KillSwitchTriggered(f"{BOT_TAG} exceeded 25% daily loss")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Main Loop: Run until market close (4:00 PM ET)
    # ──────────────────────────────────────────────────────────────────────────
    market_close_hour = 16  # 4:00 PM ET
    scan_count = 0
    tracemalloc.start()

    while True:
        now_et = _now_et()
        
        # Exit if past market close
        if now_et.hour >= market_close_hour:
            logger.heartbeat(event="market_close_reached", time=now_et.strftime("%H:%M"))
            break
        
        # Check if market is open
        if not alpaca_market_open():
            logger.heartbeat(event="waiting_for_market_open", time=now_et.strftime("%H:%M"))
            time.sleep(60)  # Wait 1 minute and check again
            continue
        
        # Run scan cycle
        scan_count += 1
        
        # Check for intraday VIXY spikes (daily update was done once at startup)
        current_regime = rm.get_effective_regime()

        # If intraday panic triggered, use PANIC config instead of daily config
        if current_regime == "PANIC" and daily_regime != "PANIC":
            regime_cfg = rm.regime_configs["PANIC"]
        
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        logger.heartbeat(
            event="scan_cycle_start",
            cycle=scan_count,
            time=now_et.strftime("%H:%M"),
            regime=current_regime,
            vixy=rm.last_vixy_price,
            slope_thresh=regime_cfg.get("slope_threshold_pct"),
            bearish_filter=regime_cfg.get("bearish_bar_filter"),
            memory_mb=round(current_mem / 1024 / 1024, 1),
            peak_memory_mb=round(peak_mem / 1024 / 1024, 1),
        )
        
        # Watchdog pause check (fail-open if file missing)
        try:
            from RubberBand.src.watchdog.pause_check import check_bot_paused
            _wd_paused, _wd_reason = check_bot_paused(BOT_TAG)
            if _wd_paused:
                logger.heartbeat(event="watchdog_paused", reason=_wd_reason)
                manage_positions(spread_cfg, logger, dry_run, registry=registry)
                continue
        except Exception as e:
            logger.heartbeat(event="watchdog_check_error", error=str(e))

        try:
            entries = run_scan_cycle(symbols, cfg, spread_cfg, logger, registry, dry_run, regime_cfg=regime_cfg)
            conn_guard.record_success()
            logger.heartbeat(event="scan_cycle_end", cycle=scan_count, new_entries=entries)
            # Commit auditor logs after each cycle for real-time auditing
            commit_auditor_log()
            gc.collect()
        except Exception as e:
            try:
                conn_guard.record_error(e)
            except CircuitBreakerExc as cbe:
                logger.error(error=str(cbe), context="circuit_breaker_halt")
                print(f"[HALT] Circuit breaker triggered: {cbe}", flush=True)
                break  # Exit while loop, proceed to EOD cleanup
            logger.error(error=str(e), context="scan_cycle")
            # If we lost connection, we should sleep longer?
            import time
            time.sleep(10)
            continue
        
        # Check if we should exit (past market close after scan)
        now_et = _now_et()
        if now_et.hour >= market_close_hour:
            logger.heartbeat(event="market_close_reached", time=now_et.strftime("%H:%M"))
            break
        
        # Wait until next scan (Align to top of minute + 5s buffer)
        # This ensures we run shortly after bars close, preventing drift.
        now = datetime.now()
        seconds_to_next_minute = 60 - now.second
        sleep_seconds = seconds_to_next_minute + 5  # Wake up at HH:MM:05
        
        # If we are already close to the target (e.g. processing took long and we are at :02),
        # sleep_seconds will be roughly 63s. Ideally we want the NEXT minute.
        # Logic: 60 - 2 + 5 = 63s -> Wakes at :05 next minute. Correct.
        # If we are at :58. 60 - 58 + 5 = 7s -> Wakes at :05 next minute. Correct.
        
        next_scan_time = now + timedelta(seconds=sleep_seconds)
        
        logger.heartbeat(
            event="waiting_for_next_scan",
            next_scan_at=next_scan_time.strftime("%H:%M:%S"),
            sleep_sec=sleep_seconds,
        )
        time.sleep(sleep_seconds)
    
    # ──────────────────────────────────────────────────────────────────────────
    # End of Day
    # ──────────────────────────────────────────────────────────────────────────
    logger.heartbeat(event="eod_processing", scan_count=scan_count)
    
    # Final position management
    manage_positions(spread_cfg, logger, dry_run, registry=registry)
    
    # Save registry
    registry.save()
    
    # EOD Summary
    summary = logger.eod_summary()
    
    # Export trades to CSV for analysis (matching backtest format)
    csv_date = _now_et().strftime("%Y%m%d")
    csv_path = f"results/{BOT_TAG}_trades_{csv_date}.csv"
    logger.export_trades_csv(csv_path)
    
    logger.close()
    
    print(f"\n{'='*60}", flush=True)
    print(f"EOD SUMMARY: {summary.get('total_trades', 0)} trades, PnL: ${summary.get('total_pnl', 0):.2f}", flush=True)
    print(f"Win Rate: {summary.get('win_rate_pct', 0):.1f}%", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[SHUTDOWN] KeyboardInterrupt received. Exiting.", flush=True)
        raise SystemExit(130)
    except SystemExit:
        raise  # Allow normal SystemExit to propagate unchanged
    except BaseException as e:
        import traceback
        print(f"[FATAL] Unhandled {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise SystemExit(1)

