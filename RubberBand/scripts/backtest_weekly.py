#!/usr/bin/env python3
"""
Weekly Mean Reversion Backtest
==============================
Standalone weekly timeframe strategy - completely separate from 15-min RubberBandBot.

Usage:
    python RubberBand/scripts/backtest_weekly.py --days 365 --symbols AAPL,NVDA,AMZN
    python RubberBand/scripts/backtest_weekly.py --days 180,365 --tickers RubberBand/tickers.txt
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

# --- Ensure repo root on sys.path ---
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.data import fetch_latest_bars

# =============================================================================
# Configuration Loading
# =============================================================================

def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def read_tickers(path: str) -> List[str]:
    """Read tickers from file, one per line."""
    with open(path, "r") as f:
        return [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]

# =============================================================================
# Indicator Calculations (Self-Contained)
# =============================================================================

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Calculate Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def compute_keltner_channels(df: pd.DataFrame, ema_period: int = 10, atr_period: int = 10, atr_mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Keltner Channels."""
    ema = df["close"].ewm(span=ema_period, adjust=False).mean()
    atr = compute_atr(df, atr_period)
    
    upper = ema + (atr * atr_mult)
    lower = ema - (atr * atr_mult)
    
    return lower, ema, upper

def attach_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Attach all indicators to dataframe.
    
    Pure Weekly RubberBand uses 3 core factors:
    1. RSI Oversold (< 45)
    2. Mean Deviation (> 10% below 20-week SMA) - the "rubber band stretch"
    3. Trend Filter (Price > 20-week SMA > 50-week SMA) - only buy in uptrends
    """
    df = df.copy()
    
    # Get config
    ind_cfg = cfg.get("indicators", {})
    kc_cfg = ind_cfg.get("keltner", {})
    rsi_cfg = ind_cfg.get("rsi", {})
    filter_cfg = cfg.get("filters", {})
    
    # Keltner Channels (for exit signal - reversion to mean)
    ema_period = int(kc_cfg.get("ema_period", 10))
    atr_period = int(kc_cfg.get("atr_period", 10))
    atr_mult = float(kc_cfg.get("atr_multiplier", 2.0))
    
    df["kc_lower"], df["kc_middle"], df["kc_upper"] = compute_keltner_channels(
        df, ema_period, atr_period, atr_mult
    )
    
    # RSI
    rsi_period = int(rsi_cfg.get("period", 14))
    df["rsi"] = compute_rsi(df["close"], rsi_period)
    
    # ATR for position sizing
    df["atr"] = compute_atr(df, atr_period)
    
    # ==========================================================================
    # CORE RUBBERBAND FACTORS FOR WEEKLY
    # ==========================================================================
    
    # Factor 1: RSI Oversold
    rsi_oversold = float(filter_cfg.get("rsi_oversold", 45))
    df["rsi_signal"] = df["rsi"] < rsi_oversold
    
    # Factor 2: Mean Deviation - the "rubber band stretch"
    # Calculate 20-week SMA for mean deviation
    sma_20w = df["close"].rolling(window=20).mean()
    df["sma_20w"] = sma_20w
    
    # Mean deviation = how far below the 20-week SMA (percentage)
    df["mean_deviation_pct"] = (df["close"] - sma_20w) / sma_20w * 100
    
    # Signal: Price is 5%+ below the 20-week SMA (stretched rubber band)
    # Note: Default -5 matches config_weekly.yaml and live_weekly_loop.py
    mean_dev_threshold = float(filter_cfg.get("mean_deviation_threshold", -5))
    df["mean_dev_signal"] = df["mean_deviation_pct"] < mean_dev_threshold
    
    # Factor 3: Trend Filter - only buy dips in uptrends
    # 20-week SMA > 50-week SMA = confirmed uptrend (golden cross structure)
    sma_50w = df["close"].rolling(window=50).mean()
    df["sma_50w"] = sma_50w
    
    # Trend is bullish when: 20-week SMA > 50-week SMA
    # This allows price to dip BELOW 20w while still being in an uptrend
    df["trend_signal"] = sma_20w > sma_50w
    
    # ==========================================================================
    # COMBINED ENTRY SIGNAL
    # ==========================================================================
    
    # Test: RSI + Mean Deviation ONLY (no trend filter)
    # This should give more signals while still being selective
    
    df["long_signal"] = df["rsi_signal"] & df["mean_dev_signal"]
    
    # Alternative: Full 3-factor (uncomment to use):
    # df["long_signal"] = df["rsi_signal"] & df["mean_dev_signal"] & df["trend_signal"]
    
    # Short Signal (not used in long-only mode)
    rsi_overbought = float(filter_cfg.get("rsi_overbought", 55))
    df["short_signal"] = df["rsi"] > rsi_overbought
    
    return df

# =============================================================================
# Data Loading
# =============================================================================

def load_weekly_data(symbol: str, cfg: dict, history_weeks: int = 104) -> pd.DataFrame:
    """
    Fetch weekly bars for a symbol.
    
    Args:
        symbol: Stock symbol
        cfg: Config dict
        history_weeks: Number of weeks of history to fetch
    """
    feed = cfg.get("feed", "iex")
    
    # Convert weeks to days (plus buffer for SMA calculation)
    trend_cfg = cfg.get("trend_filter", {})
    sma_period = int(trend_cfg.get("sma_period", 50))
    history_days = max(history_weeks * 7, (sma_period + 20) * 7)
    
    print(f"[{symbol}] Fetching weekly bars (history_days={history_days})...")
    
    bars_map, meta = fetch_latest_bars(
        symbols=[symbol],
        timeframe="1Week",
        history_days=history_days,
        feed=feed,
        rth_only=False,  # Weekly bars don't need RTH filter
        verbose=False
    )
    
    df = bars_map.get(symbol, pd.DataFrame())
    if df.empty:
        return df
    
    # Add trend filter (SMA on weekly close)
    if trend_cfg.get("enabled", False):
        sma_period_2 = int(trend_cfg.get("secondary_sma_period", 20))
        
        df["trend_sma"] = df["close"].rolling(window=sma_period).mean().shift(1)
        df["trend_sma_2"] = df["close"].rolling(window=sma_period_2).mean().shift(1)
    else:
        df["trend_sma"] = float("nan")
        df["trend_sma_2"] = float("nan")
    
    return df

# =============================================================================
# Backtest Simulator
# =============================================================================

def simulate_weekly_mean_reversion(
    df: pd.DataFrame, 
    cfg: dict, 
    symbol: str,
    start_cash: float = 10_000.0
) -> Dict[str, Any]:
    """
    Weekly Mean Reversion Backtest.
    
    Entry: Price < Lower Keltner & RSI < 40 (oversold)
    Exit: Price > Middle Keltner (mean) OR SL/TP brackets
    """
    if df is None or df.empty or len(df) < 20:
        return dict(
            trades=0, gross=0.0, net=0.0, win_rate=0.0, 
            ret_pct=0.0, equity=start_cash, detailed_trades=[]
        )
    
    # Attach indicators
    df = attach_indicators(df, cfg)
    
    # Diagnostic output - show factor breakdown
    rsi_count = df['rsi_signal'].sum()
    mean_dev_count = df['mean_dev_signal'].sum()
    trend_count = df['trend_signal'].sum()
    combined_count = df['long_signal'].sum()
    
    print(f"  [{symbol}] Factor breakdown: RSI={rsi_count} | MeanDev={mean_dev_count} | Trend={trend_count} | Combined={combined_count}")
    
    # Bracket params
    bcfg = cfg.get("brackets", {})
    atr_mult_sl = float(bcfg.get("atr_mult_sl", 2.0))
    take_profit_r = float(bcfg.get("take_profit_r", 2.5))
    
    # Sizing
    max_notional = float(cfg.get("max_notional_per_trade", 5000))
    max_shares = int(cfg.get("max_shares_per_trade", 100))
    
    # Trend filter
    trend_enabled = cfg.get("trend_filter", {}).get("enabled", False)
    
    # Trend filter
    trend_enabled = cfg.get("trend_filter", {}).get("enabled", False)
    
    # Regime Map (passed in cfg or separate arg? We'll inject it into cfg for now)
    daily_vix_map = cfg.get("daily_vix_map", {}) # Date -> VIXY Close
    # Regime Configs
    regime_params = {
        "CALM": {"rsi": 50, "dev": -3.0},
        "NORMAL": {"rsi": 45, "dev": -5.0},
        "PANIC": {"rsi": 30, "dev": -10.0}
    }
    
    # State
    equity = float(start_cash)
    in_pos = False
    qty = 0
    entry_px = 0.0
    entry_ts = None
    entry_state = {}
    
    wins = 0
    losses = 0
    gross = 0.0
    detailed_trades = []
    
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        cur = df.iloc[i]
        open_px = float(cur["open"])
        
        if not in_pos:
            # Trend Filter Check
            is_bull_trend = True  # Default
            if trend_enabled and not pd.isna(prev.get("trend_sma", float("nan"))):
                is_bull_trend = prev["close"] > prev["trend_sma"]
            
            # Long Entry Signal with Regime Logic
            # Determine Regime for this date
            current_date = cur.name.date()
            # Lookback 1 day for VIXY (simulating T-1 knowledge)
            # Since this is weekly data, 'cur.name' is a Friday. 
            # We can check VIXY from that Friday or PREV Friday? 
            # Entry is on Open of THIS week (based on PREV week signal).
            # So we need Regime at time of Entry (Monday Open).
            # We can use VIXY from prev.name (Previous Friday).
            
            vix_val = daily_vix_map.get(prev.name.date(), 40.0) # Default Normal
            if vix_val < 35:
                regime = "CALM"
            elif vix_val > 55:
                regime = "PANIC"
            else:
                regime = "NORMAL"
                
            p = regime_params.get(regime)
            
            # Use Dynamic Thresholds
            # prev.get("rsi") is already computed. 
            # But "long_signal" in dataframe was computed with STATIC thresholds!
            # We must re-evaluate signal dynamically here.
            
            prev_rsi = float(prev.get("rsi", 100))
            prev_dev = float(prev.get("mean_deviation_pct", 0))
            
            # Re-eval:
            is_regime_signal = (prev_rsi < p["rsi"]) and (prev_dev < p["dev"])
            
            # ==========================================
            # FILTER HYPOTHESIS TESTING
            # ==========================================
            
            # Filter A: Capitulation Filter
            # Skip entries when previous week shows extreme bearishness
            capitulation_filter_enabled = cfg.get("capitulation_filter", False)
            if capitulation_filter_enabled and is_regime_signal:
                prev_high = float(prev.get("high", 0))
                prev_low = float(prev.get("low", 0))
                prev_close = float(prev.get("close", 0))
                prev_open = float(prev.get("open", 1))
                bar_range = prev_high - prev_low
                if bar_range > 0:
                    close_position = (prev_close - prev_low) / bar_range
                    pct_change = (prev_close - prev_open) / prev_open * 100
                    if close_position < 0.20 and pct_change < -8:
                        continue
            
            # Filter B: Multi-Week Confirmation
            # Require 2 consecutive weeks of oversold conditions
            multi_week_filter_enabled = cfg.get("multi_week_filter", False)
            if multi_week_filter_enabled and is_regime_signal and i >= 2:
                prev_2 = df.iloc[i - 2]  # Two weeks ago
                prev_2_rsi = float(prev_2.get("rsi", 100))
                prev_2_dev = float(prev_2.get("mean_deviation_pct", 0))
                # Both weeks must be oversold
                if not (prev_2_rsi < p["rsi"] and prev_2_dev < p["dev"]):
                    continue  # Skip - not confirmed over 2 weeks
            
            # Filter C: Volume Spike Filter
            # Skip when volume is significantly above average (panic selling)
            volume_spike_filter_enabled = cfg.get("volume_spike_filter", False)
            if volume_spike_filter_enabled and is_regime_signal:
                # Calculate 20-week avg volume
                if i >= 20:
                    avg_volume = df.iloc[i-20:i]["volume"].mean()
                    prev_volume = float(prev.get("volume", 0))
                    if prev_volume > 2.0 * avg_volume:
                        continue  # Skip - panic selling
            
            # Filter D: Bullish Close Filter
            # Only enter when prev week closed in top 50% of range (showing buying interest)
            bullish_close_filter_enabled = cfg.get("bullish_close_filter", False)
            if bullish_close_filter_enabled and is_regime_signal:
                prev_high = float(prev.get("high", 0))
                prev_low = float(prev.get("low", 0))
                prev_close = float(prev.get("close", 0))
                bar_range = prev_high - prev_low
                if bar_range > 0:
                    close_position = (prev_close - prev_low) / bar_range
                    if close_position < 0.50:
                        continue  # Skip - closed in bottom half
            
            if is_bull_trend and is_regime_signal:
                atr_val = float(prev.get("atr", 0.0))
                if atr_val <= 0:
                    continue
                
                # Position sizing
                qty = min(max_shares, int(max_notional / open_px)) if open_px > 0 else 0
                if qty <= 0:
                    continue
                
                entry_px = open_px
                in_pos = True
                entry_ts = cur.name
                
                entry_state = {
                    "rsi": float(prev.get("rsi", 0)),
                    "atr": atr_val,
                    "sl_px": entry_px - (atr_val * atr_mult_sl),
                    "tp_px": entry_px + (atr_val * take_profit_r * atr_mult_sl)
                }
                continue
        
        else:
            # Exit Logic
            exit_signal = False
            reason = ""
            exit_px = 0.0
            
            # 1. Stop Loss
            if cur["low"] <= entry_state["sl_px"]:
                exit_signal = True
                reason = "SL"
                exit_px = entry_state["sl_px"]
            # 2. Take Profit
            elif cur["high"] >= entry_state["tp_px"]:
                exit_signal = True
                reason = "TP"
                exit_px = entry_state["tp_px"]
            # 3. Mean Reversion (Close > Middle Keltner)
            elif cur["close"] > cur["kc_middle"]:
                exit_signal = True
                reason = "MeanRev"
                exit_px = cur["close"]
            
            if exit_signal:
                pnl = (exit_px - entry_px) * qty
                gross += pnl
                
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                
                # Hold duration in weeks
                hold_weeks = (cur.name - entry_ts).days / 7
                
                detailed_trades.append({
                    "symbol": symbol,
                    "entry_time": entry_ts,
                    "exit_time": cur.name,
                    "entry_price": round(entry_px, 2),
                    "exit_price": round(exit_px, 2),
                    "qty": qty,
                    "pnl": round(pnl, 2),
                    "exit_reason": reason,
                    "hold_weeks": round(hold_weeks, 1),
                    "entry_rsi": round(entry_state.get("rsi", 0), 1)
                })
                
                in_pos = False
                qty = 0
    
    trades = wins + losses
    net = gross
    ret_pct = (net / start_cash) * 100.0 if start_cash > 0 else 0.0
    wr = (100.0 * wins / max(trades, 1))
    
    return dict(
        trades=trades,
        gross=round(gross, 2),
        net=round(net, 2),
        win_rate=round(wr, 1),
        ret_pct=round(ret_pct, 2),
        equity=round(start_cash + net, 2),
        detailed_trades=detailed_trades
    )

# =============================================================================
# Main CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Weekly Mean Reversion Backtest")
    ap.add_argument("--config", default="RubberBand/config_weekly.yaml", help="Config file path")
    ap.add_argument("--tickers", default="", help="Tickers file path")
    ap.add_argument("--symbols", default="", help="Comma-separated list of symbols")
    ap.add_argument("--days", default="365", help="Comma-separated days to backtest (e.g., 90,180,365)")
    ap.add_argument("--cash", type=float, default=10_000, help="Starting cash")
    ap.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    ap.add_argument("--adx-max", type=float, default=0, help="Skip entries when ADX > this value (0=disabled, try 50-60)")
    ap.add_argument("--capitulation-filter", action="store_true", help="Skip entries when prev week closed in bottom 20%% AND was >8%% down")
    ap.add_argument("--multi-week-filter", action="store_true", help="Require 2 consecutive weeks of oversold conditions")
    ap.add_argument("--volume-spike-filter", action="store_true", help="Skip when volume is >2x avg (panic selling)")
    ap.add_argument("--bullish-close-filter", action="store_true", help="Only enter when prev week closed in top 50%% of range")
    args = ap.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Determine symbols
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.tickers.strip():
        symbols = read_tickers(args.tickers)
    else:
        # Default test symbols
        symbols = ["AAPL", "NVDA", "AMZN", "META", "GOOGL", "MSFT", "AMD", "TSLA"]
    
    # Parse days
    try:
        days_list = [int(d.strip()) for d in args.days.split(",") if d.strip()]
    except ValueError:
        print("Error: --days must be comma-separated integers")
        return
    
    max_days = max(days_list)
    history_weeks = int(max_days / 7) + 60  # Extra buffer for SMA
    
    print(f"=" * 70)
    print(f"WEEKLY MEAN REVERSION BACKTEST")
    print(f"=" * 70)
    print(f"Symbols: {len(symbols)}")
    print(f"Days: {days_list}")
    print(f"Config: {args.config}")
    print(f"RSI Threshold: {cfg.get('filters', {}).get('rsi_oversold', 40)}")
    print(f"=" * 70)
    
    rows = []
    all_trades = []
    
    # Pre-fetch VIXY for Regime Logic
    print("Fetching VIXY data for Regime Detection...")
    vixy_map, _ = fetch_latest_bars(["VIXY"], "1Day", history_days=max_days+100, feed="alpaca", verbose=False)
    daily_vix_map = {}
    if "VIXY" in vixy_map and not vixy_map["VIXY"].empty:
        vdf = vixy_map["VIXY"]
        daily_vix_map = {row.Index.date(): row.close for row in vdf.itertuples()}
    
    # Inject map into cfg
    cfg["daily_vix_map"] = daily_vix_map
    
    # Inject filter settings from CLI
    cfg["capitulation_filter"] = getattr(args, "capitulation_filter", False)
    cfg["multi_week_filter"] = getattr(args, "multi_week_filter", False)
    cfg["volume_spike_filter"] = getattr(args, "volume_spike_filter", False)
    cfg["bullish_close_filter"] = getattr(args, "bullish_close_filter", False)

    for sym in symbols:
        try:
            df_full = load_weekly_data(sym, cfg, history_weeks)
        except Exception as e:
            if not args.quiet:
                print(f"[{sym}] Error: {e}")
            continue
        
        if df_full.empty:
            if not args.quiet:
                print(f"[{sym}] No data")
            continue
        
        if not args.quiet:
            print(f"[{sym}] Loaded {len(df_full)} weekly bars")
        
        for d in days_list:
            # Slice to approximate rows
            approx_weeks = d // 7
            if len(df_full) > approx_weeks:
                df_run = df_full.tail(approx_weeks).copy()
            else:
                df_run = df_full.copy()
            
            res = simulate_weekly_mean_reversion(df_run, cfg, sym, start_cash=args.cash)
            rows.append({"symbol": sym, "days": d, **{k: v for k, v in res.items() if k != "detailed_trades"}})
            
            for t in res["detailed_trades"]:
                t["backtest_days"] = d
                all_trades.append(t)
    
    if not rows:
        print("\nNo results.")
        return
    
    out = pd.DataFrame(rows)
    
    print(f"\n{'='*70}")
    print("RESULTS BY SYMBOL")
    print(out.to_string(index=False))
    
    # Save Detailed Trades
    if all_trades:
        df_trades = pd.DataFrame(all_trades)
        df_trades.to_csv("weekly_stock_backtest.csv", index=False)
        print("\nSaved detailed trades to weekly_stock_backtest.csv")
    print(f"{'='*70}")
    print(out.to_string(index=False))
    
    # Summary by Days
    print(f"\n{'='*70}")
    print("SUMMARY BY PERIOD")
    print(f"{'='*70}")
    
    for d in days_list:
        subset = out[out["days"] == d]
        total_trades = int(subset["trades"].sum())
        total_net = round(float(subset["net"].sum()), 2)
        avg_wr = round((subset["trades"] * subset["win_rate"]).sum() / max(total_trades, 1), 1) if total_trades > 0 else 0
        
        print(f"\n{d}-Day Window:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Net PnL: ${total_net:,.2f}")
        print(f"  Avg Win Rate: {avg_wr}%")
    
    # Overall
    total_all = int(out["trades"].sum())
    net_all = round(float(out["net"].sum()), 2)
    
    print(f"\n{'='*70}")
    print(f"OVERALL: {total_all} trades | ${net_all:,.2f} net PnL")
    print(f"{'='*70}")
    
    # Save results
    out.to_csv("weekly_backtest_summary.csv", index=False)
    print(f"\nSaved summary to weekly_backtest_summary.csv")
    
    if all_trades:
        df_trades = pd.DataFrame(all_trades)
        df_trades.to_csv("weekly_detailed_trades.csv", index=False)
        print(f"Saved {len(all_trades)} trades to weekly_detailed_trades.csv")

if __name__ == "__main__":
    main()
