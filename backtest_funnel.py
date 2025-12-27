
from __future__ import annotations
import os
import sys
import argparse
import pandas as pd
from collections import Counter
import json

# Setup Path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.utils import load_config, read_tickers
from RubberBand.src.data import fetch_latest_bars
from RubberBand.strategy import attach_verifiers
from RubberBand.src.ticker_health import TickerHealthManager

# Funnel Stats
FUNNEL = {
    "total_bars": 0,
    "has_signal_rsi": 0,
    "passed_trend": 0,
    "passed_slope": 0,
    "trades_taken": 0,
    "rejected_by_trend": 0,
    "rejected_by_slope": 0
}

def load_bars_for_symbol_dummy(symbol, cfg, days, **kwargs):
    # Re-use logic from backtest.py via import or copy?
    # Copying essential logic here for standalone script
    key = os.getenv("APCA_API_KEY_ID", "")
    secret = os.getenv("APCA_API_SECRET_KEY", "")
    feed = cfg.get("feed", "iex")
    
    # 1Day first for Trend
    daily_map, _ = fetch_latest_bars(
        symbols=[symbol], timeframe="1Day", history_days=500, feed=feed, rth_only=False,
        key=key, secret=secret, verbose=False
    )
    df_daily = daily_map.get(symbol, pd.DataFrame())
    
    # 15Min for Strat
    bars_map, _ = fetch_latest_bars(
        symbols=[symbol], timeframe="15Min", history_days=int(days*1.6), feed=feed, rth_only=True,
        key=key, secret=secret, verbose=False
    )
    df = bars_map.get(symbol, pd.DataFrame())
    
    if df.empty: return df
    
    # Calculate Trend SMA
    if not df_daily.empty:
        df_daily["trend_sma"] = df_daily["close"].rolling(window=200).mean().shift(1)
        df_daily["date_only"] = df_daily.index.date
        df["date_only"] = df.index.date
        sma_map = df_daily.set_index("date_only")["trend_sma"].to_dict()
        df["trend_sma"] = df["date_only"].map(sma_map)
        df.drop(columns=["date_only"], inplace=True)
    
    return df

def simulate_funnel(df, cfg, sym):
    global FUNNEL
    
    if df.empty: return

    # Ensure indicators
    # RSI, KC, ATR
    from RubberBand.strategy import attach_verifiers
    df = attach_verifiers(df, cfg).copy()

    # Iterating last few bars (Dec 18 focus)
    # Just iterate whole DF
    for i in range(11, len(df)):
        FUNNEL["total_bars"] += 1
        
        cur = df.iloc[i]
        prev = df.iloc[i-1]
        
        # 1. RSI / Signal Check (The "Candidate" Layer)
        # Entry requires: RSI < 25 (or 30) AND Close < Lower KC (Usually)
        # Let's check pure basic signal first
        rsi = prev.get("rsi", 100)
        kc_lower = prev.get("kc_lower", 0)
        
        # Check config limits
        rsi_threshold = cfg.get("filters", {}).get("rsi_oversold", 30)
        
        is_candidate = False
        if rsi < rsi_threshold and prev["close"] < kc_lower:
            is_candidate = True
        
        if not is_candidate:
            continue
            
        FUNNEL["has_signal_rsi"] += 1
        
        # 2. Trend Check
        # Bull Trend: Close > SMA200
        trend_sma = prev.get("trend_sma", float('nan'))
        is_bull_trend = False
        if not pd.isna(trend_sma) and prev["close"] > trend_sma:
            is_bull_trend = True
        
        # If trend filter enabled and not bull trend -> Reject
        trend_enabled = cfg.get("trend_filter", {}).get("enabled", True)
        if trend_enabled and not is_bull_trend:
            FUNNEL["rejected_by_trend"] += 1
            # print(f"[{sym}] REJECTED_TREND: RSI={rsi:.1f} Close={prev['close']} SMA={trend_sma}")
            continue
            
        FUNNEL["passed_trend"] += 1
        
        # 3. Slope Check (The "Guardrail" Layer)
        slope_threshold = cfg.get("slope_threshold", -0.20)
        slope_threshold_10 = cfg.get("slope_threshold_10", -0.15)
        
        # Calculate Slopes
        slope3 = (df["kc_middle"].iloc[i-1] - df["kc_middle"].iloc[i-4]) / 3
        slope10 = (df["kc_middle"].iloc[i-1] - df["kc_middle"].iloc[i-11]) / 10
        
        slope_fail = False
        if slope3 > slope_threshold:
            slope_fail = True
            # print(f"[{sym}] REJECTED_SLOPE3: Slope={slope3:.4f}")
        elif slope10 > slope_threshold_10:
            slope_fail = True
            # print(f"[{sym}] REJECTED_SLOPE10: Slope={slope10:.4f}")
            
        if slope_fail:
            FUNNEL["rejected_by_slope"] += 1
            continue
            
        FUNNEL["passed_slope"] += 1
        FUNNEL["trades_taken"] += 1
        print(f"[{sym}] TRADE TAKEN: {cur.name} RSI={rsi:.1f}")

def main():
    cfg = load_config("RubberBand/config.yaml")
    
    # Override for test if needed
    # cfg["slope_threshold"] = -0.20 
    
    tickers = read_tickers("RubberBand/tickers.txt")
    # Add FIX/CLH to be sure
    extras = ["FIX", "CLH"]
    for e in extras:
        if e not in tickers: tickers.append(e)

    print(f"Running Funnel Analysis on {len(tickers)} tickers for last 2 days...")
    
    for sym in tickers:
        try:
            df = load_bars_for_symbol_dummy(sym, cfg, days=2)
            simulate_funnel(df, cfg, sym)
        except Exception as e:
            # print(f"Error {sym}: {e}")
            pass
            
    print("\n=== FUNNEL RESULTS (Dec 18 Focus) ===")
    print(f"1. Candidates (RSI < {cfg.get('filters', {}).get('rsi_oversold', 30)} & <KC): {FUNNEL['has_signal_rsi']}")
    print(f"2. Passed Trend Filter (>SMA200):     {FUNNEL['passed_trend']} (Rejected {FUNNEL['rejected_by_trend']})")
    print(f"3. Passed Slope Filter (Steep Drop):  {FUNNEL['passed_slope']} (Rejected {FUNNEL['rejected_by_slope']})")
    print(f"4. Trades Taken:                      {FUNNEL['trades_taken']}")

if __name__ == "__main__":
    main()
