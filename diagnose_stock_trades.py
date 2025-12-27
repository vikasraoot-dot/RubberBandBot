
from __future__ import annotations
import os
import sys
import pandas as pd
import json

_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.utils import load_config
from RubberBand.src.data import fetch_latest_bars
from RubberBand.scripts.backtest import simulate_mean_reversion
from RubberBand.src.ticker_health import TickerHealthManager

def diagnose():
    cfg = load_config("RubberBand/config.yaml")
    
    # Override for diagnosis
    cfg["trend_filter"]["enabled"] = True
    cfg["trend_filter"]["sma_period"] = 50  # Use the winner
    
    # Check EOD setting
    print(f"Config Check: flatten_minutes_before_close = {cfg.get('flatten_minutes_before_close')}")
    # Note: simulate_mean_reversion uses _flatten_eod, let's see what logic sets or defaults it.
    
    # Use Top 5 Elite Tickers
    tickers = ["NVDA", "MARA", "AMD", "SOXL", "MSTR"]
    
    key = os.getenv("APCA_API_KEY_ID", "")
    secret = os.getenv("APCA_API_SECRET_KEY", "")
    feed = cfg.get("feed", "iex")
    
    print("Fetching data...")
    stock_df_map, _ = fetch_latest_bars(tickers, "15Min", 60, feed, True, key=key, secret=secret, verbose=False)
    daily_df_map, _ = fetch_latest_bars(tickers, "1Day", 400, feed, False, key=key, secret=secret, verbose=False)

    print("\n--- TRADE DIAGNOSIS (SMA 50) ---")
    
    hm = TickerHealthManager("temp_diag_health.json", cfg.get("resilience", {}))
    
    all_trades = []
    
    for sym in tickers:
        df = stock_df_map.get(sym)
        df_d = daily_df_map.get(sym)
        
        if df is None or df_d is None or df.empty or df_d.empty:
            continue
            
        # Inject SMA 50
        df_d["sma50"] = df_d["close"].rolling(50).mean().shift(1)
        df_d["date"] = df_d.index.date
        df["date"] = df.index.date
        sma_map = df_d.set_index("date")["sma50"].to_dict()
        df["trend_sma"] = df["date"].map(sma_map)
        df.drop(columns=["date"], inplace=True)
        
        # Run with VERBOSE=True
        print(f"\nRunning {sym}...")
        res = simulate_mean_reversion(df, cfg, hm, sym, start_cash=10000, risk_pct=0.01, verbose=True)
        all_trades.extend(res.get("detailed_trades", []))
        
    print("\n--- EXIT REASON ANALYSIS ---")
    exit_reasons = {}
    total_pnl = 0
    for t in all_trades:
        reason = t["exit_reason"]
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        total_pnl += t["pnl"]
        
    print(f"Total Trades: {len(all_trades)}")
    print(f"Total PnL: ${total_pnl:.2f}")
    print("Reasons Breakdown:")
    for r, count in exit_reasons.items():
        print(f"  {r}: {count}")

if __name__ == "__main__":
    diagnose()
