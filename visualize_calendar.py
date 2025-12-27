
from __future__ import annotations
import os
import sys
import pandas as pd
import datetime as dt
import calendar

# Setup Path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.utils import load_config, read_tickers
from RubberBand.src.data import fetch_latest_bars
from RubberBand.scripts.backtest import simulate_mean_reversion
from RubberBand.src.ticker_health import TickerHealthManager

def generate_calendar():
    print("\n--- Generating Trade Calendar (SMA 50 | Last 220 Days) ---")
    cfg = load_config("RubberBand/config.yaml")
    tickers = read_tickers("RubberBand/tickers.txt")
    
    # Force SMA 50
    cfg["trend_filter"]["enabled"] = True
    cfg["trend_filter"]["sma_period"] = 50
    
    feed = cfg.get("feed", "iex")
    key = os.getenv("APCA_API_KEY_ID", "")
    secret = os.getenv("APCA_API_SECRET_KEY", "")
    
    # Fetch Data
    days = 220
    print(f"Loading data for {len(tickers)} tickers...")
    stock_df_map, _ = fetch_latest_bars(tickers, "15Min", int(days*1.6), feed, True, key=key, secret=secret, verbose=False)
    daily_df_map, _ = fetch_latest_bars(tickers, "1Day", days + 250, feed, False, key=key, secret=secret, verbose=False)
    
    # Run Simulation
    hm = TickerHealthManager("temp_cal_health.json", cfg.get("resilience", {}))
    all_trades = []
    
    print("Simulating trades...")
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
        
        res = simulate_mean_reversion(df, cfg, hm, sym, start_cash=10000, risk_pct=0.01, verbose=False)
        all_trades.extend(res.get("detailed_trades", []))
        
    # Cleanup
    if os.path.exists("temp_cal_health.json"):
        try: os.remove("temp_cal_health.json")
        except: pass

    # Organize by Date
    trades_by_date = {} # "YYYY-MM-DD": {"count": 0, "pnl": 0.0}
    
    for t in all_trades:
        date_str = str(t["entry_time"].date())
        if date_str not in trades_by_date:
            trades_by_date[date_str] = {"count": 0, "pnl": 0.0, "wins": 0}
            
        trades_by_date[date_str]["count"] += 1
        trades_by_date[date_str]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            trades_by_date[date_str]["wins"] += 1

    # Print Calendar
    # Find min/max date
    dates = [dt.datetime.strptime(d, "%Y-%m-%d").date() for d in trades_by_date.keys()]
    if not dates:
        print("No trades found.")
        return

    start_date = min(dates)
    end_date = max(dates)
    # Start from beginning of that month
    curr_date = start_date.replace(day=1)
    
    print("\nLEGEND: [Trades] PnL_Symbol (W=Win, L=Loss, M=Mixed)")
    print("=" * 60)
    
    while curr_date <= end_date:
        year = curr_date.year
        month = curr_date.month
        month_name = calendar.month_name[month]
        month_matrix = calendar.monthcalendar(year, month)
        
        print(f"\n{month_name.upper()} {year}")
        print("Mon        Tue        Wed        Thu        Fri")
        print("-" * 50)
        
        for week in month_matrix:
            line_str = ""
            has_trading_days = False
            
            # Check if this week is relevant (trading days are Mon-Fri: indices 0-4)
            week_relevant = False
            for day in week[:5]:
                if day != 0: week_relevant = True
            if not week_relevant: continue
                
            for i, day in enumerate(week[:5]): # Only Mon-Fri
                if day == 0:
                    line_str += "           "
                    continue
                
                d_obj = dt.date(year, month, day)
                d_str = str(d_obj)
                
                cell = f"{day:02d} | ...  " # Default empty
                
                if d_str in trades_by_date:
                    data = trades_by_date[d_str]
                    cnt = data["count"]
                    pnl = data["pnl"]
                    wins = data["wins"]
                    
                    # Status Icon
                    if pnl > 0: icon = "✅"
                    elif pnl < 0: icon = "❌"
                    else: icon = "➖"
                    
                    # Compact: "05 | 2✅  "
                    cell = f"{day:02d} | {cnt:<1}{icon}  "
                    
                line_str += f"{cell:<11}"
            
            print(line_str)
            
        # Move to next month
        if month == 12:
            curr_date = dt.date(year + 1, 1, 1)
        else:
            curr_date = dt.date(year, month + 1, 1)

    print("\nSummary:")
    print(f"Total Days Traded: {len(trades_by_date)}")
    print(f"Total Trades: {len(all_trades)}")

if __name__ == "__main__":
    generate_calendar()
