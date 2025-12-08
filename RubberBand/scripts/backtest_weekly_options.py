
import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Ensure repo root is on path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.utils import load_config
from RubberBand.src.data import fetch_latest_bars
from RubberBand.scripts.backtest_weekly import attach_indicators

def simulate_weekly_options(
    df: pd.DataFrame, 
    cfg: dict, 
    ticker: str,
    days: int
) -> list:
    """
    Simulate Weekly Options Trading.
    Strategy: Buy ITM Call (~45 DTE) when Weekly Signal hits.
    Exit: Mean Reversion, Stop Loss, or 4-week Time Stop.
    Model: Delta 0.65, Theta Decay -10% of Extrinsic Value per week.
    """
    if df is None or df.empty or len(df) < 20:
        return []

    # Attach Strategy Indicators
    df = attach_indicators(df, cfg)
    
    trades = []
    
    # Weekly Config
    rsi_thresh = float(cfg["filters"]["rsi_oversold"])
    mean_dev_thresh = float(cfg["filters"].get("mean_deviation_threshold", -5)) / 100.0
    
    # Pre-calculate Indicators
    df["sma_20"] = df["close"].rolling(20).mean()
    df["mean_dev"] = (df["close"] - df["sma_20"]) / df["sma_20"]
    
    # Normalize shift for "confirmed" signals (previous bar)
    # We want to enter based on PREV bar signal
    df["prev_rsi"] = df["rsi"].shift(1)
    df["prev_mean_dev"] = df["mean_dev"].shift(1)
    df["prev_close"] = df["close"].shift(1)

    # DEBUG: Print first few rows of indicators
    # print(f"DEBUG {ticker}: RSI head: {df['prev_rsi'].head(30).values}")

    # Option Parameters (Simulated ITM Call, ~45 DTE)
    # Higher Delta (0.65) = More intrinsic value, less theta decay
    DELTA = 0.65
    THETA_DECAY_WEEKLY = 0.10  # Lose 10% of premium per week due to time
    LEVERAGE_COST = 0.06       # Premium costs ~6% of stock price (ITM 45DTE)
    
    in_trade = False
    trade = {}
    
    # Iterate
    for i in range(25, len(df)):
        cur = df.iloc[i]
        


        if not in_trade:
            # Check Signal (on previous closed bar)
            if cur["prev_rsi"] < rsi_thresh and cur["prev_mean_dev"] < mean_dev_thresh:
                # ENTRY
                entry_price = float(cur["open"]) # Open of current week
                
                # Option Simulation
                # Buy 1 ATM Call
                # Premium Cost
                opt_premium = entry_price * LEVERAGE_COST
                contracts = 1 # Per simulate unit
                
                trade = {
                    "symbol": ticker,
                    "entry_date": cur.name,
                    "entry_price": entry_price,
                    "opt_cost": opt_premium,
                    "contracts": contracts,
                    "max_profit": 0,
                    "max_loss": 0,
                    "weeks_held": 0
                }
                in_trade = True
        
        else:
            # MANAGING TRADE
            trade["weeks_held"] += 1
            
            # Update Option Value
            # Change in Stock
            stock_change = cur["close"] - trade["entry_price"]
            
            # Intrinsic + Extrinsic
            # Delta component
            delta_pnl = stock_change * DELTA
            
            # Theta component (Decay on the original premium/extrinsic)
            # Simple model: Value = Original_Prem + Delta_PnL - (Original_Prem * Theta * Weeks)
            theta_loss = trade["opt_cost"] * THETA_DECAY_WEEKLY * trade["weeks_held"]
            
            current_opt_val = trade["opt_cost"] + delta_pnl - theta_loss
            
            # Exit Conditions
            exit_signal = False
            reason = ""
            
            # 1. Mean Reversion (Close > KC Middle)
            kc_mid = cur["kc_middle"] if "kc_middle" in cur.index else cur["close"] * 1.05  # Fallback
            if cur["close"] > kc_mid:
                exit_signal = True
                reason = "MeanRev"
            
            # 2. Stop Loss (Option value down 50%?)
            # Or use Stock ATR stop? Let's use Stock ATR stop like the stock strategy
            atr = cur["atr"] if "atr" in cur.index else entry_price * 0.05
            sl_stock_price = trade["entry_price"] - (2.0 * atr)
            
            if cur["low"] < sl_stock_price:
                exit_signal = True
                reason = "SL_Stock"
                # If SL hit, estimate option value at SL price
                stock_change_sl = sl_stock_price - trade["entry_price"]
                current_opt_val = trade["opt_cost"] + (stock_change_sl * DELTA) - theta_loss
                
            # 3. Take Profit (Option up 100%? or Stock Target?)
            # Stock Target = 2.5 Risk (ATR)
            tp_stock_price = trade["entry_price"] + (2.5 * atr * 2.0) # Wait, 2.5R means 2.5 * (2*ATR) = 5 ATR?
            # Config says TP R:R 2.5, SL is 2.0 ATR. So Risk=2ATR. Reward=5ATR.
            if cur["high"] > tp_stock_price:
                exit_signal = True
                reason = "TP_Stock"
                stock_change_tp = tp_stock_price - trade["entry_price"]
                current_opt_val = trade["opt_cost"] + (stock_change_tp * DELTA) - theta_loss

            if exit_signal or trade["weeks_held"] > 3: # Time stop - max 4 weeks to reduce theta decay
                if not exit_signal: reason = "TimeStop"
                
                pnl = current_opt_val - trade["opt_cost"]
                roi = (pnl / trade["opt_cost"]) * 100
                
                trade.update({
                    "exit_date": cur.name,
                    "exit_price": cur["close"],
                    "exit_opt_val": current_opt_val,
                    "pnl": pnl,
                    "roi": roi,
                    "reason": reason
                })
                trades.append(trade)
                in_trade = False

    return trades

import yfinance as yf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=1095)
    parser.add_argument("--tickers", default="RubberBand/tickers_weekly.txt")
    args = parser.parse_args()
    
    cfg = load_config("RubberBand/config_weekly.yaml")
    
    with open(args.tickers, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
        
    print(f"Backtesting Weekly Options on {len(tickers)} tickers for {args.days} days...")
    
    all_trades = []
    
    # Calculate start date
    start_date = datetime.now() - pd.Timedelta(days=args.days + 100)
    
    for t in tickers:
        try:
            # Direct YF Download for reliability
            df = yf.download(t, start=start_date, interval="1wk", progress=False)
            
            if df is None or df.empty:
                print(f"[{t}] No data")
                continue
            
            # Standardize columns to lowercase
            # Handle MultiIndex if present (e.g. ('Close', 'AAPL'))
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df.columns = [c.lower() for c in df.columns]
            # If MultiIndex (Price, Ticker), drop level
            # yfinance > 0.2 returns MultiIndex sometimes?
            # Usually single ticker download returns standard DataFrame
            
            # Run Sim
            t_trades = simulate_weekly_options(df, cfg, t, args.days)
            all_trades.extend(t_trades)
            
        except Exception as e:
            print(f"[{t}] Error: {e}")
            continue
        
    # Summary
    if not all_trades:
        print("No trades.")
        return
        
    df_res = pd.DataFrame(all_trades)
    
    total_pnl = df_res["pnl"].sum()
    win_rate = len(df_res[df_res["pnl"] > 0]) / len(df_res) * 100
    avg_roi = df_res["roi"].mean()
    
    print("="*60)
    print("WEEKLY OPTIONS RESULTS (Simulated ATM Calls)")
    print("="*60)
    print(f"Trades: {len(df_res)}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Avg ROI per trade: {avg_roi:.1f}%")
    # Assuming $2000 position size (fixed Notional for option allocation)
    # If we put $2000 into options each trade:
    capital_per_trade = 2000
    # Total Profit = Average ROI * Capital * Trades
    estimated_total_profit = (avg_roi/100) * capital_per_trade * len(df_res)
    
    print(f"Est. Total Profit (Allocating ${capital_per_trade}/trade): ${estimated_total_profit:,.0f}")
    print("="*60)
    
    # Save
    df_res.to_csv("weekly_options_backtest.csv", index=False)
    print("Saved to weekly_options_backtest.csv")

if __name__ == "__main__":
    main()
