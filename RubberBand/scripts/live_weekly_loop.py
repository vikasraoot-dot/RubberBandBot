
import os
import sys
import time
import logging
import pandas as pd
import yaml
from datetime import datetime
import traceback

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RubberBand.src.data import (
    fetch_latest_bars,
    alpaca_market_open,
    submit_bracket_order,
    get_positions,
)
from RubberBand.src.trade_logger import TradeLogger
from RubberBand.scripts.backtest_weekly import attach_indicators

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] WEEKLY: %(message)s",
    handlers=[
        logging.FileHandler("weekly_bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def load_config():
    with open("RubberBand/config_weekly.yaml", "r") as f:
        return yaml.safe_load(f)

def load_tickers():
    # Load tickers IN ORDER of performance (assumes file is sorted)
    with open("RubberBand/tickers_weekly.txt", "r") as f:
        return [line.strip() for line in f if line.strip()]

def run_weekly_cycle():
    cfg = load_config()
    tickers = load_tickers()
    
    logging.info("="*60)
    logging.info(f"Starting Weekly Strategy Cycle - {len(tickers)} Tickers")
    logging.info("="*60)

    # Check if market is open first (save API calls)
    if not alpaca_market_open():
        logging.info("Market is closed. Skipping cycle.")
        return

    # 1. Check Existing Positions (using env vars for credentials)
    positions = get_positions()  # Uses env: APCA_API_BASE_URL, APCA_API_KEY_ID, APCA_API_SECRET_KEY
    open_symbols = {p.get("symbol") for p in positions if p.get("symbol")}
    
    logging.info(f"Open Positions: {len(open_symbols)}")

    max_capital_per_trade = float(cfg.get("max_notional_per_trade", 2000.0))
    limit_pos = int(cfg.get("max_concurrent_positions", 5))
    
    if len(open_symbols) >= limit_pos:
        logging.info("Max positions reached. No new entries.")
        return

    # Initialize Trade Logger
    from datetime import timezone
    log_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    logger = TradeLogger(path=f"logs/weekly_live_{log_date}.jsonl")
    
    # Config for exit brackets
    bcfg = cfg.get("brackets", {})
    atr_mult_sl = float(bcfg.get("atr_mult_sl", 2.0))
    take_profit_r = float(bcfg.get("take_profit_r", 2.5))

    # 2. Iterate Tickers in ORDER of Performance
    entries_made = 0
    for symbol in tickers:
        try:
            # Skip if already in position
            if symbol in open_symbols:
                continue
            
            # Check position limit
            if len(open_symbols) + entries_made >= limit_pos:
                logging.info("Max positions limit reached. Stopping.")
                break

            logging.info(f"Analyzing {symbol}...")

            # 3. Fetch Weekly Data
            bars_map, meta = fetch_latest_bars(
                symbols=[symbol], 
                timeframe="1Week", 
                history_days=365,
                feed=cfg.get("feed", "iex")
            )
            
            df = bars_map.get(symbol)
            if df is None or df.empty or len(df) < 30:
                logging.warning(f"No sufficient data for {symbol}")
                continue

            # 4. Attach Indicators
            df = attach_indicators(df, cfg)
            cur = df.iloc[-1]
            
            # Re-implement core logic check explicitly
            sma_20 = df["close"].rolling(20).mean().iloc[-1]
            rsi = float(cur["rsi"])
            close = float(cur["close"])
            atr = float(cur["atr"]) if "atr" in cur.index else close * 0.03  # Fallback 3% ATR
            
            # Conditions
            rsi_oversold = float(cfg["filters"]["rsi_oversold"])  # 45
            mean_dev_thresh = float(cfg["filters"].get("mean_deviation_threshold", -5)) / 100.0
            
            mean_dev_pct = (close - sma_20) / sma_20
            
            is_oversold = rsi < rsi_oversold
            is_stretched = mean_dev_pct < mean_dev_thresh
            
            if is_oversold and is_stretched:
                logging.info(f"🔥 SIGNAL FOUND for {symbol}: RSI={rsi:.1f}, Dev={mean_dev_pct*100:.1f}%")
                
                # Calculate order parameters
                qty = int(max_capital_per_trade // close)
                if qty <= 0:
                    logging.warning(f"Qty is 0 for {symbol} at price {close}")
                    continue
                
                # Calculate bracket levels
                sl_price = close - (atr_mult_sl * atr)
                risk = close - sl_price
                tp_price = close + (take_profit_r * risk)
                
                logging.info(f"Submitting Bracket Order: {symbol} x {qty} @ Market, SL={sl_price:.2f}, TP={tp_price:.2f}")
                
                # Submit bracket order (uses env credentials)
                result = submit_bracket_order(
                    base_url=None,  # Uses APCA_API_BASE_URL env
                    key=None,       # Uses APCA_API_KEY_ID env
                    secret=None,    # Uses APCA_API_SECRET_KEY env
                    symbol=symbol,
                    qty=qty,
                    side="buy",
                    limit_price=None,  # Market order
                    take_profit_price=tp_price,
                    stop_loss_price=sl_price,
                    tif="day"
                )
                
                if result.get("id"):
                    logging.info(f"✅ Order placed: {result.get('id')}")
                    logger.entry_submit(
                        symbol=symbol,
                        side="buy",
                        qty=qty,
                        entry_price=close,
                        stop_loss_price=sl_price,
                        take_profit_price=tp_price,
                        entry_reason=f"Weekly RB: RSI={rsi:.1f}, Dev={mean_dev_pct*100:.1f}%"
                    )
                    entries_made += 1
                else:
                    logging.error(f"Order failed: {result}")
                    
            else:
                logging.debug(f"{symbol}: No signal (RSI={rsi:.1f}, Dev={mean_dev_pct*100:.1f}%)")

        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    while True:
        # Simple loop: Run once every hour around market times or just sleep
        # For now, simplistic loop
        now = datetime.now()
        
        # Only run during market hours (or extended) to fetch quotes
        # Or just run once and let the user cron it. 
        # User requested "Loop", so:
        try:
             # Run logic
            run_weekly_cycle()
            
            # Sleep 1 hour
            logging.info("Cycle complete. Sleeping 60 mins...")
            time.sleep(3600)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.error(f"Main loop error: {e}")
            time.sleep(60)
