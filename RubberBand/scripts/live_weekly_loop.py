
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
    get_daily_fills,
)
from RubberBand.src.trade_logger import TradeLogger
from RubberBand.scripts.backtest_weekly import attach_indicators
from RubberBand.src.position_registry import PositionRegistry

# Bot tag for position attribution
BOT_TAG = "WK_STK"

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
    
    # Position Registry for this bot
    registry = PositionRegistry(bot_tag=BOT_TAG)
    
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

            # 3. Fetch Daily Data and Resample to Weekly
            # '1Week' bars often fail on IEX/free plans, so we build them from 1Day
            bars_map, meta = fetch_latest_bars(
                symbols=[symbol], 
                timeframe="1Day", 
                history_days=400, # Fetch enough days for >52 weeks
                feed=cfg.get("feed", "iex")
            )
            
            df_daily = bars_map.get(symbol)
            if df_daily is None or df_daily.empty:
                logging.warning(f"No daily data for {symbol}")
                continue

            # Resample to Weekly (Ending Friday)
            df_weekly = df_daily.resample('W-FRI').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            if len(df_weekly) < 30:
                logging.warning(f"Not enough weekly bars for {symbol} (found {len(df_weekly)})")
                continue
            
            # Use resampled weekly dataframe
            df = df_weekly

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
                
                # Generate client_order_id for position attribution
                coid = registry.generate_order_id(symbol)
                
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
                    tif="day",
                    client_order_id=coid,
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
                    # Record in registry for position attribution
                    registry.record_entry(
                        symbol=symbol,
                        client_order_id=coid,
                        qty=qty,
                        entry_price=close,
                        order_id=result.get("id", ""),
                    )
                    entries_made += 1
                else:
                    logging.error(f"Order failed: {result}")
                    
            else:
                logging.debug(f"{symbol}: No signal (RSI={rsi:.1f}, Dev={mean_dev_pct*100:.1f}%)")

        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")
            traceback.print_exc()

    # --- Session Summary ---
    logging.info("\n=== Weekly Stock Session Summary ===")
    try:
        fills = get_daily_fills(bot_tag=BOT_TAG)
        if not fills:
            logging.info("No trades filled today for WK_STK.")
        else:
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

            header = f"{'Ticker':<8} {'Bought':<8} {'Avg Ent':<10} {'Basis':<12} {'Sold':<8} {'Avg Ex':<10} {'Day PnL':<10}"
            logging.info("-" * len(header))
            logging.info(header)
            logging.info("-" * len(header))

            total_pnl = 0.0
            total_vol = 0.0
            for sym in sorted(stats.keys()):
                s = stats[sym]
                b_qty, b_val = s["buy_qty"], s["buy_val"]
                s_qty, s_val = s["sell_qty"], s["sell_val"]
                avg_ent = (b_val / b_qty) if b_qty > 0 else 0.0
                avg_ex = (s_val / s_qty) if s_qty > 0 else 0.0
                matched_qty = min(b_qty, s_qty)
                realized_pnl = (avg_ex - avg_ent) * matched_qty if matched_qty > 0 else 0.0
                total_pnl += realized_pnl
                total_vol += (b_val + s_val)
                pnl_str = f"{realized_pnl:,.2f}" if matched_qty > 0 else "-"
                logging.info(f"{sym:<8} {int(b_qty):<8} {avg_ent:<10.2f} {b_val:<12.2f} {int(s_qty):<8} {avg_ex:<10.2f} {pnl_str:<10}")

            logging.info("-" * len(header))
            logging.info(f"TOTAL Day PnL: ${total_pnl:,.2f} | TOTAL VOL: ${total_vol:,.2f}")
    except Exception as e:
        logging.error(f"Failed to generate summary: {e}")
    logging.info("=== End Summary ===")

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
