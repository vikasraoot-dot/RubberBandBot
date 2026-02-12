
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
    check_kill_switch,
    check_capital_limit,
    order_exists_today,
    close_position,
    KillSwitchTriggered,
    CapitalLimitExceeded,
)
from RubberBand.src.trade_logger import TradeLogger
from RubberBand.scripts.backtest_weekly import attach_indicators
from RubberBand.src.position_registry import PositionRegistry
from RubberBand.src.regime_manager import RegimeManager

# Bot tag for position attribution
BOT_TAG = "WK_STK"

# Time stop: Close positions held longer than this (matches backtest behavior)
TIME_STOP_WEEKS = 20  # 20 weeks = ~5 months max hold

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] WEEKLY: %(message)s",
    handlers=[
        logging.FileHandler("weekly_bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Trade logger for structured JSONL logging and CSV export
from zoneinfo import ZoneInfo
ET = ZoneInfo("US/Eastern")
logger = TradeLogger(f"logs/{BOT_TAG}/trade_{datetime.now(ET).strftime('%Y%m%d')}.jsonl")

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
    
    # --- Dynamic Regime Detection (Daily + Intraday) ---
    rm = RegimeManager(verbose=True)
    daily_regime = rm.update()  # Sets reference values for intraday checks
    current_regime = rm.get_effective_regime()  # Checks for intraday VIXY spikes
    regime_cfg = rm.get_config_overrides()

    # If intraday panic triggered, use PANIC config
    if current_regime == "PANIC" and daily_regime != "PANIC":
        regime_cfg = rm.regime_configs["PANIC"]
        logging.warning(f"INTRADAY PANIC DETECTED - Daily: {daily_regime}, Effective: {current_regime}")

    logging.info(f"Regime: {current_regime} (Daily: {daily_regime}, VIXY={rm.last_vixy_price:.2f})")
    # --------------------------------
    
    logging.info("="*60)
    logging.info(f"Starting Weekly Strategy Cycle - {len(tickers)} Tickers")
    logging.info("="*60)

    # Check if market is open first (save API calls)
    if not alpaca_market_open():
        logging.info("Market is closed. Skipping cycle.")
        return

    # Initialize Trade Logger
    from datetime import timezone
    log_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    logger = TradeLogger(path=f"logs/weekly_live_{log_date}.jsonl")
    
    # Position Registry for this bot
    registry = PositionRegistry(bot_tag=BOT_TAG)
    
    # Debug: Log registry state
    logging.info(f"Registry loaded: {len(registry.positions)} positions in {BOT_TAG} registry")
    if registry.positions:
        logging.debug(f"Registry symbols: {list(registry.positions.keys())}")

    # 1. Check Existing Positions (using env vars for credentials)
    all_positions = get_positions()  # Uses env: APCA_API_BASE_URL, APCA_API_KEY_ID, APCA_API_SECRET_KEY

    # PHASE 2 FIX (GAP-008): Use reconcile_or_halt() instead of sync_with_alpaca()
    # This prevents silent cleanup of orphaned positions and alerts on mismatch.
    is_clean, registry_orphans, broker_untracked = registry.reconcile_or_halt(
        all_positions,
        auto_clean=False,  # Do NOT auto-clean - we want to know about mismatches
    )

    if not is_clean:
        # Registry has positions that broker doesn't - this is a critical mismatch
        logging.critical(f"POSITION_MISMATCH_AT_STARTUP: Registry has {len(registry_orphans)} orphaned positions")
        logging.critical(f"Orphaned symbols: {registry_orphans}")
        logging.critical("Auto-cleaning to allow trading to continue...")

        # Re-run with auto_clean=True to fix the state
        registry.reconcile_or_halt(all_positions, auto_clean=True)

    logging.info(f"Registry reconciled: {len(registry.positions)} positions, clean={is_clean}, orphans={len(registry_orphans) if not is_clean else 0}")
    
    # Kill Switch Check - RE-ENABLED Dec 13, 2025
    # Halts trading if daily loss exceeds 25% of invested capital
    if check_kill_switch(bot_tag=BOT_TAG, max_loss_pct=25.0):
        logging.critical(f"[KILL SWITCH] {BOT_TAG} exceeded 25% daily loss - HALTING")
        raise KillSwitchTriggered(f"{BOT_TAG} exceeded 25% daily loss")
    
    # Filter to only OUR positions (based on registry)
    positions = registry.filter_positions(all_positions)
    
    # Additional validation: Only show positions that have WK_STK client_order_id
    # This catches cases where registry may have been contaminated
    validated_positions = []
    for p in positions:
        sym = p.get("symbol", "")
        # Try to find client_order_id from our registry
        reg_entry = registry.positions.get(sym, {})
        coid = reg_entry.get("client_order_id", "")
        if coid.startswith(f"{BOT_TAG}_"):
            validated_positions.append(p)
        else:
            logging.warning(f"Skipping {sym}: client_order_id '{coid}' does not match {BOT_TAG}")
    
    positions = validated_positions
    open_symbols = {p.get("symbol") for p in positions if p.get("symbol")}

    # CROSS-BOT AWARENESS: Track ALL broker equity positions (any bot) to prevent
    # position stacking when multiple bots (15M_STK, WK_STK) target the same ticker.
    all_broker_equity_symbols = {
        p.get("symbol") for p in all_positions
        if p.get("symbol") and p.get("asset_class", "us_equity") == "us_equity"
    }

    logging.info(f"Open Positions ({BOT_TAG}): {len(open_symbols)}")
    logging.info(f"All Broker Equity Positions: {len(all_broker_equity_symbols)} ({all_broker_equity_symbols})")

    if positions:
        logging.info("--- Current Holdings ---")
        header = f"{'Ticker':<8} {'Qty':<6} {'Entry':<10} {'Current':<10} {'PnL $':<10} {'PnL %':<8}"
        logging.info(header)
        logging.info("-" * len(header))
        
        for p in positions:
            sym = p.get("symbol", "N/A")
            qty = float(p.get("qty", 0))
            entry = float(p.get("avg_entry_price", 0))
            current = float(p.get("current_price", 0))
            pnl = float(p.get("unrealized_pl", 0))
            pnl_pct = float(p.get("unrealized_plpc", 0)) * 100
            
            logging.info(f"{sym:<8} {int(qty):<6} {entry:<10.2f} {current:<10.2f} {pnl:<10.2f} {pnl_pct:<8.2f}%")
        logging.info("-" * len(header))
    
    # --- Time Stop Check ---
    # Close positions held longer than TIME_STOP_WEEKS
    from datetime import timedelta
    from dateutil import parser as dateparser
    
    time_stop_days = TIME_STOP_WEEKS * 7
    now = datetime.now()
    
    for p in positions:
        sym = p.get("symbol", "")
        reg_entry = registry.positions.get(sym, {})
        entry_date_str = reg_entry.get("entry_date", "")
        
        if not entry_date_str:
            continue
        
        try:
            entry_date = dateparser.parse(entry_date_str)
            if entry_date.tzinfo:
                entry_date = entry_date.replace(tzinfo=None)
            
            days_held = (now - entry_date).days
            
            if days_held >= time_stop_days:
                logging.warning(
                    f"⏰ TIME STOP: {sym} held {days_held} days (>{time_stop_days}). Closing position."
                )
                result = close_position(
                    base_url=None,  # Uses APCA_API_BASE_URL env
                    key=None,       # Uses APCA_API_KEY_ID env
                    secret=None,    # Uses APCA_API_SECRET_KEY env
                    symbol=sym
                )
                if result.get("ok"):
                    logging.info(f"✅ Time stop exit: {sym} closed after {days_held} days")
                    registry.record_exit(sym, exit_reason="TIME_STOP", pnl=p.get("unrealized_pl", 0))
                    # Update open_symbols so freed slot is available for new entries
                    open_symbols.discard(sym)
                else:
                    logging.error(f"❌ Failed to close {sym}: {result}")
        except Exception as e:
            logging.error(f"Error checking time stop for {sym}: {e}")

    # Watchdog pause check (fail-open if file missing)
    try:
        from RubberBand.src.watchdog.pause_check import check_bot_paused
        _wd_paused, _wd_reason = check_bot_paused(BOT_TAG)
        if _wd_paused:
            logging.warning(f"[WATCHDOG] {BOT_TAG} paused: {_wd_reason}")
            return
    except Exception as e:
        logging.debug("[WATCHDOG] non-fatal: %s", e)

    max_capital_per_trade = float(cfg.get("max_notional_per_trade", 2000.0))
    limit_pos = int(cfg.get("max_concurrent_positions", 5))
    
    if len(open_symbols) >= limit_pos:
        logging.info("Max positions reached. No new entries.")
        return

    # Config for exit brackets
    bcfg = cfg.get("brackets", {})
    atr_mult_sl = float(bcfg.get("atr_mult_sl", 2.0))
    take_profit_r = float(bcfg.get("take_profit_r", 2.5))

    # 2. Iterate Tickers in ORDER of Performance
    entries_made = 0
    for symbol in tickers:
        try:
            # Skip if already in position (own registry)
            if symbol in open_symbols:
                continue

            # CROSS-BOT GATE: Skip if ANY other bot already holds this ticker
            # Prevents position stacking (e.g., 15M_STK + WK_STK both holding NFLX)
            # Note: own positions are already caught by `open_symbols` check above
            if symbol in all_broker_equity_symbols:
                logging.info(f"CROSS-BOT BLOCK: {symbol} already held by another bot — skipping")
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
                feed=cfg.get("feed", "iex"),
                rth_only=False # Daily bars have 00:00 timestamp, so RTH filter would kill them
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
            # Use Regime Config if available, else fallback to static config
            rsi_oversold = float(regime_cfg.get("weekly_rsi_oversold", cfg["filters"]["rsi_oversold"]))
            
            # Regime dev is pct (e.g. -5.0), static config might be int (-5)
            # Normalize to decimal (e.g. -0.05)
            regime_dev_pct = regime_cfg.get("weekly_mean_dev_pct")
            if regime_dev_pct is not None:
                mean_dev_thresh = float(regime_dev_pct) / 100.0
            else:
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
                
                # Capital limit check
                trade_value = qty * close
                max_capital = float(cfg.get("max_capital", 100000))
                try:
                    check_capital_limit(
                        base_url=None, key=None, secret=None,
                        proposed_trade_value=trade_value,
                        max_capital=max_capital,
                        bot_tag=BOT_TAG,
                    )
                except CapitalLimitExceeded as e:
                    logging.warning(f"Capital limit exceeded for {symbol}: {e}")
                    continue
                
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
                    tif="gtc",  # GTC so TP/SL persist overnight
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
    
    # Force save registry to ensure artifact exists (even if empty)
    registry.save()
    
    logging.info("=== End Summary ===")

if __name__ == "__main__":
    # Load registry once at startup for persistence across cycles
    registry = PositionRegistry(bot_tag=BOT_TAG)
    
    while True:
        # Simple loop: Run once every hour around market times or just sleep
        now = datetime.now()
        
        try:
            # Check if market is still open before running cycle
            if not alpaca_market_open():
                logging.info("Market is closed. Saving registry and exporting logs.")
                # EOD: Export trades to CSV and summary
                try:
                    logger.eod_summary()
                    csv_date = datetime.now(ET).strftime("%Y%m%d")
                    logger.export_trades_csv(f"results/{BOT_TAG}_trades_{csv_date}.csv")
                    logger.close()
                except Exception as e:
                    logging.error(f"Error exporting trade logs: {e}")
                registry.save()
                break
            
            # Run logic
            run_weekly_cycle()
            
            # Sleep 1 hour
            logging.info("Cycle complete. Sleeping 60 mins...")
            time.sleep(3600)
            
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt. Saving registry and exiting.")
            registry.save()
            break
        except KillSwitchTriggered as e:
            logging.critical(f"Kill switch triggered: {e}. Saving registry and exiting.")
            registry.save()
            break
        except Exception as e:
            logging.error(f"Main loop error: {e}")
            time.sleep(60)

