#!/usr/bin/env python3
"""
Auditor Bot (Shadow Ledger)
---------------------------
Monitors live bot logs and simulates a "Parallel Universe" where decisions were inverted.
- If Bot SKIPS -> Auditor ENTERS (Shadow Position).
- If Bot ENTERS -> Auditor SKIPS (Shadow Skip).

Tracks PnL of this "Shadow Portfolio" to benchmark whether the Bot's filtering logic is adding value (avoiding losses) or missing value (avoiding profits).
"""

import sys
import os
import time
import json
import logging
import argparse
import glob
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from zoneinfo import ZoneInfo

import pandas as pd

# Add Repo Root to Path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.options_data import (
    select_spread_contracts,
    get_option_quote,
    get_underlying_price,
    format_option_symbol
)

# Configuration
LOG_DIR = os.path.join(_REPO_ROOT, "results") # Verify this path
STATE_FILE = os.path.join(_REPO_ROOT, "auditor_state.json")
AUDIT_LOG_CSV = os.path.join(_REPO_ROOT, "auditor_log.csv")
ET = ZoneInfo("US/Eastern")

# Logger Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(_REPO_ROOT, "auditor.log"))
    ]
)
logger = logging.getLogger("Auditor")

class AuditorBot:
    def __init__(self, dry_run=False, log_dir=None, single_pass=False):
        self.dry_run = dry_run
        self.single_pass = single_pass
        self.log_dir = log_dir or LOG_DIR
        self.state = self._load_state()
        self.files_map = {} # {filepath: file_handle}
        self.positions = self.state.get("positions", {}) # {id: position_dict}
        self.closed_positions = self.state.get("closed_positions", [])
        self.processed_lines = self.state.get("processed_lines", {})  # {filepath: line_count}
        
        # Determine latest log files
        self.log_files = self._discover_logs()
        
        logger.info(f"Auditor initialized. Log Dir: {self.log_dir}, Files: {len(self.log_files)}")

    def _discover_logs(self) -> List[str]:
        """Discover all log files in the log directory."""
        patterns = [
            "15M_STK_*.jsonl",  # Stock bot logs
            "15M_OPT_*.jsonl",  # Options bot logs
            "live_*.jsonl",     # Legacy stock logs
            "options_trades_*.jsonl",  # Legacy options logs
        ]
        
        candidates = []
        for pattern in patterns:
            search_path = os.path.join(self.log_dir, pattern)
            candidates.extend(glob.glob(search_path))
        
        # Filter to today's logs only
        today = datetime.now().strftime("%Y%m%d")
        today_logs = [f for f in candidates if today in os.path.basename(f)]
        
        if today_logs:
            logger.info(f"Found {len(today_logs)} logs for today: {[os.path.basename(f) for f in today_logs]}")
            return today_logs
        
        # Fallback to most recent logs
        if candidates:
            candidates.sort(key=os.path.getmtime, reverse=True)
            logger.info(f"No logs for today, using most recent: {os.path.basename(candidates[0])}")
            return candidates[:2]  # Max 2 (stock + options)
        
        return []

    def _get_latest_log(self, pattern: str) -> Optional[str]:
        # Log dir might be in RubberBandBot/results or just results
        # Based on config.yaml "results_dir: results", it's relative to CWD usually.
        # Let's try explicit paths.
        search_paths = [
            os.path.join(self.log_dir, pattern),
            os.path.join(_REPO_ROOT, "RubberBandBot", "results", pattern),
            os.path.join(_REPO_ROOT, "results", pattern),
             os.path.join(_REPO_ROOT, "RubberBandBot", "logs", pattern) # Check logs dir too
        ]
        
        candidates = []
        for p in search_paths:
            candidates.extend(glob.glob(p))
            
        if not candidates:
            return None
            
        # Sort by modification time
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]

    def _load_state(self) -> Dict:
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        return {"positions": {}, "closed_positions": [], "processed_lines": {}}

    def _save_state(self):
        with open(STATE_FILE, "w") as f:
            json.dump({
                "positions": self.positions,
                "closed_positions": self.closed_positions,
                "processed_lines": self.processed_lines
            }, f, indent=2, default=str)

    def _append_to_csv(self, row: Dict):
        file_exists = os.path.exists(AUDIT_LOG_CSV)
        mode = "a" if file_exists else "w"
        
        # Flatten dict
        flat_row = {}
        for k, v in row.items():
            if isinstance(v, (dict, list)):
                flat_row[k] = json.dumps(v)
            else:
                flat_row[k] = v
                
        df = pd.DataFrame([flat_row])
        df.to_csv(AUDIT_LOG_CSV, mode=mode, header=not file_exists, index=False)

    def run(self):
        """Main processing loop."""
        if self.single_pass:
            self._process_once()
        else:
            self.tail_logs()

    def _process_once(self):
        """Single pass: process all logs once and exit."""
        for log_file in self.log_files:
            self._process_log_file(log_file)
        
        # Update shadow positions
        self.update_shadow_positions()
        self._save_state()
        
        # Print summary
        open_count = len(self.positions)
        closed_count = len(self.closed_positions)
        total_pnl = sum(p.get("realized_pnl", 0) for p in self.closed_positions)
        logger.info(f"Single pass complete. Open: {open_count}, Closed: {closed_count}, Total PnL: ${total_pnl:.2f}")

    def _process_log_file(self, log_file: str):
        """Process all entries in a log file."""
        # Use relative path for state tracking to ensure consistency across runners
        try:
            rel_path = os.path.relpath(log_file, _REPO_ROOT)
        except ValueError:
            rel_path = os.path.basename(log_file) # Fallback

        basename = os.path.basename(log_file)
        
        # Determine bot type from filename
        if "15M_STK" in basename or "live_" in basename:
            default_bot_type = "STOCK"
        elif "15M_OPT" in basename or "options_" in basename:
            default_bot_type = "OPTION"
        else:
            default_bot_type = "STOCK"
        
        # Track which lines we've already processed
        start_line = self.processed_lines.get(rel_path, 0)
        
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i < start_line:
                        continue
                    
                    try:
                        entry = json.loads(line.strip())
                        # Use bot_tag from entry if present, otherwise use filename-based default
                        bot_tag = entry.get("bot_tag", "")
                        if "STK" in bot_tag:
                            bot_type = "STOCK"
                        elif "OPT" in bot_tag:
                            bot_type = "OPTION"
                        else:
                            bot_type = default_bot_type
                        
                        self.process_log_entry(entry, bot_type)
                    except json.JSONDecodeError:
                        continue
                    
                    self.processed_lines[rel_path] = i + 1
                    
        except Exception as e:
            logger.error(f"Error processing {log_file}: {e}")

    def tail_logs(self):
        """Main loop to tail logs."""
        # Initialize file pointers to end? No, maybe we want to process catching up.
        # For now, let's process from start of file if we just started, 
        # but tracking processed lines would be better.
        # Simplifying: Read all, identifying new lines by some ID or just keeping file pointer.
        
        files_to_watch = self.log_files
        
        # Open files
        for fp in files_to_watch:
            if fp not in self.files_map:
                f = open(fp, "r", encoding="utf-8")
                # Seek to end? If we want to catch up, start from 0.
                # Let's start from 0 to catch up on today's session.
                # But we need to avoid double-processing if we restart auditor.
                # We can check if 'cid' is already in our state/history.
                self.files_map[fp] = f

        while True:
            processed_any = False
            for fp, f in self.files_map.items():
                line = f.readline()
                while line:
                    processed_any = True
                    try:
                        entry = json.loads(line)
                        # Determine bot type from bot_tag or filename
                        bot_tag = entry.get("bot_tag", "")
                        if "STK" in bot_tag or "live_" in fp:
                            bot_type = "STOCK"
                        else:
                            bot_type = "OPTION"
                        self.process_log_entry(entry, bot_type)
                    except json.JSONDecodeError:
                        pass
                    line = f.readline()
            
            # After processing new log lines, update shadow positions
            self.update_shadow_positions()
            self._save_state()
            
            if not processed_any:
                time.sleep(5) # Sleep if no new logs
            else:
                time.sleep(0.1)

    def process_log_entry(self, entry: Dict, bot_type: str):
        event_type = entry.get("type")
        cid = entry.get("cid") or entry.get("symbol") # Fallback to symbol if no cid, but cid is best
        
        # Check if we already processed this CID for this specific action?
        # A CID might have multiple log lines (signal, gate, order).
        
        # SKIP events -> Shadow ENTRY
        skip_events = ["SKIP_SLOPE3", "DKF_SKIP", "SPREAD_SKIP"]
        if event_type in skip_events or (event_type == "GATE" and entry.get("decision") == "BLOCK"):
            self.handle_bot_skip(entry, bot_type)
            
        elif event_type == "ENTRY_SUBMIT" or event_type == "SPREAD_ENTRY":
            # Bot ENTRY -> Shadow SKIP
            # We just log this for comparison, maybe create a "Shadow Skip" record 
            # to say "We missed out on this trade in the shadow universe"
            pass

    def handle_bot_skip(self, entry: Dict, bot_type: str):
        symbol = entry.get("symbol") or entry.get("underlying")
        reason = entry.get("reason") or entry.get("skip_reason")
        ts = entry.get("ts") or datetime.now(timezone.utc).isoformat()
        bot_tag = entry.get("bot_tag", bot_type)
        
        # Validate symbol
        if not symbol or not isinstance(symbol, str) or len(symbol) > 10:
            logger.debug(f"Skipping invalid symbol: {symbol}")
            return
        
        # Check if we already have an open position for this symbol
        existing = [p for p in self.positions.values() if p.get("symbol") == symbol and p.get("status") == "OPEN"]
        if existing:
            return  # Don't double-enter
        
        # Deterministic ID for this shadow trade to prevent duplicates on re-processing
        # Use TS + Symbol + Reason as unique key
        safe_ts = ts.replace(":", "").replace("-", "").replace(".", "")
        shadow_id = f"SHADOW_{bot_tag}_{symbol}_{safe_ts}"
        
        # Check if this ID was already processed (in open OR closed positions)
        if any(p.get("id") == shadow_id for p in self.closed_positions):
            return # Already closed this exact shadow trade
        if shadow_id in self.positions:
            return # Already open

        if bot_type == "STOCK":
            # Simulate a Stock Buy
            try:
                price = get_underlying_price(symbol)
            except Exception as e:
                logger.warning(f"Error fetching price for {symbol}: {e}")
                return
            if not price:
                logger.warning(f"Could not get price for shadow stock entry: {symbol}")
                return

            # Calc TP/SL
            # Default to 15m params: 1.5R TP, 1.5 ATR SL? 
            # Need ATR. Log might have it.
            # "SKIP_SLOPE3" log doesn't have ATR. We might need to fetch bars or use a default.
            # For simplicity, let's use percent based if ATR missing.
            atr = entry.get("atr")
            
            # Simple simulation parameters
            tp_pct = 0.01 # 1%
            sl_pct = 0.01 # 1%
            
            if atr:
                # Use ATR if available (rarely in skip logs)
                pass 
            
            self.positions[shadow_id] = {
                "id": shadow_id,
                "type": "STOCK",
                "bot_tag": bot_tag,
                "symbol": symbol,
                "entry_price": price,
                "entry_time": ts,
                "reason": reason,
                "qty": 100, # Mock qty
                "tp_price": price * (1 + tp_pct),
                "sl_price": price * (1 - sl_pct),
                "status": "OPEN",
                "pnl": 0.0
            }
            logger.info(f"[{bot_tag}] Opened Shadow STOCK Position: {symbol} @ {price} (Bot skipped: {reason})")
            self._append_to_csv({"event": "SHADOW_ENTRY", "bot_tag": bot_tag, "symbol": symbol, "price": price, "reason": reason, "ts": ts})

        elif bot_type == "OPTION":
            # Simulate Option Spread
            # Need to select contract
            # DTE 6 (default), Width 1.5 ATR.
            try:
                spread = select_spread_contracts(symbol, dte=6, spread_width_atr=1.5)
            except Exception as e:
                logger.warning(f"Error selecting spread for {symbol}: {e}")
                return
            if not spread:
                logger.warning(f"Could not find shadow spread for {symbol}")
                return
                
            long_leg = spread["long"]
            short_leg = spread["short"]
            
            # Get current quotes
            try:
                l_quote = get_option_quote(long_leg["symbol"])
                s_quote = get_option_quote(short_leg["symbol"])
            except Exception as e:
                logger.warning(f"Error quoting spread for {symbol}: {e}")
                return
            
            if not l_quote or not s_quote:
                logger.warning(f"Could not quote shadow spread for {symbol}")
                return
                
            debit = l_quote["ask"] - s_quote["bid"]
            
            self.positions[shadow_id] = {
                "id": shadow_id,
                "type": "SPREAD",
                "bot_tag": bot_tag,
                "symbol": symbol,
                "long_symbol": long_leg["symbol"],
                "short_symbol": short_leg["symbol"],
                "entry_debit": debit,
                "entry_time": ts,
                "reason": reason,
                "qty": 1,
                "status": "OPEN",
                "pnl": 0.0
            }
            logger.info(f"[{bot_tag}] Opened Shadow SPREAD: {symbol} Debit={debit:.2f} (Bot skipped: {reason})")
            self._append_to_csv({"event": "SHADOW_ENTRY", "bot_tag": bot_tag, "symbol": symbol, "debit": debit, "reason": reason, "ts": ts})

    def update_shadow_positions(self):
        # Check TP/SL for all open positions
        # Fetch current prices
        
        if not self.positions:
            return  # No positions to update
        
        for pid, pos in list(self.positions.items()):
            if pos["status"] != "OPEN":
                continue
                
            if pos["type"] == "STOCK":
                try:
                    curr = get_underlying_price(pos["symbol"])
                except Exception as e:
                    logger.debug(f"Error updating {pos['symbol']}: {e}")
                    continue
                if not curr: continue
                
                # Check Exit
                pnl_pct = (curr - pos["entry_price"]) / pos["entry_price"]
                pos["current_price"] = curr
                pos["unrealized_pnl"] = (curr - pos["entry_price"]) * pos["qty"]
                
                if curr >= pos["tp_price"]:
                    self.close_position(pid, curr, "TP_HIT")
                elif curr <= pos["sl_price"]:
                    self.close_position(pid, curr, "SL_HIT")
                    
            elif pos["type"] == "SPREAD":
                try:
                    l_quote = get_option_quote(pos["long_symbol"])
                    s_quote = get_option_quote(pos["short_symbol"])
                except Exception as e:
                    logger.debug(f"Error updating spread {pos['symbol']}: {e}")
                    continue
                
                if l_quote and s_quote:
                    curr_val = l_quote["bid"] - s_quote["ask"] # Sell to close (Bid on long, Ask on short)
                    pos["current_val"] = curr_val
                    pos["unrealized_pnl"] = (curr_val - pos["entry_debit"]) * 100
                    
                    # 50% profit target, -50% stop loss (approx)
                    # Use defaults from config if possible
                    pnl_pct = (curr_val - pos["entry_debit"]) / pos["entry_debit"] if pos["entry_debit"] > 0 else 0
                    
                    if pnl_pct >= 0.50:
                         self.close_position(pid, curr_val, "TP_HIT")
                    elif pnl_pct <= -0.50:
                         self.close_position(pid, curr_val, "SL_HIT")

    def close_position(self, pid, exit_price, reason):
        pos = self.positions[pid]
        pos["status"] = "CLOSED"
        pos["exit_price"] = exit_price
        pos["exit_time"] = datetime.now(timezone.utc).isoformat()
        pos["exit_reason"] = reason
        
        if pos["type"] == "STOCK":
            pnl = (exit_price - pos["entry_price"]) * pos["qty"]
        else:
            pnl = (exit_price - pos["entry_debit"]) * 100 * pos["qty"]
            
        pos["realized_pnl"] = pnl
        
        self.closed_positions.append(pos)
        del self.positions[pid]
        
        bot_tag = pos.get("bot_tag", "UNKNOWN")
        logger.info(f"[{bot_tag}] Closed Shadow {pos['type']} {pos['symbol']}: PnL=${pnl:.2f} ({reason})")
        self._append_to_csv({"event": "SHADOW_EXIT", "bot_tag": bot_tag, "symbol": pos["symbol"], "pnl": pnl, "reason": reason})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auditor Bot - Shadow Ledger")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (no actual shadow trades)")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory containing bot log files")
    parser.add_argument("--single-pass", action="store_true", help="Process logs once and exit (for GitHub Actions)")
    args = parser.parse_args()
    
    # Resolve log-dir
    log_dir = args.log_dir
    if log_dir and not os.path.isabs(log_dir):
        log_dir = os.path.join(_REPO_ROOT, log_dir)
    
    bot = AuditorBot(dry_run=args.dry_run, log_dir=log_dir, single_pass=args.single_pass)
    try:
        bot.run()
    except KeyboardInterrupt:
        logger.info("Auditor stopped.")
