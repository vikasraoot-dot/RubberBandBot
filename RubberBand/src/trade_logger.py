"""
Trade Logger: JSONL logging for stock trades with comprehensive details.
Includes entry/exit reasons, P&L tracking, and EOD summaries.
"""
from __future__ import annotations
import os
import json
import threading
import datetime as dt
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("US/Eastern")
_ISO = "%Y-%m-%dT%H:%M:%SZ"


def _ts() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime(_ISO)


def _ts_et() -> str:
    return dt.datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")


def _date_et() -> str:
    """Return current date in YYYY-MM-DD format (Eastern time)."""
    return dt.datetime.now(ET).strftime("%Y-%m-%d")


def _time_et() -> str:
    """Return current time in HH:MM:SS format (Eastern time)."""
    return dt.datetime.now(ET).strftime("%H:%M:%S")


class TradeLogger:
    """
    Line-buffered JSONL logger with a compact, auditable schema.
    One event per line. Designed to be safe to call from multiple places.

    Enhancement: mirror each JSONL line to stdout so CI (GitHub Actions) shows
    live progress, without changing the on-disk audit format.
    
    Tracks trades for EOD summary with entry/exit reasons.
    """
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # line-buffered
        self._fp = open(path, "a", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()
        self._trades: List[Dict[str, Any]] = []  # Track all trades for EOD summary

    def _write(self, obj: Dict[str, Any]):
        obj.setdefault("ts", _ts())
        obj.setdefault("ts_et", _ts_et())
        line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False, default=str)
        with self._lock:
            # Always try to write the file log first (audit source of truth)
            try:
                self._fp.write(line + "\n")
            except Exception:
                # Never let logging crash trading; still try to emit to console
                pass
            # Mirror to console so it appears in GitHub Actions live logs
            try:
                print(line, flush=True)
            except Exception:
                # Ignore console failures to keep trading resilient
                pass

    # ---- Emitters (normalized "type" values) ----
    def heartbeat(self, **kw): kw.setdefault("type","HEARTBEAT"); self._write(kw)
    def signal(self, **kw): kw.setdefault("type","SIGNAL"); self._write(kw)
    def gate(self, **kw): kw.setdefault("type","GATE"); self._write(kw)

    def entry_submit(self, **kw):
        kw.setdefault("type", "ENTRY_SUBMIT")
        # Track trade for EOD summary
        trade = {
            "symbol": kw.get("symbol"),
            "side": kw.get("side"),
            "qty": kw.get("qty"),
            "entry_price": kw.get("entry_price"),
            "stop_loss": kw.get("stop_loss_price"),
            "take_profit": kw.get("take_profit_price"),
            "entry_reason": kw.get("entry_reason", "RubberBand_signal"),
            "entry_ts": _ts(),
            "entry_date": _date_et(),        # YYYY-MM-DD for filtering
            "entry_time_et": _time_et(),     # HH:MM:SS Eastern
            "exit_ts": None,
            "exit_date": None,
            "exit_time_et": None,
            "exit_reason": None,
            "exit_price": None,
            "pnl": None,
        }
        self._trades.append(trade)
        self._write(kw)

    def entry_ack(self, **kw): kw.setdefault("type","ENTRY_ACK"); self._write(kw)
    def entry_reject(self, **kw): kw.setdefault("type","ENTRY_REJECT"); self._write(kw)
    def entry_fill(self, **kw): kw.setdefault("type","ENTRY_FILL"); self._write(kw)

    def oco_submit(self, **kw): kw.setdefault("type","OCO_SUBMIT"); self._write(kw)
    def oco_ack(self, **kw): kw.setdefault("type","OCO_ACK"); self._write(kw)
    
    def exit_fill(self, **kw):
        kw.setdefault("type", "EXIT_FILL")
        # Update corresponding trade
        symbol = kw.get("symbol")
        exit_reason = kw.get("exit_reason", "Unknown")
        exit_price = kw.get("exit_price", 0)
        pnl = kw.get("pnl", 0)
        
        for trade in self._trades:
            if trade.get("symbol") == symbol and trade.get("exit_ts") is None:
                trade["exit_ts"] = _ts()
                trade["exit_date"] = _date_et()
                trade["exit_time_et"] = _time_et()
                trade["exit_reason"] = exit_reason
                trade["exit_price"] = exit_price
                trade["pnl"] = pnl
                break
        self._write(kw)

    def cancel(self, **kw): kw.setdefault("type","CANCEL"); self._write(kw)

    def error(self, **kw): kw.setdefault("type","ERROR"); self._write(kw)
    def snapshot(self, **kw): kw.setdefault("type","SNAPSHOT"); self._write(kw)

    def eod_summary(self, total_pnl: float = 0, total_vol: float = 0, **kw):
        """Generate and log end-of-day summary with all trades."""
        closed_trades = [t for t in self._trades if t.get("exit_ts") is not None]
        open_trades = [t for t in self._trades if t.get("exit_ts") is None]
        
        wins = [t for t in closed_trades if (t.get("pnl", 0) or 0) > 0]
        losses = [t for t in closed_trades if (t.get("pnl", 0) or 0) < 0]
        
        calc_pnl = sum(t.get("pnl", 0) or 0 for t in closed_trades)
        win_rate = len(wins) / len(closed_trades) * 100 if closed_trades else 0
        avg_win = sum(t.get("pnl", 0) or 0 for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.get("pnl", 0) or 0 for t in losses) / len(losses) if losses else 0
        
        # Exit reason breakdown
        exit_reasons = {}
        for t in closed_trades:
            reason = t.get("exit_reason", "Unknown")
            if reason not in exit_reasons:
                exit_reasons[reason] = {"count": 0, "pnl": 0}
            exit_reasons[reason]["count"] += 1
            exit_reasons[reason]["pnl"] += t.get("pnl", 0) or 0
        
        summary = {
            "type": "EOD_SUMMARY",
            "date": dt.datetime.now(ET).strftime("%Y-%m-%d"),
            "total_trades": len(self._trades),
            "closed_trades": len(closed_trades),
            "open_trades": len(open_trades),
            "total_pnl": round(total_pnl if total_pnl else calc_pnl, 2),
            "total_volume": round(total_vol, 2),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate_pct": round(win_rate, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "exit_reasons": exit_reasons,
            "trades": [
                {
                    "symbol": t.get("symbol"),
                    "side": t.get("side"),
                    "qty": t.get("qty"),
                    "entry_price": t.get("entry_price"),
                    "exit_price": t.get("exit_price"),
                    "entry_reason": t.get("entry_reason"),
                    "exit_reason": t.get("exit_reason"),
                    "pnl": t.get("pnl"),
                }
                for t in self._trades
            ],
            **kw
        }
        
        self._write(summary)
        return summary

    def close(self):
        try:
            self._fp.close()
        except Exception:
            pass
    
    def export_trades_csv(self, path: str):
        """
        Export all trades to CSV format for post-analysis.
        
        Called at EOD for analysis similar to backtest output.
        """
        import csv
        
        if not self._trades:
            print(f"[logger] No trades to export")
            return
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        # Define columns for stock trades
        columns = [
            "symbol", "side", "entry_time", "exit_time",
            "entry_price", "exit_price", "qty",
            "stop_loss", "take_profit",
            "entry_reason", "exit_reason", "pnl",
        ]
        
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                
                for t in self._trades:
                    row = [
                        t.get("symbol", ""),
                        t.get("side", ""),
                        t.get("entry_ts", ""),
                        t.get("exit_ts", ""),
                        t.get("entry_price", 0),
                        t.get("exit_price", 0),
                        t.get("qty", 0),
                        t.get("stop_loss", 0),
                        t.get("take_profit", 0),
                        t.get("entry_reason", ""),
                        t.get("exit_reason", ""),
                        t.get("pnl", 0),
                    ]
                    writer.writerow(row)
            
            print(f"[logger] Exported {len(self._trades)} trades to {path}")
        except Exception as e:
            print(f"[logger] Error exporting CSV: {e}")
