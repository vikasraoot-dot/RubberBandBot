"""
Options Trade Logger: JSONL logging for options spread trades with comprehensive details.
Includes entry/exit reasons, spread details, P&L tracking, and EOD summaries.
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


class OptionsTradeLogger:
    """
    Line-buffered JSONL logger for options spread trades.
    
    Logs:
    - SPREAD_SIGNAL: When a spread opportunity is identified
    - SPREAD_ENTRY: When a spread order is submitted
    - SPREAD_FILL: When spread legs are filled
    - SPREAD_EXIT: When a spread is closed (with reason)
    - SPREAD_SKIP: When a spread is skipped (with reason)
    - EOD_SUMMARY: End of day summary with all trades and P&L
    """
    
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fp = open(path, "a", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()
        self._trades: List[Dict[str, Any]] = []  # Track all trades for EOD summary
    
    def _write(self, obj: Dict[str, Any]):
        obj.setdefault("ts", _ts())
        obj.setdefault("ts_et", _ts_et())
        line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False, default=str)
        with self._lock:
            try:
                self._fp.write(line + "\n")
            except Exception:
                pass
            try:
                print(line, flush=True)
            except Exception:
                pass
    
    # ──────────────────────────────────────────────────────────────────────────
    # Signal Events
    # ──────────────────────────────────────────────────────────────────────────
    def spread_signal(
        self,
        underlying: str,
        signal_reason: str,
        entry_price: float,
        rsi: float = 0,
        atr: float = 0,
        **kw
    ):
        """Log when a spread signal is generated."""
        self._write({
            "type": "SPREAD_SIGNAL",
            "underlying": underlying,
            "signal_reason": signal_reason,
            "entry_price": entry_price,
            "rsi": round(rsi, 2),
            "atr": round(atr, 4),
            **kw
        })
    
    def spread_skip(
        self,
        underlying: str,
        skip_reason: str,
        **kw
    ):
        """Log when a spread is skipped."""
        self._write({
            "type": "SPREAD_SKIP",
            "underlying": underlying,
            "skip_reason": skip_reason,
            **kw
        })
    
    # ──────────────────────────────────────────────────────────────────────────
    # Entry Events
    # ──────────────────────────────────────────────────────────────────────────
    def spread_entry(
        self,
        underlying: str,
        long_symbol: str,
        short_symbol: str,
        atm_strike: float,
        otm_strike: float,
        spread_width: float,
        net_debit: float,
        contracts: int,
        expiration: str,
        entry_reason: str,
        signal_rsi: float = 0,
        signal_atr: float = 0,
        **kw
    ):
        """Log a spread entry."""
        trade = {
            "type": "SPREAD_ENTRY",
            "underlying": underlying,
            "long_symbol": long_symbol,
            "short_symbol": short_symbol,
            "atm_strike": atm_strike,
            "otm_strike": otm_strike,
            "spread_width": spread_width,
            "net_debit": round(net_debit, 2),
            "total_cost": round(net_debit * 100 * contracts, 2),
            "contracts": contracts,
            "expiration": expiration,
            "entry_reason": entry_reason,
            "signal_rsi": round(signal_rsi, 2),
            "signal_atr": round(signal_atr, 4),
            "entry_ts": _ts(),
            "exit_ts": None,
            "exit_reason": None,
            "exit_value": None,
            "pnl": None,
            **kw
        }
        self._trades.append(trade)
        self._write(trade)
    
    # ──────────────────────────────────────────────────────────────────────────
    # Exit Events
    # ──────────────────────────────────────────────────────────────────────────
    def spread_exit(
        self,
        underlying: str,
        long_symbol: str,
        short_symbol: str,
        exit_value: float,
        exit_reason: str,
        pnl: float,
        pnl_pct: float,
        **kw
    ):
        """Log a spread exit."""
        # Update the corresponding trade in our list
        for trade in self._trades:
            if (trade.get("underlying") == underlying and 
                trade.get("long_symbol") == long_symbol and
                trade.get("exit_ts") is None):
                trade["exit_ts"] = _ts()
                trade["exit_reason"] = exit_reason
                trade["exit_value"] = round(exit_value, 2)
                trade["pnl"] = round(pnl, 2)
                trade["pnl_pct"] = round(pnl_pct, 1)
                break
        
        self._write({
            "type": "SPREAD_EXIT",
            "underlying": underlying,
            "long_symbol": long_symbol,
            "short_symbol": short_symbol,
            "exit_value": round(exit_value, 2),
            "exit_reason": exit_reason,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 1),
            **kw
        })
    
    # ──────────────────────────────────────────────────────────────────────────
    # EOD Summary
    # ──────────────────────────────────────────────────────────────────────────
    def eod_summary(self, **kw):
        """Generate and log end-of-day summary."""
        closed_trades = [t for t in self._trades if t.get("exit_ts") is not None]
        open_trades = [t for t in self._trades if t.get("exit_ts") is None]
        
        total_pnl = sum(t.get("pnl", 0) or 0 for t in closed_trades)
        wins = [t for t in closed_trades if (t.get("pnl", 0) or 0) > 0]
        losses = [t for t in closed_trades if (t.get("pnl", 0) or 0) < 0]
        
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
            "total_pnl": round(total_pnl, 2),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate_pct": round(win_rate, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "exit_reasons": exit_reasons,
            "trades": [
                {
                    "underlying": t.get("underlying"),
                    "entry_reason": t.get("entry_reason"),
                    "exit_reason": t.get("exit_reason"),
                    "net_debit": t.get("net_debit"),
                    "pnl": t.get("pnl"),
                    "pnl_pct": t.get("pnl_pct"),
                }
                for t in self._trades
            ],
            **kw
        }
        
        self._write(summary)
        return summary
    
    def heartbeat(self, **kw):
        """Log a heartbeat."""
        kw.setdefault("type", "HEARTBEAT")
        self._write(kw)
    
    def error(self, **kw):
        """Log an error."""
        kw.setdefault("type", "ERROR")
        self._write(kw)
    
    def close(self):
        try:
            self._fp.close()
        except Exception:
            pass
