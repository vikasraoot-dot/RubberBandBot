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
        # NEW: Enhanced fields for analysis
        entry_close: float = 0,       # Underlying price at entry
        kc_lower: float = 0,          # Keltner Channel lower band
        dte: int = 0,                 # Days to expiration
        long_premium: float = 0,      # Ask price of long leg
        short_premium: float = 0,     # Bid price of short leg  
        long_iv: float = 0,           # IV of long leg
        short_iv: float = 0,          # IV of short leg
        long_theta: float = 0,        # Theta of long leg
        short_theta: float = 0,       # Theta of short leg
        long_delta: float = 0,        # Delta of long leg
        short_delta: float = 0,       # Delta of short leg
        **kw
    ):
        """Log a spread entry with enhanced fields for analysis."""
        # Calculate max profit/loss for the spread
        max_loss = round(net_debit * 100 * contracts, 2)  # Total cost
        max_profit = round((spread_width - net_debit) * 100 * contracts, 2)
        
        trade = {
            "type": "SPREAD_ENTRY",
            "underlying": underlying,
            "long_symbol": long_symbol,
            "short_symbol": short_symbol,
            "atm_strike": atm_strike,
            "otm_strike": otm_strike,
            "spread_width": round(spread_width, 2),
            "net_debit": round(net_debit, 2),
            "total_cost": max_loss,
            "max_profit": max_profit,
            "max_loss": -max_loss,  # Negative to show as loss
            "contracts": contracts,
            "expiration": expiration,
            "dte": dte,
            "entry_reason": entry_reason,
            "signal_rsi": round(signal_rsi, 2),
            "signal_atr": round(signal_atr, 4),
            "entry_close": round(entry_close, 2),
            "kc_lower": round(kc_lower, 2),
            # Option-specific fields
            "long_premium": round(long_premium, 2),
            "short_premium": round(short_premium, 2),
            "long_iv": round(long_iv, 4),
            "short_iv": round(short_iv, 4),
            "long_theta": round(long_theta, 4),
            "short_theta": round(short_theta, 4),
            "long_delta": round(long_delta, 4),
            "short_delta": round(short_delta, 4),
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
        holding_minutes: int = 0,  # NEW: Trade duration in minutes
        **kw
    ):
        """Log a spread exit with duration tracking."""
        exit_ts = _ts()
        
        # Update the corresponding trade in our list
        for trade in self._trades:
            if (trade.get("underlying") == underlying and 
                trade.get("long_symbol") == long_symbol and
                trade.get("exit_ts") is None):
                trade["exit_ts"] = exit_ts
                trade["exit_reason"] = exit_reason
                trade["exit_value"] = round(exit_value, 2)
                trade["pnl"] = round(pnl, 2)
                trade["pnl_pct"] = round(pnl_pct, 1)
                trade["holding_minutes"] = holding_minutes
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
            "holding_minutes": holding_minutes,
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
    
    def export_trades_csv(self, path: str):
        """
        Export all trades to CSV format matching backtest output.
        
        Called at EOD for analysis.
        """
        import csv
        
        if not self._trades:
            print(f"[logger] No trades to export")
            return
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        # Define columns matching backtest format
        columns = [
            "symbol", "entry_time", "exit_time",
            "entry_close", "atm_strike", "otm_strike", "spread_width",
            "entry_debit", "exit_value", "cost", "max_profit", "max_loss",
            "pnl", "pnl_pct", "reason", "dte", "holding_minutes",
            "entry_rsi", "entry_atr", "kc_lower", "entry_reason",
            "long_symbol", "short_symbol",
            "long_premium", "short_premium",
            "long_iv", "short_iv", "long_theta", "short_theta",
            "long_delta", "short_delta",
        ]
        
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                
                for t in self._trades:
                    row = [
                        t.get("underlying", ""),
                        t.get("entry_ts", ""),
                        t.get("exit_ts", ""),
                        t.get("entry_close", 0),
                        t.get("atm_strike", 0),
                        t.get("otm_strike", 0),
                        t.get("spread_width", 0),
                        t.get("net_debit", 0),
                        t.get("exit_value", 0),
                        t.get("total_cost", 0),
                        t.get("max_profit", 0),
                        t.get("max_loss", 0),
                        t.get("pnl", 0),
                        t.get("pnl_pct", 0),
                        t.get("exit_reason", ""),
                        t.get("dte", 0),
                        t.get("holding_minutes", 0),
                        t.get("signal_rsi", 0),
                        t.get("signal_atr", 0),
                        t.get("kc_lower", 0),
                        t.get("entry_reason", ""),
                        t.get("long_symbol", ""),
                        t.get("short_symbol", ""),
                        t.get("long_premium", 0),
                        t.get("short_premium", 0),
                        t.get("long_iv", 0),
                        t.get("short_iv", 0),
                        t.get("long_theta", 0),
                        t.get("short_theta", 0),
                        t.get("long_delta", 0),
                        t.get("short_delta", 0),
                    ]
                    writer.writerow(row)
            
            print(f"[logger] Exported {len(self._trades)} trades to {path}")
        except Exception as e:
            print(f"[logger] Error exporting CSV: {e}")
