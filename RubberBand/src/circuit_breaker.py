from decimal import Decimal
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict

class CircuitBreakerExc(Exception):
    pass

class PortfolioGuard:
    """
    Implements drawdown protection.
    Tracks peak equity and halts if current equity drops below MAX_DRAWDOWN_PCT.
    """
    def __init__(self, state_file: str, max_drawdown_pct: float = 0.10):
        self.state_file = state_file
        self.max_drawdown_pct = Decimal(str(max_drawdown_pct))
        self.peak_equity = Decimal("0.00")
        self.halted = False
        self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.peak_equity = Decimal(str(data.get("peak_equity", 0)))
                    self.halted = data.get("halted", False)
            except Exception as e:
                print(f"[Guard] Error loading state: {e}")

    def _save_state(self):
        try:
            with open(self.state_file, "w") as f:
                json.dump({
                    "peak_equity": float(self.peak_equity),
                    "halted": self.halted,
                    "updated": datetime.now(timezone.utc).isoformat()
                }, f)
        except Exception as e:
            print(f"[Guard] Error saving state: {e}")

    def update(self, current_equity: float):
        equity = Decimal(str(current_equity))
        
        if self.halted:
            raise CircuitBreakerExc(f"TRADING HALTED: Previous Drawdown Triggered. Manual Reset Required.")

        # Update Peak
        if equity > self.peak_equity:
            self.peak_equity = equity
            self._save_state()
            
        if self.peak_equity <= 0:
            return

        # Check Drawdown
        drawdown = (self.peak_equity - equity) / self.peak_equity
        if drawdown > self.max_drawdown_pct:
            self.halted = True
            msg = f"CRITICAL: Drawdown {drawdown*100:.2f}% exceeds limit {self.max_drawdown_pct*100:.1f}%. Halting."
            self._save_state()
            raise CircuitBreakerExc(msg)

class ConnectivityGuard:
    """
    Tracks consecutive API errors and halts if threshold exceeded.
    Simple in-memory counter since we want to stop the CURRENT process loop.
    """
    def __init__(self, max_errors: int = 5):
        self.max_errors = max_errors
        self.errors = 0
        
    def record_success(self):
        self.errors = 0
        
    def record_error(self, error: Exception):
        self.errors += 1
        pct = (self.errors / self.max_errors) * 100
        print(f"[Guard] API Error {self.errors}/{self.max_errors} ({error})", flush=True)
        
        if self.errors >= self.max_errors:
            raise CircuitBreakerExc(f"CRITICAL: {self.errors} consecutive API errors. Halting for safety.")
