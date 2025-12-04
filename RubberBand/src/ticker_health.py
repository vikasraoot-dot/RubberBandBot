import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class TickerHealthManager:
    def __init__(self, health_file: str, config: dict):
        self.health_file = health_file
        self.config = config
        self.health_data = self._load_health()
        
        # Config values
        self.enabled = config.get("enabled", True)
        self.lookback = config.get("lookback_trades", 5)
        self.max_consecutive_losses = config.get("max_consecutive_losses", 3)
        self.drawdown_threshold = config.get("drawdown_threshold_usd", -100.0)
        self.probation_days = config.get("probation_period_days", 7)

    def _load_health(self) -> Dict:
        if os.path.exists(self.health_file):
            try:
                with open(self.health_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load ticker health: {e}")
                return {}
        return {}

    def _save_health(self):
        try:
            with open(self.health_file, "w") as f:
                json.dump(self.health_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save ticker health: {e}")

    def is_paused(self, symbol: str, now: Optional[datetime] = None) -> tuple[bool, str]:
        """
        Check if a ticker is currently paused.
        Returns (is_paused, reason)
        """
        if not self.enabled:
            return False, ""

        if now is None:
            now = datetime.now()

        data = self.health_data.get(symbol)
        if not data:
            return False, ""

        if data.get("status") == "paused":
            paused_until = data.get("paused_until")
            if paused_until:
                paused_dt = datetime.fromisoformat(paused_until)
                # Ensure timezone awareness compatibility
                if now.tzinfo is None and paused_dt.tzinfo is not None:
                     now = now.replace(tzinfo=paused_dt.tzinfo)
                elif now.tzinfo is not None and paused_dt.tzinfo is None:
                     paused_dt = paused_dt.replace(tzinfo=now.tzinfo)

                if now < paused_dt:
                    return True, f"Paused until {paused_until}"
                else:
                    # Probation over
                    self.reset_status(symbol)
                    return False, "Probation ended"
            return True, "Paused indefinitely"
        
        return False, ""

    def reset_status(self, symbol: str):
        """Reset a ticker to active status."""
        if symbol in self.health_data:
            self.health_data[symbol]["status"] = "active"
            self.health_data[symbol]["paused_until"] = None
            # We keep history to avoid immediate re-trigger if logic changes,
            # but usually we might want to clear consecutive losses?
            # Let's clear consecutive losses to give it a fresh start.
            self.health_data[symbol]["consecutive_losses"] = 0
            self._save_health()
            logger.info(f"[{symbol}] Status reset to ACTIVE.")

    def update_trade(self, symbol: str, pnl: float, trade_id: str, now: Optional[datetime] = None):
        """
        Update health with a new closed trade result.
        """
        if not self.enabled:
            return

        if now is None:
            now = datetime.now()

        if symbol not in self.health_data:
            self.health_data[symbol] = {
                "status": "active",
                "consecutive_losses": 0,
                "recent_trades": [], # List of {pnl, id}
                "paused_until": None
            }
        
        data = self.health_data[symbol]
        
        # Avoid duplicate processing
        for t in data["recent_trades"]:
            if t.get("id") == trade_id:
                return

        # Add trade
        data["recent_trades"].append({"pnl": pnl, "id": trade_id, "ts": now.isoformat()})
        # Keep only lookback
        data["recent_trades"] = data["recent_trades"][-self.lookback:]
        
        # Update metrics
        if pnl < 0:
            data["consecutive_losses"] += 1
        else:
            data["consecutive_losses"] = 0
            
        # Check Triggers
        recent_pnl_sum = sum(t["pnl"] for t in data["recent_trades"])
        
        reason = None
        if data["consecutive_losses"] >= self.max_consecutive_losses:
            reason = f"{data['consecutive_losses']} consecutive losses"
        elif recent_pnl_sum < self.drawdown_threshold:
            reason = f"Drawdown ${recent_pnl_sum:.2f} < ${self.drawdown_threshold}"
            
        if reason:
            self._pause_ticker(symbol, reason, now)
        
        self._save_health()

    def _pause_ticker(self, symbol: str, reason: str, now: datetime):
        data = self.health_data[symbol]
        data["status"] = "paused"
        until = now + timedelta(days=self.probation_days)
        data["paused_until"] = until.isoformat()
        logger.warning(f"⛔ [{symbol}] PAUSED for {self.probation_days} days. Reason: {reason}")
