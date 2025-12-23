
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import os
import sys

# Ensure we can import from src
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.data import fetch_latest_bars

class RegimeManager:
    """
    Manages Dynamic Market Regimes based on VIXY (Volatility ETF) levels.
    
    Regimes (Calibrated to VIXY ETF):
    - CALM   (VIXY < 35):    Aggressive Entry (Slope -0.08%)
    - NORMAL (35 <= VIXY <= 55): Baseline Entry (Slope -0.12%)
    - PANIC  (VIXY > 55):    Defensive Entry (Slope -0.20% + DKF Enabled)
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.current_regime = "NORMAL" # Default
        self.last_vixy = 0.0
        self.last_update = None
        
        # Config Map
        self.regime_configs = {
            "CALM": {
                "description": "Low Volatility (<35). Aggressive Entry.",
                "slope_threshold_pct": -0.08,
                "dead_knife_filter": False
            },
            "NORMAL": {
                "description": "Normal Volatility (35-55). Baseline.",
                "slope_threshold_pct": -0.12,
                "dead_knife_filter": False # Optional, dependent on bot strictness
            },
            "PANIC": {
                "description": "High Volatility (>55). Defensive.",
                "slope_threshold_pct": -0.20,
                "dead_knife_filter": True
            }
        }

    def update(self) -> str:
        """
        Fetches latest VIXY data and updates the current regime.
        Returns the regime name.
        """
        try:
            # Fetch last 2 days to ensure we get a close
            bars_map, _ = fetch_latest_bars(["VIXY"], "1Day", 2, feed="iex", verbose=self.verbose)
            vixy_df = bars_map.get("VIXY")
            
            if vixy_df is None or vixy_df.empty:
                if self.verbose: 
                    print("[RegimeManager] Warning: Could not fetch VIXY. Keeping current regime.")
                return self.current_regime
                
            last_price = float(vixy_df.iloc[-1]["close"])
            self.last_vixy = last_price
            self.last_update = datetime.now()
            
            # Determine Regime
            if last_price < 35.0:
                self.current_regime = "CALM"
            elif last_price > 55.0:
                self.current_regime = "PANIC"
            else:
                self.current_regime = "NORMAL"
                
            if self.verbose:
                print(f"[RegimeManager] VIXY=${last_price:.2f} -> {self.current_regime}")
                
        except Exception as e:
            print(f"[RegimeManager] Error updating regime: {e}")
            
        return self.current_regime

    def get_config_overrides(self) -> Dict[str, Any]:
        """Returns the configuration overrides for the current regime."""
        return self.regime_configs.get(self.current_regime, self.regime_configs["NORMAL"])

if __name__ == "__main__":
    # Test
    rm = RegimeManager()
    regime = rm.update()
    config = rm.get_config_overrides()
    print(f"Current Regime: {regime}")
    print(f"Config: {config}")
