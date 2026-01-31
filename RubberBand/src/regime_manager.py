
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
    Manages Dynamic Market Regimes based on VIXY (Volatility ETF) behavior.
    
    Refactored Jan 2026: "Hybrid Dynamic Logic"
    - Uses Relative Metrics (Bollinger Bands, Daily Delta) to handle VIXY decay/splits.
    - Uses Volume Confirmation to filter noise.
    - Uses Hysteresis (3-day buffer) to prevent premature "Calm" signals.
    
    Regimes:
    - PANIC:  (Price > Upper Band OR Spike > +8%) AND (Vol > 1.5x Avg).
    - CALM:   Price < SMA_20 for 3 consecutive days.
    - NORMAL: Everything else.
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.current_regime = "NORMAL" # Default
        self.last_vixy_price = 0.0
        self.last_vixy_vol = 0
        self.last_update = None
        
        # Hysteresis Tracking
        # We need to track how many consecutive days conditions are met.
        # But since this class might be instantiated fresh daily by a script, 
        # complex state persistence is hard without a DB.
        # SOLUTION: We fetch enough history (30 days) to re-calculate the streak dynamically every time.
        
        # Config Map
        self.regime_configs = {
            "CALM": {
                "description": "Low Volatility (Trend Down). Aggressive Entry.",
                "slope_threshold_pct": -0.08,
                "dead_knife_filter": False,
                "bearish_bar_filter": False,
                "weekly_rsi_oversold": 50,
                "weekly_mean_dev_pct": -3.0
            },
            "NORMAL": {
                "description": "Normal Volatility. Baseline.",
                "slope_threshold_pct": -0.12,
                "dead_knife_filter": False,
                "bearish_bar_filter": True,
                "weekly_rsi_oversold": 45,
                "weekly_mean_dev_pct": -5.0
            },
            "PANIC": {
                "description": "High Volatility (Breakout/Spike). Defensive.",
                "slope_threshold_pct": -0.20,
                "dead_knife_filter": True,
                "bearish_bar_filter": True,
                "weekly_rsi_oversold": 30,
                "weekly_mean_dev_pct": -10.0
            }
        }

    def update(self) -> str:
        """
        Fetches latest VIXY data (30 days) and updates the current regime.
        Returns the regime name.
        """
        try:
            # Fetch 30 days to establish 20-day SMA/Bollinger Baseline + buffer
            # Use 'iex' feed as 'sip' requires subscription/permissions often missing in paper envs
            bars_map, _ = fetch_latest_bars(["VIXY"], "1Day", 35, feed="iex", verbose=self.verbose)
            df = bars_map.get("VIXY")
            
            if df is None or df.empty or len(df) < 20:
                if self.verbose: 
                    print("[RegimeManager] Warning: Insufficient VIXY data. Keeping current regime.")
                return self.current_regime

            # Calculate Indicators
            df["sma_20"] = df["close"].rolling(window=20).mean()
            df["std_20"] = df["close"].rolling(window=20).std()
            df["vol_sma_20"] = df["volume"].rolling(window=20).mean()
            df["upper_band"] = df["sma_20"] + (2.0 * df["std_20"])
            
            # Daily Delta (Percentage Change)
            # (Close - PrevClose) / PrevClose
            df["prev_close"] = df["close"].shift(1)
            df["delta_pct"] = ((df["close"] - df["prev_close"]) / df["prev_close"]) * 100.0
            
            # Get latest complete bar (Last row)
            # Note: fetch_latest_bars handles "today vs yesterday" based on market hours, 
            # assuming df.iloc[-1] is the most recent actionable bar.
            latest = df.iloc[-1]
            self.last_vixy_price = latest["close"]
            self.last_vixy_vol = latest["volume"]
            
            sma_20 = latest["sma_20"]
            upper_band = latest["upper_band"]
            avg_vol = latest["vol_sma_20"]
            delta = latest["delta_pct"]
            
            # --- LOGIC EVALUATION ---
            
            # 1. PANIC Check
            # Trigger: Breakout > UpperBand OR Massive Spike > 8%
            # Confirmation: Volume > 1.5x Avg
            is_panic_price = (latest["close"] > upper_band) or (delta > 8.0)
            is_high_volume = (latest["volume"] > 1.5 * avg_vol)
            
            if is_panic_price and is_high_volume:
                self.current_regime = "PANIC"
                reason = f"Breakout ({latest['close']:.2f} > {upper_band:.2f}) & Vol Spike"
            
            elif is_panic_price and not is_high_volume:
                 # Price is high but Volume is weak -> "Fake out" or Exhaustion?
                 # Treat as NORMAL (Suspicious)
                 self.current_regime = "NORMAL"
                 reason = "High Price but Low Volume (Trap?)"
                 
            else:
                # 2. CALM Check (Hysteresis)
                # Requirement: Price < SMA_20 for 3 consecutive days (inclusive)
                # We need to look back 3 rows.
                if len(df) >= 3:
                     subset = df.iloc[-3:]
                     all_below_sma = True
                     for idx, row in subset.iterrows():
                         if row["close"] >= row["sma_20"]:
                             all_below_sma = False
                             break
                     
                     if all_below_sma:
                         self.current_regime = "CALM"
                         reason = "Price < SMA_20 for 3 days"
                     else:
                         self.current_regime = "NORMAL"
                         reason = "Baseline"
                else:
                     self.current_regime = "NORMAL"
                     reason = "Insufficient history for Hysteresis"

            # Panic Persistence:
            # If we didn't trigger fresh Panic, but price is STILL > Upper Band, 
            # we should probably stay in Panic (or at least Normal-Defensive).
            # But the user logic said "Panic Trigger only valid if Vol > 1.5x".
            # If Vol drops, we exit Panic. This is consistent with "VSA Confirmation".
            # If Vol drops, it might be an UpThrust (reversal sign), so exiting Panic is correct.

            self.last_update = datetime.now()
                
            if self.verbose:
                p = self.regime_configs.get(self.current_regime, {})
                print("\n" + "="*60)
                print(f" [RegimeManager] Market Environment Update (Hybrid)")
                print("="*60)
                print(f"  • Date            : {latest.name.date()}")
                print(f"  • VIXY Price      : ${latest['close']:.2f} (Delta: {delta:+.2f}%)")
                print(f"  • VIXY Stats      : SMA20=${sma_20:.2f} | UpperBand=${upper_band:.2f}")
                print(f"  • VIXY Volume     : {int(latest['volume']):,} (Avg: {int(avg_vol):,})")
                print(f"  • Regime Verdict  : {self.current_regime} ({reason})")
                print("-" * 60)
                print(f"  • Slope Threshold : {p.get('slope_threshold_pct', 'N/A')}")
                print(f"  • Dead Knife Fltr : {'ENABLED' if p.get('dead_knife_filter') else 'DISABLED'}")
                print(f"  • Weekly RSI Lim  : < {p.get('weekly_rsi_oversold', 'N/A')}")
                print("="*60 + "\n")
                
        except Exception as e:
            print(f"[RegimeManager] Error updating regime: {e}")
            import traceback
            traceback.print_exc()
            
        return self.current_regime

    def get_config_overrides(self) -> Dict[str, Any]:
        """Returns the configuration overrides for the current regime."""
        return self.regime_configs.get(self.current_regime, self.regime_configs["NORMAL"])

if __name__ == "__main__":
    # Test Run
    rm = RegimeManager()
    regime = rm.update()
    config = rm.get_config_overrides()
    print(f"Final Regime: {regime}")
