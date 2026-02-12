
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import os
import sys

logger = logging.getLogger(__name__)

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

        # Intraday monitoring reference values (set by daily update())
        self._reference_close = None  # Yesterday's close for intraday delta calc
        self._upper_band = None       # Upper Bollinger Band for breakout detection
        self._sma_20 = None           # SMA for reference
        self._intraday_panic = False  # Track if we triggered intraday panic
        
        # Hysteresis Tracking
        # We need to track how many consecutive days conditions are met.
        # But since this class might be instantiated fresh daily by a script, 
        # complex state persistence is hard without a DB.
        # SOLUTION: We fetch enough history (30 days) to re-calculate the streak dynamically every time.
        
        # Config Map
        self.regime_configs = {
            "CALM": {
                "description": "Low Volatility (Trend Down). Aggressive Entry.",
                "slope_threshold_pct": -0.20,
                "dead_knife_filter": False,
                "bearish_bar_filter": False,
                "weekly_rsi_oversold": 50,
                "weekly_mean_dev_pct": -3.0
            },
            "NORMAL": {
                "description": "Normal Volatility. Baseline.",
                "slope_threshold_pct": -0.20,
                "dead_knife_filter": False,
                "bearish_bar_filter": False,
                "weekly_rsi_oversold": 45,
                "weekly_mean_dev_pct": -5.0
            },
            "PANIC": {
                "description": "High Volatility (Breakout/Spike). Defensive.",
                "slope_threshold_pct": -0.20,
                "dead_knife_filter": True,
                "bearish_bar_filter": False,
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
            # SIP = consolidated tape from all exchanges (free-tier: 15-min delay, handled by fetch_latest_bars)
            bars_map, _ = fetch_latest_bars(["VIXY"], "1Day", 35, feed="sip", verbose=self.verbose)
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

            # Store reference values for intraday monitoring
            self._reference_close = latest["close"]
            self._upper_band = upper_band
            self._sma_20 = sma_20
            self._intraday_panic = False  # Reset on daily update
            
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
            logger.error(f"[RegimeManager] Error updating regime: {e}", exc_info=True)
            if self.verbose:
                print(f"[RegimeManager] Error updating regime: {e}")

        return self.current_regime

    def get_config_overrides(self) -> Dict[str, Any]:
        """Returns the configuration overrides for the current regime."""
        return self.regime_configs.get(self.current_regime, self.regime_configs["NORMAL"])

    def check_intraday(self) -> str:
        """
        Check for intraday volatility spikes that should trigger PANIC.

        This method fetches the current VIXY price and compares it to the
        reference close from the daily update(). If VIXY spikes > 8% intraday
        OR breaks above the upper Bollinger band, we return PANIC.

        Returns:
            Current effective regime (may be PANIC even if daily regime is NORMAL/CALM)

        Note:
            - Must call update() at least once before this method works
            - Does NOT require volume confirmation for intraday (speed matters)
            - Once intraday PANIC triggers, it persists until next daily update()
        """
        # If we already triggered intraday panic, stay in panic
        if self._intraday_panic:
            return "PANIC"

        # If no reference values yet, return current regime
        if self._reference_close is None or self._upper_band is None:
            if self.verbose:
                print("[RegimeManager] No reference values - call update() first")
            return self.current_regime

        try:
            # Fetch latest VIXY price (1 bar of 5-minute data for speed)
            bars_map, _ = fetch_latest_bars(
                ["VIXY"], "5Min", 1, feed="iex", verbose=False  # IEX for real-time intraday panic detection (SIP has 15-min delay)
            )
            df = bars_map.get("VIXY")

            if df is None or df.empty:
                # Can't get data, return current regime
                return self.current_regime

            current_price = df.iloc[-1]["close"]

            # Calculate intraday delta vs reference (yesterday's close)
            intraday_delta = ((current_price - self._reference_close) / self._reference_close) * 100.0

            # Check PANIC conditions (no volume confirmation for speed)
            is_spike = intraday_delta > 8.0
            is_breakout = current_price > self._upper_band

            if is_spike or is_breakout:
                self._intraday_panic = True
                reason = []
                if is_spike:
                    reason.append(f"Intraday spike +{intraday_delta:.1f}%")
                if is_breakout:
                    reason.append(f"Breakout ${current_price:.2f} > ${self._upper_band:.2f}")

                if self.verbose:
                    print("\n" + "!" * 60)
                    print(" [RegimeManager] INTRADAY PANIC TRIGGERED!")
                    print("!" * 60)
                    print(f"  • VIXY Current    : ${current_price:.2f}")
                    print(f"  • Reference Close : ${self._reference_close:.2f}")
                    print(f"  • Intraday Delta  : {intraday_delta:+.2f}%")
                    print(f"  • Upper Band      : ${self._upper_band:.2f}")
                    print(f"  • Trigger         : {', '.join(reason)}")
                    print("!" * 60 + "\n")

                return "PANIC"

            # No intraday panic, return daily regime
            return self.current_regime

        except Exception as e:
            # Always log errors (per CLAUDE.md Section 2.3 - never swallow silently)
            logger.error(f"[RegimeManager] Error in intraday check: {e}", exc_info=True)
            if self.verbose:
                print(f"[RegimeManager] Error in intraday check: {e}")
            # On error, return current regime (fail-safe)
            return self.current_regime

    def get_effective_regime(self) -> str:
        """
        Get the effective regime considering both daily and intraday conditions.

        This is the recommended method for live trading loops to call.
        It returns PANIC if intraday conditions triggered, otherwise daily regime.

        Returns:
            Effective regime name (CALM, NORMAL, or PANIC)
        """
        return self.check_intraday()

    def classify_market_condition(self) -> Dict[str, Any]:
        """
        Classify current market condition into finer-grained categories
        (CHOPPY, TRENDING_UP, TRENDING_DOWN, RANGE, BREAKOUT) using SPY data.

        This is additive to the existing PANIC/NORMAL/CALM regime logic.
        Delegates to MarketConditionClassifier.

        Returns:
            Dict with market_condition, overrides, reason, updated_at.
            Returns neutral RANGE result on import or classification errors.
        """
        try:
            from RubberBand.src.watchdog.market_classifier import MarketConditionClassifier
            classifier = MarketConditionClassifier(verbose=self.verbose)
            return classifier.classify()
        except Exception as e:
            logger.error("[RegimeManager] classify_market_condition failed: %s", e, exc_info=True)
            if self.verbose:
                print(f"[RegimeManager] classify_market_condition error: {e}")
            return {
                "market_condition": "RANGE",
                "overrides": {
                    "position_size_multiplier": 1.0,
                    "tp_r_multiple_adjustment": 0.0,
                    "breakeven_trigger_r_adjustment": 0.0,
                },
                "reason": f"Error: {e}",
            }


# ──────────────────────────────────────────────────────────────────────────────
# Shared Backtest Logic (Refactored from script)
# ──────────────────────────────────────────────────────────────────────────────
def calculate_regime_map(df: pd.DataFrame) -> Dict[Any, str]:
    """
    Calculate regime based on VIXY history using production logic (Hybrid Dynamic).
    Returns a map of Date -> Regime Name (Effective for that trade date).
    """
    if df is None or df.empty or len(df) < 20:
        return {}
    
    # Copy to avoid modifying original
    df = df.copy()
    
    # Calculate Indicators
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["std_20"] = df["close"].rolling(window=20).std()
    df["vol_sma_20"] = df["volume"].rolling(window=20).mean()
    df["upper_band"] = df["sma_20"] + (2.0 * df["std_20"])
    
    df["prev_close"] = df["close"].shift(1)
    df["delta_pct"] = ((df["close"] - df["prev_close"]) / df["prev_close"]) * 100.0
    
    # Iterate to determine regime based on EACH DAY'S close
    regime_map = {}
    below_sma_streak = 0
    
    # Pre-calculate conditions to speed up loop
    is_panic_price = (df["close"] > df["upper_band"]) | (df["delta_pct"] > 8.0)
    is_high_vol = (df["volume"] > 1.5 * df["vol_sma_20"])
    closes = df["close"].values
    smas = df["sma_20"].values
    
    # Extract dates carefully
    if hasattr(df.index, 'date'):        dates = df.index.date
    else:                               dates = [pd.to_datetime(d).date() for d in df.index]

    for i in range(len(df)):
        # Skip if indicators not ready (first 20 bars)
        if pd.isna(smas[i]):
            regime_map[dates[i]] = "NORMAL"
            continue
            
        panic = is_panic_price.iloc[i]
        vol = is_high_vol.iloc[i]
        
        row_regime = "NORMAL"
        
        if panic and vol:
            row_regime = "PANIC"
            below_sma_streak = 0
        elif panic and not vol:
            # Fakeout -> Normal
            row_regime = "NORMAL"
            if closes[i] < smas[i]:
                 below_sma_streak += 1
            else:
                 below_sma_streak = 0
        else:
            # Check CALM
            if closes[i] < smas[i]:
                below_sma_streak += 1
            else:
                below_sma_streak = 0
            
            if below_sma_streak >= 3:
                row_regime = "CALM"
                
        regime_map[dates[i]] = row_regime
        
    # shift regime map by 1 day
    # We want: Map[Date T] = Regime based on Date T-1
    # Current Map[Date T] = Regime based on Date T
    # Convert keys to sorted list
    sorted_dates = sorted(list(regime_map.keys()))
    shifted_map = {}
    
    # Init first day as NORMAL (no prior day)
    if sorted_dates:
        shifted_map[sorted_dates[0]] = "NORMAL"
        
    for i in range(1, len(sorted_dates)):
        curr_date = sorted_dates[i]
        prev_date = sorted_dates[i-1]
        # Regime for Curr Date is determined by Prev Date's close
        shifted_map[curr_date] = regime_map[prev_date]
        
    return shifted_map


if __name__ == "__main__":
    # Test Run
    rm = RegimeManager()
    regime = rm.update()
    config = rm.get_config_overrides()
    print(f"Final Regime: {regime}")
