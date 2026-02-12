"""
Market Condition Classifier — Phase 3A of the AI Watchdog system.

Classifies SPY market conditions into CHOPPY, TRENDING_UP, TRENDING_DOWN,
RANGE, or BREAKOUT. Produces dynamic override multipliers that live loops
read from results/watchdog/dynamic_overrides.json.

Also computes market breadth (% of universe tickers above SMA-100) to
apply a bearish sizing reduction when breadth is weak (<30%).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from RubberBand.src.data import fetch_latest_bars
from RubberBand.src.indicators import ta_add_atr, ta_add_adx_di, ta_add_sma

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OVERRIDES_PATH = os.path.join(_REPO_ROOT, "results", "watchdog", "dynamic_overrides.json")
BREADTH_PATH = os.path.join(_REPO_ROOT, "results", "watchdog", "market_breadth.json")

# Market breadth thresholds
_BREADTH_BEARISH_PCT = 30.0   # Below this: bearish sizing reduction
_BREADTH_NORMAL_PCT = 70.0    # Above this: normal sizing
_BREADTH_BEARISH_MULTIPLIER = 0.5  # Size multiplier when breadth is bearish
_BREADTH_SMA_LENGTH = 100     # SMA period for breadth calculation
_BREADTH_HISTORY_DAYS = 140   # Days of history to fetch (need 100+ for SMA)
_BREADTH_BATCH_SIZE = 30      # Symbols per API call to avoid rate limits

# Override multipliers per condition (from plan)
CONDITION_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "CHOPPY": {
        "position_size_multiplier": 0.5,
        "tp_r_multiple_adjustment": -0.5,
        "breakeven_trigger_r_adjustment": -0.5,
    },
    "TRENDING_UP": {
        "position_size_multiplier": 1.0,
        "tp_r_multiple_adjustment": 0.5,
        "breakeven_trigger_r_adjustment": 0.5,
    },
    "TRENDING_DOWN": {
        "position_size_multiplier": 0.5,
        "tp_r_multiple_adjustment": -1.0,
        "breakeven_trigger_r_adjustment": 0.0,
    },
    "RANGE": {
        "position_size_multiplier": 1.0,
        "tp_r_multiple_adjustment": 0.0,
        "breakeven_trigger_r_adjustment": 0.0,
    },
    "BREAKOUT": {
        "position_size_multiplier": 0.5,
        "tp_r_multiple_adjustment": 0.0,
        "breakeven_trigger_r_adjustment": 0.0,
    },
}


class MarketConditionClassifier:
    """
    Classifies SPY market conditions to drive dynamic parameter overrides.

    Classification rules:
        CHOPPY:        SPY 5d ATR < 20d ATR x 0.8, high bar-reversal count
        TRENDING_UP:   SPY above SMA-20, ADX > 25, +DI > -DI
        TRENDING_DOWN: SPY below SMA-20, ADX > 25, -DI > +DI
        RANGE:         SPY 5d ATR ~ 20d ATR, ADX < 20
        BREAKOUT:      SPY 5d ATR > 20d ATR x 1.5
    """

    def __init__(
        self,
        verbose: bool = False,
        universe_symbols: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the market condition classifier.

        Args:
            verbose: If True, print diagnostic output to stdout.
            universe_symbols: Ticker symbols for breadth calculation.
                If None, breadth is skipped and no sizing adjustment is applied.
        """
        self.verbose = verbose
        self.universe_symbols = universe_symbols

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self) -> Dict[str, Any]:
        """
        Fetch SPY data, compute indicators, classify condition, and write overrides.

        If ``universe_symbols`` was provided, also computes market breadth
        (% of tickers above SMA-100) and applies a bearish sizing multiplier
        when breadth drops below 30%.

        Returns:
            Dict with keys: market_condition, overrides, reason, updated_at.
            On error returns a RANGE (neutral) result so bots continue normally.
        """
        try:
            result = self._classify_internal()
        except Exception as exc:
            logger.error("MarketConditionClassifier.classify() failed: %s", exc, exc_info=True)
            if self.verbose:
                print(f"[MarketClassifier] Error: {exc}")
            result = self._neutral_result(f"Error: {exc}")

        # Compute market breadth if universe was provided
        breadth = self._compute_breadth()
        if breadth is not None:
            result["breadth"] = breadth
            # Apply bearish sizing reduction when breadth is low
            if breadth["pct_above_sma100"] < _BREADTH_BEARISH_PCT:
                current_mult = result["overrides"].get("position_size_multiplier", 1.0)
                result["overrides"]["position_size_multiplier"] = min(
                    current_mult, _BREADTH_BEARISH_MULTIPLIER,
                )
                result["overrides"]["breadth_override"] = True
                logger.info(
                    "Breadth %.1f%% < %.1f%% — sizing reduced to %.1fx",
                    breadth["pct_above_sma100"],
                    _BREADTH_BEARISH_PCT,
                    result["overrides"]["position_size_multiplier"],
                )
            self._write_breadth(breadth)

        # Persist to disk
        self._write_overrides(result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_internal(self) -> Dict[str, Any]:
        """Core classification logic — may raise on API/data errors."""
        bars_map, meta = fetch_latest_bars(
            ["SPY"],
            timeframe="1Day",
            history_days=40,
            feed="iex",
            rth_only=False,
            verbose=self.verbose,
        )
        df = bars_map.get("SPY")
        if df is None or df.empty or len(df) < 25:
            return self._neutral_result("Insufficient SPY data")

        # ---- Compute indicators ----
        df = ta_add_atr(df, length=14)
        df = ta_add_adx_di(df, period=14)
        df = ta_add_sma(df, length=20)

        # 5-day and 20-day ATR (rolling)
        tr = pd.concat([
            (df["high"] - df["low"]),
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_5 = tr.rolling(window=5, min_periods=5).mean()
        atr_20 = tr.rolling(window=20, min_periods=20).mean()

        latest = df.iloc[-1]
        atr_5_val = float(atr_5.iloc[-1]) if pd.notna(atr_5.iloc[-1]) else 0.0
        atr_20_val = float(atr_20.iloc[-1]) if pd.notna(atr_20.iloc[-1]) else 0.0
        adx_val = float(latest.get("adx", 0.0))
        pdi_val = float(latest.get("+DI", latest.get("pdi", 0.0)))
        mdi_val = float(latest.get("-DI", latest.get("mdi", 0.0)))
        sma_20_val = float(latest.get("sma_20", 0.0))
        close_val = float(latest["close"])

        # Bar-reversal count over last 20 bars
        reversal_count = self._count_reversals(df, lookback=20)

        # ATR ratio
        atr_ratio = atr_5_val / atr_20_val if atr_20_val > 0 else 1.0

        # ---- Classification (evaluated in priority order) ----
        condition, reason = self._evaluate_condition(
            close_val, sma_20_val, adx_val, pdi_val, mdi_val,
            atr_ratio, atr_5_val, atr_20_val, reversal_count,
        )

        overrides = CONDITION_OVERRIDES.get(condition, CONDITION_OVERRIDES["RANGE"])

        result: Dict[str, Any] = {
            "market_condition": condition,
            "overrides": overrides,
            "reason": reason,
            "diagnostics": {
                "spy_close": round(close_val, 2),
                "sma_20": round(sma_20_val, 2),
                "adx": round(adx_val, 1),
                "pdi": round(pdi_val, 1),
                "mdi": round(mdi_val, 1),
                "atr_5": round(atr_5_val, 2),
                "atr_20": round(atr_20_val, 2),
                "atr_ratio": round(atr_ratio, 3),
                "reversal_count_20": reversal_count,
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if self.verbose:
            print(f"\n{'='*60}")
            print(f" [MarketClassifier] Condition: {condition}")
            print(f"{'='*60}")
            print(f"  SPY Close : ${close_val:.2f}  SMA-20: ${sma_20_val:.2f}")
            print(f"  ADX       : {adx_val:.1f}  +DI: {pdi_val:.1f}  -DI: {mdi_val:.1f}")
            print(f"  ATR Ratio : {atr_ratio:.3f} (5d={atr_5_val:.2f}, 20d={atr_20_val:.2f})")
            print(f"  Reversals : {reversal_count}/20 bars")
            print(f"  Reason    : {reason}")
            print(f"  Overrides : {overrides}")
            print(f"{'='*60}\n")

        return result

    def _compute_breadth(self) -> Optional[Dict[str, Any]]:
        """Compute market breadth: % of universe tickers above SMA-100.

        Fetches daily bars for all universe symbols in batches, computes
        SMA-100 for each, and returns the percentage above.

        Returns:
            Dict with breadth stats, or None if universe_symbols is empty/unset.
        """
        if not self.universe_symbols:
            return None

        symbols = list(self.universe_symbols)
        above_count = 0
        total_valid = 0
        failed_symbols: List[str] = []

        for i in range(0, len(symbols), _BREADTH_BATCH_SIZE):
            batch = symbols[i : i + _BREADTH_BATCH_SIZE]
            try:
                bars_map, _ = fetch_latest_bars(
                    batch,
                    timeframe="1Day",
                    history_days=_BREADTH_HISTORY_DAYS,
                    feed="iex",
                    rth_only=False,
                    verbose=False,
                )
            except Exception as exc:
                logger.warning("Breadth fetch failed for batch %d: %s", i, exc)
                failed_symbols.extend(batch)
                continue

            for sym in batch:
                df = bars_map.get(sym)
                if df is None or df.empty or len(df) < _BREADTH_SMA_LENGTH:
                    failed_symbols.append(sym)
                    continue

                df = ta_add_sma(df, length=_BREADTH_SMA_LENGTH)
                sma_col = f"sma_{_BREADTH_SMA_LENGTH}"
                latest_close = float(df["close"].iloc[-1])
                latest_sma = df[sma_col].iloc[-1]

                if pd.isna(latest_sma):
                    failed_symbols.append(sym)
                    continue

                total_valid += 1
                if latest_close > float(latest_sma):
                    above_count += 1

        pct = (above_count / total_valid * 100.0) if total_valid > 0 else 50.0

        breadth_data: Dict[str, Any] = {
            "pct_above_sma100": round(pct, 1),
            "above_count": above_count,
            "total_valid": total_valid,
            "total_universe": len(symbols),
            "failed_count": len(failed_symbols),
            "regime": (
                "BEARISH" if pct < _BREADTH_BEARISH_PCT
                else "NORMAL" if pct >= _BREADTH_NORMAL_PCT
                else "CAUTIOUS"
            ),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if self.verbose:
            print(f"  Breadth   : {above_count}/{total_valid} "
                  f"({pct:.1f}%) above SMA-100 => {breadth_data['regime']}")
            if failed_symbols:
                print(f"  Breadth skipped: {len(failed_symbols)} symbols")

        return breadth_data

    @staticmethod
    def _write_breadth(breadth: Dict[str, Any]) -> None:
        """Persist breadth data to results/watchdog/market_breadth.json.

        Args:
            breadth: The breadth data dict.
        """
        try:
            os.makedirs(os.path.dirname(BREADTH_PATH), exist_ok=True)
            with open(BREADTH_PATH, "w", encoding="utf-8") as fh:
                json.dump(breadth, fh, indent=2, default=str)
            logger.info("Wrote market breadth to %s", BREADTH_PATH)
        except Exception as exc:
            logger.error("Failed to write market breadth: %s", exc)

    @staticmethod
    def _evaluate_condition(
        close: float,
        sma_20: float,
        adx: float,
        pdi: float,
        mdi: float,
        atr_ratio: float,
        atr_5: float,
        atr_20: float,
        reversal_count: int,
    ) -> tuple[str, str]:
        """
        Apply classification rules in priority order.

        Args:
            close: Latest SPY close price.
            sma_20: 20-period SMA of SPY close.
            adx: Latest ADX value.
            pdi: Latest +DI value.
            mdi: Latest -DI value.
            atr_ratio: 5d ATR / 20d ATR.
            atr_5: 5-day ATR value.
            atr_20: 20-day ATR value.
            reversal_count: Number of bar reversals in last 20 bars.

        Returns:
            Tuple of (condition_name, reason_string).
        """
        # BREAKOUT: SPY 5d ATR > 20d ATR x 1.5
        if atr_ratio > 1.5:
            return (
                "BREAKOUT",
                f"SPY 5d ATR {atr_5:.2f} > 20d ATR {atr_20:.2f} x 1.5 "
                f"(ratio {atr_ratio:.2f})",
            )

        # CHOPPY: SPY 5d ATR < 20d ATR x 0.8, high bar-reversal count (>= 12/20)
        if atr_ratio < 0.8 and reversal_count >= 12:
            return (
                "CHOPPY",
                f"SPY 5d ATR {atr_5:.2f} < 20d ATR {atr_20:.2f} x 0.8 "
                f"(ratio {atr_ratio:.2f}). {reversal_count}/20 bars reversed.",
            )

        # TRENDING_UP: SPY above SMA-20, ADX > 25, +DI > -DI
        if close > sma_20 and adx > 25 and pdi > mdi:
            return (
                "TRENDING_UP",
                f"SPY ${close:.2f} > SMA-20 ${sma_20:.2f}, ADX {adx:.1f} > 25, "
                f"+DI {pdi:.1f} > -DI {mdi:.1f}",
            )

        # TRENDING_DOWN: SPY below SMA-20, ADX > 25, -DI > +DI
        if close < sma_20 and adx > 25 and mdi > pdi:
            return (
                "TRENDING_DOWN",
                f"SPY ${close:.2f} < SMA-20 ${sma_20:.2f}, ADX {adx:.1f} > 25, "
                f"-DI {mdi:.1f} > +DI {pdi:.1f}",
            )

        # RANGE: ADX < 20 (weak trend)
        if adx < 20:
            return (
                "RANGE",
                f"ADX {adx:.1f} < 20, ATR ratio {atr_ratio:.2f} near baseline.",
            )

        # Default: RANGE
        return (
            "RANGE",
            f"No strong condition detected. ADX={adx:.1f}, ATR ratio={atr_ratio:.2f}.",
        )

    @staticmethod
    def _count_reversals(df: pd.DataFrame, lookback: int = 20) -> int:
        """
        Count bar reversals in the last *lookback* bars.

        A reversal is defined as a bar whose close-vs-open direction is
        opposite to the previous bar's close-vs-open direction.

        Args:
            df: DataFrame with 'open' and 'close' columns.
            lookback: Number of recent bars to inspect.

        Returns:
            Number of reversals detected.
        """
        if len(df) < lookback + 1:
            lookback = max(len(df) - 1, 1)

        tail = df.iloc[-(lookback + 1):]
        direction = np.sign(tail["close"].values - tail["open"].values)
        # A reversal occurs when consecutive bars have opposite signs
        # (ignoring doji bars where direction == 0)
        reversals = 0
        for i in range(1, len(direction)):
            if direction[i] != 0 and direction[i - 1] != 0:
                if direction[i] != direction[i - 1]:
                    reversals += 1
        return reversals

    @staticmethod
    def _neutral_result(reason: str) -> Dict[str, Any]:
        """
        Return a RANGE (neutral / no-op) result for fail-open behavior.

        Args:
            reason: Explanation string for why neutral was returned.

        Returns:
            Dict matching the classify() output schema.
        """
        return {
            "market_condition": "RANGE",
            "overrides": CONDITION_OVERRIDES["RANGE"],
            "reason": reason,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _write_overrides(result: Dict[str, Any]) -> None:
        """
        Persist the classification result to dynamic_overrides.json.

        Args:
            result: The classification dict to write.
        """
        try:
            os.makedirs(os.path.dirname(OVERRIDES_PATH), exist_ok=True)
            with open(OVERRIDES_PATH, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, default=str)
            logger.info("Wrote dynamic overrides to %s", OVERRIDES_PATH)
        except Exception as exc:
            logger.error("Failed to write dynamic overrides: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Public helper: read overrides (used by live loops)
# Re-exported from dynamic_overrides.py (canonical reader)
# ---------------------------------------------------------------------------

from RubberBand.src.watchdog.dynamic_overrides import read_dynamic_overrides  # noqa: F401, E402


# ---------------------------------------------------------------------------
# Public helper: read breadth (used by live loops / dashboard)
# ---------------------------------------------------------------------------

def read_market_breadth() -> Dict[str, Any]:
    """Read market breadth from results/watchdog/market_breadth.json.

    Fail-open: returns neutral defaults if file is missing or corrupt.

    Returns:
        Dict with at least 'pct_above_sma100' and 'regime' keys.
    """
    defaults: Dict[str, Any] = {
        "pct_above_sma100": 50.0,
        "regime": "CAUTIOUS",
    }
    if not os.path.isfile(BREADTH_PATH):
        return defaults
    try:
        with open(BREADTH_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if "pct_above_sma100" not in data:
            return defaults
        return data
    except (json.JSONDecodeError, OSError):
        return defaults


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Market Condition Classifier")
    parser.add_argument(
        "--tickers",
        default=os.path.join(_REPO_ROOT, "RubberBand", "tickers.txt"),
        help="Path to tickers file for breadth calculation.",
    )
    parser.add_argument(
        "--no-breadth",
        action="store_true",
        help="Skip market breadth calculation.",
    )
    args = parser.parse_args()

    universe: Optional[List[str]] = None
    if not args.no_breadth and os.path.isfile(args.tickers):
        with open(args.tickers, "r", encoding="utf-8") as fh:
            universe = [
                line.strip() for line in fh
                if line.strip() and not line.strip().startswith("#")
            ]
        print(f"[MarketClassifier] Loaded {len(universe)} tickers for breadth")

    classifier = MarketConditionClassifier(verbose=True, universe_symbols=universe)
    result = classifier.classify()
    print(json.dumps(result, indent=2, default=str))
