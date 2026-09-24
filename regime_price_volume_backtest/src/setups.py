"""
Entry-signal definitions for the three setup families plus a location-free
"candle only" family used by the Strategy 0-5 ladder.

A signal is True at the *close* of session t; the trade (if any) is entered at
the open of t+1 by the engine.  Everything here reads only the feature frame,
which is itself free of look-ahead.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Dict, Optional

import pandas as pd

from . import config as C


@dataclass(frozen=True)
class EntrySpec:
    """Declarative description of one entry rule (all filters optional)."""

    family: str = "A"                    # A=pullback, B=failed breakdown, C=contraction breakout, CANDLE=no location
    candle: str = "any"                  # any|engulf|hammer|wrb|strong|upclose|none
    clv_min: Optional[float] = None      # override strong-close threshold (sensitivity)
    trend: str = "full"                  # none|above200|50_200|above200_50_200|full|ema9_20
    trig_vol: Optional[float] = C.TRIGGER_VOL_RATIO
    # Setup A
    pb_dist: float = C.PULLBACK_DIST
    pb_mode: str = "near"                # near|touch
    pb_vol: bool = True                  # require pullback-volume contraction
    pb_vol_max: float = C.PB_VOL_MAX_RATIO
    ext_max: Optional[float] = C.EXTENSION_MAX
    ext_atr_max: Optional[float] = None
    # Setup B
    b_level: str = "swing"               # swing|ema20
    # Setup C
    base_max_range_atr: float = C.BASE_MAX_RANGE_ATR
    atr_contraction: Optional[float] = C.ATR_CONTRACTION
    base_vol_decline: bool = True

    def label(self) -> str:
        """Compact human-readable label."""
        d = asdict(self)
        return "|".join(f"{k}={v}" for k, v in d.items())

    def with_(self, **kw: object) -> "EntrySpec":
        """Return a copy with fields replaced."""
        return replace(self, **kw)


def _trend(f: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "none":
        return pd.Series(True, index=f.index)
    if mode == "above200":
        return f["above_200"]
    if mode == "50_200":
        return f["sma50_gt_200"]
    if mode == "above200_50_200":
        return f["above_200"] & f["sma50_gt_200"]
    if mode == "full":
        return f["trend_full"]
    if mode == "ema9_20":
        return f["above_200"] & f["ema9_gt_20"]
    raise KeyError(mode)


def _candle(f: pd.DataFrame, kind: str, clv_min: Optional[float]) -> pd.Series:
    if kind == "none":
        return pd.Series(True, index=f.index)
    if kind == "upclose":
        return f["up_close"]
    if kind == "strong":
        if clv_min is None:
            return f["cdl_strong"]
        return f["bull"] & (f["clv"] >= clv_min)
    if kind == "any":
        if clv_min is None:
            return f["cdl_any"]
        strong = f["bull"] & (f["clv"] >= clv_min)
        return f["cdl_engulf"] | f["cdl_hammer"] | f["cdl_wrb"] | strong
    col = {"engulf": "cdl_engulf", "hammer": "cdl_hammer", "wrb": "cdl_wrb"}[kind]
    return f[col]


def signals(f: pd.DataFrame, spec: EntrySpec) -> pd.Series:
    """
    Evaluate ``spec`` on one stock's feature frame.

    Returns:
        Boolean Series, True on sessions whose close produces an entry signal.
    """
    ok = _trend(f, spec.trend) & _candle(f, spec.candle, spec.clv_min)
    if spec.trig_vol is not None:
        ok &= f["vol_ratio"] >= spec.trig_vol

    if spec.family == "A":
        if spec.pb_mode == "near":
            near_today = f["low_to_ema20"] <= spec.pb_dist
            loc = near_today.rolling(C.PULLBACK_WINDOW, min_periods=1).max().astype(bool)
            loc &= f["close"] > f["ema20"]
        elif spec.pb_mode == "touch":
            loc = (f["low"] <= f["ema20"]) & (f["close"] > f["ema20"])
        else:
            raise KeyError(spec.pb_mode)
        ok &= loc
        if spec.pb_vol:
            ok &= f["pb_vol_ratio"] < spec.pb_vol_max
    elif spec.family == "B":
        if spec.b_level == "swing":
            ok &= f["undercut"] & (f["close"] > f["support20"])
        elif spec.b_level == "ema20":
            ok &= f["ema20_break_recent"] & (f["close"] > f["ema20"])
        else:
            raise KeyError(spec.b_level)
        ok &= f["clv"] >= C.RECLAIM_MIN_CLV
    elif spec.family == "C":
        ok &= f["base_range_atr"] <= spec.base_max_range_atr
        if spec.atr_contraction is not None:
            ok &= f["atr_contraction"] < spec.atr_contraction
        if spec.base_vol_decline:
            ok &= f["base_vol_ratio"] < 1.0
        ok &= f["close"] > f["hh_base"]
    elif spec.family == "CANDLE":
        pass
    else:
        raise KeyError(spec.family)

    if spec.ext_max is not None:
        ok &= f["ext"] <= spec.ext_max
    if spec.ext_atr_max is not None:
        ok &= f["ext_atr"] <= spec.ext_atr_max
    # indicators must be warmed up
    ok &= f["sma200"].notna() & f["vol_avg20_prev"].notna() & f["atr_prev"].notna()
    return ok.fillna(False).astype(bool)


# A-priori baselines for each family (defined before any results were seen).
BASE_A = EntrySpec(family="A")
BASE_B = EntrySpec(family="B", b_level="swing", trend="above200", pb_vol=False, ext_max=None, candle="none")
BASE_B_EMA = EntrySpec(family="B", b_level="ema20", trend="above200", pb_vol=False, ext_max=None, candle="none")
BASE_C = EntrySpec(family="C", candle="strong", clv_min=0.70, trig_vol=C.BREAKOUT_VOL_RATIO,
                   trend="full", pb_vol=False, ext_max=None)
BASE_CANDLE = EntrySpec(family="CANDLE", candle="any", trend="none", trig_vol=None, pb_vol=False, ext_max=None)

FAMILY_BASELINES: Dict[str, EntrySpec] = {
    "A_pullback_reversal": BASE_A,
    "B_failed_breakdown_swing": BASE_B,
    "B_reclaim_ema20": BASE_B_EMA,
    "C_contraction_breakout": BASE_C,
}
