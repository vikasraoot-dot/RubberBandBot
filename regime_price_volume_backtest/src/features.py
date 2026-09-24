"""
Per-stock indicators, candle geometry, volume metrics and exit conditions.

No look-ahead: every column at row *t* depends only on bars <= *t*.  Columns
named ``*_prev`` or built with ``shift(1)`` deliberately exclude the current
bar (e.g. the trigger-volume denominator is the 20 sessions *before* the
trigger so the trigger's own volume cannot inflate it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def ema(s: pd.Series, n: int) -> pd.Series:
    """Exponential moving average (recursive, alpha = 2/(n+1), seeded at the first value)."""
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    """Simple moving average with a full window required."""
    return s.rolling(n, min_periods=n).mean()


def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """True range: max(H-L, |H-C_prev|, |L-C_prev|)."""
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def wilder_atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
    """Wilder's ATR (RMA of true range, alpha = 1/n)."""
    return true_range(h, l, c).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature frame for one stock.

    Args:
        df: adjusted daily bars with open/high/low/close/volume.

    Returns:
        DataFrame (same index) with indicators, candle metrics, volume metrics,
        setup building blocks and exit-condition booleans.
    """
    o, h, l, c, v = (df[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    f = pd.DataFrame(index=df.index)
    f["open"], f["high"], f["low"], f["close"], f["volume"] = o, h, l, c, v

    # ── Trend indicators ────────────────────────────────────────────────────
    f["ema9"] = ema(c, C.EMA_FAST)
    f["ema20"] = ema(c, C.EMA_MID)
    f["sma50"] = sma(c, C.SMA_MID)
    f["sma200"] = sma(c, C.SMA_LONG)
    f["atr14"] = wilder_atr(h, l, c, C.ATR_LEN)
    f["atr_prev"] = f["atr14"].shift(1)
    tr = true_range(h, l, c)
    f["atr5"] = tr.rolling(5).mean()
    f["atr20"] = tr.rolling(20).mean()
    f["atr_pct"] = f["atr14"] / c
    f["ema20_rising"] = f["ema20"] > f["ema20"].shift(C.SLOPE_LOOKBACK)
    f["above_200"] = c > f["sma200"]
    f["sma50_gt_200"] = f["sma50"] > f["sma200"]
    f["ema9_gt_20"] = f["ema9"] > f["ema20"]
    f["trend_full"] = f["above_200"] & f["sma50_gt_200"] & f["ema20_rising"]
    f["ret_20d"] = c / c.shift(20) - 1.0

    # ── Extension ──────────────────────────────────────────────────────────
    f["ext"] = (c - f["ema20"]) / f["ema20"]
    f["ext_atr"] = (c - f["ema20"]) / f["atr14"]

    # ── Candle geometry ────────────────────────────────────────────────────
    rng = h - l
    body = (c - o).abs()
    f["range"] = rng
    f["body"] = body
    f["upper_wick"] = h - np.maximum(o, c)
    f["lower_wick"] = np.minimum(o, c) - l
    f["clv"] = np.where(rng > 0, (c - l) / rng.replace(0, np.nan), 0.5)
    f["body_frac"] = np.where(rng > 0, body / rng.replace(0, np.nan), 0.0)
    f["range_atr"] = rng / f["atr_prev"]
    f["bull"] = c > o
    f["bear"] = c < o
    o1, c1, body1 = o.shift(1), c.shift(1), body.shift(1)
    prev_bear = c1 < o1
    f["cdl_engulf"] = f["bull"] & prev_bear & (c >= o1) & (o <= c1) & (body >= body1)
    f["cdl_hammer"] = (rng > 0) & (f["lower_wick"] >= C.HAMMER_LOWER_WICK_FRAC * rng) & (f["clv"] >= C.HAMMER_MIN_CLV)
    f["cdl_wrb"] = (f["bull"] & (f["range_atr"] >= C.WRB_RANGE_ATR) & (f["clv"] >= C.WRB_MIN_CLV)
                    & (f["body_frac"] >= C.WRB_MIN_BODY_FRAC))
    f["cdl_strong"] = f["bull"] & (f["clv"] >= C.STRONG_CLOSE_CLV)
    f["cdl_any"] = f["cdl_engulf"] | f["cdl_hammer"] | f["cdl_wrb"] | f["cdl_strong"]
    f["up_close"] = c > c1          # price-only "reversal" with no candle-shape requirement

    # ── Volume ─────────────────────────────────────────────────────────────
    f["vol_avg20_prev"] = v.shift(1).rolling(C.VOL_AVG_LEN, min_periods=C.VOL_AVG_LEN).mean()
    f["vol_ratio"] = v / f["vol_avg20_prev"]
    # pullback volume: mean of the PB_VOL_DAYS sessions before the trigger vs the
    # 20 sessions before those (both windows exclude the trigger bar)
    pb_num = v.shift(1).rolling(C.PB_VOL_DAYS).mean()
    pb_den = v.shift(1 + C.PB_VOL_DAYS).rolling(C.VOL_AVG_LEN).mean()
    f["pb_vol_ratio"] = pb_num / pb_den
    f["dollar_vol"] = c * v

    # ── Pullback location building blocks ──────────────────────────────────
    f["low_to_ema20"] = l / f["ema20"] - 1.0           # <= d means the low came within d of EMA20
    f["close_to_ema20"] = c / f["ema20"] - 1.0
    hh10_prev = h.shift(1).rolling(10).max()
    f["pb_depth"] = (hh10_prev - l.rolling(C.PULLBACK_WINDOW).min()) / hh10_prev
    f["pb_depth_atr"] = (hh10_prev - l.rolling(C.PULLBACK_WINDOW).min()) / f["atr14"]

    # ── Setup B building blocks (failed breakdown) ─────────────────────────
    f["support20"] = l.shift(C.UNDERCUT_WINDOW).rolling(C.SUPPORT_LOOKBACK).min()
    f["undercut"] = l.rolling(C.UNDERCUT_WINDOW).min() < f["support20"]
    f["ema20_break_recent"] = (c.shift(1) < f["ema20"].shift(1)) | (c.shift(2) < f["ema20"].shift(2)) | (l < f["ema20"])

    # ── Setup C building blocks (contraction breakout) ─────────────────────
    f["hh_base"] = h.shift(1).rolling(C.BASE_LEN).max()
    f["ll_base"] = l.shift(1).rolling(C.BASE_LEN).min()
    f["base_range_atr"] = (f["hh_base"] - f["ll_base"]) / f["atr14"].shift(C.BASE_LEN + 1)
    f["atr_contraction"] = (f["atr5"] / f["atr20"]).shift(1)
    f["base_vol_ratio"] = v.shift(1).rolling(C.BASE_LEN).mean() / v.shift(C.BASE_LEN + 1).rolling(50).mean()

    # ── Structure for stops/exits ──────────────────────────────────────────
    f["swing_low_prev"] = l.shift(1).rolling(C.SWING_LOOKBACK).min()
    f["ll5_prev"] = l.shift(1).rolling(5).min()

    # ── Exit conditions (evaluated at the close; exit fills next open) ─────
    below9 = c < f["ema9"]
    below20 = c < f["ema20"]
    bear_hv = f["bear"] & (f["vol_ratio"] >= 1.2)
    distribution = below9 & f["bear"] & (f["clv"] <= 0.30) & (f["vol_ratio"] >= 1.2)
    f["x_A_ema9"] = below9
    f["x_A2_2xema9"] = below9 & below9.shift(1, fill_value=False)
    f["x_W_ema9_warn"] = below9 & (below9.shift(1, fill_value=False) | bear_hv | below20 | (c < f["ll5_prev"]))
    f["x_B_ema20"] = below20
    f["x_C_cross"] = (f["ema9"] < f["ema20"]) & (f["ema9"].shift(1) >= f["ema20"].shift(1))
    f["x_D_swing"] = c < f["swing_low_prev"]
    f["x_E_pv"] = below20 | distribution
    f["x_none"] = pd.Series(False, index=f.index)   # used with trailing-stop-only / fixed-horizon exits
    return f


EXIT_COLUMNS = {
    "A_close<EMA9": "x_A_ema9",
    "A2_2closes<EMA9": "x_A2_2xema9",
    "W_EMA9warn+confirm": "x_W_ema9_warn",
    "B_close<EMA20": "x_B_ema20",
    "C_EMA9xEMA20": "x_C_cross",
    "D_swinglow": "x_D_swing",
    "E_pricevol": "x_E_pv",
    "F_chandelier": "x_none",
    "T10_time": "x_none",
}
