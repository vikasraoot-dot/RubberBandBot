"""
Market regime layer: SPY/QQQ trend, VIX volatility state and S&P 500 breadth.

All regime values at session *t* use closes up to and including *t*.  A signal
formed at the close of *t* is gated by the regime at *t* and filled at the open
of *t+1*, so there is no look-ahead.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from .features import ema, sma
from .membership import RENAMES, data_usable_for, load_membership

log = logging.getLogger(__name__)


def _trend_block(px: pd.DataFrame, prefix: str) -> pd.DataFrame:
    c = px["close"]
    out = pd.DataFrame(index=px.index)
    out[f"{prefix}_close"] = c
    out[f"{prefix}_sma50"] = sma(c, C.SMA_MID)
    out[f"{prefix}_sma200"] = sma(c, C.SMA_LONG)
    e20 = ema(c, C.EMA_MID)
    out[f"{prefix}_ema20"] = e20
    out[f"{prefix}_above_200"] = c > out[f"{prefix}_sma200"]
    out[f"{prefix}_50_gt_200"] = out[f"{prefix}_sma50"] > out[f"{prefix}_sma200"]
    out[f"{prefix}_ema20_rising"] = e20 > e20.shift(C.SLOPE_LOOKBACK)
    return out


def rolling_percentile(s: pd.Series, window: int) -> pd.Series:
    """Percent of the trailing ``window`` observations (incl. today) that are <= today's value."""
    vals = s.to_numpy()
    out = np.full(len(vals), np.nan)
    for i in range(window - 1, len(vals)):
        w = vals[i - window + 1:i + 1]
        out[i] = 100.0 * np.mean(w <= vals[i])
    return pd.Series(out, index=s.index)


def compute_breadth(sessions: pd.DatetimeIndex) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    % of point-in-time S&P 500 members above their 50/200-day SMA each session.

    Returns:
        (breadth frame with pct_above_50/pct_above_200/n_members/coverage, diagnostics dict)
    """
    mem = load_membership()
    tickers = sorted(mem.tickers_since(C.UNIVERSE_SELECTION_START))
    member_mat = mem.daily_matrix(sessions, tickers)
    above50 = pd.DataFrame(np.nan, index=sessions, columns=tickers)
    above200 = pd.DataFrame(np.nan, index=sessions, columns=tickers)
    used, rejected = [], []
    for t in tickers:
        first_member = member_mat.index[member_mat[t].to_numpy().argmax()] if member_mat[t].any() else None
        if first_member is None:
            continue
        df = None
        for sym in (RENAMES.get(t), t):
            if sym is None:
                continue
            try:
                cand = D.load_cached(sym, C.BREADTH_RAW_DIR)
            except FileNotFoundError:
                continue
            if len(cand) and data_usable_for(cand.index.min(), first_member):
                df = cand
                break
        if df is None:
            rejected.append(t)
            continue
        used.append(t)
        c = df["close"].reindex(sessions)
        s50 = df["close"].rolling(C.SMA_MID, min_periods=C.SMA_MID).mean().reindex(sessions)
        s200 = df["close"].rolling(C.SMA_LONG, min_periods=C.SMA_LONG).mean().reindex(sessions)
        above50[t] = np.where(c.notna() & s50.notna(), (c > s50).astype(float), np.nan)
        above200[t] = np.where(c.notna() & s200.notna(), (c > s200).astype(float), np.nan)
    m = member_mat.astype(bool)
    a50 = above50.where(m)
    a200 = above200.where(m)
    out = pd.DataFrame(index=sessions)
    out["pct_above_50"] = 100.0 * a50.mean(axis=1, skipna=True)
    out["pct_above_200"] = 100.0 * a200.mean(axis=1, skipna=True)
    out["breadth_n"] = a50.notna().sum(axis=1)
    out["breadth_members"] = m.sum(axis=1)
    out["breadth_coverage"] = out["breadth_n"] / out["breadth_members"].replace(0, np.nan)
    diag = {"tickers_considered": len(tickers), "tickers_used": len(used),
            "tickers_rejected_no_usable_data": len(rejected), "rejected": rejected}
    return out, diag


def build_regime_frame(with_breadth: bool = True) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Assemble the daily market-regime frame on the SPY session calendar.

    Returns:
        (regime DataFrame, diagnostics)
    """
    spy = D.load_cached("SPY", C.RAW_DIR)
    qqq = D.load_cached("QQQ", C.RAW_DIR)
    vix = D.load_cached("^VIX", C.RAW_DIR)
    reg = _trend_block(spy, "spy").join(_trend_block(qqq, "qqq"), how="left")
    reg["spy_open"] = spy["open"]
    reg["qqq_open"] = qqq["open"]
    v = vix["close"].reindex(reg.index).ffill()
    reg["vix"] = v
    reg["vix_sma20"] = sma(v, C.VIX_SMA_LEN)
    reg["vix_pct"] = rolling_percentile(v, C.VIX_PCT_WINDOW)
    reg["vix_roc5"] = v / v.shift(C.VIX_ROC_LEN) - 1.0
    reg["vix_gt_sma"] = v > reg["vix_sma20"]

    shock = (reg["vix_pct"] >= C.VIX_SHOCK_PCT) | (reg["vix_roc5"] >= C.VIX_SHOCK_ROC)
    high = ~shock & (v >= C.VIX_HIGH_LEVEL)
    low = ~shock & ~high & (reg["vix_pct"] <= C.VIX_LOW_PCT)
    reg["vol_regime"] = np.select([shock, high, low], ["SHOCK", "HIGH", "LOW"], default="NORMAL")

    bull = reg["spy_above_200"] & reg["spy_50_gt_200"]
    mixed = reg["spy_above_200"] & ~reg["spy_50_gt_200"]
    reg["market_state"] = np.select([bull, mixed], ["BULLISH", "MIXED"], default="BEARISH")

    diag: Dict[str, object] = {}
    if with_breadth:
        br, diag = compute_breadth(reg.index)
        reg = reg.join(br)
        reg["breadth_state"] = np.select(
            [reg["pct_above_50"] >= C.BREADTH_STRONG, reg["pct_above_50"] < C.BREADTH_WEAK],
            ["STRONG", "WEAK"], default="NEUTRAL")
    return reg, diag


# ──────────────────────────────────────────────────────────────────────────────
# Gate library.  Each gate maps the regime frame to a boolean "longs allowed" series.
# ──────────────────────────────────────────────────────────────────────────────
def gate_series(reg: pd.DataFrame, name: str) -> pd.Series:
    """
    Evaluate a named gate on the regime frame.

    Gate names are composed with '+', e.g. ``"spy200+vix_noshock+br50_40"``.
    """
    if name in ("none", "", None):
        return pd.Series(True, index=reg.index)
    out = pd.Series(True, index=reg.index)
    for part in name.split("+"):
        out &= _atomic_gate(reg, part)
    return out


def _atomic_gate(reg: pd.DataFrame, part: str) -> pd.Series:
    table = {
        "spy200": lambda r: r["spy_above_200"],
        "spy50_200": lambda r: r["spy_50_gt_200"],
        "spyema20": lambda r: r["spy_ema20_rising"],
        "qqq200": lambda r: r["qqq_above_200"],
        "vix_lt20": lambda r: r["vix"] < 20,
        "vix_lt25": lambda r: r["vix"] < 25,
        "vix_lt30": lambda r: r["vix"] < 30,
        "vix_pct_lt80": lambda r: r["vix_pct"] < 80,
        "vix_pct_lt90": lambda r: r["vix_pct"] < 90,
        "vix_le_sma": lambda r: ~r["vix_gt_sma"],
        "vix_roc_lt20": lambda r: r["vix_roc5"] < 0.20,
        "vix_roc_lt10": lambda r: r["vix_roc5"] < 0.10,
        "vix_noshock": lambda r: r["vol_regime"] != "SHOCK",
        "vix_calm": lambda r: r["vol_regime"].isin(["LOW", "NORMAL"]),
        "br50_40": lambda r: r["pct_above_50"] >= 40,
        "br50_50": lambda r: r["pct_above_50"] >= 50,
        "br50_60": lambda r: r["pct_above_50"] >= 60,
        "br200_50": lambda r: r["pct_above_200"] >= 50,
    }
    if part not in table:
        raise KeyError(f"unknown gate component '{part}'")
    return table[part](reg).fillna(False).astype(bool)
