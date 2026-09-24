"""
Step 6: independent manual audit of sample trades (Section 59).

For a reproducible sample of FINAL-model trades this script recomputes, with
plain-Python loops working directly from the cached bars (not the pandas
feature pipeline), every quantity that determined the trade:

* EMA9/EMA20 (recursive), SMA50/SMA200, Wilder ATR14
* stock-trend and pullback qualification on the signal day
* market regime at the signal date: SPY vs SMA200 and SMA50, VIX level,
  % of point-in-time S&P members above their 50-day SMA (breadth)
* entry date = next session after the signal, entry price = that open
* initial stop = entry - 1.5 x ATR14(signal day)
* exit: first close < EMA20 (filled next open) or intraday stop
* MFE / MAE / peak close / giveback

and compares them with the engine's trades.csv row.  Writes qc/audit_report.md.

Run:  python -m src.audit
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from .membership import RENAMES, data_usable_for, load_membership


def ema_loop(vals: List[float], n: int) -> List[float]:
    """Recursive EMA seeded with the first value; NaN until n values seen (matches min_periods)."""
    a = 2.0 / (n + 1)
    out, e = [], None
    for i, v in enumerate(vals):
        e = v if e is None else a * v + (1 - a) * e
        out.append(e if i >= n - 1 else math.nan)
    return out


def sma_at(vals: List[float], i: int, n: int) -> float:
    """Simple average of the n values ending at i."""
    return sum(vals[i - n + 1:i + 1]) / n if i >= n - 1 else math.nan


def atr_loop(h: List[float], l: List[float], c: List[float], n: int) -> List[float]:
    """Wilder ATR as an RMA of true range seeded with the first TR (matches ewm(alpha=1/n, adjust=False))."""
    out, a, prev = [], 1.0 / n, None
    for i in range(len(c)):
        tr = h[i] - l[i] if i == 0 else max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        prev = tr if prev is None else a * tr + (1 - a) * prev
        out.append(prev if i >= n - 1 else math.nan)
    return out


_FIRST_MEMBER: Dict[str, pd.Timestamp] = {}


def _first_member_dates() -> Dict[str, pd.Timestamp]:
    """
    First date each ticker is a member within the study window (from the regime
    frame's first session, 2015-01-02), computed once.  Membership spells before
    that are ignored so a ticker reused by a new company (e.g. CEG: Constellation
    Energy Group 1996-2012, Constellation Energy spin-off 2022) is judged on its
    current spell -- the same convention as ``regime.compute_breadth``.
    """
    if not _FIRST_MEMBER:
        start = pd.Timestamp(C.MARKET_DOWNLOAD_START)
        tbl = load_membership().table
        in_force = tbl[tbl["date"] <= start]
        if len(in_force):
            for t in in_force.iloc[-1]["members"]:
                _FIRST_MEMBER[t] = start
        for d, mem in tbl[tbl["date"] > start].itertuples(index=False):
            for t in mem:
                _FIRST_MEMBER.setdefault(t, d)
    return _FIRST_MEMBER


def breadth_on(date: pd.Timestamp) -> float:
    """% of members on ``date`` (with usable data) whose close is above their 50-day SMA."""
    mem = load_membership()
    members = mem.members_on(date)
    first = _first_member_dates()
    above, n = 0, 0
    for t in members:
        df = None
        for sym in (RENAMES.get(t), t):
            if sym is None:
                continue
            try:
                cand = D.load_cached(sym, C.BREADTH_RAW_DIR)
            except FileNotFoundError:
                continue
            if len(cand) and data_usable_for(cand.index.min(), first[t]):
                df = cand
                break
        if df is None or date not in df.index:
            continue
        closes = df.loc[:date, "close"].tolist()
        if len(closes) < 50:
            continue
        n += 1
        above += closes[-1] > sum(closes[-50:]) / 50
    return 100.0 * above / n if n else math.nan


def audit_trade(row: pd.Series, cost: float) -> Dict[str, object]:
    """Recompute one trade from raw bars."""
    df = D.load_cached(row["ticker"], C.RAW_DIR)
    dates = list(df.index)
    o, h, l, c = (df[k].tolist() for k in ("open", "high", "low", "close"))
    e9, e20 = ema_loop(c, 9), ema_loop(c, 20)
    atr = atr_loop(h, l, c, 14)
    s = dates.index(pd.Timestamp(row["signal_date"]))
    trend = (c[s] > sma_at(c, s, 200)) and (sma_at(c, s, 50) > sma_at(c, s, 200)) and (e20[s] > e20[s - 5])
    near = any(l[j] / e20[j] - 1.0 <= 0.02 for j in range(s - 4, s + 1))
    pullback_ok = near and c[s] > e20[s]
    e = s + 1
    entry = o[e]
    stop = entry - 1.5 * atr[s]
    period_end = pd.Timestamp(row["period_end"])
    end_i = max(i for i, d in enumerate(dates) if d <= period_end)
    mfe_hi, mae_lo, peak = entry, entry, -math.inf
    j = e
    while True:
        if l[j] < stop:
            gap = j > e and o[j] < stop
            px = o[j] if gap else stop
            mfe_hi, mae_lo = max(mfe_hi, o[j]), min(mae_lo, px)
            x, reason = j, "stop"
            break
        mfe_hi, mae_lo, peak = max(mfe_hi, h[j]), min(mae_lo, l[j]), max(peak, c[j])
        if j >= end_i:
            px, x, reason = c[j], j, "period_end"
            break
        if c[j] < e20[j]:
            px, x, reason = o[j + 1], j + 1, "signal"
            mfe_hi, mae_lo = max(mfe_hi, px), min(mae_lo, px)
            break
        j += 1
    ret = (px * (1 - cost)) / (entry * (1 + cost)) - 1.0
    peak_ret = peak / entry - 1.0
    spy = D.load_cached("SPY", C.RAW_DIR)
    vix = D.load_cached("^VIX", C.RAW_DIR)
    sc = spy["close"].tolist()
    si = list(spy.index).index(pd.Timestamp(row["signal_date"]))
    return {
        "ticker": row["ticker"], "signal_date": str(row["signal_date"])[:10],
        "trend_ok": trend, "pullback_ok": pullback_ok,
        "ema20_signal": e20[s], "atr14_signal": atr[s],
        "entry_date": dates[e].date().isoformat(), "entry_px": entry, "stop_px": stop,
        "exit_date": dates[x].date().isoformat(), "exit_px": px, "exit_reason": reason, "ret": ret,
        "mfe": mfe_hi / entry - 1.0, "mae": mae_lo / entry - 1.0, "peak_close_ret": peak_ret,
        "giveback_pp": peak_ret - (px / entry - 1.0) if peak_ret > 0 else math.nan,
        "spy_above_200": sc[si] > sma_at(sc, si, 200), "spy_50_gt_200": sma_at(sc, si, 50) > sma_at(sc, si, 200),
        "vix": float(vix.loc[pd.Timestamp(row["signal_date"]), "close"]),
        "breadth50": breadth_on(pd.Timestamp(row["signal_date"])),
    }


def main() -> None:
    """Audit 8 FINAL trades (4 dev, 4 OOS; mix of stop/signal exits) and write the report."""
    tr = pd.read_csv(C.OUT_DIR / "trades.csv", parse_dates=["signal_date", "entry_date", "exit_date"])
    tr = tr[tr["model"] == "FINAL"]
    ends = {"dev": pd.Timestamp(C.DEV_END),
            "oos": pd.Timestamp(pd.read_json(C.META_DIR / "data_end.json", typ="series")["data_end"])}
    rng = np.random.default_rng(11)
    picks = []
    for p in ("dev", "oos"):
        for reason in ("stop", "signal"):
            sub = tr[(tr["period"] == p) & (tr["exit_reason"] == reason)]
            picks.append(sub.iloc[rng.choice(len(sub), size=2, replace=False)])
    sample = pd.concat(picks)
    reg = pd.read_csv(C.DERIVED_DIR / "regime.csv.gz", index_col="date", parse_dates=["date"])
    lines = ["# Manual trade audit (independent recomputation)", "",
             "Each sampled FINAL trade is recomputed with plain-Python loops from the cached bars "
             "(`src/audit.py`) and compared with the engine output (`trades.csv`). "
             "Tolerance: 1e-6 relative on prices/returns; breadth within 0.5 pp (the engine uses "
             "pandas rolling means over the same members).", ""]
    all_ok = True
    for _, row in sample.iterrows():
        row = row.copy()
        row["period_end"] = ends[row["period"]]
        a = audit_trade(row, C.COST_BPS_PER_SIDE / 1e4)
        rg = reg.loc[row["signal_date"]]
        checks = [
            ("stock trend qualifies", True, a["trend_ok"]),
            ("pullback qualifies (low within 2% of EMA20 in 5 sessions, close > EMA20)", True, a["pullback_ok"]),
            ("entry date (next session)", str(row["entry_date"])[:10], a["entry_date"]),
            ("entry price (next open)", row["entry_px"], a["entry_px"]),
            ("initial stop (entry - 1.5 ATR14)", row["stop_px"], a["stop_px"]),
            ("exit date", str(row["exit_date"])[:10], a["exit_date"]),
            ("exit price", row["exit_px"], a["exit_px"]),
            ("exit reason", row["exit_reason"], a["exit_reason"]),
            ("net return", row["ret"], a["ret"]),
            ("MFE", row["mfe"], a["mfe"]),
            ("MAE", row["mae"], a["mae"]),
            ("peak close return", row["peak_close_ret"], a["peak_close_ret"]),
            ("giveback (pp)", row["giveback_pp"], a["giveback_pp"]),
            ("SPY > SMA200 at signal", bool(rg["spy_above_200"]), a["spy_above_200"]),
            ("SPY SMA50 > SMA200 at signal", bool(rg["spy_50_gt_200"]), a["spy_50_gt_200"]),
            ("VIX close at signal", float(rg["vix"]), a["vix"]),
            ("breadth % > SMA50 at signal", float(rg["pct_above_50"]), a["breadth50"]),
        ]
        lines += [f"## {row['ticker']} — signal {a['signal_date']} ({row['period']})", "",
                  f"EMA20(signal) = {a['ema20_signal']:.4f}, ATR14(signal) = {a['atr14_signal']:.4f}", "",
                  "| check | engine | independent | ok |", "|---|---|---|---|"]
        for name, eng, ind in checks:
            if isinstance(eng, (bool, np.bool_)) or isinstance(eng, str):
                ok = str(eng) == str(ind)
            elif name.startswith("breadth"):
                ok = abs(float(eng) - float(ind)) <= 0.5
            else:
                both_nan = (isinstance(eng, float) and math.isnan(eng)) and (isinstance(ind, float) and math.isnan(ind))
                ok = both_nan or math.isclose(float(eng), float(ind), rel_tol=1e-6, abs_tol=1e-9)
            all_ok &= ok
            fmt = (lambda v: f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v))
            lines.append(f"| {name} | {fmt(eng)} | {fmt(ind)} | {'✅' if ok else '❌'} |")
        lines.append("")
    lines.insert(2, f"**Overall: {'ALL CHECKS PASSED' if all_ok else 'MISMATCHES FOUND — see ❌ rows'}** "
                    f"({len(sample)} trades).\n")
    (C.ROOT / "qc" / "audit_report.md").write_text("\n".join(lines))
    print("audit:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
