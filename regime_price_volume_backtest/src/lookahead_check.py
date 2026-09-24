"""
Automated look-ahead test.

For a random sample of (ticker, session) pairs, recompute the full feature
frame using ONLY bars up to and including that session and compare every
feature and every setup signal with the values from the full-history frame.
Any difference means a feature depended on future bars.  Also checks the
regime frame the same way for SPY/VIX-derived columns.

Run:  python -m src.lookahead_check
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from .features import compute_features
from .regime import _trend_block, rolling_percentile
from .research import load_universe
from .setups import BASE_A, BASE_B, BASE_B_EMA, BASE_C, BASE_CANDLE, signals

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("lookahead")


def check_stock(ticker: str, n_points: int, rng: np.random.Generator) -> dict:
    """Compare truncated-history features against full-history features at sampled sessions."""
    df = D.load_cached(ticker, C.RAW_DIR)
    full = compute_features(df)
    specs = {"A": BASE_A, "B": BASE_B, "Bema": BASE_B_EMA, "C": BASE_C, "CANDLE": BASE_CANDLE}
    full_sig = {k: signals(full, s) for k, s in specs.items()}
    start = full.index.searchsorted(pd.Timestamp(C.BACKTEST_START))
    pts = rng.choice(np.arange(start, len(full) - 1), size=n_points, replace=False)
    mismatches = 0
    checked = 0
    num_cols = [c for c in full.columns if full[c].dtype != bool]
    bool_cols = [c for c in full.columns if full[c].dtype == bool]
    for i in pts:
        trunc = compute_features(df.iloc[:i + 1])
        a, b = full.iloc[i], trunc.iloc[-1]
        num_ok = np.allclose(a[num_cols].astype(float), b[num_cols].astype(float), rtol=1e-9, atol=1e-9,
                             equal_nan=True)
        bool_ok = bool((a[bool_cols] == b[bool_cols]).all())
        sig_ok = all(bool(full_sig[k].iloc[i]) == bool(signals(trunc, s).iloc[-1]) for k, s in specs.items())
        checked += 1
        if not (num_ok and bool_ok and sig_ok):
            mismatches += 1
            log.error("LOOK-AHEAD mismatch %s @ %s", ticker, full.index[i].date())
    return {"ticker": ticker, "points": checked, "mismatches": mismatches}


def check_regime(n_points: int, rng: np.random.Generator) -> dict:
    """Same test for the SPY and QQQ trend blocks, the VIX percentile and the VIX 5-day ROC."""
    spy = D.load_cached("SPY", C.RAW_DIR)
    qqq = D.load_cached("QQQ", C.RAW_DIR)
    vix = D.load_cached("^VIX", C.RAW_DIR)["close"]
    full_t = _trend_block(spy, "spy")
    full_q = _trend_block(qqq, "qqq")
    full_p = rolling_percentile(vix, C.VIX_PCT_WINDOW)
    full_roc = vix / vix.shift(C.VIX_ROC_LEN) - 1.0
    start = spy.index.searchsorted(pd.Timestamp(C.BACKTEST_START))
    bad = 0
    pts = rng.choice(np.arange(start, len(spy) - 1), size=n_points, replace=False)
    for i in pts:
        d = spy.index[i]
        t = _trend_block(spy.iloc[:i + 1], "spy").iloc[-1]
        ok_t = np.allclose(full_t.loc[d].astype(float), t.astype(float), equal_nan=True)
        q = _trend_block(qqq.loc[:d], "qqq").iloc[-1]
        ok_q = np.allclose(full_q.loc[d].astype(float), q.astype(float), equal_nan=True)
        vp = rolling_percentile(vix.loc[:d], C.VIX_PCT_WINDOW).iloc[-1]
        vt = vix.loc[:d]
        ok_v = np.isclose(full_p.loc[d], vp, equal_nan=True) and np.isclose(
            full_roc.loc[d], vt.iloc[-1] / vt.iloc[-1 - C.VIX_ROC_LEN] - 1.0, equal_nan=True)
        if not (ok_t and ok_q and ok_v):
            bad += 1
            log.error("regime look-ahead mismatch at %s", d.date())
    return {"ticker": "REGIME(SPY,QQQ,VIX pct,VIX roc)", "points": int(len(pts)), "mismatches": bad}


def check_breadth(n_points: int, rng: np.random.Generator) -> dict:
    """Recompute breadth on sampled dates from member histories truncated at that date."""
    from .audit import breadth_on
    reg = pd.read_csv(C.DERIVED_DIR / "regime.csv.gz", index_col="date", parse_dates=["date"])
    end = json.loads((C.META_DIR / "data_end.json").read_text())["data_end"]
    dates = reg.loc[C.BACKTEST_START:end].index
    bad = 0
    for d in rng.choice(dates, size=n_points, replace=False):
        if not np.isclose(reg.loc[d, "pct_above_50"], breadth_on(pd.Timestamp(d)), atol=1e-6):
            bad += 1
            log.error("breadth look-ahead mismatch at %s", pd.Timestamp(d).date())
    return {"ticker": "BREADTH(% members > SMA50)", "points": int(n_points), "mismatches": bad}


def main() -> None:
    """Run the check on 25 random universe stocks x 12 sessions each, plus the regime frame."""
    rng = np.random.default_rng(7)
    uni = load_universe()
    tickers = rng.choice(uni["ticker"].to_numpy(), size=25, replace=False)
    res = [check_stock(t, 12, rng) for t in tickers]
    res.append(check_regime(40, rng))
    res.append(check_breadth(6, rng))
    out = pd.DataFrame(res)
    out.to_csv(C.ROOT / "qc" / "lookahead_check.csv", index=False)
    log.info("look-ahead check: %d points, %d mismatches", out["points"].sum(), out["mismatches"].sum())


if __name__ == "__main__":
    main()
