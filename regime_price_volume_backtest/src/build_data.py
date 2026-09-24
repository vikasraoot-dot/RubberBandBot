"""
Step 2-3: freeze the universe, stage its bars, build the regime frame and
validate every series.

Run:  python -m src.build_data
"""
from __future__ import annotations

import logging
import shutil

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import universe as U
from .regime import build_regime_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_data")


def main() -> None:
    """Freeze universe, copy bars, compute regime/breadth, write validation reports."""
    uni = U.build()
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for t in uni["ticker"]:
        shutil.copy(D.cache_path(t, C.BREADTH_RAW_DIR), D.cache_path(t, C.RAW_DIR))

    # ── validation ─────────────────────────────────────────────────────────
    checks = []
    spy = D.load_cached("SPY", C.RAW_DIR)
    for t in list(uni["ticker"]) + ["SPY", "QQQ", "^VIX"]:
        df = D.load_cached(t, C.RAW_DIR)
        chk = D.validate_bars(df, t)
        expected = spy.loc[max(pd.Timestamp(C.BACKTEST_START), df.index.min()):df.index.max()].index
        chk["missing_vs_spy_calendar"] = int(len(expected.difference(df.index)))
        checks.append(chk)
    val = pd.DataFrame(checks)
    val.to_csv(C.META_DIR / "data_validation.csv", index=False)
    log.info("validation: %d series; any H<L=%d; any OHLC outside=%d; max missing sessions=%d",
             len(val), val["high_lt_low"].sum(), val["oc_outside_hl"].sum(), val["missing_vs_spy_calendar"].max())

    # ── common data end: last session where SPY, QQQ, VIX and every universe stock has a bar ──
    sessions = spy.index
    for t in list(uni["ticker"]) + ["QQQ", "^VIX"]:
        sessions = sessions.intersection(D.load_cached(t, C.RAW_DIR).index)
    cal = spy.loc[C.BACKTEST_START:].index
    incomplete = cal.difference(sessions)
    data_end = cal[cal < incomplete.min()].max() if len(incomplete) else cal.max()
    tail_gaps = cal[cal > data_end]
    D.write_json({"data_end": data_end.date().isoformat(),
                  "dropped_trailing_sessions": [d.date().isoformat() for d in tail_gaps],
                  "reason": "provider returned null bars for most stocks on the dropped session(s)"},
                 C.META_DIR / "data_end.json")
    log.info("data_end=%s (dropped trailing %s)", data_end.date(), [d.date() for d in tail_gaps])

    # ── regime frame ───────────────────────────────────────────────────────
    reg, diag = build_regime_frame(with_breadth=True)
    C.DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    reg.to_csv(C.DERIVED_DIR / "regime.csv.gz", compression="gzip")
    D.write_json({k: v for k, v in diag.items()}, C.META_DIR / "breadth_diagnostics.json")
    cov = reg.loc[C.BACKTEST_START:, "breadth_coverage"]
    log.info("breadth coverage 2020+: mean %.3f min %.3f", cov.mean(), cov.min())

    # ── VIX cross-check vs FRED (if the download succeeded) ────────────────
    fred_path = C.META_DIR / "fred_vixcls.csv.gz"
    if fred_path.exists():
        fred = pd.read_csv(fred_path, index_col=0, parse_dates=True)["vixcls"]
        both = pd.concat([reg["vix"], fred], axis=1, join="inner").loc[C.BACKTEST_START:].dropna()
        diff = (both.iloc[:, 0] - both.iloc[:, 1]).abs()
        D.write_json({"days": int(len(both)), "max_abs_diff": float(diff.max()),
                      "mean_abs_diff": float(diff.mean())}, C.META_DIR / "vix_crosscheck.json")
        log.info("VIX vs FRED: max abs diff %.3f over %d days", diff.max(), len(both))


if __name__ == "__main__":
    main()
