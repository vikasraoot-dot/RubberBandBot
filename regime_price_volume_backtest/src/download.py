"""
Step 1: download and cache all market data needed by the study.

* SPY, QQQ, ^VIX (from 2015 for a full 252-day VIX percentile warm-up)
* every ticker that was an S&P 500 member at any time from 2019-01-01 onward
  (point-in-time membership from fja05680/sp500).  These feed both the breadth
  indicator and the universe selection.
* FRED VIXCLS, used only to cross-check the Yahoo VIX series.

Run:  python -m src.download
"""
from __future__ import annotations

import logging
import sys
from typing import Dict, List

import pandas as pd
import requests

from . import config as C
from . import data as D
from .membership import RENAMES, load_membership

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download")

MARKET_SYMBOLS = ["SPY", "QQQ", "^VIX"]


def main() -> None:
    """Download market series and all point-in-time S&P members; write a coverage log."""
    log.info("provider=%s", D.provider_name())
    for sym in MARKET_SYMBOLS:
        df = D.load_or_fetch(sym, C.MARKET_DOWNLOAD_START, C.RAW_DIR)
        log.info("%s rows=%d %s..%s", sym, len(df), df.index.min().date(), df.index.max().date())

    try:
        fred = D.fetch_fred_vix()
        fred.to_frame("vixcls").to_csv(C.META_DIR / "fred_vixcls.csv.gz", compression="gzip")
    except requests.RequestException as exc:  # cross-check only; Yahoo ^VIX is the primary series
        log.warning("FRED VIX cross-check unavailable: %s", exc)

    mem = load_membership()
    tickers = sorted(mem.tickers_since(C.UNIVERSE_SELECTION_START))
    fetch_syms = sorted(set(tickers) | set(RENAMES.values()))
    log.info("membership tickers since %s: %d (fetch symbols %d)",
             C.UNIVERSE_SELECTION_START, len(tickers), len(fetch_syms))
    status: List[Dict[str, object]] = []
    for i, sym in enumerate(fetch_syms):
        try:
            df = D.load_or_fetch(sym, C.DOWNLOAD_START, C.BREADTH_RAW_DIR)
            status.append({"symbol": sym, "status": "ok", "rows": len(df),
                           "first": df.index.min().date(), "last": df.index.max().date()})
        except D.DataUnavailable as exc:
            status.append({"symbol": sym, "status": "unavailable", "rows": 0,
                           "first": None, "last": None, "note": str(exc)[:80]})
        except RuntimeError as exc:
            status.append({"symbol": sym, "status": "error", "rows": 0,
                           "first": None, "last": None, "note": str(exc)[:80]})
        if i % 50 == 0:
            log.info("%d/%d %s", i, len(fetch_syms), sym)
    pd.DataFrame(status).to_csv(C.META_DIR / "download_status.csv", index=False)
    ok = sum(1 for s in status if s["status"] == "ok")
    log.info("done: %d ok / %d", ok, len(status))
    if ok == 0:
        sys.exit("no data downloaded")


if __name__ == "__main__":
    main()
