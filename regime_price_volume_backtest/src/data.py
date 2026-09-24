"""
Daily OHLCV acquisition, caching and validation.

Provider order
--------------
1. **Alpaca** (preferred by the research brief) when ``APCA_API_KEY_ID`` /
   ``APCA_API_SECRET_KEY`` are present in the environment.  Requests go through
   the repo's ``AlpacaHttpClient`` (retries/backoff/timeouts) and use
   ``adjustment=all`` so OHLC is split *and* dividend adjusted.
2. **Yahoo Finance chart API** otherwise.  Yahoo's ``close`` is split-adjusted
   only, ``adjclose`` is split+dividend adjusted.  We scale O/H/L/C by
   ``adjclose / close`` so the whole bar is total-return adjusted, matching
   Alpaca's ``adjustment=all``.

Every cached file stores both the adjusted bar (used by the strategy) and the
*as-traded* close (``raw_close``: split adjustment undone using the split
events) which is needed for point-in-time price filters such as "price > $10".
Dollar volume is invariant to split adjustment so it is computed from the
split-adjusted close x split-adjusted volume.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from . import config as C

log = logging.getLogger(__name__)

YAHOO_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{sym}"
_UA = {"User-Agent": "Mozilla/5.0 (research backtest; daily bars)"}
COLUMNS = ["open", "high", "low", "close", "volume", "raw_close", "adj_factor"]


class DataUnavailable(Exception):
    """Raised when a provider has no data for a symbol (e.g. delisted)."""


def yahoo_symbol(ticker: str) -> str:
    """Map an index-style ticker (BRK.B) to Yahoo's convention (BRK-B)."""
    return ticker.replace(".", "-")


def _to_epoch(date_str: str) -> int:
    return int(pd.Timestamp(date_str, tz="UTC").timestamp())


def fetch_yahoo(ticker: str, start: str, end: Optional[str] = None,
                max_retries: int = 5) -> pd.DataFrame:
    """
    Download daily bars from Yahoo's chart API and return total-return adjusted OHLCV.

    Args:
        ticker: Index-style ticker (dots allowed).
        start: First date (inclusive, YYYY-MM-DD).
        end: Last date (exclusive); defaults to tomorrow.
        max_retries: Retries on HTTP 429/5xx/network errors with exponential backoff.

    Returns:
        DataFrame indexed by session date with ``COLUMNS``.

    Raises:
        DataUnavailable: if Yahoo reports no data for the symbol.
    """
    end = end or (pd.Timestamp.utcnow() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    params = {
        "period1": _to_epoch(start),
        "period2": _to_epoch(end),
        "interval": "1d",
        "events": "div,split",
        "includeAdjustedClose": "true",
    }
    url = YAHOO_URL.format(sym=yahoo_symbol(ticker))
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=_UA, timeout=20)
            if r.status_code == 404:
                raise DataUnavailable(f"{ticker}: 404 from Yahoo")
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            payload = r.json()
            break
        except DataUnavailable:
            raise
        except (requests.RequestException, ValueError) as exc:
            last_err = exc
            time.sleep(2.0 * (2 ** attempt))
    else:
        raise RuntimeError(f"{ticker}: Yahoo request failed after retries: {last_err}")

    chart = payload.get("chart", {})
    if chart.get("error") or not chart.get("result"):
        raise DataUnavailable(f"{ticker}: {chart.get('error')}")
    res = chart["result"][0]
    ts = res.get("timestamp")
    if not ts:
        raise DataUnavailable(f"{ticker}: empty timestamp array")
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    tz = res.get("meta", {}).get("exchangeTimezoneName", "America/New_York")
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    df = pd.DataFrame({
        "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
        "volume": q["volume"], "adjclose": adj if adj is not None else q["close"],
    }, index=idx, dtype="float64")
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["open", "high", "low", "close"])

    # Undo split adjustment to recover the as-traded close (for price filters only).
    splits = (res.get("events") or {}).get("splits") or {}
    split_factor = pd.Series(1.0, index=df.index)
    for ev in splits.values():
        d = pd.to_datetime(ev["date"], unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
        ratio = float(ev["numerator"]) / float(ev["denominator"])
        if ratio > 0:
            split_factor[df.index < d] *= ratio
    df["raw_close"] = df["close"] * split_factor

    factor = (df["adjclose"] / df["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col] * factor
    df["adj_factor"] = factor
    df["volume"] = df["volume"].fillna(0.0)
    return df[COLUMNS]


def fetch_alpaca(ticker: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    """
    Download adjusted daily bars from Alpaca market data (``adjustment=all``, SIP feed).

    Uses the repo's ``AlpacaHttpClient`` so the standard retry/backoff policy applies.
    Only used when Alpaca credentials are present in the environment.

    Args:
        ticker: Ticker symbol.
        start: First date (inclusive).
        end: Last date (exclusive); defaults to tomorrow.

    Returns:
        DataFrame indexed by session date with ``COLUMNS``.
    """
    repo_root = C.ROOT.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from RubberBand.src.alpaca_creds import resolve_credentials  # noqa: WPS433
    from RubberBand.src.core.http_client import AlpacaHttpClient  # noqa: WPS433

    key, secret, _ = resolve_credentials()
    client = AlpacaHttpClient("https://data.alpaca.markets", key, secret)
    frames: List[dict] = []
    raw_frames: List[dict] = []
    try:
        for adjustment, sink in (("all", frames), ("raw", raw_frames)):
            token: Optional[str] = None
            while True:
                params = {"timeframe": "1Day", "start": start, "adjustment": adjustment,
                          "feed": "sip", "limit": 10000}
                if end:
                    params["end"] = end
                if token:
                    params["page_token"] = token
                j = client.get(f"/v2/stocks/{ticker}/bars", params=params)
                sink.extend(j.get("bars") or [])
                token = j.get("next_page_token")
                if not token:
                    break
    finally:
        client.close()
    if not frames:
        raise DataUnavailable(f"{ticker}: no Alpaca bars")
    df = pd.DataFrame(frames)
    df["date"] = pd.to_datetime(df["t"]).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    df = df.set_index("date").rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    raw = pd.DataFrame(raw_frames)
    raw["date"] = pd.to_datetime(raw["t"]).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    raw = raw.set_index("date")
    df["raw_close"] = raw["c"].reindex(df.index)
    df["adj_factor"] = df["close"] / df["raw_close"]
    return df[COLUMNS].astype("float64")


def provider_name() -> str:
    """Return the provider that ``fetch`` will use in this environment."""
    has_alpaca = bool(os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_KEY_ID"))
    return "alpaca" if has_alpaca else "yahoo"


def fetch(ticker: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    """Fetch adjusted daily bars from the active provider (see module docstring)."""
    if provider_name() == "alpaca":
        return fetch_alpaca(ticker, start, end)
    return fetch_yahoo(ticker, start, end)


def cache_path(ticker: str, directory: Path) -> Path:
    """Path of the cached CSV for ``ticker`` inside ``directory``."""
    return directory / f"{ticker.replace('.', '_').replace('^', 'IDX_')}.csv.gz"


def load_or_fetch(ticker: str, start: str, directory: Path, refresh: bool = False,
                  pause: float = 0.25) -> pd.DataFrame:
    """
    Return cached bars for ``ticker`` or download and cache them.

    Args:
        ticker: Ticker symbol.
        start: Download start date.
        directory: Cache directory.
        refresh: Force re-download.
        pause: Politeness delay after a network fetch (seconds).
    """
    path = cache_path(ticker, directory)
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col="date", parse_dates=["date"])
    df = fetch(ticker, start)
    directory.mkdir(parents=True, exist_ok=True)
    df.round(6).to_csv(path, compression="gzip")
    time.sleep(pause)
    return df


def load_cached(ticker: str, directory: Path) -> pd.DataFrame:
    """Load a cached ticker file (raises FileNotFoundError if missing)."""
    return pd.read_csv(cache_path(ticker, directory), index_col="date", parse_dates=["date"])


def fetch_fred_vix() -> pd.Series:
    """Download VIXCLS from FRED (used only as an independent cross-check of Yahoo ^VIX)."""
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                     params={"id": "VIXCLS"}, headers=_UA, timeout=30)
    r.raise_for_status()
    from io import StringIO
    s = pd.read_csv(StringIO(r.text), parse_dates=["observation_date"], index_col="observation_date")["VIXCLS"]
    return pd.to_numeric(s, errors="coerce").dropna()


def validate_bars(df: pd.DataFrame, ticker: str) -> Dict[str, object]:
    """
    Run basic integrity checks on a bar DataFrame.

    Returns a dict of counts: non-positive prices, high<low, open/close outside
    [low, high], zero-volume days, max absolute daily return and the largest
    calendar gap between sessions.
    """
    o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    tol = 1e-6 * c
    ret = c.pct_change().abs()
    gaps = df.index.to_series().diff().dt.days
    return {
        "ticker": ticker,
        "rows": int(len(df)),
        "first": df.index.min().date().isoformat() if len(df) else None,
        "last": df.index.max().date().isoformat() if len(df) else None,
        "nonpositive_px": int(((df[["open", "high", "low", "close"]] <= 0).any(axis=1)).sum()),
        "high_lt_low": int((h < l - tol).sum()),
        "oc_outside_hl": int(((o > h + tol) | (o < l - tol) | (c > h + tol) | (c < l - tol)).sum()),
        "zero_volume_days": int((v <= 0).sum()),
        "max_abs_daily_ret": float(ret.max()) if len(ret) > 1 else 0.0,
        "max_gap_days": int(gaps.max()) if len(gaps) > 1 else 0,
    }


def write_json(obj: object, path: Path) -> None:
    """Write ``obj`` as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
