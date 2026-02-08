"""
Persistent SQLite cache for market data bars.
Fetches from Alpaca API and caches locally to avoid throttling.
Supports 1m, 5m, 15m, 1h, 1d timeframes.
"""
import os
import sys
import json
import time
import sqlite3
import datetime as dt
from typing import List, Dict, Optional, Tuple

import pandas as pd
import requests

# Reuse Alpaca helpers from RubberBand
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "market_cache.db")

# Alpaca data API
DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"
ISO_UTC = "%Y-%m-%dT%H:%M:%SZ"

def _alpaca_headers() -> Dict[str, str]:
    k = os.getenv("APCA_API_KEY_ID", "").strip()
    s = os.getenv("APCA_API_SECRET_KEY", "").strip()
    return {
        "APCA-API-KEY-ID": k,
        "APCA-API-SECRET-KEY": s,
    }

def _init_db(db_path: str = None) -> sqlite3.Connection:
    db_path = db_path or DB_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bars (
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            vwap REAL,
            trade_count INTEGER,
            PRIMARY KEY (symbol, timeframe, timestamp)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_bars_sym_tf_ts
        ON bars(symbol, timeframe, timestamp)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            bar_count INTEGER,
            PRIMARY KEY (symbol, timeframe, start_date, end_date)
        )
    """)
    conn.commit()
    return conn

def _normalize_timeframe(tf: str) -> str:
    """Normalize timeframe string to Alpaca format."""
    tf = tf.lower().strip()
    mapping = {
        "1m": "1Min", "1min": "1Min",
        "5m": "5Min", "5min": "5Min",
        "15m": "15Min", "15min": "15Min",
        "30m": "30Min", "30min": "30Min",
        "1h": "1Hour", "1hour": "1Hour", "60m": "1Hour", "60min": "1Hour",
        "1d": "1Day", "1day": "1Day", "day": "1Day", "daily": "1Day",
    }
    return mapping.get(tf, tf)

def _is_cached(conn: sqlite3.Connection, symbol: str, timeframe: str,
               start_date: str, end_date: str) -> bool:
    """Check if we already have data for this range."""
    cur = conn.execute(
        "SELECT bar_count FROM fetch_log WHERE symbol=? AND timeframe=? AND start_date<=? AND end_date>=?",
        (symbol.upper(), timeframe, start_date, end_date)
    )
    row = cur.fetchone()
    return row is not None and row[0] > 0

def _fetch_from_alpaca(symbol: str, timeframe: str, start_iso: str, end_iso: str,
                       feed: str = "iex", max_pages: int = 30) -> List[Dict]:
    """Fetch bars from Alpaca with pagination."""
    headers = _alpaca_headers()
    all_bars = []
    page_token = None

    for page in range(max_pages):
        params = {
            "symbols": symbol.upper(),
            "timeframe": timeframe,
            "start": start_iso,
            "end": end_iso,
            "limit": 10000,
            "feed": feed,
        }
        if page_token:
            params["page_token"] = page_token

        for attempt in range(3):
            try:
                r = requests.get(DATA_URL, headers=headers, params=params, timeout=30)
                if r.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    print(f"  [cache] Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                break
            except requests.Timeout:
                if attempt < 2:
                    time.sleep(2)
                    continue
                raise
        else:
            print(f"  [cache] Max retries for {symbol} page {page}")
            break

        j = r.json() or {}
        bars_data = j.get("bars", {})

        # Handle both dict and list shapes
        if isinstance(bars_data, dict):
            sym_bars = bars_data.get(symbol.upper(), [])
        elif isinstance(bars_data, list):
            sym_bars = [b for b in bars_data if (b.get("S") or b.get("symbol", "")).upper() == symbol.upper()]
        else:
            sym_bars = []

        all_bars.extend(sym_bars)

        page_token = j.get("next_page_token")
        if not page_token:
            break

    return all_bars

def _store_bars(conn: sqlite3.Connection, symbol: str, timeframe: str, bars: List[Dict]):
    """Store bars in cache, upserting on conflict."""
    if not bars:
        return

    rows = []
    for b in bars:
        ts = b.get("t") or b.get("time", "")
        rows.append((
            symbol.upper(),
            timeframe,
            ts,
            float(b.get("o", 0) or b.get("open", 0)),
            float(b.get("h", 0) or b.get("high", 0)),
            float(b.get("l", 0) or b.get("low", 0)),
            float(b.get("c", 0) or b.get("close", 0)),
            float(b.get("v", 0) or b.get("volume", 0)),
            float(b.get("vw", 0) or b.get("vwap", 0)),
            int(b.get("n", 0) or b.get("trade_count", 0)),
        ))

    conn.executemany("""
        INSERT OR REPLACE INTO bars (symbol, timeframe, timestamp, open, high, low, close, volume, vwap, trade_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()

def get_bars(symbol: str, timeframe: str = "5Min", days: int = 30,
             feed: str = "iex", end_date: Optional[str] = None,
             rth_only: bool = True, db_path: str = None,
             force_refresh: bool = False) -> pd.DataFrame:
    """
    Get bars for a symbol, using cache when available.

    Args:
        symbol: Ticker symbol
        timeframe: Bar timeframe (1m, 5m, 15m, 1h, 1d)
        days: Number of calendar days of history
        feed: Alpaca feed (iex or sip)
        end_date: End date (YYYY-MM-DD), defaults to now
        rth_only: Filter to regular trading hours only
        db_path: Custom database path
        force_refresh: Skip cache and re-fetch

    Returns:
        DataFrame with OHLCV + vwap columns, UTC DatetimeIndex
    """
    conn = _init_db(db_path)
    tf = _normalize_timeframe(timeframe)
    symbol = symbol.upper()

    if end_date:
        end_dt = dt.datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=dt.timezone.utc)
    else:
        end_dt = dt.datetime.now(dt.timezone.utc)

    start_dt = end_dt - dt.timedelta(days=days)
    start_iso = start_dt.strftime(ISO_UTC)
    end_iso = end_dt.strftime(ISO_UTC)
    start_date_str = start_dt.strftime("%Y-%m-%d")
    end_date_str = end_dt.strftime("%Y-%m-%d")

    # Check cache
    if not force_refresh and _is_cached(conn, symbol, tf, start_date_str, end_date_str):
        pass  # Will read from DB below
    else:
        # Fetch from API
        print(f"  [cache] Fetching {symbol} {tf} {start_date_str} to {end_date_str}...")
        bars = _fetch_from_alpaca(symbol, tf, start_iso, end_iso, feed)
        _store_bars(conn, symbol, tf, bars)

        # Log the fetch
        conn.execute("""
            INSERT OR REPLACE INTO fetch_log (symbol, timeframe, start_date, end_date, fetched_at, bar_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (symbol, tf, start_date_str, end_date_str,
              dt.datetime.now(dt.timezone.utc).isoformat(), len(bars)))
        conn.commit()
        print(f"  [cache] Stored {len(bars)} bars for {symbol}")

    # Read from DB
    cur = conn.execute("""
        SELECT timestamp, open, high, low, close, volume, vwap, trade_count
        FROM bars
        WHERE symbol=? AND timeframe=? AND timestamp>=? AND timestamp<=?
        ORDER BY timestamp
    """, (symbol, tf, start_iso, end_iso))

    rows = cur.fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "vwap", "trade_count"])
    df.index = pd.to_datetime(df["timestamp"], utc=True)
    df.drop(columns=["timestamp"], inplace=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # RTH filter
    if rth_only and tf not in ("1Day", "1Week"):
        eastern = df.index.tz_convert("US/Eastern")
        mask = (eastern.hour * 60 + eastern.minute >= 9 * 60 + 30) & \
               (eastern.hour * 60 + eastern.minute <= 15 * 60 + 55)
        df = df.loc[mask]

    return df

def bulk_fetch(symbols: List[str], timeframe: str = "5Min", days: int = 30,
               feed: str = "iex", end_date: Optional[str] = None,
               rth_only: bool = True, db_path: str = None,
               force_refresh: bool = False, delay: float = 0.2) -> Dict[str, pd.DataFrame]:
    """
    Fetch bars for multiple symbols with rate limiting.

    Args:
        symbols: List of ticker symbols
        delay: Seconds between API calls (rate limiting)

    Returns:
        Dict mapping symbol -> DataFrame
    """
    result = {}
    total = len(symbols)

    for i, sym in enumerate(symbols):
        try:
            df = get_bars(sym, timeframe, days, feed, end_date, rth_only, db_path, force_refresh)
            if not df.empty:
                result[sym] = df
            if i < total - 1:
                time.sleep(delay)
        except Exception as e:
            print(f"  [cache] Error fetching {sym}: {e}")
            continue

        if (i + 1) % 25 == 0:
            print(f"  [cache] Progress: {i+1}/{total} symbols fetched")

    print(f"  [cache] Done: {len(result)}/{total} symbols have data")
    return result

def get_cache_stats(db_path: str = None) -> Dict:
    """Get cache statistics."""
    conn = _init_db(db_path)

    stats = {}
    cur = conn.execute("SELECT COUNT(DISTINCT symbol) FROM bars")
    stats["unique_symbols"] = cur.fetchone()[0]

    cur = conn.execute("SELECT COUNT(*) FROM bars")
    stats["total_bars"] = cur.fetchone()[0]

    cur = conn.execute("SELECT timeframe, COUNT(*) FROM bars GROUP BY timeframe")
    stats["bars_by_timeframe"] = dict(cur.fetchall())

    cur = conn.execute("SELECT symbol, timeframe, bar_count, fetched_at FROM fetch_log ORDER BY fetched_at DESC LIMIT 10")
    stats["recent_fetches"] = [
        {"symbol": r[0], "timeframe": r[1], "bars": r[2], "when": r[3]}
        for r in cur.fetchall()
    ]

    conn.close()
    return stats
