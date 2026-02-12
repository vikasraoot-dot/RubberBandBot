"""
News Scanner: Rule-based scanning of Alpaca News API for high-impact events.

Uses the Alpaca News API (free on paper accounts, 200 calls/min) to detect
headlines containing adverse keywords.  Tickers flagged by negative news are
written to ``results/watchdog/news_pause.json`` so live loops can skip them.

This is **rule-based** (keyword matching), not LLM-based, for reliability
and zero additional cost.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger(__name__)

ET = ZoneInfo("US/Eastern")

# Default output path
_NEWS_PAUSE_PATH = "results/watchdog/news_pause.json"

# Keywords indicating potentially adverse events (case-insensitive match)
_ADVERSE_KEYWORDS: List[str] = [
    "downgrade",
    "miss",
    "SEC",
    "layoff",
    "recall",
    "guidance cut",
    "investigation",
    "lawsuit",
    "fraud",
    "bankruptcy",
    "delisted",
    "halt",
    "warning",
    "revenue miss",
    "earnings miss",
    "cuts forecast",
    "lowers guidance",
]


def _alpaca_creds() -> Tuple[str, str, str]:
    """Read Alpaca API credentials from environment variables.

    Returns:
        Tuple of (base_url, key_id, secret_key).

    Raises:
        EnvironmentError: If credentials are not set.
    """
    base = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    key = os.environ.get("APCA_API_KEY_ID", "")
    secret = os.environ.get("APCA_API_SECRET_KEY", "")
    if not key or not secret:
        raise EnvironmentError("APCA_API_KEY_ID / APCA_API_SECRET_KEY not set")
    return base, key, secret


def fetch_news(
    symbols: List[str],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Fetch recent news articles for a list of symbols from Alpaca.

    Args:
        symbols: List of ticker symbols.
        limit: Max articles to fetch per request (max 50).

    Returns:
        List of news article dicts from Alpaca API.
    """
    _, key, secret = _alpaca_creds()
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }

    # Alpaca News API endpoint
    url = "https://data.alpaca.markets/v1beta1/news"
    all_articles: List[Dict[str, Any]] = []

    # Batch symbols (API accepts comma-separated, but keep batches reasonable)
    batch_size = 20
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        params = {
            "symbols": ",".join(batch),
            "limit": limit,
            "sort": "desc",
        }
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=15)
                resp.raise_for_status()
                articles = resp.json().get("news", [])
                all_articles.extend(articles)
                break
            except requests.RequestException as exc:
                logger.warning("News API attempt %d failed for %s: %s", attempt + 1, batch, exc)
                if attempt < 2:
                    time.sleep(2 ** attempt)

    return all_articles


def scan_for_adverse_news(
    symbols: List[str],
    keywords: List[str] | None = None,
    limit: int = 10,
) -> Dict[str, List[Dict[str, str]]]:
    """Scan recent news for adverse keywords per ticker.

    Args:
        symbols: List of ticker symbols to scan.
        keywords: Override keyword list (defaults to ``_ADVERSE_KEYWORDS``).
        limit: Max articles per API call.

    Returns:
        Dict mapping ticker -> list of ``{headline, keyword, published_at}``
        for articles that matched.
    """
    kw_list = [k.lower() for k in (keywords or _ADVERSE_KEYWORDS)]
    articles = fetch_news(symbols, limit=limit)
    symbols_upper = {s.upper() for s in symbols}

    flagged: Dict[str, List[Dict[str, str]]] = {}

    for article in articles:
        headline = (article.get("headline") or "").lower()
        summary = (article.get("summary") or "").lower()
        combined = f"{headline} {summary}"
        published = article.get("created_at", "")
        article_symbols = article.get("symbols", [])

        for kw in kw_list:
            if kw in combined:
                for sym in article_symbols:
                    sym_upper = sym.upper()
                    if sym_upper in symbols_upper:
                        flagged.setdefault(sym_upper, []).append({
                            "headline": article.get("headline", ""),
                            "keyword": kw,
                            "published_at": published,
                        })
                break  # One match per article is enough

    return flagged


def update_news_pause_file(
    symbols: List[str],
    output_path: str = _NEWS_PAUSE_PATH,
    keywords: List[str] | None = None,
) -> Dict[str, Any]:
    """Scan news and write flagged tickers to the news pause file.

    Args:
        symbols: List of tickers in the trading universe.
        output_path: Path to write ``news_pause.json``.
        keywords: Override keyword list.

    Returns:
        Dict of the pause file contents.
    """
    flagged = scan_for_adverse_news(symbols, keywords=keywords)
    now = datetime.now(ET)

    pause_data: Dict[str, Any] = {
        "updated_at": now.isoformat(),
        "flagged_tickers": {},
    }

    for sym, matches in flagged.items():
        pause_data["flagged_tickers"][sym] = {
            "paused": True,
            "reason": f"Adverse news: {matches[0]['keyword']}",
            "headline": matches[0]["headline"],
            "published_at": matches[0]["published_at"],
            "flagged_at": now.isoformat(),
            "match_count": len(matches),
        }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(pause_data, fh, indent=2, default=str)
        logger.info(
            "News scan complete: %d flagged out of %d tickers",
            len(flagged),
            len(symbols),
        )
    except OSError as exc:
        logger.error("Failed to write news pause file: %s", exc)

    return pause_data


def check_news_paused(
    symbol: str,
    pause_path: str = _NEWS_PAUSE_PATH,
) -> Tuple[bool, str]:
    """Check if a ticker is paused due to adverse news.

    Fail-open: returns ``(False, "")`` if the file is missing or corrupt.

    Args:
        symbol: Ticker symbol.
        pause_path: Path to ``news_pause.json``.

    Returns:
        Tuple of ``(is_paused, reason)``.
    """
    if not os.path.exists(pause_path):
        return False, ""

    try:
        with open(pause_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return False, ""

    entry = data.get("flagged_tickers", {}).get(symbol.upper())
    if entry and entry.get("paused"):
        return True, entry.get("reason", "Adverse news detected")

    return False, ""
