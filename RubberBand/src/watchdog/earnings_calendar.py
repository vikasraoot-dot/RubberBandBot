"""
Earnings Calendar: Auto-pause tickers with upcoming earnings.

Mean-reversion strategies fail around earnings — the move is fundamental,
not technical.  This module fetches upcoming earnings dates and flags
tickers within 2 trading days of their earnings for pausing.

Primary data source: ``yfinance.Ticker.get_earnings_dates()`` (free, no key).
Fallback: Alpaca calendar data if yfinance is unavailable.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("US/Eastern")

# Default output path
_EARNINGS_PAUSE_PATH = "results/watchdog/earnings_pause.json"

# Number of trading days before earnings to pause
_BUFFER_DAYS = 2


def _get_earnings_dates_yfinance(symbol: str) -> List[datetime]:
    """Fetch upcoming earnings dates using yfinance.

    Args:
        symbol: Ticker symbol.

    Returns:
        List of upcoming earnings datetimes, or empty list on failure.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        dates_df = ticker.get_earnings_dates(limit=4)
        if dates_df is None or dates_df.empty:
            return []
        # Index contains the earnings dates as Timestamps
        return [d.to_pydatetime() for d in dates_df.index if d is not None]
    except ImportError:
        logger.debug("yfinance not installed — skipping earnings fetch for %s", symbol)
        return []
    except Exception as exc:
        logger.debug("yfinance earnings fetch failed for %s: %s", symbol, exc)
        return []


def check_earnings_proximity(
    symbol: str,
    buffer_days: int = _BUFFER_DAYS,
) -> Tuple[bool, Optional[str]]:
    """Check if a ticker has earnings within *buffer_days* trading days.

    Args:
        symbol: Ticker symbol.
        buffer_days: Number of calendar days to look ahead.

    Returns:
        Tuple of ``(is_near_earnings, earnings_date_str)``.
        ``earnings_date_str`` is ``YYYY-MM-DD`` or ``None``.
    """
    now = datetime.now(ET)
    # Use calendar days as a conservative proxy for trading days
    # 2 trading days ~ 4 calendar days (covers weekends)
    lookahead = timedelta(days=buffer_days * 2 + 1)

    dates = _get_earnings_dates_yfinance(symbol)

    for dt in dates:
        # Normalise timezone
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ET)
        elif dt.tzinfo != ET:
            try:
                dt = dt.astimezone(ET)
            except Exception as e:
                logger.warning("Timezone conversion failed for %s: %s", dt, e)

        delta = dt - now
        if timedelta(days=-1) <= delta <= lookahead:
            return True, dt.strftime("%Y-%m-%d")

    return False, None


def scan_earnings(
    symbols: List[str],
    buffer_days: int = _BUFFER_DAYS,
    output_path: str = _EARNINGS_PAUSE_PATH,
) -> Dict[str, Any]:
    """Scan all tickers for upcoming earnings and write a pause file.

    Args:
        symbols: List of ticker symbols in the universe.
        buffer_days: Days before earnings to flag.
        output_path: Path to write ``earnings_pause.json``.

    Returns:
        Dict of the pause file contents.
    """
    now = datetime.now(ET)
    flagged: Dict[str, Dict[str, Any]] = {}

    for sym in symbols:
        near, date_str = check_earnings_proximity(sym, buffer_days)
        if near and date_str:
            flagged[sym] = {
                "paused": True,
                "earnings_date": date_str,
                "reason": f"Earnings within {buffer_days} trading days ({date_str})",
                "flagged_at": now.isoformat(),
            }
            logger.info("Flagged %s: earnings on %s", sym, date_str)

    pause_data: Dict[str, Any] = {
        "updated_at": now.isoformat(),
        "buffer_days": buffer_days,
        "flagged_tickers": flagged,
        "total_scanned": len(symbols),
        "total_flagged": len(flagged),
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(pause_data, fh, indent=2, default=str)
        logger.info(
            "Earnings scan complete: %d flagged out of %d tickers",
            len(flagged),
            len(symbols),
        )
    except OSError as exc:
        logger.error("Failed to write earnings pause file: %s", exc)

    return pause_data


def check_earnings_paused(
    symbol: str,
    pause_path: str = _EARNINGS_PAUSE_PATH,
) -> Tuple[bool, str]:
    """Check if a ticker is paused due to upcoming earnings.

    Fail-open: returns ``(False, "")`` if the file is missing or corrupt.

    Args:
        symbol: Ticker symbol.
        pause_path: Path to ``earnings_pause.json``.

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
        return True, entry.get("reason", "Upcoming earnings")

    return False, ""
