"""Tests for yfinance fallback in fetch_latest_bars().

Verifies that:
- yfinance recovers empty symbols when Alpaca SIP returns nothing
- Fallback is skipped when HTTP errors are present (401/403)
- Fallback is skipped when yf_fallback=False
- Graceful handling when yfinance is not installed
- Column normalization and UTC timezone conversion
- RTH filtering is applied to yfinance data
- dollar_vol_avg column is calculated
- Unsupported timeframes are gracefully skipped
- Partial batch failures don't block other symbols
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from RubberBand.src.data import _yf_fetch_bars, fetch_latest_bars


# ---------------------------------------------------------------------------
# Fixtures / Helpers
# ---------------------------------------------------------------------------

def _make_yf_df(
    sym: str,
    rows: int = 50,
    interval_min: int = 15,
    start: dt.datetime | None = None,
) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame mimicking yfinance output."""
    if start is None:
        start = dt.datetime(2026, 2, 18, 9, 30, tzinfo=dt.timezone.utc)
    idx = pd.date_range(start=start, periods=rows, freq=f"{interval_min}min", tz="UTC")
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.standard_normal(rows) * 0.5)
    return pd.DataFrame({
        "Open": close - rng.uniform(0.1, 0.5, rows),
        "High": close + rng.uniform(0.1, 0.5, rows),
        "Low": close - rng.uniform(0.1, 0.8, rows),
        "Close": close,
        "Volume": rng.integers(1000, 50000, rows),
    }, index=idx)


def _make_multi_yf_df(syms: List[str], rows: int = 50) -> pd.DataFrame:
    """Build a MultiIndex DataFrame like yf.download with multiple tickers."""
    frames = {}
    for sym in syms:
        frames[sym] = _make_yf_df(sym, rows=rows)

    # Combine into MultiIndex columns (ticker, field)
    combined = pd.concat(frames, axis=1)
    return combined


# ---------------------------------------------------------------------------
# Tests for _yf_fetch_bars helper
# ---------------------------------------------------------------------------

class TestYfFetchBars:
    """Tests for the _yf_fetch_bars() helper function."""

    def test_yf_fallback_recovers_empty_symbols(self):
        """yfinance should return data for requested symbols."""
        syms = ["AAPL", "MSFT", "GOOG"]
        mock_df = _make_multi_yf_df(syms)

        with patch("yfinance.download", return_value=mock_df) as mock_dl:
            result = _yf_fetch_bars(syms, timeframe="15Min", history_days=10)

        assert len(result) == 3
        for sym in syms:
            assert sym in result
            df = result[sym]
            assert "close" in df.columns
            assert "volume" in df.columns
            assert df.index.tz is not None  # Should be UTC

    def test_yf_fallback_graceful_when_not_installed(self):
        """ImportError should be handled gracefully (returns empty dict)."""
        import builtins
        import sys
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "yfinance":
                raise ImportError("No module named 'yfinance'")
            return real_import(name, *args, **kwargs)

        # Remove yfinance from sys.modules cache so the lazy import is forced
        # to go through __import__ again (prevents test-order-dependent flakiness)
        saved = sys.modules.pop("yfinance", None)
        try:
            with patch("builtins.__import__", side_effect=mock_import):
                result = _yf_fetch_bars(["AAPL"], timeframe="15Min")
        finally:
            if saved is not None:
                sys.modules["yfinance"] = saved

        assert result == {}

    def test_yf_unsupported_timeframe_skipped(self):
        """Unknown timeframe should return empty dict without error."""
        result = _yf_fetch_bars(["AAPL"], timeframe="3Min")
        assert result == {}

    def test_yf_column_and_tz_normalization(self):
        """Columns should be lowercase, index should be UTC."""
        mock_df = _make_multi_yf_df(["SPY"])

        with patch("yfinance.download", return_value=mock_df):
            result = _yf_fetch_bars(["SPY"], timeframe="15Min")

        df = result["SPY"]
        # All columns should be lowercase
        for col in df.columns:
            assert col == col.lower(), f"Column '{col}' should be lowercase"
        # Index should be UTC
        assert str(df.index.tz) == "UTC"

    def test_yf_partial_batch_failure(self):
        """One symbol failing should not block others."""
        # Build df for AAPL only, BADTICKER missing from MultiIndex
        mock_df = _make_multi_yf_df(["AAPL"])

        with patch("yfinance.download", return_value=mock_df):
            result = _yf_fetch_bars(["AAPL", "BADTICKER"], timeframe="15Min")

        assert "AAPL" in result
        assert "BADTICKER" not in result


# ---------------------------------------------------------------------------
# Tests for yfinance fallback inside fetch_latest_bars
# ---------------------------------------------------------------------------

class TestFetchLatestBarsYfFallback:
    """Tests for the yfinance fallback integration in fetch_latest_bars."""

    def _mock_alpaca_empty(self, *args, **kwargs):
        """Mock requests.get to return empty bars JSON."""
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"bars": {}, "next_page_token": None}
        return resp

    def _mock_alpaca_401(self, *args, **kwargs):
        """Mock requests.get to return 401."""
        resp = MagicMock()
        resp.status_code = 401
        resp.raise_for_status = MagicMock()
        return resp

    def test_yf_fallback_fires_on_empty_alpaca(self):
        """When Alpaca returns empty, yfinance should be tried."""
        syms = ["AAPL", "MSFT"]
        mock_yf_df = _make_multi_yf_df(syms)

        with patch("requests.get", side_effect=self._mock_alpaca_empty), \
             patch("RubberBand.src.data._yf_fetch_bars") as mock_yf:
            # Return normalized DataFrames (lowercase cols, UTC index)
            normalized = {}
            for sym in syms:
                df = mock_yf_df[sym].copy()
                df.columns = [c.lower() for c in df.columns]
                normalized[sym] = df
            mock_yf.return_value = normalized

            # Pass key/secret to avoid missing-api-key http_error
            bars_map, meta = fetch_latest_bars(
                syms, timeframe="15Min", verbose=False, yf_fallback=True,
                key="test-key", secret="test-secret",
            )

        # yfinance should have been called
        mock_yf.assert_called_once()
        assert "AAPL" in bars_map
        assert "MSFT" in bars_map

    def test_yf_fallback_skipped_when_http_errors(self):
        """When Alpaca returns 401/403, yfinance should NOT be tried."""
        with patch("requests.get", side_effect=self._mock_alpaca_401), \
             patch("RubberBand.src.data._yf_fetch_bars") as mock_yf:
            mock_yf.return_value = {}
            bars_map, meta = fetch_latest_bars(
                ["AAPL"], timeframe="15Min", verbose=False, yf_fallback=True,
                key="test-key", secret="test-secret",
            )

        # yfinance should NOT have been called (http_errors present)
        mock_yf.assert_not_called()
        assert len(meta["http_errors"]) > 0

    def test_yf_fallback_skipped_when_disabled(self):
        """yf_fallback=False should skip the fallback entirely."""
        with patch("requests.get", side_effect=self._mock_alpaca_empty), \
             patch("RubberBand.src.data._yf_fetch_bars") as mock_yf:
            mock_yf.return_value = {}
            bars_map, meta = fetch_latest_bars(
                ["AAPL"], timeframe="15Min", verbose=False, yf_fallback=False,
                key="test-key", secret="test-secret",
            )

        mock_yf.assert_not_called()
        assert len(bars_map) == 0

    def test_yf_rth_filtering_applied(self):
        """yfinance bars should have RTH filtering applied."""
        # Create bars spanning outside RTH (e.g., 4 AM UTC = before market)
        sym = "SPY"
        early_start = dt.datetime(2026, 2, 18, 4, 0, tzinfo=dt.timezone.utc)  # ~11 PM ET
        df = _make_yf_df(sym, rows=100, start=early_start)
        df.columns = [c.lower() for c in df.columns]

        with patch("requests.get", side_effect=self._mock_alpaca_empty), \
             patch("RubberBand.src.data._yf_fetch_bars", return_value={sym: df}):
            bars_map, _ = fetch_latest_bars(
                [sym], timeframe="15Min", verbose=False, rth_only=True,
                key="test-key", secret="test-secret", yf_fallback=True,
            )

        assert sym in bars_map, f"Expected {sym} in bars_map after yf fallback"
        result_df = bars_map[sym]
        # Check all bars are within RTH (09:30-15:55 ET)
        local = result_df.tz_convert("US/Eastern")
        for ts in local.index:
            minutes = ts.hour * 60 + ts.minute
            assert 570 <= minutes <= 955, \
                f"Bar at {ts} is outside RTH window"

    def test_yf_dollar_vol_calculated(self):
        """yfinance bars should have dollar_vol_avg column."""
        sym = "AAPL"
        # Create bars during RTH hours
        rth_start = dt.datetime(2026, 2, 18, 14, 30, tzinfo=dt.timezone.utc)  # 9:30 AM ET
        df = _make_yf_df(sym, rows=30, start=rth_start)
        df.columns = [c.lower() for c in df.columns]

        with patch("requests.get", side_effect=self._mock_alpaca_empty), \
             patch("RubberBand.src.data._yf_fetch_bars", return_value={sym: df}):
            bars_map, _ = fetch_latest_bars(
                [sym], timeframe="15Min", verbose=False,
                key="test-key", secret="test-secret", yf_fallback=True,
            )

        assert sym in bars_map, f"Expected {sym} in bars_map after yf fallback"
        assert "dollar_vol_avg" in bars_map[sym].columns
