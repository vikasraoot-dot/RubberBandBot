# tests/unit/test_sip_feed.py
"""Tests for SIP feed delay handling and feed migration guards."""
from __future__ import annotations

import ast
import datetime as dt
import os
from unittest.mock import patch, MagicMock, call
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
FAKE_NOW = dt.datetime(2026, 2, 12, 18, 0, 0, tzinfo=dt.timezone.utc)  # 1 PM ET


def _patch_now(fake_now):
    """Patch _now_utc in data module."""
    return patch("RubberBand.src.data._now_utc", return_value=fake_now)


def _patch_http_ok():
    """Patch requests.get to return an empty-bars 200 response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"bars": {}, "next_page_token": None}
    return patch("requests.get", return_value=mock_resp)


# ---------------------------------------------------------------------------
# SIP Delay Cap Tests
# ---------------------------------------------------------------------------
class TestSipDelayCap:
    """Verify that fetch_latest_bars caps end_dt for SIP feed."""

    def test_sip_feed_caps_end_to_16min_ago(self):
        """When feed='sip' and end=None, end_dt should be capped to now-16min."""
        from RubberBand.src.data import fetch_latest_bars

        with _patch_now(FAKE_NOW), _patch_http_ok() as mock_get:
            fetch_latest_bars(
                ["AAPL"], "15Min", history_days=1,
                feed="sip", verbose=False,
                key="fake", secret="fake",
            )
            call_args = mock_get.call_args
            params = call_args.kwargs.get("params") or call_args[1].get("params", {})
            end_sent = params.get("end", "")
            expected_end = (FAKE_NOW - dt.timedelta(minutes=16)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            assert end_sent == expected_end, (
                f"SIP end should be capped to {expected_end}, got {end_sent}"
            )

    def test_iex_feed_does_not_cap_end(self):
        """When feed='iex' and end=None, end_dt should be current time."""
        from RubberBand.src.data import fetch_latest_bars

        with _patch_now(FAKE_NOW), _patch_http_ok() as mock_get:
            fetch_latest_bars(
                ["AAPL"], "15Min", history_days=1,
                feed="iex", verbose=False,
                key="fake", secret="fake",
            )
            call_args = mock_get.call_args
            params = call_args.kwargs.get("params") or call_args[1].get("params", {})
            end_sent = params.get("end", "")
            expected_end = FAKE_NOW.replace(microsecond=0).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            assert end_sent == expected_end, (
                f"IEX end should be current time {expected_end}, got {end_sent}"
            )

    def test_sip_explicit_end_not_overridden(self):
        """When caller provides explicit end, SIP delay cap should NOT apply."""
        from RubberBand.src.data import fetch_latest_bars

        explicit_end = "2026-02-12T16:00:00Z"
        with _patch_now(FAKE_NOW), _patch_http_ok() as mock_get:
            fetch_latest_bars(
                ["AAPL"], "15Min", history_days=1,
                feed="sip", verbose=False,
                key="fake", secret="fake",
                end=explicit_end,
            )
            call_args = mock_get.call_args
            params = call_args.kwargs.get("params") or call_args[1].get("params", {})
            end_sent = params.get("end", "")
            # Explicit end must pass through exactly (preserved as ISO Z suffix)
            assert end_sent == "2026-02-12T16:00:00Z", (
                f"Explicit end should be preserved as-is, got {end_sent}"
            )

    def test_sip_uppercase_feed_still_caps(self):
        """Feed='SIP' (uppercase) should still trigger the delay cap."""
        from RubberBand.src.data import fetch_latest_bars

        with _patch_now(FAKE_NOW), _patch_http_ok() as mock_get:
            fetch_latest_bars(
                ["AAPL"], "15Min", history_days=1,
                feed="SIP", verbose=False,
                key="fake", secret="fake",
            )
            call_args = mock_get.call_args
            params = call_args.kwargs.get("params") or call_args[1].get("params", {})
            end_sent = params.get("end", "")
            expected_end = (FAKE_NOW - dt.timedelta(minutes=16)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            assert end_sent == expected_end
            # Also verify feed is sent as lowercase to Alpaca
            assert params.get("feed") == "sip", (
                f"Feed should be normalized to lowercase, got {params.get('feed')}"
            )

    def test_sip_start_before_end(self):
        """Start param must always be before end param (valid time window)."""
        from RubberBand.src.data import fetch_latest_bars

        with _patch_now(FAKE_NOW), _patch_http_ok() as mock_get:
            fetch_latest_bars(
                ["AAPL"], "15Min", history_days=5,
                feed="sip", verbose=False,
                key="fake", secret="fake",
            )
            call_args = mock_get.call_args
            params = call_args.kwargs.get("params") or call_args[1].get("params", {})
            start_sent = params.get("start", "")
            end_sent = params.get("end", "")
            assert start_sent < end_sent, (
                f"Start ({start_sent}) must be before end ({end_sent})"
            )

    def test_bars_use_split_adjustment(self):
        """Bars must request split-adjusted data to handle stock splits correctly.

        Without adjustment='split', raw bars mix pre-split and post-split prices,
        causing corrupted indicators (e.g. RSI=6.7 on a 5:1 split) and invalid
        bracket orders (negative stop-loss prices).
        """
        from RubberBand.src.data import fetch_latest_bars

        with _patch_now(FAKE_NOW), _patch_http_ok() as mock_get:
            fetch_latest_bars(
                ["AAPL"], "1Week", history_days=90,
                feed="sip", verbose=False,
                key="fake", secret="fake",
            )
            call_args = mock_get.call_args
            params = call_args.kwargs.get("params") or call_args[1].get("params", {})
            adjustment = params.get("adjustment")
            assert adjustment == "split", (
                f"Bars must use adjustment='split' for stock split safety, "
                f"got adjustment='{adjustment}'"
            )


# ---------------------------------------------------------------------------
# Config Validation Tests
# ---------------------------------------------------------------------------
class TestSipConfig:
    """Verify config files specify SIP feed."""

    def test_config_yaml_uses_sip(self):
        """config.yaml should specify 'sip' feed."""
        import yaml
        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "RubberBand", "config.yaml"
        )
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg.get("feed") == "sip"

    def test_config_weekly_uses_sip(self):
        """config_weekly.yaml should specify 'sip' feed."""
        import yaml
        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "RubberBand", "config_weekly.yaml"
        )
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg.get("feed") == "sip"

    def test_fetch_latest_bars_default_feed_is_sip(self):
        """fetch_latest_bars default feed parameter should be 'sip'."""
        from RubberBand.src.data import fetch_latest_bars
        import inspect
        sig = inspect.signature(fetch_latest_bars)
        default_feed = sig.parameters["feed"].default
        assert default_feed == "sip", (
            f"fetch_latest_bars default feed should be 'sip', got '{default_feed}'"
        )


# ---------------------------------------------------------------------------
# Guard Tests — prevent accidental SIP on real-time endpoints
# ---------------------------------------------------------------------------
class TestRealtimeQuoteGuard:
    """Ensure get_latest_quote does NOT use SIP feed (would add 15-min delay)."""

    def test_get_latest_quote_no_feed_param(self):
        """get_latest_quote must NOT pass a 'feed' param to the API.

        Real-time quotes on free tier use Alpaca's default (IEX).
        If someone accidentally adds feed='sip', exit quotes would be
        15 minutes stale — a capital-risk bug.
        """
        from RubberBand.src.data import get_latest_quote

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "quote": {"bp": 150.0, "ap": 150.05, "bs": 100, "as": 200}
        }

        with patch("requests.get", return_value=mock_resp) as mock_get:
            get_latest_quote(
                base_url="https://paper-api.alpaca.markets",
                key="fake", secret="fake",
                symbol="AAPL",
            )
            call_args = mock_get.call_args
            # Verify no 'params' kwarg at all, OR if present, no 'feed' key
            params = call_args.kwargs.get("params") or {}
            assert "feed" not in params, (
                "get_latest_quote must NOT pass feed param (would delay real-time quotes)"
            )
            # Also check the URL doesn't embed feed as query string
            url_called = call_args.args[0] if call_args.args else call_args.kwargs.get("url", "")
            assert "feed=" not in url_called, (
                f"get_latest_quote URL must not contain feed query param, got: {url_called}"
            )

    def test_get_latest_quote_source_code_no_feed(self):
        """AST check: get_latest_quote must not contain 'feed' or 'sip' string literals."""
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "RubberBand", "src", "data.py"
        )
        with open(src_path, encoding="utf-8") as f:
            tree = ast.parse(f.read())

        blocked_strings = {"sip", "feed"}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "get_latest_quote":
                for child in ast.walk(node):
                    if isinstance(child, ast.Constant) and isinstance(child.value, str):
                        assert child.value not in blocked_strings, (
                            f"get_latest_quote must NOT contain '{child.value}' string — "
                            "real-time quotes must not reference any data feed parameter"
                        )
                break


# ---------------------------------------------------------------------------
# RegimeManager Feed Tests
# ---------------------------------------------------------------------------
class TestRegimeManagerFeed:
    """Verify regime manager uses correct feeds for daily vs intraday checks."""

    def test_daily_update_uses_sip_feed(self):
        """RegimeManager.update() should use feed='sip' for daily VIXY bars."""
        from RubberBand.src.regime_manager import RegimeManager
        import pandas as pd

        rm = RegimeManager(verbose=False)
        mock_df = pd.DataFrame({
            "open": [30.0] * 25,
            "high": [31.0] * 25,
            "low": [29.0] * 25,
            "close": [30.0] * 25,
            "volume": [1000000] * 25,
        }, index=pd.date_range("2026-01-01", periods=25, freq="D"))

        with patch("RubberBand.src.regime_manager.fetch_latest_bars") as mock_fetch:
            mock_fetch.return_value = ({"VIXY": mock_df}, [])
            rm.update()

            # First call should be the daily update — extract feed then assert
            daily_call = mock_fetch.call_args_list[0]
            feed_used = None
            if "feed" in (daily_call.kwargs or {}):
                feed_used = daily_call.kwargs["feed"]
            elif len(daily_call.args) >= 4:
                feed_used = daily_call.args[3]
            assert feed_used == "sip", (
                f"Daily update should use feed='sip', got feed='{feed_used}'"
            )

    def test_intraday_check_uses_iex_feed(self):
        """RegimeManager.check_intraday() should use feed='iex' for real-time detection."""
        from RubberBand.src.regime_manager import RegimeManager
        import pandas as pd

        rm = RegimeManager(verbose=False)
        # Set up daily state first
        mock_daily = pd.DataFrame({
            "open": [40.0] * 25,
            "high": [41.0] * 25,
            "low": [39.0] * 25,
            "close": [40.0] * 25,
            "volume": [1000000] * 25,
        }, index=pd.date_range("2026-01-01", periods=25, freq="D"))

        mock_intraday = pd.DataFrame({
            "close": [40.5],
        }, index=[dt.datetime.now() - dt.timedelta(minutes=5)])

        with patch("RubberBand.src.regime_manager.fetch_latest_bars") as mock_fetch:
            # First call: daily update
            mock_fetch.return_value = ({"VIXY": mock_daily}, [])
            rm.update()

            # Second call: intraday check
            mock_fetch.return_value = ({"VIXY": mock_intraday}, [])
            rm.get_effective_regime()

            # The intraday call should use feed='iex'
            intraday_call = mock_fetch.call_args_list[-1]
            # check_intraday passes feed as keyword or positional arg #4
            feed_used = None
            if "feed" in (intraday_call.kwargs or {}):
                feed_used = intraday_call.kwargs["feed"]
            elif len(intraday_call.args) >= 4:
                feed_used = intraday_call.args[3]
            assert feed_used == "iex", (
                f"Intraday VIXY check should use feed='iex' for real-time data, "
                f"got feed='{feed_used}'"
            )
