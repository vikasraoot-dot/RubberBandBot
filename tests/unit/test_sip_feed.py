# tests/unit/test_sip_feed.py
"""Tests for SIP feed delay handling in fetch_latest_bars."""
from __future__ import annotations

import datetime as dt
from unittest.mock import patch, MagicMock
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
# Tests
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
            # Inspect the 'end' param sent to the API
            call_args = mock_get.call_args
            params = call_args.kwargs.get("params") or call_args[1].get("params", {})
            end_sent = params.get("end", "")
            # Should be 16 minutes before FAKE_NOW
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
            # Explicit end should pass through (may be reformatted but must match the date/time)
            assert "2026-02-12" in end_sent and "16:00:00" in end_sent, (
                f"Explicit end should be preserved, got {end_sent}"
            )

    def test_config_yaml_uses_sip(self):
        """config.yaml should specify 'sip' feed."""
        import yaml
        import os
        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "RubberBand", "config.yaml"
        )
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg.get("feed") == "sip"

    def test_config_weekly_uses_sip(self):
        """config_weekly.yaml should specify 'sip' feed."""
        import yaml
        import os
        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "RubberBand", "config_weekly.yaml"
        )
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg.get("feed") == "sip"
