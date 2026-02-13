# tests/unit/test_order_rejection_tracking.py
"""Tests for order rejection alert tracking and counting."""
from __future__ import annotations

import json
import os
import tempfile

import pytest


# ---------------------------------------------------------------------------
# emit_order_rejection_alert
# ---------------------------------------------------------------------------
class TestEmitOrderRejectionAlert:
    """Verify that rejection alerts are written to alerts.jsonl."""

    def test_writes_alert_to_jsonl(self, tmp_path):
        """emit_order_rejection_alert should append a JSONL line."""
        from RubberBand.src.watchdog.intraday_monitor import emit_order_rejection_alert

        alerts_file = str(tmp_path / "alerts.jsonl")
        emit_order_rejection_alert(
            bot_tag="15M_STK",
            symbol="AAPL",
            side="buy",
            qty=10,
            reason="insufficient_buying_power",
            error_code="40310000",
            alerts_path=alerts_file,
        )

        assert os.path.exists(alerts_file)
        with open(alerts_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1

        alert = json.loads(lines[0])
        assert alert["bot_tag"] == "15M_STK"
        assert alert["event"] == "ORDER_REJECTED"
        assert alert["level"] == "WARNING"
        assert alert["symbol"] == "AAPL"
        assert alert["side"] == "buy"
        assert alert["qty"] == 10
        assert alert["reason"] == "insufficient_buying_power"
        assert alert["error_code"] == "40310000"
        assert "ts" in alert  # timestamp auto-set

    def test_appends_multiple_alerts(self, tmp_path):
        """Multiple calls should append, not overwrite."""
        from RubberBand.src.watchdog.intraday_monitor import emit_order_rejection_alert

        alerts_file = str(tmp_path / "alerts.jsonl")
        for sym in ["AAPL", "MSFT", "GOOG"]:
            emit_order_rejection_alert(
                bot_tag="WK_STK", symbol=sym, side="buy",
                qty=5, reason="test", alerts_path=alerts_file,
            )

        with open(alerts_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 3
        symbols = [json.loads(l)["symbol"] for l in lines]
        assert symbols == ["AAPL", "MSFT", "GOOG"]


# ---------------------------------------------------------------------------
# _count_rejections_today
# ---------------------------------------------------------------------------
class TestCountRejectionsToday:
    """Verify the IntraDayMonitor rejection counting method."""

    def _make_monitor(self, tmp_path, alerts_content=""):
        """Create a monitor with a temporary alerts file."""
        from RubberBand.src.watchdog.intraday_monitor import IntraDayMonitor

        alerts_file = str(tmp_path / "alerts.jsonl")
        if alerts_content:
            with open(alerts_file, "w") as f:
                f.write(alerts_content)

        monitor = IntraDayMonitor.__new__(IntraDayMonitor)
        monitor._alerts_path = alerts_file
        return monitor

    def test_counts_rejections_for_today(self, tmp_path):
        """Should count ORDER_REJECTED events matching today's date."""
        lines = [
            json.dumps({"event": "ORDER_REJECTED", "bot_tag": "15M_STK", "ts": "2026-02-13T10:00:00"}),
            json.dumps({"event": "ORDER_REJECTED", "bot_tag": "15M_STK", "ts": "2026-02-13T11:00:00"}),
            json.dumps({"event": "ORDER_REJECTED", "bot_tag": "WK_STK", "ts": "2026-02-13T10:30:00"}),
            json.dumps({"event": "WARNING", "bot_tag": "15M_STK", "ts": "2026-02-13T10:00:00"}),  # not rejection
            json.dumps({"event": "ORDER_REJECTED", "bot_tag": "15M_STK", "ts": "2026-02-12T15:00:00"}),  # yesterday
        ]
        monitor = self._make_monitor(tmp_path, "\n".join(lines) + "\n")

        counts = monitor._count_rejections_today("2026-02-13")
        assert counts == {"15M_STK": 2, "WK_STK": 1}

    def test_empty_file_returns_empty(self, tmp_path):
        """No alerts file should return empty dict."""
        monitor = self._make_monitor(tmp_path)
        counts = monitor._count_rejections_today("2026-02-13")
        assert counts == {}

    def test_no_rejections_returns_empty(self, tmp_path):
        """File with no ORDER_REJECTED events returns empty."""
        lines = [
            json.dumps({"event": "PAUSE", "bot_tag": "15M_STK", "ts": "2026-02-13T10:00:00"}),
        ]
        monitor = self._make_monitor(tmp_path, "\n".join(lines) + "\n")
        counts = monitor._count_rejections_today("2026-02-13")
        assert counts == {}


# ---------------------------------------------------------------------------
# Scanner SIP feed guard
# ---------------------------------------------------------------------------
class TestScannerFeedSip:
    """Verify scanner files use SIP feed, not IEX."""

    @pytest.mark.parametrize("scanner_file", [
        "scan_for_bot.py",
        "scan_weekly_candidates.py",
        "scan_rsi_snapshot.py",
    ])
    def test_scanner_uses_sip_not_iex(self, scanner_file):
        """Scanner files must not contain feed='iex' hardcoded calls."""
        import ast

        scanner_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "RubberBand", "scripts", scanner_file
        )
        with open(scanner_path, encoding="utf-8") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in getattr(node, "keywords", []):
                    if kw.arg == "feed" and isinstance(kw.value, ast.Constant):
                        assert kw.value.value != "iex", (
                            f"{scanner_file} has feed='iex' — should be 'sip' "
                            f"to match live trading data feed"
                        )

    def test_scan_universe_default_is_sip(self):
        """scan_universe.py fallback default should be 'sip', not 'iex'."""
        scanner_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "RubberBand", "scripts", "scan_universe.py"
        )
        with open(scanner_path, encoding="utf-8") as f:
            source = f.read()

        # Should not have cfg.get("feed", "iex") anywhere
        assert 'cfg.get("feed", "iex")' not in source, (
            "scan_universe.py default feed fallback should be 'sip', not 'iex'"
        )
