"""Tests for automatic daily profit lock reset in IntraDayMonitor.

Verifies that:
- Day change triggers automatic reset_daily()
- Same-day cycles do NOT trigger reset_daily()
- First-ever cycle (no prior date) does NOT trigger reset_daily()
- AUTO_DAILY_RESET alert is emitted on day change
- Weekend-to-Monday transitions trigger reset
- Reproduces the Feb 16 stuck lock bug scenario
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import MagicMock, call, patch

import pytest

from RubberBand.src.watchdog.intraday_monitor import (
    IntraDayMonitor,
    _load_json,
    _save_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_monitor(
    tmpdir: str,
    health_date: str | None = None,
    pause_flags: Dict[str, Any] | None = None,
    profit_locks: Dict[str, Any] | None = None,
) -> IntraDayMonitor:
    """Create an IntraDayMonitor wired to temp files for isolated tests."""
    config_path = os.path.join(tmpdir, "watchdog_config.yaml")
    health_path = os.path.join(tmpdir, "intraday_health.json")
    pause_flags_path = os.path.join(tmpdir, "bot_pause_flags.json")
    profit_locks_path = os.path.join(tmpdir, "profit_locks.json")
    alerts_path = os.path.join(tmpdir, "alerts.jsonl")

    # Minimal config so _load_config doesn't blow up
    with open(config_path, "w") as f:
        f.write("thresholds:\n  warn_loss: -50\n  pause_loss: -100\n")

    # Seed health with a date if provided
    if health_date is not None:
        _save_json(health_path, {"date": health_date})

    # Seed pause flags / profit locks
    if pause_flags is not None:
        _save_json(pause_flags_path, pause_flags)
    if profit_locks is not None:
        _save_json(profit_locks_path, profit_locks)

    mon = IntraDayMonitor(
        config_path=config_path,
        health_path=health_path,
        pause_flags_path=pause_flags_path,
        profit_locks_path=profit_locks_path,
        alerts_path=alerts_path,
    )
    return mon


def _read_alerts(tmpdir: str):
    """Read all alerts from the JSONL file."""
    alerts_path = os.path.join(tmpdir, "alerts.jsonl")
    if not os.path.exists(alerts_path):
        return []
    lines = []
    with open(alerts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines


# Patch external Alpaca calls inside run_cycle so we don't hit the network.
_ACCOUNT = {"equity": "66000", "last_equity": "66000"}
_EMPTY_POSITIONS = []
_EMPTY_FILLS = []


def _patch_run_cycle_deps():
    """Context manager to patch Alpaca calls used inside run_cycle."""
    return patch.multiple(
        "RubberBand.src.data",
        get_account=MagicMock(return_value=_ACCOUNT),
        get_positions=MagicMock(return_value=_EMPTY_POSITIONS),
        get_daily_fills=MagicMock(return_value=_EMPTY_FILLS),
    )


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# Tests: Day-change detection logic
# ---------------------------------------------------------------------------

class TestDayChangeDetection:
    """Verify that run_cycle calls reset_daily at the right time."""

    def test_auto_reset_called_on_day_change(self, tmpdir):
        """Yesterday -> today should call reset_daily()."""
        mon = _make_monitor(tmpdir, health_date="2026-02-17")

        with patch("RubberBand.src.watchdog.intraday_monitor._now_et") as mock_now, \
             _patch_run_cycle_deps(), \
             patch.object(mon, "reset_daily", wraps=mon.reset_daily) as spy:
            mock_now.return_value = datetime(2026, 2, 18, 9, 30)
            mon.run_cycle()

        spy.assert_called_once()

    def test_no_reset_on_same_day(self, tmpdir):
        """Same date should NOT call reset_daily()."""
        mon = _make_monitor(tmpdir, health_date="2026-02-18")

        with patch("RubberBand.src.watchdog.intraday_monitor._now_et") as mock_now, \
             _patch_run_cycle_deps(), \
             patch.object(mon, "reset_daily") as spy:
            mock_now.return_value = datetime(2026, 2, 18, 10, 10)
            mon.run_cycle()

        spy.assert_not_called()

    def test_no_reset_on_first_cycle(self, tmpdir):
        """First cycle ever (no prior health date) should NOT call reset_daily()."""
        mon = _make_monitor(tmpdir, health_date=None)
        assert mon._last_cycle_date is None

        with patch("RubberBand.src.watchdog.intraday_monitor._now_et") as mock_now, \
             _patch_run_cycle_deps(), \
             patch.object(mon, "reset_daily") as spy:
            mock_now.return_value = datetime(2026, 2, 18, 9, 30)
            mon.run_cycle()

        spy.assert_not_called()
        # After cycle, date should be set for future comparisons
        assert mon._last_cycle_date == "2026-02-18"

    def test_weekend_to_monday_triggers_reset(self, tmpdir):
        """Friday -> Monday (2+ day gap) should call reset_daily()."""
        mon = _make_monitor(tmpdir, health_date="2026-02-13")  # Friday

        with patch("RubberBand.src.watchdog.intraday_monitor._now_et") as mock_now, \
             _patch_run_cycle_deps(), \
             patch.object(mon, "reset_daily", wraps=mon.reset_daily) as spy:
            mock_now.return_value = datetime(2026, 2, 16, 9, 30)  # Monday
            mon.run_cycle()

        spy.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Alert emission
# ---------------------------------------------------------------------------

class TestResetAlerts:
    """Verify AUTO_DAILY_RESET alert is emitted correctly."""

    def test_auto_reset_emits_alert(self, tmpdir):
        """AUTO_DAILY_RESET should appear in alerts.jsonl on day change."""
        mon = _make_monitor(tmpdir, health_date="2026-02-17")

        with patch("RubberBand.src.watchdog.intraday_monitor._now_et") as mock_now, \
             _patch_run_cycle_deps():
            mock_now.return_value = datetime(2026, 2, 18, 9, 30)
            mon.run_cycle()

        alerts = _read_alerts(tmpdir)
        reset_alerts = [a for a in alerts if a.get("event") == "AUTO_DAILY_RESET"]
        assert len(reset_alerts) == 1
        assert reset_alerts[0]["prev_date"] == "2026-02-17"
        assert reset_alerts[0]["new_date"] == "2026-02-18"
        assert reset_alerts[0]["level"] == "INFO"

    def test_no_alert_on_same_day(self, tmpdir):
        """No AUTO_DAILY_RESET alert when date hasn't changed."""
        mon = _make_monitor(tmpdir, health_date="2026-02-18")

        with patch("RubberBand.src.watchdog.intraday_monitor._now_et") as mock_now, \
             _patch_run_cycle_deps():
            mock_now.return_value = datetime(2026, 2, 18, 10, 0)
            mon.run_cycle()

        alerts = _read_alerts(tmpdir)
        reset_alerts = [a for a in alerts if a.get("event") == "AUTO_DAILY_RESET"]
        assert len(reset_alerts) == 0


# ---------------------------------------------------------------------------
# Tests: Integration — full reset_daily behavior
# ---------------------------------------------------------------------------

class TestResetDailyIntegration:
    """Verify reset_daily clears state correctly (called via run_cycle)."""

    def test_pause_flags_cleared_on_day_change(self, tmpdir):
        """Paused bot with resume_tomorrow=True should be unpaused after reset."""
        mon = _make_monitor(
            tmpdir,
            health_date="2026-02-17",
            pause_flags={
                "15M_STK": {
                    "paused": True,
                    "reason": "P&L dropped below lock floor",
                    "paused_at": "2026-02-16T09:38:20",
                    "resume_tomorrow": True,
                }
            },
            profit_locks={
                "15M_STK": {
                    "peak_pnl_today": 86.63,
                    "lock_floor": 43.32,
                    "lock_active": True,
                    "lock_triggered": True,
                }
            },
        )

        with patch("RubberBand.src.watchdog.intraday_monitor._now_et") as mock_now, \
             _patch_run_cycle_deps():
            mock_now.return_value = datetime(2026, 2, 18, 9, 30)
            mon.run_cycle()

        # After reset + re-evaluation with P&L=0, the bot should be unpaused
        flags = _load_json(os.path.join(tmpdir, "bot_pause_flags.json"))
        assert flags["15M_STK"]["paused"] is False
        assert flags["15M_STK"]["reason"] == ""

        # Profit lock should be reset to fresh zeros (not the stale 86.63 peak)
        locks = _load_json(os.path.join(tmpdir, "profit_locks.json"))
        assert locks["15M_STK"]["peak_pnl_today"] == 0.0
        assert locks["15M_STK"]["lock_triggered"] is False
        assert locks["15M_STK"]["lock_active"] is False

    def test_profit_lock_cleared_reproduces_bug(self, tmpdir):
        """Reproduce exact Feb 16 bug: peak 86.63, floor 43.32, paused 3 days.

        Without the fix, this lock would persist forever because
        reset_daily() was never called automatically.
        """
        mon = _make_monitor(
            tmpdir,
            health_date="2026-02-16",  # Lock triggered on this day
            pause_flags={
                "15M_STK": {
                    "paused": True,
                    "reason": "P&L $0.0 dropped below lock floor $43.32",
                    "paused_at": "2026-02-16T09:38:20",
                    "resume_tomorrow": True,
                }
            },
            profit_locks={
                "15M_STK": {
                    "peak_pnl_today": 86.63,
                    "lock_floor": 43.32,
                    "lock_active": True,
                    "lock_triggered": True,
                    "lock_triggered_at": "2026-02-16T09:38:20.816488-05:00",
                }
            },
        )

        # Simulate Feb 17 first cycle (should reset)
        with patch("RubberBand.src.watchdog.intraday_monitor._now_et") as mock_now, \
             _patch_run_cycle_deps():
            mock_now.return_value = datetime(2026, 2, 17, 9, 0)
            mon.run_cycle()

        flags = _load_json(os.path.join(tmpdir, "bot_pause_flags.json"))
        assert flags["15M_STK"]["paused"] is False, \
            "15M_STK should have been unpaused on Feb 17 (resume_tomorrow=True)"

        locks = _load_json(os.path.join(tmpdir, "profit_locks.json"))
        assert locks["15M_STK"]["peak_pnl_today"] == 0.0, \
            "Peak P&L should be reset to 0 after daily reset"
        assert locks["15M_STK"]["lock_triggered"] is False

        # Verify the _last_cycle_date was updated
        assert mon._last_cycle_date == "2026-02-17"

    def test_weekend_reset_clears_friday_locks(self, tmpdir):
        """Friday profit lock should be cleared on Monday morning."""
        mon = _make_monitor(
            tmpdir,
            health_date="2026-02-13",  # Friday
            pause_flags={
                "WK_STK": {
                    "paused": True,
                    "reason": "profit lock",
                    "paused_at": "2026-02-13T15:00:00",
                    "resume_tomorrow": True,
                }
            },
            profit_locks={
                "WK_STK": {"peak_pnl_today": 100.0, "lock_floor": 50.0}
            },
        )

        with patch("RubberBand.src.watchdog.intraday_monitor._now_et") as mock_now, \
             _patch_run_cycle_deps():
            mock_now.return_value = datetime(2026, 2, 16, 9, 30)  # Monday
            mon.run_cycle()

        flags = _load_json(os.path.join(tmpdir, "bot_pause_flags.json"))
        assert flags["WK_STK"]["paused"] is False

        locks = _load_json(os.path.join(tmpdir, "profit_locks.json"))
        assert locks["WK_STK"]["peak_pnl_today"] == 0.0
        assert locks["WK_STK"]["lock_triggered"] is False


# ---------------------------------------------------------------------------
# Tests: Direct reset_daily behavior
# ---------------------------------------------------------------------------

class TestResetDailyDirect:
    """Test reset_daily() directly without running full cycle."""

    def test_reset_clears_pause_flags_with_resume_tomorrow(self, tmpdir):
        """Flags with resume_tomorrow=True are cleared."""
        mon = _make_monitor(
            tmpdir,
            pause_flags={
                "15M_STK": {
                    "paused": True,
                    "reason": "profit lock",
                    "paused_at": "2026-02-16T09:38:20",
                    "resume_tomorrow": True,
                },
                "WK_STK": {
                    "paused": True,
                    "reason": "manual override",
                    "paused_at": "2026-02-16T10:00:00",
                    "resume_tomorrow": False,
                },
            },
        )

        mon.reset_daily()

        flags = _load_json(os.path.join(tmpdir, "bot_pause_flags.json"))
        # resume_tomorrow=True -> unpaused
        assert flags["15M_STK"]["paused"] is False
        assert flags["15M_STK"]["reason"] == ""
        # resume_tomorrow=False -> stays paused
        assert flags["WK_STK"]["paused"] is True

    def test_emergency_sets_resume_tomorrow_false(self, tmpdir):
        """EMERGENCY status must set resume_tomorrow=False (CLAUDE.md 3.3)."""
        mon = _make_monitor(tmpdir)

        now = datetime(2026, 2, 18, 10, 0)
        mon._evaluate_bot(
            bot_tag="15M_STK",
            daily_pnl=Decimal("-300"),       # exceeds emergency threshold
            warn_loss=Decimal("-50"),
            pause_loss=Decimal("-100"),
            emergency_loss=Decimal("-200"),
            account_paused=False,
            profit_lock_activation=Decimal("20"),
            profit_lock_pct=Decimal("0.50"),
            now=now,
        )

        flag = mon._pause_flags["15M_STK"]
        assert flag["paused"] is True
        assert flag["resume_tomorrow"] is False, \
            "EMERGENCY pause must NOT auto-resume (requires manual intervention)"

    def test_account_paused_sets_resume_tomorrow_false(self, tmpdir):
        """ACCOUNT_PAUSED status must set resume_tomorrow=False (CLAUDE.md 3.3)."""
        mon = _make_monitor(tmpdir)

        now = datetime(2026, 2, 18, 10, 0)
        mon._evaluate_bot(
            bot_tag="15M_STK",
            daily_pnl=Decimal("0"),          # no loss — pause comes from account
            warn_loss=Decimal("-50"),
            pause_loss=Decimal("-100"),
            emergency_loss=Decimal("-200"),
            account_paused=True,             # account-level override
            profit_lock_activation=Decimal("20"),
            profit_lock_pct=Decimal("0.50"),
            now=now,
        )

        flag = mon._pause_flags["15M_STK"]
        assert flag["paused"] is True
        assert flag["resume_tomorrow"] is False, \
            "ACCOUNT_PAUSED must NOT auto-resume (requires manual intervention)"

    def test_normal_pause_keeps_resume_tomorrow_true(self, tmpdir):
        """Regular PAUSED status should still auto-resume (resume_tomorrow=True)."""
        mon = _make_monitor(tmpdir)

        now = datetime(2026, 2, 18, 10, 0)
        mon._evaluate_bot(
            bot_tag="15M_STK",
            daily_pnl=Decimal("-150"),       # exceeds pause but not emergency
            warn_loss=Decimal("-50"),
            pause_loss=Decimal("-100"),
            emergency_loss=Decimal("-200"),
            account_paused=False,
            profit_lock_activation=Decimal("20"),
            profit_lock_pct=Decimal("0.50"),
            now=now,
        )

        flag = mon._pause_flags["15M_STK"]
        assert flag["paused"] is True
        assert flag["resume_tomorrow"] is True, \
            "Regular PAUSED should auto-resume the next day"

    def test_reset_clears_all_profit_locks(self, tmpdir):
        """All profit locks should be cleared to empty dict."""
        mon = _make_monitor(
            tmpdir,
            profit_locks={
                "15M_STK": {"peak_pnl_today": 86.63, "lock_floor": 43.32},
                "WK_STK": {"peak_pnl_today": 50.0, "lock_floor": 25.0},
            },
        )

        mon.reset_daily()

        locks = _load_json(os.path.join(tmpdir, "profit_locks.json"))
        assert locks == {}
