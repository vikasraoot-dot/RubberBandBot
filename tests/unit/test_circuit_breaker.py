# === tests/unit/test_circuit_breaker.py ===
"""Unit tests for PortfolioGuard and ConnectivityGuard circuit breakers."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from RubberBand.src.circuit_breaker import (
    CircuitBreakerExc,
    ConnectivityGuard,
    PortfolioGuard,
)


# ---------------------------------------------------------------------------
# PortfolioGuard
# ---------------------------------------------------------------------------

class TestPortfolioGuardPeakTracking:
    """Verify peak equity tracking and state persistence."""

    def test_update_tracks_new_peak(self, tmp_path: str) -> None:
        state_file = os.path.join(str(tmp_path), "guard.json")
        guard = PortfolioGuard(state_file, max_drawdown_pct=0.10)

        guard.update(10_000.0)
        assert float(guard.peak_equity) == 10_000.0

        guard.update(11_000.0)
        assert float(guard.peak_equity) == 11_000.0

    def test_peak_does_not_decrease(self, tmp_path: str) -> None:
        state_file = os.path.join(str(tmp_path), "guard.json")
        guard = PortfolioGuard(state_file, max_drawdown_pct=0.10)

        guard.update(10_000.0)
        guard.update(9_500.0)  # 5% drawdown, within limit
        assert float(guard.peak_equity) == 10_000.0

    def test_state_persists_across_instances(self, tmp_path: str) -> None:
        state_file = os.path.join(str(tmp_path), "guard.json")

        guard1 = PortfolioGuard(state_file, max_drawdown_pct=0.10)
        guard1.update(50_000.0)

        guard2 = PortfolioGuard(state_file, max_drawdown_pct=0.10)
        assert float(guard2.peak_equity) == 50_000.0


class TestPortfolioGuardDrawdown:
    """Verify drawdown detection and halt behavior."""

    def test_halts_on_drawdown_exceeding_threshold(self, tmp_path: str) -> None:
        state_file = os.path.join(str(tmp_path), "guard.json")
        guard = PortfolioGuard(state_file, max_drawdown_pct=0.10)

        guard.update(10_000.0)  # Set peak

        with pytest.raises(CircuitBreakerExc, match="Drawdown"):
            guard.update(8_900.0)  # 11% drawdown > 10% limit

    def test_does_not_halt_within_threshold(self, tmp_path: str) -> None:
        state_file = os.path.join(str(tmp_path), "guard.json")
        guard = PortfolioGuard(state_file, max_drawdown_pct=0.10)

        guard.update(10_000.0)
        guard.update(9_100.0)  # 9% drawdown, within limit — should not raise

    def test_halted_state_persists_and_blocks_future_updates(self, tmp_path: str) -> None:
        state_file = os.path.join(str(tmp_path), "guard.json")
        guard = PortfolioGuard(state_file, max_drawdown_pct=0.10)

        guard.update(10_000.0)
        with pytest.raises(CircuitBreakerExc):
            guard.update(8_000.0)  # Triggers halt

        # New instance — halted state should persist from file
        guard2 = PortfolioGuard(state_file, max_drawdown_pct=0.10)
        assert guard2.halted is True

        with pytest.raises(CircuitBreakerExc, match="Manual Reset Required"):
            guard2.update(12_000.0)  # Even recovery doesn't clear halt

    def test_zero_peak_equity_skips_drawdown_check(self, tmp_path: str) -> None:
        state_file = os.path.join(str(tmp_path), "guard.json")
        guard = PortfolioGuard(state_file, max_drawdown_pct=0.10)

        # First update with 0 — should not raise
        guard.update(0.0)

    def test_exact_threshold_does_not_halt(self, tmp_path: str) -> None:
        state_file = os.path.join(str(tmp_path), "guard.json")
        guard = PortfolioGuard(state_file, max_drawdown_pct=0.10)

        guard.update(10_000.0)
        guard.update(9_000.0)  # Exactly 10% drawdown — uses > not >=

    def test_missing_state_file_starts_fresh(self, tmp_path: str) -> None:
        state_file = os.path.join(str(tmp_path), "nonexistent_guard.json")
        guard = PortfolioGuard(state_file, max_drawdown_pct=0.10)

        assert float(guard.peak_equity) == 0.0
        assert guard.halted is False


# ---------------------------------------------------------------------------
# ConnectivityGuard
# ---------------------------------------------------------------------------

class TestConnectivityGuard:
    """Verify consecutive error tracking and halt behavior."""

    def test_halts_on_max_consecutive_errors(self) -> None:
        guard = ConnectivityGuard(max_errors=3)

        guard.record_error(Exception("err1"))
        guard.record_error(Exception("err2"))

        with pytest.raises(CircuitBreakerExc, match="consecutive API errors"):
            guard.record_error(Exception("err3"))

    def test_success_resets_error_counter(self) -> None:
        guard = ConnectivityGuard(max_errors=3)

        guard.record_error(Exception("err1"))
        guard.record_error(Exception("err2"))
        guard.record_success()  # Reset

        # Should not halt — counter was reset
        guard.record_error(Exception("err3"))
        guard.record_error(Exception("err4"))

        # NOW it should halt (2 errors after reset → err5 = 3rd)
        with pytest.raises(CircuitBreakerExc):
            guard.record_error(Exception("err5"))

    def test_single_error_does_not_halt(self) -> None:
        guard = ConnectivityGuard(max_errors=5)
        guard.record_error(Exception("one"))
        # Should not raise — only 1/5
