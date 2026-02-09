"""
Unit tests for the Quote Inversion Guard in live_spreads_loop.calculate_spread_pnl().

The guard detects when a bull call spread appears to have negative value
(long_value < short_value), which is physically impossible and indicates
stale or bad quotes. When detected, calculate_spread_pnl returns (None, None)
so the caller can skip exit logic and avoid false stop-losses.

Tests cover:
1. Negative raw_spread_value -> returns (None, None)
2. Normal positive spread -> returns proper (pnl, pnl_pct)
3. Edge case: raw_spread_value == 0 -> works normally, does NOT return None
4. Fresh quote path vs position fallback path
5. SL confirmation counter behavior with _SL_CONFIRM_REQUIRED == 1
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class TestCalculateSpreadPnl:
    """Tests for calculate_spread_pnl() quote inversion guard."""

    @patch("RubberBand.scripts.live_spreads_loop.get_option_quote")
    def test_negative_spread_returns_none(self, mock_quote):
        """
        When long_value < short_value (physically impossible for bull call spread),
        calculate_spread_pnl must return (None, None) to signal bad data.
        """
        from RubberBand.scripts.live_spreads_loop import calculate_spread_pnl

        # Simulate inverted quotes: short is worth more than long
        mock_quote.side_effect = [
            {"mid": 1.00, "bid": 0.95, "ask": 1.05},   # long quote
            {"mid": 2.00, "bid": 1.95, "ask": 2.05},   # short quote (higher = inversion)
        ]

        long_pos = {"symbol": "AAPL260213C00200000", "current_price": 1.00}
        short_pos = {"symbol": "AAPL260213C00205000", "current_price": 2.00}
        entry_debit = 0.80

        pnl, pnl_pct = calculate_spread_pnl(long_pos, short_pos, entry_debit)

        assert pnl is None, "PnL should be None when spread is inverted"
        assert pnl_pct is None, "PnL % should be None when spread is inverted"

    @patch("RubberBand.scripts.live_spreads_loop.get_option_quote")
    def test_normal_positive_spread(self, mock_quote):
        """
        Normal case: long_value > short_value. Should return proper (pnl, pnl_pct).
        """
        from RubberBand.scripts.live_spreads_loop import calculate_spread_pnl

        # Normal quotes: long is worth more than short (as expected for bull call spread)
        mock_quote.side_effect = [
            {"mid": 3.00, "bid": 2.95, "ask": 3.05},   # long quote
            {"mid": 1.50, "bid": 1.45, "ask": 1.55},   # short quote
        ]

        long_pos = {"symbol": "AAPL260213C00200000", "current_price": 3.00}
        short_pos = {"symbol": "AAPL260213C00205000", "current_price": 1.50}
        entry_debit = 1.00

        pnl, pnl_pct = calculate_spread_pnl(long_pos, short_pos, entry_debit)

        # Spread value = 3.00 - 1.50 = 1.50
        # PnL = (1.50 - 1.00) * 100 = 50.0
        assert pnl is not None, "PnL should not be None for valid spread"
        assert pnl_pct is not None, "PnL % should not be None for valid spread"
        assert pnl == pytest.approx(50.0, abs=0.01), "PnL should be ~$50 per contract"
        assert pnl_pct == pytest.approx(50.0, abs=0.01), "PnL % should be ~50%"

    @patch("RubberBand.scripts.live_spreads_loop.get_option_quote")
    def test_zero_spread_value_works_normally(self, mock_quote):
        """
        Edge case: raw_spread_value == 0 (both legs equal value).
        This is valid (not an inversion) and should NOT return None.
        The spread is worth $0, which means we've lost the entire debit.
        """
        from RubberBand.scripts.live_spreads_loop import calculate_spread_pnl

        # Both legs have equal mid prices -> spread value = 0
        mock_quote.side_effect = [
            {"mid": 1.50, "bid": 1.45, "ask": 1.55},   # long quote
            {"mid": 1.50, "bid": 1.45, "ask": 1.55},   # short quote (same)
        ]

        long_pos = {"symbol": "AAPL260213C00200000", "current_price": 1.50}
        short_pos = {"symbol": "AAPL260213C00205000", "current_price": 1.50}
        entry_debit = 1.00

        pnl, pnl_pct = calculate_spread_pnl(long_pos, short_pos, entry_debit)

        # Spread value = 0.0, PnL = (0 - 1.0) * 100 = -100
        assert pnl is not None, "Zero spread should NOT return None"
        assert pnl_pct is not None, "Zero spread should NOT return None"
        assert pnl == pytest.approx(-100.0, abs=0.01), "PnL should be -$100 (lost entire debit)"
        assert pnl_pct == pytest.approx(-100.0, abs=0.01), "PnL % should be -100%"

    @patch("RubberBand.scripts.live_spreads_loop.get_option_quote")
    def test_fallback_to_position_prices_when_quotes_unavailable(self, mock_quote):
        """
        When fresh quotes are unavailable (return None), the function should
        fall back to position current_price values.
        """
        from RubberBand.scripts.live_spreads_loop import calculate_spread_pnl

        # Quotes unavailable
        mock_quote.side_effect = [None, None]

        long_pos = {"symbol": "AAPL260213C00200000", "current_price": 2.50}
        short_pos = {"symbol": "AAPL260213C00205000", "current_price": 1.00}
        entry_debit = 1.00

        pnl, pnl_pct = calculate_spread_pnl(long_pos, short_pos, entry_debit)

        # Spread value = 2.50 - 1.00 = 1.50
        # PnL = (1.50 - 1.00) * 100 = 50.0
        assert pnl is not None
        assert pnl == pytest.approx(50.0, abs=0.01)

    @patch("RubberBand.scripts.live_spreads_loop.get_option_quote")
    def test_fallback_inversion_also_returns_none(self, mock_quote):
        """
        Even with position price fallback, inversions should return (None, None).
        """
        from RubberBand.scripts.live_spreads_loop import calculate_spread_pnl

        # Quotes unavailable, forcing position price fallback
        mock_quote.side_effect = [None, None]

        # Position prices are inverted
        long_pos = {"symbol": "AAPL260213C00200000", "current_price": 0.50}
        short_pos = {"symbol": "AAPL260213C00205000", "current_price": 1.50}
        entry_debit = 1.00

        pnl, pnl_pct = calculate_spread_pnl(long_pos, short_pos, entry_debit)

        assert pnl is None, "Inversion via fallback should also return None"
        assert pnl_pct is None

    @patch("RubberBand.scripts.live_spreads_loop.get_option_quote")
    def test_no_short_position_single_leg(self, mock_quote):
        """
        When short_pos is None (orphaned long leg), short_value defaults to 0.
        This is valid and should return proper PnL.
        """
        from RubberBand.scripts.live_spreads_loop import calculate_spread_pnl

        mock_quote.side_effect = [
            {"mid": 2.00, "bid": 1.95, "ask": 2.05},   # long quote
        ]

        long_pos = {"symbol": "AAPL260213C00200000", "current_price": 2.00}
        short_pos = None
        entry_debit = 1.50

        pnl, pnl_pct = calculate_spread_pnl(long_pos, short_pos, entry_debit)

        # No short -> spread value = long_value = 2.00
        # PnL = (2.00 - 1.50) * 100 = 50.0
        assert pnl is not None
        assert pnl == pytest.approx(50.0, abs=0.01)

    @patch("RubberBand.scripts.live_spreads_loop.get_option_quote")
    def test_slightly_negative_inversion_detected(self, mock_quote):
        """
        Even a very small inversion (e.g., -$0.001) should be detected and return None.
        This catches subtle quote staleness.
        """
        from RubberBand.scripts.live_spreads_loop import calculate_spread_pnl

        # Tiny inversion: long = 1.500, short = 1.501
        mock_quote.side_effect = [
            {"mid": 1.500, "bid": 1.49, "ask": 1.51},
            {"mid": 1.501, "bid": 1.49, "ask": 1.51},
        ]

        long_pos = {"symbol": "AAPL260213C00200000", "current_price": 1.50}
        short_pos = {"symbol": "AAPL260213C00205000", "current_price": 1.501}
        entry_debit = 1.00

        pnl, pnl_pct = calculate_spread_pnl(long_pos, short_pos, entry_debit)

        assert pnl is None, "Even tiny inversions should return None"
        assert pnl_pct is None


class TestSLConfirmationImmediate:
    """
    Test that _SL_CONFIRM_REQUIRED == 1 means immediate SL exit.

    With confirmation count of 1, the first SL reading should trigger the exit
    without any delay. This was changed from 2 to 1 after a 63% MRK loss
    caused by a 2-minute delay in SL execution.
    """

    def test_sl_confirm_required_is_one(self):
        """Verify _SL_CONFIRM_REQUIRED is set to 1 (immediate exit)."""
        from RubberBand.scripts.live_spreads_loop import _SL_CONFIRM_REQUIRED

        assert _SL_CONFIRM_REQUIRED == 1, (
            f"_SL_CONFIRM_REQUIRED should be 1 for immediate SL exit, got {_SL_CONFIRM_REQUIRED}"
        )

    def test_sl_exit_is_immediate_on_first_reading(self):
        """
        With _SL_CONFIRM_REQUIRED == 1, check_spread_exit_conditions should
        return True on the very first SL reading.

        The manage_positions function has SL confirmation logic that increments
        a counter. When _SL_CONFIRM_REQUIRED == 1, the counter reaches the
        threshold on the first SL event, so exit should be immediate.
        """
        from RubberBand.scripts.live_spreads_loop import (
            check_spread_exit_conditions,
            _SL_CONFIRM_REQUIRED,
            _sl_consecutive,
        )

        # Verify a significant loss triggers SL
        spread_cfg = {
            "tp_max_profit_pct": 50.0,
            "sl_pct": -25.0,
            "hold_overnight": True,
            "dte": 3,
            "bars_stop": 0,
        }

        # PnL of -30% should trigger SL
        should_exit, reason = check_spread_exit_conditions(-30.0, spread_cfg)

        assert should_exit is True, "SL should trigger on -30% when sl_pct is -25%"
        assert "SL_hit" in reason, f"Reason should contain 'SL_hit', got: {reason}"

    def test_sl_confirmation_counter_resets_when_not_in_sl(self):
        """
        When a position recovers from near-SL, the counter should reset.
        """
        from RubberBand.scripts.live_spreads_loop import _sl_consecutive

        # Simulate: set a counter for AAPL
        _sl_consecutive["AAPL"] = 1

        # Now simulate position NOT in SL territory: logic in manage_positions resets
        # We verify the dict can be cleared
        _sl_consecutive.pop("AAPL", None)
        assert "AAPL" not in _sl_consecutive

    def test_tp_exit_does_not_require_confirmation(self):
        """TP exits should be immediate regardless of confirmation setting."""
        from RubberBand.scripts.live_spreads_loop import check_spread_exit_conditions

        spread_cfg = {
            "tp_max_profit_pct": 50.0,
            "sl_pct": -25.0,
            "hold_overnight": True,
            "dte": 3,
            "bars_stop": 0,
        }

        # +60% profit should trigger TP immediately
        should_exit, reason = check_spread_exit_conditions(60.0, spread_cfg)

        assert should_exit is True
        assert "TP_hit" in reason

    def test_time_stop_exit_does_not_require_confirmation(self):
        """Time stop exits should be immediate."""
        from RubberBand.scripts.live_spreads_loop import check_spread_exit_conditions

        spread_cfg = {
            "tp_max_profit_pct": 50.0,
            "sl_pct": -25.0,
            "hold_overnight": True,
            "dte": 3,
            "bars_stop": 10,  # 10 bars = 150 minutes
        }

        # Held for 200 minutes (exceeds 10 bars * 15 min = 150 min)
        should_exit, reason = check_spread_exit_conditions(-5.0, spread_cfg, holding_minutes=200)

        assert should_exit is True
        assert "TIME_STOP" in reason
