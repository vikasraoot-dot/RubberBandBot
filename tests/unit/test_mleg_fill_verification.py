"""
Unit tests for Mleg Fill Verification in options_execution.py.

The mleg (multi-leg) fill verification system ensures that both legs of
a spread order are properly filled and exist as positions. It handles:

1. _verify_mleg_fill():
   - 'filled' when order fills and both positions exist
   - 'partial_fill' when order fills but only one leg in positions
   - 'failed' when order is canceled/rejected/expired
   - 'pending' on timeout

2. _close_naked_legs():
   - Closes any existing naked positions from partial fills

Tests use mocks to avoid real API calls.
"""

import pytest
from unittest.mock import patch, MagicMock, call
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class TestVerifyMlegFill:
    """Tests for _verify_mleg_fill() function."""

    @patch("RubberBand.src.options_execution.requests")
    @patch("RubberBand.src.options_execution._get_order_by_id")
    @patch("RubberBand.src.options_execution.time")
    def test_filled_both_legs_exist(self, mock_time, mock_get_order, mock_requests):
        """
        When order status is 'filled' and both legs exist in positions,
        should return 'filled'.
        """
        from RubberBand.src.options_execution import _verify_mleg_fill

        # Order fills on first check
        mock_get_order.return_value = {"status": "filled"}

        # Both legs exist in positions
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"symbol": "AAPL260213C00200000", "qty": "1"},
            {"symbol": "AAPL260213C00205000", "qty": "-1"},
        ]
        mock_requests.get.return_value = mock_response

        result = _verify_mleg_fill(
            base="https://paper-api.alpaca.markets",
            key="test-key",
            secret="test-secret",
            order_id="order-123",
            long_symbol="AAPL260213C00200000",
            short_symbol="AAPL260213C00205000",
            timeout=5,
        )

        assert result == "filled", f"Expected 'filled', got '{result}'"

    @patch("RubberBand.src.options_execution.requests")
    @patch("RubberBand.src.options_execution._get_order_by_id")
    @patch("RubberBand.src.options_execution.time")
    def test_filled_trusts_broker_when_one_leg_missing(self, mock_time, mock_get_order, mock_requests):
        """
        When order status is 'filled' but only the long leg exists in positions,
        should still return 'filled' (trust broker over positions API latency).
        """
        from RubberBand.src.options_execution import _verify_mleg_fill

        mock_get_order.return_value = {"status": "filled"}

        # Only long leg exists (short is missing)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"symbol": "AAPL260213C00200000", "qty": "1"},
            # Short leg missing!
        ]
        mock_requests.get.return_value = mock_response

        result = _verify_mleg_fill(
            base="https://paper-api.alpaca.markets",
            key="test-key",
            secret="test-secret",
            order_id="order-123",
            long_symbol="AAPL260213C00200000",
            short_symbol="AAPL260213C00205000",
            timeout=5,
        )

        # After the race condition fix, the broker's "filled" status is trusted even if
        # positions API doesn't show both legs (settlement latency). We do NOT return
        # "partial_fill" because that would trigger _close_naked_legs on a valid position.
        assert result == "filled", f"Expected 'filled' (trust broker), got '{result}'"

    @patch("RubberBand.src.options_execution._get_order_by_id")
    @patch("RubberBand.src.options_execution.time")
    def test_canceled_returns_failed(self, mock_time, mock_get_order):
        """Order canceled should return 'failed'."""
        from RubberBand.src.options_execution import _verify_mleg_fill

        mock_get_order.return_value = {"status": "canceled"}

        result = _verify_mleg_fill(
            base="https://paper-api.alpaca.markets",
            key="test-key",
            secret="test-secret",
            order_id="order-123",
            long_symbol="AAPL260213C00200000",
            short_symbol="AAPL260213C00205000",
            timeout=5,
        )

        assert result == "failed", f"Expected 'failed', got '{result}'"

    @patch("RubberBand.src.options_execution._get_order_by_id")
    @patch("RubberBand.src.options_execution.time")
    def test_rejected_returns_failed(self, mock_time, mock_get_order):
        """Order rejected should return 'failed'."""
        from RubberBand.src.options_execution import _verify_mleg_fill

        mock_get_order.return_value = {"status": "rejected"}

        result = _verify_mleg_fill(
            base="https://paper-api.alpaca.markets",
            key="test-key",
            secret="test-secret",
            order_id="order-123",
            long_symbol="AAPL260213C00200000",
            short_symbol="AAPL260213C00205000",
            timeout=5,
        )

        assert result == "failed"

    @patch("RubberBand.src.options_execution._get_order_by_id")
    @patch("RubberBand.src.options_execution.time")
    def test_expired_returns_failed(self, mock_time, mock_get_order):
        """Order expired should return 'failed'."""
        from RubberBand.src.options_execution import _verify_mleg_fill

        mock_get_order.return_value = {"status": "expired"}

        result = _verify_mleg_fill(
            base="https://paper-api.alpaca.markets",
            key="test-key",
            secret="test-secret",
            order_id="order-123",
            long_symbol="AAPL260213C00200000",
            short_symbol="AAPL260213C00205000",
            timeout=5,
        )

        assert result == "failed"

    @patch("RubberBand.src.options_execution._get_order_by_id")
    @patch("RubberBand.src.options_execution.time")
    def test_timeout_returns_pending(self, mock_time, mock_get_order):
        """
        When order stays 'new' through the entire timeout period,
        should return 'pending'.
        """
        from RubberBand.src.options_execution import _verify_mleg_fill

        # Order stays in 'new' status for all checks
        mock_get_order.return_value = {"status": "new"}

        result = _verify_mleg_fill(
            base="https://paper-api.alpaca.markets",
            key="test-key",
            secret="test-secret",
            order_id="order-123",
            long_symbol="AAPL260213C00200000",
            short_symbol="AAPL260213C00205000",
            timeout=3,  # Short timeout for test speed
        )

        assert result == "pending", f"Expected 'pending', got '{result}'"

    @patch("RubberBand.src.options_execution.requests")
    @patch("RubberBand.src.options_execution._get_order_by_id")
    @patch("RubberBand.src.options_execution.time")
    def test_filled_position_check_fails_returns_filled(self, mock_time, mock_get_order, mock_requests):
        """
        When order fills but the position check API call fails,
        should still return 'filled' (assume OK as per code comment).
        """
        from RubberBand.src.options_execution import _verify_mleg_fill

        mock_get_order.return_value = {"status": "filled"}

        # Position check fails (non-200 status)
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_requests.get.return_value = mock_response

        result = _verify_mleg_fill(
            base="https://paper-api.alpaca.markets",
            key="test-key",
            secret="test-secret",
            order_id="order-123",
            long_symbol="AAPL260213C00200000",
            short_symbol="AAPL260213C00205000",
            timeout=5,
        )

        # Code falls through to "return 'filled'" when position check fails
        assert result == "filled"

    @patch("RubberBand.src.options_execution._get_order_by_id")
    @patch("RubberBand.src.options_execution.time")
    def test_fills_on_second_attempt(self, mock_time, mock_get_order):
        """
        Order is 'new' on first check, then 'filled' on second check.
        Should handle the transition correctly.
        """
        from RubberBand.src.options_execution import _verify_mleg_fill

        # First call: new, second call: filled
        mock_get_order.side_effect = [
            {"status": "new"},
            {"status": "filled"},
        ]

        # Mock requests.get for position verification
        with patch("RubberBand.src.options_execution.requests") as mock_req:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = [
                {"symbol": "AAPL260213C00200000", "qty": "1"},
                {"symbol": "AAPL260213C00205000", "qty": "-1"},
            ]
            mock_req.get.return_value = mock_resp

            result = _verify_mleg_fill(
                base="https://paper-api.alpaca.markets",
                key="test-key",
                secret="test-secret",
                order_id="order-123",
                long_symbol="AAPL260213C00200000",
                short_symbol="AAPL260213C00205000",
                timeout=5,
            )

        assert result == "filled"


class TestCloseNakedLegs:
    """Tests for _close_naked_legs() emergency cleanup function."""

    @patch("RubberBand.src.options_execution.submit_option_order")
    @patch("RubberBand.src.options_execution.requests")
    def test_closes_existing_naked_long(self, mock_requests, mock_submit):
        """
        If only the long leg exists after partial fill, it should be closed
        with a sell order.
        """
        from RubberBand.src.options_execution import _close_naked_legs

        # Only long leg exists in positions
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"symbol": "AAPL260213C00200000", "qty": "1"},  # Long leg only
        ]
        mock_requests.get.return_value = mock_response

        _close_naked_legs(
            base="https://paper-api.alpaca.markets",
            key="test-key",
            secret="test-secret",
            long_symbol="AAPL260213C00200000",
            short_symbol="AAPL260213C00205000",
        )

        # Should close the long leg with a sell order
        mock_submit.assert_called_once_with(
            "AAPL260213C00200000", qty=1, side="sell", order_type="market"
        )

    @patch("RubberBand.src.options_execution.submit_option_order")
    @patch("RubberBand.src.options_execution.requests")
    def test_closes_existing_naked_short(self, mock_requests, mock_submit):
        """
        If only the short leg exists after partial fill, it should be closed
        with a buy order.
        """
        from RubberBand.src.options_execution import _close_naked_legs

        # Only short leg exists in positions (negative qty means short)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"symbol": "AAPL260213C00205000", "qty": "-1"},  # Short leg only
        ]
        mock_requests.get.return_value = mock_response

        _close_naked_legs(
            base="https://paper-api.alpaca.markets",
            key="test-key",
            secret="test-secret",
            long_symbol="AAPL260213C00200000",
            short_symbol="AAPL260213C00205000",
        )

        # Should close the short leg with a buy order
        mock_submit.assert_called_once_with(
            "AAPL260213C00205000", qty=1, side="buy", order_type="market"
        )

    @patch("RubberBand.src.options_execution.submit_option_order")
    @patch("RubberBand.src.options_execution.requests")
    def test_closes_both_naked_legs(self, mock_requests, mock_submit):
        """
        If both legs exist as separate positions (not as a spread), both
        should be individually closed.
        """
        from RubberBand.src.options_execution import _close_naked_legs

        # Both legs exist
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"symbol": "AAPL260213C00200000", "qty": "1"},   # Long
            {"symbol": "AAPL260213C00205000", "qty": "-1"},  # Short
        ]
        mock_requests.get.return_value = mock_response

        _close_naked_legs(
            base="https://paper-api.alpaca.markets",
            key="test-key",
            secret="test-secret",
            long_symbol="AAPL260213C00200000",
            short_symbol="AAPL260213C00205000",
        )

        # Both legs should be closed
        assert mock_submit.call_count == 2

    @patch("RubberBand.src.options_execution.submit_option_order")
    @patch("RubberBand.src.options_execution.requests")
    def test_no_positions_does_nothing(self, mock_requests, mock_submit):
        """
        If neither leg exists in positions, no close orders should be placed.
        """
        from RubberBand.src.options_execution import _close_naked_legs

        # No positions exist
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_requests.get.return_value = mock_response

        _close_naked_legs(
            base="https://paper-api.alpaca.markets",
            key="test-key",
            secret="test-secret",
            long_symbol="AAPL260213C00200000",
            short_symbol="AAPL260213C00205000",
        )

        mock_submit.assert_not_called()

    @patch("RubberBand.src.options_execution.submit_option_order")
    @patch("RubberBand.src.options_execution.requests")
    def test_api_failure_does_not_crash(self, mock_requests, mock_submit):
        """
        If the positions API call fails, _close_naked_legs should not raise.
        It prints an error but continues (fail-safe).
        """
        from RubberBand.src.options_execution import _close_naked_legs

        # API returns error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_requests.get.return_value = mock_response

        # Should not raise any exception
        _close_naked_legs(
            base="https://paper-api.alpaca.markets",
            key="test-key",
            secret="test-secret",
            long_symbol="AAPL260213C00200000",
            short_symbol="AAPL260213C00205000",
        )

        mock_submit.assert_not_called()

    @patch("RubberBand.src.options_execution.submit_option_order")
    @patch("RubberBand.src.options_execution.requests")
    def test_exception_during_close_does_not_propagate(self, mock_requests, mock_submit):
        """
        If requests.get raises an exception, _close_naked_legs catches it
        and does not propagate. Critical safety function must not crash.
        """
        from RubberBand.src.options_execution import _close_naked_legs

        mock_requests.get.side_effect = ConnectionError("Network failure")

        # Should not raise
        _close_naked_legs(
            base="https://paper-api.alpaca.markets",
            key="test-key",
            secret="test-secret",
            long_symbol="AAPL260213C00200000",
            short_symbol="AAPL260213C00205000",
        )

        mock_submit.assert_not_called()
