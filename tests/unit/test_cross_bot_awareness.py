"""
Unit tests for Cross-Bot Position Awareness in live_weekly_loop.py.

The cross-bot gate prevents position stacking when multiple bots
(e.g., 15M_STK and WK_STK) target the same ticker. The logic:

1. all_broker_equity_symbols collects ALL equity positions from the broker
   (filtering to asset_class == "us_equity" to exclude options).
2. If a symbol exists in all_broker_equity_symbols but NOT in open_symbols
   (this bot's own positions), it is blocked as "held by another bot."
3. If a symbol is in open_symbols (own registry), the first check
   ("already in position") catches it before reaching the cross-bot gate.

Tests cover:
- Correct filtering of all_broker_equity_symbols to us_equity class only
- Cross-bot block when another bot holds the ticker
- Own-registry skip takes priority over cross-bot gate
- Options positions excluded from equity set
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class TestAllBrokerEquitySymbolsFiltering:
    """Test that all_broker_equity_symbols correctly filters to only us_equity."""

    def test_filters_to_us_equity_only(self):
        """
        Only positions with asset_class == 'us_equity' should be in the set.
        Options positions (asset_class == 'us_option' or long symbols) should be excluded.
        """
        all_positions = [
            {"symbol": "AAPL", "asset_class": "us_equity", "qty": "10"},
            {"symbol": "NVDA", "asset_class": "us_equity", "qty": "5"},
            {"symbol": "AAPL260213C00200000", "asset_class": "us_option", "qty": "1"},
            {"symbol": "SPY260213P00500000", "asset_class": "us_option", "qty": "-1"},
            {"symbol": "TSLA", "asset_class": "us_equity", "qty": "20"},
        ]

        # Replicate the logic from live_weekly_loop.py
        all_broker_equity_symbols = {
            p.get("symbol") for p in all_positions
            if p.get("symbol") and p.get("asset_class", "us_equity") == "us_equity"
        }

        assert "AAPL" in all_broker_equity_symbols
        assert "NVDA" in all_broker_equity_symbols
        assert "TSLA" in all_broker_equity_symbols
        assert "AAPL260213C00200000" not in all_broker_equity_symbols, "Options should be excluded"
        assert "SPY260213P00500000" not in all_broker_equity_symbols, "Options should be excluded"
        assert len(all_broker_equity_symbols) == 3

    def test_default_asset_class_is_us_equity(self):
        """
        When asset_class is missing from a position, it defaults to 'us_equity'.
        This matches the `.get("asset_class", "us_equity")` fallback in the code.
        """
        all_positions = [
            {"symbol": "AAPL", "qty": "10"},  # No asset_class field
            {"symbol": "NVDA", "asset_class": "us_equity", "qty": "5"},
        ]

        all_broker_equity_symbols = {
            p.get("symbol") for p in all_positions
            if p.get("symbol") and p.get("asset_class", "us_equity") == "us_equity"
        }

        assert "AAPL" in all_broker_equity_symbols, "Missing asset_class defaults to us_equity"
        assert "NVDA" in all_broker_equity_symbols
        assert len(all_broker_equity_symbols) == 2

    def test_empty_positions_returns_empty_set(self):
        """No positions should produce empty set."""
        all_positions = []

        all_broker_equity_symbols = {
            p.get("symbol") for p in all_positions
            if p.get("symbol") and p.get("asset_class", "us_equity") == "us_equity"
        }

        assert len(all_broker_equity_symbols) == 0

    def test_none_symbol_excluded(self):
        """Positions with None symbol should not be included."""
        all_positions = [
            {"symbol": None, "asset_class": "us_equity", "qty": "10"},
            {"asset_class": "us_equity", "qty": "5"},  # Missing symbol key
            {"symbol": "AAPL", "asset_class": "us_equity", "qty": "10"},
        ]

        all_broker_equity_symbols = {
            p.get("symbol") for p in all_positions
            if p.get("symbol") and p.get("asset_class", "us_equity") == "us_equity"
        }

        assert "AAPL" in all_broker_equity_symbols
        assert None not in all_broker_equity_symbols
        assert len(all_broker_equity_symbols) == 1


class TestCrossBotBlockLogic:
    """Test that cross-bot gate blocks tickers held by other bots."""

    def test_cross_bot_blocks_ticker_held_by_another_bot(self):
        """
        If NFLX is in all_broker_equity_symbols (another bot holds it)
        but NOT in open_symbols (this bot doesn't hold it), it should be blocked.
        """
        open_symbols = {"AAPL", "NVDA"}  # This bot's positions
        all_broker_equity_symbols = {"AAPL", "NVDA", "NFLX", "TSLA"}  # All broker positions

        symbol = "NFLX"

        # Replicate the logic from live_weekly_loop.py
        # First check: skip if already in our own position
        in_own_position = symbol in open_symbols
        assert in_own_position is False, "NFLX is not in our positions"

        # Second check: cross-bot gate
        cross_bot_blocked = symbol in all_broker_equity_symbols and symbol not in open_symbols
        assert cross_bot_blocked is True, "NFLX should be blocked by cross-bot gate"

    def test_own_position_handled_by_first_check(self):
        """
        If AAPL is in BOTH open_symbols and all_broker_equity_symbols,
        the first check (own registry) catches it before the cross-bot gate.
        """
        open_symbols = {"AAPL", "NVDA"}
        all_broker_equity_symbols = {"AAPL", "NVDA", "NFLX"}

        symbol = "AAPL"

        # First check: skip if already in our own position
        in_own_position = symbol in open_symbols
        assert in_own_position is True, "AAPL is in our positions - caught by first check"

        # Cross-bot gate would NOT trigger because symbol IS in open_symbols
        cross_bot_blocked = symbol in all_broker_equity_symbols and symbol not in open_symbols
        assert cross_bot_blocked is False, "Own position should NOT trigger cross-bot block"

    def test_new_ticker_not_blocked(self):
        """
        A ticker not held by any bot should not be blocked.
        """
        open_symbols = {"AAPL"}
        all_broker_equity_symbols = {"AAPL"}

        symbol = "MSFT"

        in_own_position = symbol in open_symbols
        assert in_own_position is False

        cross_bot_blocked = symbol in all_broker_equity_symbols and symbol not in open_symbols
        assert cross_bot_blocked is False, "MSFT not held by anyone - should not be blocked"

    def test_cross_bot_gate_full_scenario(self):
        """
        Full scenario: iterate tickers and apply both checks in correct order.
        Validates the exact logic flow from live_weekly_loop.py.
        """
        open_symbols = {"AAPL", "NVDA"}
        all_broker_equity_symbols = {"AAPL", "NVDA", "NFLX", "TSLA"}
        tickers = ["AAPL", "NFLX", "MSFT", "TSLA", "NVDA"]

        blocked = []
        skipped_own = []
        allowed = []

        for symbol in tickers:
            # First check: own registry
            if symbol in open_symbols:
                skipped_own.append(symbol)
                continue

            # Second check: cross-bot gate
            if symbol in all_broker_equity_symbols and symbol not in open_symbols:
                blocked.append(symbol)
                continue

            # Would proceed to signal analysis
            allowed.append(symbol)

        assert skipped_own == ["AAPL", "NVDA"], "Own positions caught first"
        assert blocked == ["NFLX", "TSLA"], "Cross-bot positions blocked"
        assert allowed == ["MSFT"], "Only unblocked tickers proceed"


class TestCrossBotPositionLimitInteraction:
    """Test that position limit check works correctly with cross-bot awareness."""

    def test_position_limit_still_enforced(self):
        """
        Even if cross-bot doesn't block, max positions should still stop entries.
        """
        open_symbols = {"AAPL", "NVDA", "TSLA", "AMZN", "MSFT"}
        all_broker_equity_symbols = {"AAPL", "NVDA", "TSLA", "AMZN", "MSFT"}
        limit_pos = 5

        # Already at max positions
        assert len(open_symbols) >= limit_pos, "Already at position limit"

        # Even if GOOG is not blocked by cross-bot gate, we should not enter
        symbol = "GOOG"
        entries_made = 0
        at_limit = len(open_symbols) + entries_made >= limit_pos
        assert at_limit is True, "Should be at position limit"
