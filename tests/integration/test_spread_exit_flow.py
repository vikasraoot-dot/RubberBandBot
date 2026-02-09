"""
Integration tests for the spread exit flow in live_spreads_loop.py.

These tests verify the end-to-end behavior of the position management system
when encountering various market conditions:

1. Quote inversion guard integrates with manage_positions to skip exit logic
2. Normal exit flow works correctly when quotes are valid
3. SL confirmation counter works with _SL_CONFIRM_REQUIRED == 1
4. Cross-bot gate integrates with weekly loop position scanning
5. Mleg fill verification integrates with submit_spread_order

All external API calls are mocked.
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class TestManagePositionsQuoteInversion:
    """
    Integration test: manage_positions should skip exit logic when
    calculate_spread_pnl returns (None, None) due to inverted quotes.
    """

    @patch("RubberBand.scripts.live_spreads_loop.close_spread")
    @patch("RubberBand.scripts.live_spreads_loop.get_option_quote")
    @patch("RubberBand.scripts.live_spreads_loop.get_option_positions")
    def test_inverted_quotes_skip_exit(self, mock_get_positions, mock_quote, mock_close):
        """
        When calculate_spread_pnl returns (None, None) for a position,
        manage_positions should NOT call close_spread and should continue
        to the next cycle.
        """
        from RubberBand.scripts.live_spreads_loop import manage_positions

        # Create mock positions for a bull call spread
        mock_get_positions.return_value = [
            {
                "symbol": "AAPL260213C00200000",
                "qty": 1,
                "cost_basis": "150.00",
                "current_price": "1.00",
            },
            {
                "symbol": "AAPL260213C00205000",
                "qty": -1,
                "cost_basis": "-80.00",
                "current_price": "2.00",  # Higher than long = inversion
            },
        ]

        # Set up inverted quotes (short > long)
        def mock_quote_func(symbol):
            if "C00200000" in symbol:  # Long leg
                return {"mid": 1.00, "bid": 0.95, "ask": 1.05}
            elif "C00205000" in symbol:  # Short leg (inverted)
                return {"mid": 2.00, "bid": 1.95, "ask": 2.05}
            return None

        mock_quote.side_effect = mock_quote_func

        mock_logger = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_my_symbols.return_value = ["AAPL260213C00200000"]
        mock_registry.find_by_symbol.return_value = "AAPL260213C00200000"
        mock_registry.positions = {
            "AAPL260213C00200000": {
                "short_symbol": "AAPL260213C00205000",
                "entry_date": "2026-02-08T10:00:00-05:00",
            }
        }

        spread_cfg = {
            "tp_max_profit_pct": 50.0,
            "sl_pct": -25.0,
            "hold_overnight": True,
            "dte": 3,
            "bars_stop": 10,
        }

        manage_positions(spread_cfg, mock_logger, dry_run=True, registry=mock_registry)

        # Close should NOT be called because of inverted quotes
        mock_close.assert_not_called()


class TestManagePositionsNormalExit:
    """
    Integration test: manage_positions correctly exits spreads when
    TP/SL conditions are met and quotes are valid.
    """

    @patch("RubberBand.scripts.live_spreads_loop.close_option_position")
    @patch("RubberBand.scripts.live_spreads_loop.close_spread")
    @patch("RubberBand.scripts.live_spreads_loop.get_option_quote")
    @patch("RubberBand.scripts.live_spreads_loop.get_option_positions")
    def test_tp_exit_with_valid_quotes(self, mock_get_positions, mock_quote,
                                        mock_close_spread, mock_close_option):
        """
        When spread P&L exceeds TP threshold and quotes are valid,
        manage_positions should trigger an exit and log it.
        """
        from RubberBand.scripts.live_spreads_loop import manage_positions

        mock_get_positions.return_value = [
            {
                "symbol": "NVDA260213C00900000",
                "qty": 1,
                "cost_basis": "100.00",
                "current_price": "3.00",
            },
            {
                "symbol": "NVDA260213C00905000",
                "qty": -1,
                "cost_basis": "-50.00",
                "current_price": "0.50",
            },
        ]

        # Profitable spread: long=3.00, short=0.50, spread=2.50, entry=0.50
        # PnL% = (2.50/0.50 - 1) * 100 = 400%
        def mock_quote_func(symbol):
            if "C00900000" in symbol:
                return {"mid": 3.00, "bid": 2.95, "ask": 3.05}
            elif "C00905000" in symbol:
                return {"mid": 0.50, "bid": 0.45, "ask": 0.55}
            return None

        mock_quote.side_effect = mock_quote_func

        mock_logger = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_my_symbols.return_value = ["NVDA260213C00900000"]
        mock_registry.find_by_symbol.return_value = "NVDA260213C00900000"
        mock_registry.positions = {
            "NVDA260213C00900000": {
                "short_symbol": "NVDA260213C00905000",
                "entry_date": "2026-02-08T10:00:00-05:00",
            }
        }

        spread_cfg = {
            "tp_max_profit_pct": 50.0,  # 50% TP
            "sl_pct": -25.0,
            "hold_overnight": True,
            "dte": 3,
            "bars_stop": 0,
        }

        # Dry run to verify logging without actual API call
        manage_positions(spread_cfg, mock_logger, dry_run=True, registry=mock_registry)

        # Logger should record a spread exit (dry-run)
        mock_logger.spread_exit.assert_called_once()
        call_kwargs = mock_logger.spread_exit.call_args[1]
        assert "DRY-RUN" in call_kwargs.get("exit_reason", "")
        assert "TP_hit" in call_kwargs.get("exit_reason", "")


class TestCrossBotIntegration:
    """
    Integration test: Verify cross-bot position awareness works
    with the full ticker iteration logic.
    """

    def test_cross_bot_gate_prevents_duplicate_entries(self):
        """
        Simulate the full ticker iteration from live_weekly_loop.py
        where 15M_STK already holds NFLX and WK_STK should skip it.
        """
        # Simulate broker positions (from all bots)
        all_positions = [
            {"symbol": "AAPL", "asset_class": "us_equity", "qty": "10"},   # WK_STK
            {"symbol": "NFLX", "asset_class": "us_equity", "qty": "50"},   # 15M_STK
            {"symbol": "NVDA260213C00200000", "asset_class": "us_option", "qty": "1"},  # 15M_OPT
        ]

        # WK_STK's own positions (only AAPL)
        open_symbols = {"AAPL"}

        # Build cross-bot equity set (same as live_weekly_loop.py)
        all_broker_equity_symbols = {
            p.get("symbol") for p in all_positions
            if p.get("symbol") and p.get("asset_class", "us_equity") == "us_equity"
        }

        assert "NFLX" in all_broker_equity_symbols, "NFLX should be in equity set"
        assert "NVDA260213C00200000" not in all_broker_equity_symbols, "Options excluded"

        # Simulate ticker iteration
        tickers = ["AAPL", "NFLX", "MSFT", "GOOG"]
        entries_attempted = []

        for symbol in tickers:
            if symbol in open_symbols:
                continue  # Own position

            if symbol in all_broker_equity_symbols and symbol not in open_symbols:
                continue  # Cross-bot block

            entries_attempted.append(symbol)

        assert "AAPL" not in entries_attempted, "Own position should be skipped"
        assert "NFLX" not in entries_attempted, "Cross-bot position should be blocked"
        assert "MSFT" in entries_attempted, "MSFT should be allowed"
        assert "GOOG" in entries_attempted, "GOOG should be allowed"


class TestMlegFillIntegration:
    """
    Integration test: Verify mleg fill verification integrates with
    submit_spread_order to handle partial fills.
    """

    @patch("RubberBand.src.options_execution._verify_mleg_fill")
    @patch("RubberBand.src.options_execution.requests")
    @patch("RubberBand.src.options_execution._resolve_creds")
    @patch("RubberBand.src.options_data.get_option_quote")
    def test_failed_mleg_returns_error(
        self, mock_get_quote, mock_creds, mock_requests, mock_verify
    ):
        """
        When _verify_mleg_fill returns 'failed' (order canceled/rejected/expired),
        submit_spread_order should return an error result.

        NOTE: After the race condition fix, _verify_mleg_fill no longer returns
        'partial_fill'. When the broker confirms 'filled', we trust it (even if
        the positions API is slow). The only error path is 'failed'.
        """
        from RubberBand.src.options_execution import submit_spread_order

        mock_creds.return_value = ("https://paper-api.alpaca.markets", "key", "secret")

        # Quotes for spread pricing
        mock_get_quote.side_effect = [
            {"ask": 2.00, "bid": 1.90, "mid": 1.95},  # long
            {"bid": 0.50, "ask": 0.60, "mid": 0.55},  # short
        ]

        # Broker accepts the order
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "order-123",
            "status": "new",
        }
        mock_requests.post.return_value = mock_response

        # Verification returns failed (order was canceled/rejected/expired)
        mock_verify.return_value = "failed"

        result = submit_spread_order(
            long_symbol="AAPL260213C00200000",
            short_symbol="AAPL260213C00205000",
            qty=1,
            max_debit=3.00,
        )

        # Should return error
        assert result.get("error") is True
        assert "failed" in result.get("message", "")

    @patch("RubberBand.src.options_execution._verify_mleg_fill")
    @patch("RubberBand.src.options_execution.requests")
    @patch("RubberBand.src.options_execution._resolve_creds")
    @patch("RubberBand.src.options_data.get_option_quote")
    def test_successful_fill_returns_order_info(self, mock_get_quote, mock_creds,
                                                 mock_requests, mock_verify):
        """
        When submit_spread_order gets a successful fill, it should
        return a success result with order details.
        """
        from RubberBand.src.options_execution import submit_spread_order

        mock_creds.return_value = ("https://paper-api.alpaca.markets", "key", "secret")

        mock_get_quote.side_effect = [
            {"ask": 2.00, "bid": 1.90, "mid": 1.95},
            {"bid": 0.50, "ask": 0.60, "mid": 0.55},
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "order-456",
            "status": "accepted",
        }
        mock_requests.post.return_value = mock_response

        mock_verify.return_value = "filled"

        result = submit_spread_order(
            long_symbol="AAPL260213C00200000",
            short_symbol="AAPL260213C00205000",
            qty=1,
            max_debit=3.00,
        )

        assert result.get("error") is False
        assert result.get("order_id") == "order-456"
        assert result.get("long_symbol") == "AAPL260213C00200000"
        assert result.get("short_symbol") == "AAPL260213C00205000"


class TestSLConfirmationIntegration:
    """
    Integration test: Verify that with _SL_CONFIRM_REQUIRED == 1,
    the full manage_positions flow exits on the first SL reading.
    """

    def test_sl_confirm_required_value(self):
        """Sanity check that the module-level constant is 1."""
        from RubberBand.scripts.live_spreads_loop import _SL_CONFIRM_REQUIRED
        assert _SL_CONFIRM_REQUIRED == 1

    @patch("RubberBand.scripts.live_spreads_loop.get_option_quote")
    @patch("RubberBand.scripts.live_spreads_loop.get_option_positions")
    def test_first_sl_reading_exits_immediately(self, mock_get_positions, mock_quote):
        """
        With _SL_CONFIRM_REQUIRED == 1, the FIRST time a position hits SL
        should trigger an immediate exit (dry-run mode for safety).
        """
        from RubberBand.scripts.live_spreads_loop import (
            manage_positions,
            _sl_consecutive,
        )

        # Clear any residual state
        _sl_consecutive.clear()

        mock_get_positions.return_value = [
            {
                "symbol": "TSLA260213C00250000",
                "qty": 1,
                "cost_basis": "200.00",
                "current_price": "0.30",
            },
            {
                "symbol": "TSLA260213C00255000",
                "qty": -1,
                "cost_basis": "-100.00",
                "current_price": "0.15",
            },
        ]

        # Spread losing money: long=0.30, short=0.15, spread=0.15
        # entry_debit = (200 - 100) / 100 = 1.00
        # PnL% = (0.15/1.00 - 1)*100 = -85% (deep in SL territory)
        def mock_quote_func(symbol):
            if "C00250000" in symbol:
                return {"mid": 0.30, "bid": 0.25, "ask": 0.35}
            elif "C00255000" in symbol:
                return {"mid": 0.15, "bid": 0.10, "ask": 0.20}
            return None

        mock_quote.side_effect = mock_quote_func

        mock_logger = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_my_symbols.return_value = ["TSLA260213C00250000"]
        mock_registry.find_by_symbol.return_value = "TSLA260213C00250000"
        mock_registry.positions = {
            "TSLA260213C00250000": {
                "short_symbol": "TSLA260213C00255000",
                "entry_date": "2026-02-08T10:00:00-05:00",
            }
        }

        spread_cfg = {
            "tp_max_profit_pct": 50.0,
            "sl_pct": -25.0,  # -85% is far below -25% SL
            "hold_overnight": True,
            "dte": 3,
            "bars_stop": 0,
        }

        manage_positions(spread_cfg, mock_logger, dry_run=True, registry=mock_registry)

        # With _SL_CONFIRM_REQUIRED == 1, the first reading should trigger exit
        mock_logger.spread_exit.assert_called_once()
        call_kwargs = mock_logger.spread_exit.call_args[1]
        assert "SL_hit" in call_kwargs.get("exit_reason", ""), (
            f"Exit reason should contain 'SL_hit', got: {call_kwargs.get('exit_reason', '')}"
        )
