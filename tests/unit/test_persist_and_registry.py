"""
Tests for persist_daily_results mleg extraction and position registry bootstrap.
"""
import json
import os
import pytest
from decimal import Decimal
from unittest.mock import patch, mock_open

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
from RubberBand.src.position_registry import (
    BOT_TAGS,
    ensure_all_registries_exist,
    PositionRegistry,
)

# Import via a sys.path-safe route (persist_daily_results does its own sys.path
# manipulation; we can import the helper directly once the module is loaded).
import importlib
_pdr = importlib.import_module("RubberBand.scripts.persist_daily_results")
_extract_order_symbol_side = _pdr._extract_order_symbol_side
calculate_bot_pnl = _pdr.calculate_bot_pnl


# ============================================================================
# _extract_order_symbol_side tests
# ============================================================================

class TestExtractOrderSymbolSide:
    """Tests for the mleg symbol/side extraction helper."""

    def test_stock_order(self):
        """Standard stock order returns top-level symbol/side, not a spread."""
        order = {"symbol": "AAPL", "side": "buy", "filled_qty": 100}
        sym, side, is_spread = _extract_order_symbol_side(order)
        assert sym == "AAPL"
        assert side == "buy"
        assert is_spread is False

    def test_single_option_order(self):
        """Single-leg option order returns top-level symbol/side."""
        order = {"symbol": "AAPL260213C00200000", "side": "sell"}
        sym, side, is_spread = _extract_order_symbol_side(order)
        assert sym == "AAPL260213C00200000"
        assert side == "sell"
        assert is_spread is False

    def test_mleg_spread_both_filled(self):
        """Multi-leg spread with both legs filled returns buy leg + is_spread=True."""
        order = {
            "symbol": None,
            "side": None,
            "legs": [
                {"symbol": "NVDA260321C00130000", "side": "buy", "status": "filled"},
                {"symbol": "NVDA260321C00135000", "side": "sell", "status": "filled"},
            ],
        }
        sym, side, is_spread = _extract_order_symbol_side(order)
        assert sym == "NVDA260321C00130000"
        assert side == "buy"
        assert is_spread is True

    def test_mleg_spread_empty_string_symbol(self):
        """Multi-leg spread where symbol is '' (empty string, not None)."""
        order = {
            "symbol": "",
            "side": "",
            "legs": [
                {"symbol": "AA260227C00057000", "side": "buy", "status": "filled"},
                {"symbol": "AA260227C00058000", "side": "sell", "status": "filled"},
            ],
        }
        sym, side, is_spread = _extract_order_symbol_side(order)
        assert sym == "AA260227C00057000"
        assert side == "buy"
        assert is_spread is True

    def test_mleg_partial_fill(self):
        """Only one leg filled — returns that leg's side, not a spread."""
        order = {
            "symbol": "",
            "side": "",
            "legs": [
                {"symbol": "GOOG260227C00300000", "side": "buy", "status": "filled"},
                {"symbol": "GOOG260227C00305000", "side": "sell", "status": "pending"},
            ],
        }
        sym, side, is_spread = _extract_order_symbol_side(order)
        assert sym == "GOOG260227C00300000"
        assert side == "buy"
        assert is_spread is False

    def test_mleg_no_buy_leg(self):
        """All legs are sell — uses first filled leg, mapped to buy for spread."""
        order = {
            "symbol": None,
            "side": None,
            "legs": [
                {"symbol": "SPY260220P00450000", "side": "sell", "status": "filled"},
                {"symbol": "SPY260220P00445000", "side": "sell", "status": "filled"},
            ],
        }
        sym, side, is_spread = _extract_order_symbol_side(order)
        assert sym == "SPY260220P00450000"
        assert side == "buy"  # mapped to "buy" for P&L accounting
        assert is_spread is True

    def test_empty_order(self):
        """Order with no symbol and no legs returns empty strings."""
        order = {"symbol": "", "side": "", "legs": []}
        sym, side, is_spread = _extract_order_symbol_side(order)
        assert sym == ""
        assert side == ""
        assert is_spread is False

    def test_no_legs_key(self):
        """Order with neither symbol nor legs key."""
        order = {"filled_qty": 1}
        sym, side, is_spread = _extract_order_symbol_side(order)
        assert sym == ""
        assert side == ""
        assert is_spread is False

    def test_whitespace_symbol_stripped(self):
        """Leading/trailing whitespace in symbol is stripped."""
        order = {"symbol": "  AAPL  ", "side": "buy"}
        sym, side, is_spread = _extract_order_symbol_side(order)
        assert sym == "AAPL"
        assert is_spread is False

    def test_legs_none_value(self):
        """legs=None should not crash."""
        order = {"symbol": "", "side": "", "legs": None}
        sym, side, is_spread = _extract_order_symbol_side(order)
        assert sym == ""
        assert side == ""
        assert is_spread is False

    def test_leg_missing_symbol_key(self):
        """Leg dict without 'symbol' key should not crash."""
        order = {
            "symbol": "",
            "side": "",
            "legs": [
                {"side": "buy", "status": "filled"},
                {"symbol": "AAPL260227C00200000", "side": "sell", "status": "filled"},
            ],
        }
        sym, side, is_spread = _extract_order_symbol_side(order)
        # buy leg has no symbol → .get("symbol") returns None → stripped to ""
        # Since buy leg is preferred, symbol is empty; is_spread=True
        assert is_spread is True
        assert side == "buy"

    def test_leg_missing_side_key(self):
        """Leg dict without 'side' key should not crash."""
        order = {
            "symbol": "",
            "side": "",
            "legs": [
                {"symbol": "AAPL260227C00200000", "status": "filled"},
            ],
        }
        sym, side, is_spread = _extract_order_symbol_side(order)
        assert sym == "AAPL260227C00200000"
        assert is_spread is False
        # Single filled leg, no side key → primary.get("side") returns None → ""
        assert side == ""

    def test_single_leg_mleg(self):
        """Mleg order with exactly 1 leg in legs array (not a spread)."""
        order = {
            "symbol": "",
            "side": "",
            "legs": [
                {"symbol": "TSLA260320C00400000", "side": "buy", "status": "filled"},
            ],
        }
        sym, side, is_spread = _extract_order_symbol_side(order)
        assert sym == "TSLA260320C00400000"
        assert side == "buy"
        assert is_spread is False

    def test_leg_symbol_none(self):
        """Leg with symbol=None should not crash."""
        order = {
            "symbol": "",
            "side": "",
            "legs": [
                {"symbol": None, "side": "buy", "status": "filled"},
            ],
        }
        sym, side, is_spread = _extract_order_symbol_side(order)
        assert sym == ""
        assert is_spread is False


# ============================================================================
# calculate_bot_pnl with mleg orders
# ============================================================================

class TestCalculateBotPnlMleg:
    """Verify calculate_bot_pnl correctly handles mleg orders."""

    def _make_order(self, **kw):
        """Convenience: build a minimal order dict."""
        base = {
            "id": "ord_1",
            "filled_at": "2026-02-17T10:00:00Z",
            "filled_qty": "1",
            "filled_avg_price": "1.50",
            "client_order_id": "15M_OPT_TEST_1234567890",
        }
        base.update(kw)
        return base

    def test_mleg_order_appears_in_trades(self):
        """Mleg order should produce a trade with non-blank symbol and side=buy."""
        mleg_order = self._make_order(
            symbol="",
            side="",
            legs=[
                {"symbol": "CRM260327C00190000", "side": "buy", "status": "filled"},
                {"symbol": "CRM260327C00195000", "side": "sell", "status": "filled"},
            ],
        )

        with patch(
            "RubberBand.scripts.persist_daily_results.extract_bot_tag_from_order",
            return_value="15M_OPT",
        ):
            result = calculate_bot_pnl(
                orders=[mleg_order],
                positions=[],
                bot_tag="15M_OPT",
                target_date="2026-02-17",
            )

        assert len(result["trades"]) == 1
        assert result["trades"][0]["symbol"] == "CRM260327C00190000"
        assert result["trades"][0]["side"] == "buy"
        assert result["trades"][0]["is_spread"] is True

    def test_stock_order_unchanged(self):
        """Regular stock order still works as before."""
        stock_order = self._make_order(
            symbol="AAPL", side="buy", filled_qty="10", filled_avg_price="150.0"
        )

        with patch(
            "RubberBand.scripts.persist_daily_results.extract_bot_tag_from_order",
            return_value="15M_STK",
        ):
            result = calculate_bot_pnl(
                orders=[stock_order],
                positions=[],
                bot_tag="15M_STK",
                target_date="2026-02-17",
            )

        assert len(result["trades"]) == 1
        assert result["trades"][0]["symbol"] == "AAPL"
        assert result["trades"][0]["side"] == "buy"
        assert result["trades"][0]["is_spread"] is False

    def test_stock_round_trip_pnl(self):
        """Stock buy + sell produces correct realized P&L."""
        buy_order = self._make_order(
            id="ord_buy",
            symbol="AAPL", side="buy",
            filled_qty="10", filled_avg_price="150.00",
        )
        sell_order = self._make_order(
            id="ord_sell",
            symbol="AAPL", side="sell",
            filled_qty="10", filled_avg_price="155.00",
            client_order_id="",  # untagged bracket exit
        )

        with patch(
            "RubberBand.scripts.persist_daily_results.extract_bot_tag_from_order",
            side_effect=lambda o: "15M_STK" if o["id"] == "ord_buy" else None,
        ):
            result = calculate_bot_pnl(
                orders=[buy_order, sell_order],
                positions=[],
                bot_tag="15M_STK",
                target_date="2026-02-17",
            )

        assert len(result["trades"]) == 2
        # Realized P&L: 10 * (155 - 150) = $50
        assert Decimal(result["realized_pnl"]) == Decimal("50.00")

    def test_mleg_spread_accumulates_buy_value(self):
        """Spread order (side=buy, is_spread=True) accumulates into buy_value."""
        mleg_order = self._make_order(
            symbol="",
            side="",
            filled_qty="1",
            filled_avg_price="2.50",
            legs=[
                {"symbol": "NVDA260321C00130000", "side": "buy", "status": "filled"},
                {"symbol": "NVDA260321C00135000", "side": "sell", "status": "filled"},
            ],
        )

        with patch(
            "RubberBand.scripts.persist_daily_results.extract_bot_tag_from_order",
            return_value="15M_OPT",
        ):
            result = calculate_bot_pnl(
                orders=[mleg_order],
                positions=[],
                bot_tag="15M_OPT",
                target_date="2026-02-17",
            )

        # The spread should count as a buy, contributing to buy_value
        assert len(result["trades"]) == 1
        assert result["trades"][0]["side"] == "buy"
        assert result["trades"][0]["price"] == "2.50"
        # No matching sell → realized P&L stays 0
        assert Decimal(result["realized_pnl"]) == Decimal("0")

    def test_mixed_stock_and_mleg(self):
        """Bot with both stock and spread orders on the same day."""
        stock_buy = self._make_order(
            id="s1", symbol="AAPL", side="buy",
            filled_qty="10", filled_avg_price="150.00",
        )
        stock_sell = self._make_order(
            id="s2", symbol="AAPL", side="sell",
            filled_qty="10", filled_avg_price="152.00",
            client_order_id="",
        )
        mleg_order = self._make_order(
            id="m1",
            symbol="",
            side="",
            filled_qty="1",
            filled_avg_price="1.00",
            client_order_id="15M_OPT_CRM_1234567890",
            legs=[
                {"symbol": "CRM260327C00190000", "side": "buy", "status": "filled"},
                {"symbol": "CRM260327C00195000", "side": "sell", "status": "filled"},
            ],
        )

        def tag_router(o):
            if o["id"] == "s1":
                return "15M_STK"
            if o["id"] == "m1":
                return "15M_OPT"
            return None

        # Test 15M_STK: should only see stock round-trip
        with patch(
            "RubberBand.scripts.persist_daily_results.extract_bot_tag_from_order",
            side_effect=tag_router,
        ):
            stk_result = calculate_bot_pnl(
                orders=[stock_buy, stock_sell, mleg_order],
                positions=[],
                bot_tag="15M_STK",
                target_date="2026-02-17",
            )
        assert len(stk_result["trades"]) == 2
        assert Decimal(stk_result["realized_pnl"]) == Decimal("20.00")

        # Test 15M_OPT: should only see the spread
        with patch(
            "RubberBand.scripts.persist_daily_results.extract_bot_tag_from_order",
            side_effect=tag_router,
        ):
            opt_result = calculate_bot_pnl(
                orders=[stock_buy, stock_sell, mleg_order],
                positions=[],
                bot_tag="15M_OPT",
                target_date="2026-02-17",
            )
        assert len(opt_result["trades"]) == 1
        assert opt_result["trades"][0]["is_spread"] is True


# ============================================================================
# ensure_all_registries_exist tests
# ============================================================================

class TestEnsureAllRegistriesExist:
    """Tests for position registry bootstrap."""

    def test_creates_missing_files(self, tmp_path):
        """Creates registry files for all 5 bots when dir is empty."""
        created = ensure_all_registries_exist(str(tmp_path))
        assert set(created) == BOT_TAGS

        for tag in BOT_TAGS:
            path = tmp_path / f"{tag}_positions.json"
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["bot_tag"] == tag
            assert data["positions"] == {}
            assert data["closed_positions"] == []

    def test_skips_existing_files(self, tmp_path):
        """Does not overwrite files that already exist."""
        # Pre-create one registry with a position
        existing = {
            "bot_tag": "15M_STK",
            "updated_at": "2026-01-01T00:00:00",
            "positions": {"AAPL": {"qty": 10}},
            "closed_positions": [],
        }
        (tmp_path / "15M_STK_positions.json").write_text(json.dumps(existing))

        created = ensure_all_registries_exist(str(tmp_path))
        assert "15M_STK" not in created
        assert len(created) == len(BOT_TAGS) - 1

        # Verify the existing file was NOT overwritten
        data = json.loads((tmp_path / "15M_STK_positions.json").read_text())
        assert "AAPL" in data["positions"]

    def test_creates_directory_if_missing(self, tmp_path):
        """Creates the registry directory if it doesn't exist."""
        target = tmp_path / "subdir" / "registry"
        created = ensure_all_registries_exist(str(target))
        assert len(created) == len(BOT_TAGS)
        assert target.is_dir()

    def test_idempotent(self, tmp_path):
        """Calling twice doesn't create duplicates or errors."""
        ensure_all_registries_exist(str(tmp_path))
        created_second = ensure_all_registries_exist(str(tmp_path))
        assert created_second == []  # Nothing new created

    def test_valid_json_for_position_registry(self, tmp_path):
        """Created files are loadable by PositionRegistry."""
        ensure_all_registries_exist(str(tmp_path))
        reg = PositionRegistry("15M_STK", registry_dir=str(tmp_path))
        assert reg.bot_tag == "15M_STK"
        assert len(reg.positions) == 0

    def test_ioerror_continues_and_logs(self, tmp_path):
        """IOError on one file doesn't prevent creating the rest."""
        call_count = 0
        original_open = open

        def patched_open(path, mode="r", *args, **kwargs):
            nonlocal call_count
            if mode == "x" and "15M_OPT_positions.json" in str(path):
                call_count += 1
                raise IOError("Disk full")
            return original_open(path, mode, *args, **kwargs)

        with patch("builtins.open", side_effect=patched_open):
            created = ensure_all_registries_exist(str(tmp_path))

        assert "15M_OPT" not in created
        # All other bots should still be created
        assert len(created) == len(BOT_TAGS) - 1
        assert call_count == 1
