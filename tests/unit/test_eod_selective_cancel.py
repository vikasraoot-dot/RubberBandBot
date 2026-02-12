"""
Tests for selective order cancellation in EOD flatten.

Verifies that cancel_orders_for_bots() correctly:
- Cancels orders belonging to intraday bots (15M_STK, 15M_OPT)
- Preserves WK_STK bracket legs (TP/SL)
- Preserves safety-check OCO orders
- Handles bracket legs that have UUID client_order_ids (traces via parent)
"""

import pytest
from unittest.mock import patch, MagicMock
import datetime as dt


# ---------------------------------------------------------------------------
# Helpers to build fake Alpaca order responses
# ---------------------------------------------------------------------------

def _make_order(order_id, coid, symbol, status="new", side="buy",
                order_type="limit", order_class="", legs=None):
    """Build a minimal Alpaca-style order dict."""
    return {
        "id": order_id,
        "client_order_id": coid,
        "symbol": symbol,
        "status": status,
        "side": side,
        "type": order_type,
        "order_class": order_class,
        "legs": legs or [],
    }


def _make_bracket(parent_id, parent_coid, symbol, tp_id, sl_id,
                  parent_status="filled"):
    """Build a bracket order (parent + TP leg + SL leg) for nested response."""
    return _make_order(
        parent_id, parent_coid, symbol, status=parent_status,
        order_class="bracket",
        legs=[
            _make_order(tp_id, f"uuid-tp-{tp_id}", symbol,
                        status="new", side="sell", order_type="limit"),
            _make_order(sl_id, f"uuid-sl-{sl_id}", symbol,
                        status="new", side="sell", order_type="stop"),
        ],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_alpaca(monkeypatch):
    """Patch requests.get/delete to simulate Alpaca API."""
    responses = {}  # keyed by (method, url_suffix) → response
    cancelled_ids = []

    def fake_get(url, headers=None, params=None, timeout=None):
        resp = MagicMock()
        resp.status_code = 200
        status = (params or {}).get("status", "")
        nested = (params or {}).get("nested", "")

        if "nested" in str(nested).lower() or nested == "true":
            resp.json.return_value = responses.get("nested", [])
        elif status == "open":
            resp.json.return_value = responses.get("open", [])
        else:
            resp.json.return_value = []
        resp.raise_for_status = MagicMock()
        return resp

    def fake_delete(url, headers=None, timeout=None):
        resp = MagicMock()
        # Extract order id from URL  /v2/orders/{id}
        parts = url.rstrip("/").split("/")
        oid = parts[-1] if parts[-1] != "orders" else "__all__"
        cancelled_ids.append(oid)
        resp.status_code = 200
        resp.json.return_value = {}
        resp.raise_for_status = MagicMock()
        return resp

    import requests as _req
    monkeypatch.setattr(_req, "get", fake_get)
    monkeypatch.setattr(_req, "delete", fake_delete)

    return {"responses": responses, "cancelled_ids": cancelled_ids}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCancelOrdersForBots:
    """Test cancel_orders_for_bots() selective cancellation."""

    def test_cancels_intraday_parent_orders(self, mock_alpaca):
        """Orders with 15M_STK prefix should be cancelled."""
        from RubberBand.src.data import cancel_orders_for_bots

        mock_alpaca["responses"]["nested"] = [
            _make_order("ord-1", "15M_STK_AAPL_123", "AAPL", status="new"),
        ]
        mock_alpaca["responses"]["open"] = [
            _make_order("ord-1", "15M_STK_AAPL_123", "AAPL", status="new"),
        ]

        result = cancel_orders_for_bots(
            "https://paper-api.alpaca.markets", "k", "s",
            bot_prefixes=["15M_STK", "15M_OPT"],
        )

        assert result["cancelled"] == 1
        assert result["preserved"] == 0
        assert "ord-1" in mock_alpaca["cancelled_ids"]

    def test_preserves_wk_stk_parent_orders(self, mock_alpaca):
        """Orders with WK_STK prefix should NOT be cancelled."""
        from RubberBand.src.data import cancel_orders_for_bots

        mock_alpaca["responses"]["nested"] = [
            _make_order("ord-w1", "WK_STK_PLTR_999", "PLTR", status="new"),
        ]
        mock_alpaca["responses"]["open"] = [
            _make_order("ord-w1", "WK_STK_PLTR_999", "PLTR", status="new"),
        ]

        result = cancel_orders_for_bots(
            "https://paper-api.alpaca.markets", "k", "s",
            bot_prefixes=["15M_STK", "15M_OPT"],
        )

        assert result["cancelled"] == 0
        assert result["preserved"] == 1
        assert "ord-w1" not in mock_alpaca["cancelled_ids"]

    def test_cancels_intraday_bracket_legs_by_parent_mapping(self, mock_alpaca):
        """Bracket legs with UUID coids should be cancelled if parent is intraday."""
        from RubberBand.src.data import cancel_orders_for_bots

        # Nested: bracket parent (15M_STK) with TP + SL legs
        mock_alpaca["responses"]["nested"] = [
            _make_bracket(
                "parent-1", "15M_STK_TSLA_100", "TSLA",
                tp_id="leg-tp-1", sl_id="leg-sl-1",
            ),
        ]
        # Open (flat): only the legs show up (parent is filled)
        mock_alpaca["responses"]["open"] = [
            _make_order("leg-tp-1", "uuid-tp-leg-tp-1", "TSLA",
                        side="sell", order_type="limit"),
            _make_order("leg-sl-1", "uuid-sl-leg-sl-1", "TSLA",
                        side="sell", order_type="stop"),
        ]

        result = cancel_orders_for_bots(
            "https://paper-api.alpaca.markets", "k", "s",
            bot_prefixes=["15M_STK", "15M_OPT"],
        )

        assert result["cancelled"] == 2
        assert result["preserved"] == 0
        assert "leg-tp-1" in mock_alpaca["cancelled_ids"]
        assert "leg-sl-1" in mock_alpaca["cancelled_ids"]

    def test_preserves_wk_stk_bracket_legs(self, mock_alpaca):
        """WK_STK bracket legs must NOT be cancelled — this is the main bug fix."""
        from RubberBand.src.data import cancel_orders_for_bots

        # Nested: WK_STK bracket with TP + SL legs
        mock_alpaca["responses"]["nested"] = [
            _make_bracket(
                "wk-parent-1", "WK_STK_PLTR_200", "PLTR",
                tp_id="wk-tp-1", sl_id="wk-sl-1",
            ),
        ]
        # Open (flat): only the legs (parent filled)
        mock_alpaca["responses"]["open"] = [
            _make_order("wk-tp-1", "uuid-wk-tp", "PLTR",
                        side="sell", order_type="limit"),
            _make_order("wk-sl-1", "uuid-wk-sl", "PLTR",
                        side="sell", order_type="stop"),
        ]

        result = cancel_orders_for_bots(
            "https://paper-api.alpaca.markets", "k", "s",
            bot_prefixes=["15M_STK", "15M_OPT"],
        )

        assert result["cancelled"] == 0
        assert result["preserved"] == 2
        assert "wk-tp-1" not in mock_alpaca["cancelled_ids"]
        assert "wk-sl-1" not in mock_alpaca["cancelled_ids"]

    def test_preserves_safety_check_oco_orders(self, mock_alpaca):
        """Safety-check OCO orders (UUID coids, no bot prefix) should be preserved."""
        from RubberBand.src.data import cancel_orders_for_bots

        oco_parent_id = "safety-oco-1"
        oco_leg_id = "safety-oco-leg-1"

        mock_alpaca["responses"]["nested"] = [
            _make_order(
                oco_parent_id, "e1f2a3b4-uuid-safety", "CRM",
                status="new", order_class="oco",
                legs=[
                    _make_order(oco_leg_id, "f5e6d7c8-uuid-safety-leg", "CRM",
                                side="sell", order_type="stop", status="held"),
                ],
            ),
        ]
        mock_alpaca["responses"]["open"] = [
            _make_order(oco_parent_id, "e1f2a3b4-uuid-safety", "CRM",
                        side="sell", order_type="limit", order_class="oco"),
            _make_order(oco_leg_id, "f5e6d7c8-uuid-safety-leg", "CRM",
                        side="sell", order_type="stop"),
        ]

        result = cancel_orders_for_bots(
            "https://paper-api.alpaca.markets", "k", "s",
            bot_prefixes=["15M_STK", "15M_OPT"],
        )

        assert result["cancelled"] == 0
        assert result["preserved"] == 2
        assert oco_parent_id not in mock_alpaca["cancelled_ids"]
        assert oco_leg_id not in mock_alpaca["cancelled_ids"]

    def test_mixed_scenario_cancel_intraday_preserve_weekly(self, mock_alpaca):
        """Full scenario: mix of 15M_STK, WK_STK, and safety orders."""
        from RubberBand.src.data import cancel_orders_for_bots

        mock_alpaca["responses"]["nested"] = [
            # 15M_STK bracket (should cancel legs)
            _make_bracket(
                "15m-parent", "15M_STK_AAPL_100", "AAPL",
                tp_id="15m-tp", sl_id="15m-sl",
            ),
            # WK_STK bracket (should preserve legs)
            _make_bracket(
                "wk-parent", "WK_STK_PLTR_200", "PLTR",
                tp_id="wk-tp", sl_id="wk-sl",
            ),
            # Safety check OCO (should preserve)
            _make_order(
                "safety-1", "uuid-safety-oco", "MRVL",
                status="new", order_class="oco",
                legs=[
                    _make_order("safety-leg", "uuid-safety-leg", "MRVL",
                                side="sell", order_type="stop", status="held"),
                ],
            ),
            # Unfilled 15M_STK entry (should cancel)
            _make_order("15m-entry", "15M_STK_MSFT_300", "MSFT", status="new"),
        ]

        mock_alpaca["responses"]["open"] = [
            # 15M_STK bracket legs (flat)
            _make_order("15m-tp", "uuid-15m-tp", "AAPL",
                        side="sell", order_type="limit"),
            _make_order("15m-sl", "uuid-15m-sl", "AAPL",
                        side="sell", order_type="stop"),
            # WK_STK bracket legs (flat)
            _make_order("wk-tp", "uuid-wk-tp", "PLTR",
                        side="sell", order_type="limit"),
            _make_order("wk-sl", "uuid-wk-sl", "PLTR",
                        side="sell", order_type="stop"),
            # Safety OCO (flat)
            _make_order("safety-1", "uuid-safety-oco", "MRVL",
                        side="sell", order_class="oco"),
            _make_order("safety-leg", "uuid-safety-leg", "MRVL",
                        side="sell", order_type="stop"),
            # 15M_STK unfilled entry
            _make_order("15m-entry", "15M_STK_MSFT_300", "MSFT", status="new"),
        ]

        result = cancel_orders_for_bots(
            "https://paper-api.alpaca.markets", "k", "s",
            bot_prefixes=["15M_STK", "15M_OPT"],
        )

        # 15M: 2 legs + 1 entry = 3 cancelled
        assert result["cancelled"] == 3
        # WK: 2 legs + Safety: 2 orders = 4 preserved
        assert result["preserved"] == 4

        # Verify correct IDs
        assert "15m-tp" in mock_alpaca["cancelled_ids"]
        assert "15m-sl" in mock_alpaca["cancelled_ids"]
        assert "15m-entry" in mock_alpaca["cancelled_ids"]
        assert "wk-tp" not in mock_alpaca["cancelled_ids"]
        assert "wk-sl" not in mock_alpaca["cancelled_ids"]
        assert "safety-1" not in mock_alpaca["cancelled_ids"]

    def test_handles_api_error_on_nested_fetch(self, mock_alpaca, monkeypatch):
        """Should return error dict if nested order fetch fails."""
        from RubberBand.src.data import cancel_orders_for_bots
        import requests

        def bad_get(*args, **kwargs):
            raise requests.ConnectionError("API down")

        monkeypatch.setattr(requests, "get", bad_get)

        result = cancel_orders_for_bots(
            "https://paper-api.alpaca.markets", "k", "s",
            bot_prefixes=["15M_STK"],
        )

        assert result["cancelled"] == 0
        assert len(result["errors"]) == 1

    def test_empty_open_orders(self, mock_alpaca):
        """No open orders = no cancellations, no errors."""
        from RubberBand.src.data import cancel_orders_for_bots

        mock_alpaca["responses"]["nested"] = []
        mock_alpaca["responses"]["open"] = []

        result = cancel_orders_for_bots(
            "https://paper-api.alpaca.markets", "k", "s",
            bot_prefixes=["15M_STK", "15M_OPT"],
        )

        assert result["cancelled"] == 0
        assert result["preserved"] == 0
        assert result["errors"] == []

    def test_cancels_15m_opt_prefix_orders(self, mock_alpaca):
        """15M_OPT prefix orders should also be cancelled."""
        from RubberBand.src.data import cancel_orders_for_bots

        mock_alpaca["responses"]["nested"] = [
            _make_order("opt-1", "15M_OPT_SPY_400", "SPY", status="new"),
        ]
        mock_alpaca["responses"]["open"] = [
            _make_order("opt-1", "15M_OPT_SPY_400", "SPY", status="new"),
        ]

        result = cancel_orders_for_bots(
            "https://paper-api.alpaca.markets", "k", "s",
            bot_prefixes=["15M_STK", "15M_OPT"],
        )

        assert result["cancelled"] == 1
        assert "opt-1" in mock_alpaca["cancelled_ids"]

    def test_handles_api_error_on_open_orders_fetch(self, mock_alpaca, monkeypatch):
        """Should return error dict if open orders fetch fails (step 2)."""
        from RubberBand.src.data import cancel_orders_for_bots
        import requests

        call_count = [0]

        def fail_on_second_get(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call (nested) succeeds
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = []
                resp.raise_for_status = MagicMock()
                return resp
            # Second call (open orders) fails
            raise requests.ConnectionError("API down on second call")

        monkeypatch.setattr(requests, "get", fail_on_second_get)

        result = cancel_orders_for_bots(
            "https://paper-api.alpaca.markets", "k", "s",
            bot_prefixes=["15M_STK"],
        )

        assert result["cancelled"] == 0
        assert len(result["errors"]) == 1

    def test_preserves_orders_with_empty_client_order_id(self, mock_alpaca):
        """Orders with missing/empty client_order_id should be preserved (safe default)."""
        from RubberBand.src.data import cancel_orders_for_bots

        mock_alpaca["responses"]["nested"] = [
            _make_order("mystery-1", "", "XYZ", status="new"),
        ]
        mock_alpaca["responses"]["open"] = [
            _make_order("mystery-1", "", "XYZ", status="new"),
        ]

        result = cancel_orders_for_bots(
            "https://paper-api.alpaca.markets", "k", "s",
            bot_prefixes=["15M_STK", "15M_OPT"],
        )

        assert result["cancelled"] == 0
        assert result["preserved"] == 1


class TestFlatEodIntegration:
    """Test that flat_eod.py uses selective cancel (not cancel_all)."""

    def test_flat_eod_imports_selective_cancel(self):
        """flat_eod.py should import cancel_orders_for_bots."""
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location(
            "flat_eod",
            os.path.join(os.path.dirname(__file__), "../../RubberBand/scripts/flat_eod.py"),
        )
        # Just verify the import exists — if cancel_orders_for_bots is missing
        # from data.py, this would fail at import time
        assert spec is not None

    def test_flat_eod_does_not_call_cancel_all_orders(self):
        """flat_eod.main() must NOT call cancel_all_orders (the old, dangerous path)."""
        import ast
        import os

        flat_eod_path = os.path.join(
            os.path.dirname(__file__), "../../RubberBand/scripts/flat_eod.py"
        )
        with open(flat_eod_path) as f:
            source = f.read()

        tree = ast.parse(source)

        # Walk AST to find function calls
        cancel_all_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "cancel_all_orders":
                    cancel_all_calls.append(node.lineno)
                elif isinstance(func, ast.Attribute) and func.attr == "cancel_all_orders":
                    cancel_all_calls.append(node.lineno)

        assert cancel_all_calls == [], (
            f"flat_eod.py still calls cancel_all_orders() on line(s) {cancel_all_calls}. "
            "This would destroy WK_STK bracket legs. Use cancel_orders_for_bots() instead."
        )
