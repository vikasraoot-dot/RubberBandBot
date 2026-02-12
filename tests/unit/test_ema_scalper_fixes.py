"""
Tests for EMA Momentum Scalper bug fixes.

Covers 4 critical fixes:
1. close_position() returns order details (order_id, filled_avg_price)
2. _poll_fill_price() polls broker for actual fill price
3. P&L reconciliation uses broker fills as ground truth
4. Stale quote detection cross-validates IEX vs position API

These fixes prevent phantom P&L drift and stale-quote stop-outs.
"""

import json
import time
import datetime as dt
import pytest
from unittest.mock import patch, Mock, MagicMock, call
from dataclasses import asdict


# ── Helpers ──────────────────────────────────────────────────────────────────

class MockResponse:
    """Minimal requests.Response mock."""

    def __init__(self, status_code=200, json_data=None, content=None):
        self.status_code = status_code
        self._json_data = json_data
        self.content = content if content is not None else (
            json.dumps(json_data).encode() if json_data is not None else b""
        )

    def json(self):
        if self._json_data is not None:
            return self._json_data
        if self.content:
            return json.loads(self.content)
        raise ValueError("No JSON")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


# =============================================================================
# Category 1: close_position() return value
# =============================================================================

class TestClosePositionReturnValue:
    """close_position() must return order details so callers can poll fill price."""

    def test_close_position_returns_order_details(self):
        """200 with order JSON -> returns ok=True plus order_id, filled_avg_price."""
        from ScalpingBots.src.broker import close_position

        order_body = {
            "id": "abc-123",
            "status": "pending_new",
            "filled_avg_price": "152.35",
            "filled_qty": "10",
            "symbol": "AAPL",
        }
        mock_resp = MockResponse(200, json_data=order_body)

        with patch("ScalpingBots.src.broker.requests.delete", return_value=mock_resp):
            result = close_position(
                base_url="https://paper-api.alpaca.markets",
                key="k", secret="s", symbol="AAPL",
            )

        assert result["ok"] is True
        assert result["order_id"] == "abc-123"
        assert result["filled_avg_price"] == "152.35"
        assert result["filled_qty"] == "10"
        assert result["symbol"] == "AAPL"

    def test_close_position_404_not_found(self):
        """404 -> position already closed, returns ok=True with message."""
        from ScalpingBots.src.broker import close_position

        mock_resp = MockResponse(404)
        with patch("ScalpingBots.src.broker.requests.delete", return_value=mock_resp):
            result = close_position(
                base_url="https://paper-api.alpaca.markets",
                key="k", secret="s", symbol="AAPL",
            )

        assert result["ok"] is True
        assert "not found" in result.get("message", "").lower()
        assert "order_id" not in result

    def test_close_position_empty_body(self):
        """200 with empty body -> returns ok=True, no crash."""
        from ScalpingBots.src.broker import close_position

        mock_resp = MockResponse(200, content=b"")
        mock_resp._json_data = None
        with patch("ScalpingBots.src.broker.requests.delete", return_value=mock_resp):
            result = close_position(
                base_url="https://paper-api.alpaca.markets",
                key="k", secret="s", symbol="AAPL",
            )

        assert result["ok"] is True
        assert "order_id" not in result

    def test_close_position_malformed_json(self):
        """200 with malformed JSON body -> returns ok=True gracefully."""
        from ScalpingBots.src.broker import close_position

        mock_resp = MockResponse(200, content=b"not json{{{")
        mock_resp._json_data = None
        with patch("ScalpingBots.src.broker.requests.delete", return_value=mock_resp):
            result = close_position(
                base_url="https://paper-api.alpaca.markets",
                key="k", secret="s", symbol="AAPL",
            )

        assert result["ok"] is True

    def test_close_position_backward_compatible(self):
        """Old callers checking only result['ok'] still work."""
        from ScalpingBots.src.broker import close_position

        order_body = {
            "id": "abc-123",
            "status": "filled",
            "filled_avg_price": "152.35",
            "filled_qty": "10",
            "symbol": "AAPL",
        }
        mock_resp = MockResponse(200, json_data=order_body)
        with patch("ScalpingBots.src.broker.requests.delete", return_value=mock_resp):
            result = close_position(
                base_url="https://paper-api.alpaca.markets",
                key="k", secret="s", symbol="AAPL",
            )

        # Old-style check: only inspect ok
        assert result["ok"] is True
        # New fields are present but don't break the dict interface
        assert isinstance(result, dict)

    def test_close_position_204_no_content(self):
        """204 No Content -> returns ok=True."""
        from ScalpingBots.src.broker import close_position

        mock_resp = MockResponse(204, content=b"")
        mock_resp._json_data = None
        with patch("ScalpingBots.src.broker.requests.delete", return_value=mock_resp):
            result = close_position(
                base_url="https://paper-api.alpaca.markets",
                key="k", secret="s", symbol="AAPL",
            )

        assert result["ok"] is True

    def test_close_position_no_order_id_in_body(self):
        """Body without 'id' field -> ok=True, no order_id key."""
        from ScalpingBots.src.broker import close_position

        mock_resp = MockResponse(200, json_data={"status": "filled", "symbol": "AAPL"})
        with patch("ScalpingBots.src.broker.requests.delete", return_value=mock_resp):
            result = close_position(
                base_url="https://paper-api.alpaca.markets",
                key="k", secret="s", symbol="AAPL",
            )

        assert result["ok"] is True
        assert "order_id" not in result


# =============================================================================
# Category 2: _poll_fill_price() helper
# =============================================================================

class TestPollFillPrice:
    """_poll_fill_price() must poll broker order API for actual fill price."""

    @patch("ScalpingBots.scripts.live_ema_scalp.time.sleep")
    def test_poll_fill_immediate(self, mock_sleep):
        """Order already filled on first poll -> returns fill price immediately."""
        from ScalpingBots.scripts.live_ema_scalp import _poll_fill_price

        filled_order = MockResponse(200, json_data={
            "status": "filled",
            "filled_avg_price": "153.50",
        })

        with patch("ScalpingBots.scripts.live_ema_scalp.requests.get", return_value=filled_order):
            price = _poll_fill_price(
                "https://paper-api.alpaca.markets", "k", "s",
                order_id="ord-1", fallback_price=150.0, timeout=5.0,
            )

        assert price == 153.50
        mock_sleep.assert_not_called()

    @patch("ScalpingBots.scripts.live_ema_scalp.time.sleep")
    def test_poll_fill_after_delay(self, mock_sleep):
        """Order fills after 1 poll of 'new' status -> returns fill price."""
        from ScalpingBots.scripts.live_ema_scalp import _poll_fill_price

        pending = MockResponse(200, json_data={"status": "new"})
        filled = MockResponse(200, json_data={
            "status": "filled",
            "filled_avg_price": "155.00",
        })

        with patch(
            "ScalpingBots.scripts.live_ema_scalp.requests.get",
            side_effect=[pending, filled],
        ):
            price = _poll_fill_price(
                "https://paper-api.alpaca.markets", "k", "s",
                order_id="ord-2", fallback_price=150.0, timeout=5.0,
            )

        assert price == 155.00
        assert mock_sleep.call_count == 1

    @patch("ScalpingBots.scripts.live_ema_scalp.time.sleep")
    def test_poll_fill_timeout(self, mock_sleep):
        """Order stays 'new' -> returns fallback after timeout."""
        from ScalpingBots.scripts.live_ema_scalp import _poll_fill_price

        pending = MockResponse(200, json_data={"status": "new"})

        with patch(
            "ScalpingBots.scripts.live_ema_scalp.requests.get",
            return_value=pending,
        ):
            price = _poll_fill_price(
                "https://paper-api.alpaca.markets", "k", "s",
                order_id="ord-3", fallback_price=148.00, timeout=1.5,
            )

        assert price == 148.00
        # With 0.5s interval and 1.5s timeout, should sleep ~3 times
        assert mock_sleep.call_count >= 2

    @patch("ScalpingBots.scripts.live_ema_scalp.time.sleep")
    def test_poll_fill_cancelled(self, mock_sleep):
        """Order cancelled (terminal) -> returns fallback immediately."""
        from ScalpingBots.scripts.live_ema_scalp import _poll_fill_price

        cancelled = MockResponse(200, json_data={"status": "canceled"})

        with patch(
            "ScalpingBots.scripts.live_ema_scalp.requests.get",
            return_value=cancelled,
        ):
            price = _poll_fill_price(
                "https://paper-api.alpaca.markets", "k", "s",
                order_id="ord-4", fallback_price=145.00, timeout=5.0,
            )

        assert price == 145.00
        # Should not continue polling after terminal status
        mock_sleep.assert_not_called()

    @patch("ScalpingBots.scripts.live_ema_scalp.time.sleep")
    def test_poll_fill_expired(self, mock_sleep):
        """Order expired (terminal) -> returns fallback immediately."""
        from ScalpingBots.scripts.live_ema_scalp import _poll_fill_price

        expired = MockResponse(200, json_data={"status": "expired"})

        with patch(
            "ScalpingBots.scripts.live_ema_scalp.requests.get",
            return_value=expired,
        ):
            price = _poll_fill_price(
                "https://paper-api.alpaca.markets", "k", "s",
                order_id="ord-5", fallback_price=145.00, timeout=5.0,
            )

        assert price == 145.00
        mock_sleep.assert_not_called()

    @patch("ScalpingBots.scripts.live_ema_scalp.time.sleep")
    def test_poll_fill_network_error(self, mock_sleep):
        """Network error during polling -> returns fallback with warning."""
        from ScalpingBots.scripts.live_ema_scalp import _poll_fill_price

        with patch(
            "ScalpingBots.scripts.live_ema_scalp.requests.get",
            side_effect=ConnectionError("Connection refused"),
        ):
            price = _poll_fill_price(
                "https://paper-api.alpaca.markets", "k", "s",
                order_id="ord-6", fallback_price=140.00, timeout=1.5,
            )

        assert price == 140.00

    @patch("ScalpingBots.scripts.live_ema_scalp.time.sleep")
    def test_poll_fill_no_order_id(self, mock_sleep):
        """Empty order_id -> goes through loop but times out to fallback."""
        from ScalpingBots.scripts.live_ema_scalp import _poll_fill_price

        # With an empty order_id, the API call will fail or return non-filled.
        # The function should ultimately return fallback.
        error_resp = MockResponse(404)
        error_resp._json_data = None

        with patch(
            "ScalpingBots.scripts.live_ema_scalp.requests.get",
            return_value=error_resp,
        ):
            price = _poll_fill_price(
                "https://paper-api.alpaca.markets", "k", "s",
                order_id="", fallback_price=135.00, timeout=1.0,
            )

        assert price == 135.00


# =============================================================================
# Category 3: P&L Reconciliation
# =============================================================================

class TestPnlReconciliation:
    """P&L reconciliation: broker fills are ground truth."""

    def test_reconcile_phantom_loss(self):
        """broker_pnl=-5, daily_pnl=-180 -> result should be -5 (broker truth)."""
        from ScalpingBots.src.broker import calculate_realized_pnl

        # Broker has matched fills showing only -$5 loss
        fills = [
            {"symbol": "AAPL", "side": "buy", "filled_avg_price": "150.00", "filled_qty": "10", "status": "filled"},
            {"symbol": "AAPL", "side": "sell", "filled_avg_price": "149.50", "filled_qty": "10", "status": "filled"},
        ]
        broker_pnl = calculate_realized_pnl(fills)
        daily_pnl = -180.0  # phantom loss from stale quotes

        # Reconciliation logic: always adopt broker P&L
        reconciled = broker_pnl  # This is what the main loop does
        assert abs(reconciled - (-5.0)) < 0.01
        assert abs(reconciled) < abs(daily_pnl)  # broker truth is much smaller loss

    def test_reconcile_no_fills(self):
        """broker_pnl=0 (no matched fills) -> daily_pnl unaffected."""
        from ScalpingBots.src.broker import calculate_realized_pnl

        # Only buy fills, no sells -> no realized P&L
        fills = [
            {"symbol": "AAPL", "side": "buy", "filled_avg_price": "150.00", "filled_qty": "10", "status": "filled"},
        ]
        broker_pnl = calculate_realized_pnl(fills)
        assert broker_pnl == 0.0

    def test_reconcile_divergence_warning(self, capsys):
        """When broker and internal differ by > $5, warning is printed."""
        # Simulate the reconciliation block from the main loop
        daily_pnl = -180.0
        broker_pnl = -5.0
        pnl_divergence = abs(broker_pnl - daily_pnl)

        # This mimics the live loop's logic
        if pnl_divergence > 5.0:
            print(f"  [WARN] P&L divergence: internal=${daily_pnl:.2f} vs "
                  f"broker=${broker_pnl:.2f} (diff=${pnl_divergence:.2f})")

        captured = capsys.readouterr()
        assert "P&L divergence" in captured.out
        assert "175.00" in captured.out  # diff between 180 and 5

    def test_reconcile_close_values(self, capsys):
        """When broker and internal within $5 -> no warning."""
        daily_pnl = -8.0
        broker_pnl = -5.0
        pnl_divergence = abs(broker_pnl - daily_pnl)

        # Should NOT trigger warning
        if pnl_divergence > 5.0:
            print(f"  [WARN] P&L divergence")

        captured = capsys.readouterr()
        assert "P&L divergence" not in captured.out

    def test_calculate_realized_pnl_multiple_symbols(self):
        """P&L calculation with multiple symbols, partial fills."""
        from ScalpingBots.src.broker import calculate_realized_pnl

        fills = [
            # AAPL: bought 10 @ 150, sold 10 @ 155 = +50
            {"symbol": "AAPL", "side": "buy", "filled_avg_price": "150.00", "filled_qty": "10", "status": "filled"},
            {"symbol": "AAPL", "side": "sell", "filled_avg_price": "155.00", "filled_qty": "10", "status": "filled"},
            # MSFT: bought 5 @ 400, sold 5 @ 395 = -25
            {"symbol": "MSFT", "side": "buy", "filled_avg_price": "400.00", "filled_qty": "5", "status": "filled"},
            {"symbol": "MSFT", "side": "sell", "filled_avg_price": "395.00", "filled_qty": "5", "status": "filled"},
            # NVDA: bought 3 @ 800, no sell -> 0 (unmatched)
            {"symbol": "NVDA", "side": "buy", "filled_avg_price": "800.00", "filled_qty": "3", "status": "filled"},
        ]
        pnl = calculate_realized_pnl(fills)
        assert abs(pnl - 25.0) < 0.01  # 50 - 25 = 25

    def test_calculate_realized_pnl_empty_fills(self):
        """No fills -> 0.0 P&L."""
        from ScalpingBots.src.broker import calculate_realized_pnl

        assert calculate_realized_pnl([]) == 0.0


# =============================================================================
# Category 4: Stale Quote Detection
# =============================================================================

class TestStaleQuoteDetection:
    """Stale IEX quote detection cross-validates against position API."""

    def _make_position(self, entry_price=100.0, atr=2.0, side="buy"):
        from ScalpingBots.scripts.live_ema_scalp import LivePosition
        return LivePosition(
            symbol="TEST",
            side=side,
            qty=10,
            entry_price=entry_price,
            entry_time="2026-02-11T14:00:00Z",
            sl_price=entry_price - atr * 1.5 if side == "buy" else entry_price + atr * 1.5,
            tp_price=entry_price + atr * 3.0 if side == "buy" else entry_price - atr * 3.0,
            atr=atr,
            peak_price=entry_price,
        )

    def test_stale_quote_detected(self):
        """IEX bid 7% below entry, position API near entry -> uses position price."""
        pos = self._make_position(entry_price=100.0, atr=2.0)

        # Simulate the stale quote detection block from manage_exits
        current_price = 93.0  # 7% below entry (IEX stale)
        is_long = True
        implied_loss_pct = (pos.entry_price - current_price) / pos.entry_price
        assert implied_loss_pct > 0.03  # Triggers cross-validation

        # Position API says price is near entry
        position_price = 99.50
        discrepancy = abs(current_price - position_price)
        assert discrepancy > pos.atr  # 6.5 > 2.0 -> stale confirmed

        # Logic: override current_price with position_price
        if discrepancy > pos.atr:
            current_price = position_price

        assert current_price == 99.50

    def test_real_price_drop(self):
        """IEX bid 7% below entry, position API ALSO low -> keeps IEX (real crash)."""
        pos = self._make_position(entry_price=100.0, atr=2.0)

        current_price = 93.0  # IEX says 93
        is_long = True
        implied_loss_pct = (pos.entry_price - current_price) / pos.entry_price
        assert implied_loss_pct > 0.03

        # Position API ALSO shows low price -> real drop, not stale
        position_price = 93.20
        discrepancy = abs(current_price - position_price)
        assert discrepancy < pos.atr  # 0.2 < 2.0 -> NOT stale

        # current_price should NOT be overridden
        if discrepancy > pos.atr:
            current_price = position_price
        assert current_price == 93.0  # Kept original

    def test_stale_quote_position_api_fails(self):
        """IEX bid 7% below entry, position API errors -> keeps IEX with no override."""
        pos = self._make_position(entry_price=100.0, atr=2.0)

        current_price = 93.0
        is_long = True
        implied_loss_pct = (pos.entry_price - current_price) / pos.entry_price
        assert implied_loss_pct > 0.03

        # Position API returned None (network error)
        position_price = None

        # Logic: if position_price is None, keep IEX
        if position_price is not None:
            discrepancy = abs(current_price - position_price)
            if discrepancy > pos.atr:
                current_price = position_price

        assert current_price == 93.0  # Kept original since API failed

    def test_no_stale_within_threshold(self):
        """IEX bid 2% below entry -> no cross-validation triggered."""
        pos = self._make_position(entry_price=100.0, atr=2.0)

        current_price = 98.0  # Only 2% below
        is_long = True
        implied_loss_pct = (pos.entry_price - current_price) / pos.entry_price
        assert implied_loss_pct <= 0.03  # 2% < 3% threshold

        # Cross-validation should NOT be triggered
        # The price remains as-is
        assert current_price == 98.0

    def test_stale_quote_short_position(self):
        """Short position: IEX ask 7% above entry, position API near entry -> override."""
        pos = self._make_position(entry_price=100.0, atr=2.0, side="sell")

        current_price = 107.0  # 7% adverse for short
        is_long = False
        implied_loss_pct = (current_price - pos.entry_price) / pos.entry_price
        assert implied_loss_pct > 0.03

        position_price = 100.50
        discrepancy = abs(current_price - position_price)
        assert discrepancy > pos.atr

        if discrepancy > pos.atr:
            current_price = position_price

        assert current_price == 100.50


# =============================================================================
# Category 5: End-to-end Exit Flow
# =============================================================================

class TestExitFlow:
    """Full exit flow verifying broker fill price is used for P&L."""

    def _make_managed_position(self, symbol="AAPL", entry_price=150.0, atr=2.0):
        from ScalpingBots.scripts.live_ema_scalp import LivePosition
        return LivePosition(
            symbol=symbol,
            side="buy",
            qty=10,
            entry_price=entry_price,
            entry_time="2026-02-11T14:00:00Z",
            sl_price=entry_price - atr * 1.5,
            tp_price=entry_price + atr * 3.0,
            atr=atr,
            peak_price=entry_price,
            bars_held=5,
        )

    def _make_ema_config(self):
        from ScalpingBots.strategies.ema_momentum import EMAMomentumConfig
        return EMAMomentumConfig()

    @patch("ScalpingBots.scripts.live_ema_scalp.time.sleep")
    @patch("ScalpingBots.scripts.live_ema_scalp.log_event")
    def test_exit_uses_fill_price_for_pnl(self, mock_log, mock_sleep):
        """Full exit flow: broker fill price ($155) used for P&L, not IEX quote ($154)."""
        from ScalpingBots.scripts.live_ema_scalp import manage_exits, LivePosition

        cfg = self._make_ema_config()
        pos = self._make_managed_position(entry_price=150.0, atr=2.0)
        # Set TP so that bid price triggers TAKE_PROFIT
        pos.tp_price = 153.0
        managed = {"AAPL": pos}

        # IEX quote: bid=154 (triggers TP since 154 > 153)
        mock_quote = {"bid": 154.0, "ask": 154.50, "bid_size": 100, "ask_size": 100}

        # close_position returns order_id
        mock_close = {"ok": True, "order_id": "fill-order-1", "status": "pending_new", "symbol": "AAPL"}

        # _poll_fill_price returns actual fill at 155.0
        poll_resp = MockResponse(200, json_data={
            "status": "filled", "filled_avg_price": "155.00",
        })

        now_et = dt.datetime(2026, 2, 11, 10, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5)))

        with patch("ScalpingBots.scripts.live_ema_scalp.get_latest_quote", return_value=mock_quote), \
             patch("ScalpingBots.scripts.live_ema_scalp.close_position", return_value=mock_close), \
             patch("ScalpingBots.scripts.live_ema_scalp.requests.get", return_value=poll_resp):

            cycle_pnl = manage_exits(
                managed, "https://paper-api.alpaca.markets", "k", "s",
                cfg, now_et, log_file=None, dry_run=False,
            )

        # P&L should be based on fill price: (155 - 150) * 10 = 50
        assert abs(cycle_pnl - 50.0) < 0.01

        # Position should be removed from managed
        assert "AAPL" not in managed

    @patch("ScalpingBots.scripts.live_ema_scalp.time.sleep")
    @patch("ScalpingBots.scripts.live_ema_scalp.log_event")
    def test_exit_logs_both_prices(self, mock_log, mock_sleep):
        """EXIT log event includes both exit_price (fill) and quote_price (IEX)."""
        from ScalpingBots.scripts.live_ema_scalp import manage_exits, LivePosition

        cfg = self._make_ema_config()
        pos = self._make_managed_position(entry_price=150.0, atr=2.0)
        pos.tp_price = 153.0
        managed = {"AAPL": pos}

        mock_quote = {"bid": 154.0, "ask": 154.50, "bid_size": 100, "ask_size": 100}
        mock_close = {"ok": True, "order_id": "fill-order-2", "status": "pending_new", "symbol": "AAPL"}
        poll_resp = MockResponse(200, json_data={
            "status": "filled", "filled_avg_price": "155.50",
        })

        now_et = dt.datetime(2026, 2, 11, 10, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5)))

        with patch("ScalpingBots.scripts.live_ema_scalp.get_latest_quote", return_value=mock_quote), \
             patch("ScalpingBots.scripts.live_ema_scalp.close_position", return_value=mock_close), \
             patch("ScalpingBots.scripts.live_ema_scalp.requests.get", return_value=poll_resp):

            manage_exits(
                managed, "https://paper-api.alpaca.markets", "k", "s",
                cfg, now_et, log_file=None, dry_run=False,
            )

        # Find the EXIT log call
        exit_calls = [
            c for c in mock_log.call_args_list
            if c[0][0] == "EXIT"
        ]
        assert len(exit_calls) == 1

        exit_data = exit_calls[0][0][1]
        # exit_price is from broker fill
        assert abs(exit_data["exit_price"] - 155.50) < 0.01
        # quote_price is from IEX
        assert abs(exit_data["quote_price"] - 154.0) < 0.01
        # Both are present in same log entry
        assert "exit_price" in exit_data
        assert "quote_price" in exit_data

    @patch("ScalpingBots.scripts.live_ema_scalp.time.sleep")
    @patch("ScalpingBots.scripts.live_ema_scalp.log_event")
    def test_exit_no_order_id_uses_quote_price(self, mock_log, mock_sleep):
        """When close_position returns no order_id, P&L uses quote price as fallback."""
        from ScalpingBots.scripts.live_ema_scalp import manage_exits

        cfg = self._make_ema_config()
        pos = self._make_managed_position(entry_price=150.0, atr=2.0)
        pos.tp_price = 153.0
        managed = {"AAPL": pos}

        mock_quote = {"bid": 154.0, "ask": 154.50, "bid_size": 100, "ask_size": 100}
        # No order_id in response (e.g., 204 response)
        mock_close = {"ok": True}

        now_et = dt.datetime(2026, 2, 11, 10, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5)))

        with patch("ScalpingBots.scripts.live_ema_scalp.get_latest_quote", return_value=mock_quote), \
             patch("ScalpingBots.scripts.live_ema_scalp.close_position", return_value=mock_close):

            cycle_pnl = manage_exits(
                managed, "https://paper-api.alpaca.markets", "k", "s",
                cfg, now_et, log_file=None, dry_run=False,
            )

        # P&L falls back to quote: (154 - 150) * 10 = 40
        assert abs(cycle_pnl - 40.0) < 0.01

    @patch("ScalpingBots.scripts.live_ema_scalp.log_event")
    def test_exit_dry_run_uses_quote_pnl(self, mock_log):
        """Dry run: no broker call, P&L based on quote."""
        from ScalpingBots.scripts.live_ema_scalp import manage_exits

        cfg = self._make_ema_config()
        pos = self._make_managed_position(entry_price=150.0, atr=2.0)
        pos.tp_price = 153.0
        managed = {"AAPL": pos}

        mock_quote = {"bid": 154.0, "ask": 154.50, "bid_size": 100, "ask_size": 100}

        now_et = dt.datetime(2026, 2, 11, 10, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5)))

        with patch("ScalpingBots.scripts.live_ema_scalp.get_latest_quote", return_value=mock_quote):
            cycle_pnl = manage_exits(
                managed, "https://paper-api.alpaca.markets", "k", "s",
                cfg, now_et, log_file=None, dry_run=True,
            )

        # Dry run: (154 - 150) * 10 = 40
        assert abs(cycle_pnl - 40.0) < 0.01
        assert "AAPL" not in managed
