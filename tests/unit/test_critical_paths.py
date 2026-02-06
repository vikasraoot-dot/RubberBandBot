"""
Critical path tests for production trading bot.

These tests verify the most important code paths that handle real money:
- Order submission validation
- Spread order payload structure
- Position registry operations
- Error handling in API calls
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal


class TestSpreadOrderPayload:
    """Test that spread order payloads match production-tested format."""

    def test_submit_spread_payload_has_required_fields(self):
        """Spread order payload must have all fields Alpaca requires."""
        from RubberBand.src.core.broker import BrokerClient

        # Create mock HTTP client
        mock_http = Mock()
        mock_http.post.return_value = {"id": "test-order-id", "status": "new"}
        mock_http.api_key = "test-key"
        mock_http.api_secret = "test-secret"

        # Create broker with mocked HTTP
        broker = BrokerClient.__new__(BrokerClient)
        broker._http = mock_http
        broker._data_base_url = "https://data.alpaca.markets"
        broker.base_url = "https://paper-api.alpaca.markets"

        # Submit spread order
        broker.submit_spread_order(
            long_symbol="AAPL260213C00200000",
            short_symbol="AAPL260213C00205000",
            qty=1,
            limit_price=2.50,
        )

        # Verify payload structure
        call_args = mock_http.post.call_args
        payload = call_args.kwargs["json"]

        # Check required top-level fields
        assert payload["order_class"] == "mleg"
        assert payload["type"] == "limit"
        assert payload["limit_price"] == "2.5"
        assert payload["qty"] == "1"
        assert payload["side"] == "buy"
        assert payload["time_in_force"] == "day"

        # Check legs structure
        assert len(payload["legs"]) == 2

        # Long leg
        long_leg = payload["legs"][0]
        assert long_leg["symbol"] == "AAPL260213C00200000"
        assert long_leg["side"] == "buy"
        assert long_leg["ratio_qty"] == "1"
        assert long_leg["position_intent"] == "buy_to_open"

        # Short leg
        short_leg = payload["legs"][1]
        assert short_leg["symbol"] == "AAPL260213C00205000"
        assert short_leg["side"] == "sell"
        assert short_leg["ratio_qty"] == "1"
        assert short_leg["position_intent"] == "sell_to_open"

    def test_close_spread_payload_has_position_intent(self):
        """Close spread must have sell_to_close and buy_to_close intents."""
        from RubberBand.src.core.broker import BrokerClient

        mock_http = Mock()
        mock_http.post.return_value = {"id": "test-order-id", "status": "new"}
        mock_http.api_key = "test-key"
        mock_http.api_secret = "test-secret"

        broker = BrokerClient.__new__(BrokerClient)
        broker._http = mock_http
        broker._data_base_url = "https://data.alpaca.markets"
        broker.base_url = "https://paper-api.alpaca.markets"

        # Close spread
        broker.close_spread(
            long_symbol="AAPL260213C00200000",
            short_symbol="AAPL260213C00205000",
            qty=1,
            limit_price=0.50,
        )

        payload = mock_http.post.call_args.kwargs["json"]

        # Check closing intents
        long_leg = payload["legs"][0]
        assert long_leg["side"] == "sell"
        assert long_leg["position_intent"] == "sell_to_close"

        short_leg = payload["legs"][1]
        assert short_leg["side"] == "buy"
        assert short_leg["position_intent"] == "buy_to_close"


class TestOrderValidation:
    """Test input validation on order submission."""

    def test_submit_order_rejects_negative_qty(self):
        """Negative quantity should be rejected."""
        from RubberBand.src.core.broker import BrokerClient, BrokerError

        broker = BrokerClient.__new__(BrokerClient)
        broker._http = Mock()

        with pytest.raises(BrokerError) as exc_info:
            broker.submit_order("AAPL", qty=-10, side="buy")

        assert "positive" in str(exc_info.value).lower()

    def test_submit_order_rejects_zero_qty(self):
        """Zero quantity should be rejected."""
        from RubberBand.src.core.broker import BrokerClient, BrokerError

        broker = BrokerClient.__new__(BrokerClient)
        broker._http = Mock()

        with pytest.raises(BrokerError) as exc_info:
            broker.submit_order("AAPL", qty=0, side="buy")

        assert "positive" in str(exc_info.value).lower()

    def test_submit_order_rejects_invalid_side(self):
        """Invalid side should be rejected."""
        from RubberBand.src.core.broker import BrokerClient, BrokerError

        broker = BrokerClient.__new__(BrokerClient)
        broker._http = Mock()

        with pytest.raises(BrokerError) as exc_info:
            broker.submit_order("AAPL", qty=10, side="invalid")

        assert "side" in str(exc_info.value).lower()

    def test_submit_order_rejects_limit_without_price(self):
        """Limit order without price should be rejected."""
        from RubberBand.src.core.broker import BrokerClient, BrokerError

        broker = BrokerClient.__new__(BrokerClient)
        broker._http = Mock()

        with pytest.raises(BrokerError) as exc_info:
            broker.submit_order("AAPL", qty=10, side="buy", order_type="limit")

        assert "limit_price" in str(exc_info.value).lower()

    def test_submit_spread_rejects_zero_qty(self):
        """Spread with zero quantity should be rejected."""
        from RubberBand.src.core.broker import BrokerClient, BrokerError

        broker = BrokerClient.__new__(BrokerClient)
        broker._http = Mock()

        with pytest.raises(BrokerError) as exc_info:
            broker.submit_spread_order(
                long_symbol="AAPL260213C00200000",
                short_symbol="AAPL260213C00205000",
                qty=0,
                limit_price=2.50,
            )

        assert "positive" in str(exc_info.value).lower()

    def test_submit_spread_rejects_zero_price(self):
        """Spread with zero limit price should be rejected."""
        from RubberBand.src.core.broker import BrokerClient, BrokerError

        broker = BrokerClient.__new__(BrokerClient)
        broker._http = Mock()

        with pytest.raises(BrokerError) as exc_info:
            broker.submit_spread_order(
                long_symbol="AAPL260213C00200000",
                short_symbol="AAPL260213C00205000",
                qty=1,
                limit_price=0,
            )

        assert "positive" in str(exc_info.value).lower()


class TestCredentialResolution:
    """Test credential resolution returns correct order."""

    def test_resolve_credentials_returns_key_secret_url_order(self):
        """Credentials must return (key, secret, url) in that order."""
        from RubberBand.src.alpaca_creds import resolve_credentials

        with patch.dict("os.environ", {
            "APCA_API_KEY_ID": "test-key-123",
            "APCA_API_SECRET_KEY": "test-secret-456",
            "APCA_API_BASE_URL": "https://test.alpaca.markets",
        }):
            key, secret, url = resolve_credentials()

            assert key == "test-key-123"
            assert secret == "test-secret-456"
            assert url == "https://test.alpaca.markets"


class TestPositionRegistry:
    """Test position registry operations."""

    def test_registry_loads_for_valid_bot(self):
        """Valid bot tag should load registry."""
        from RubberBand.src.position_registry import PositionRegistry

        # Use a valid bot tag
        reg = PositionRegistry("15M_OPT")
        # Should not raise, and should have required attributes
        assert hasattr(reg, "positions")
        assert hasattr(reg, "bot_tag")
        assert reg.bot_tag == "15M_OPT"
        assert isinstance(reg.positions, dict)

    def test_registry_rejects_invalid_tag(self):
        """Invalid bot tag should be rejected."""
        from RubberBand.src.position_registry import PositionRegistry

        with pytest.raises(ValueError) as exc_info:
            PositionRegistry("INVALID_TAG")

        assert "Invalid bot_tag" in str(exc_info.value)


class TestMarketStatusErrorHandling:
    """Test error handling for market status checks."""

    def test_is_market_open_returns_none_on_error(self):
        """API error should return None, not False."""
        from RubberBand.src.core.broker import BrokerClient
        from RubberBand.src.core.http_client import AlpacaHttpError

        mock_http = Mock()
        mock_http.get.side_effect = AlpacaHttpError("Connection failed")

        broker = BrokerClient.__new__(BrokerClient)
        broker._http = mock_http

        result = broker.is_market_open()

        # Should be None (unknown), not False
        assert result is None


class TestProductionOptionsExecution:
    """Test production options_execution.py code has required fields."""

    def test_spread_order_code_has_position_intent(self):
        """Verify production code includes position_intent in spread orders."""
        # Instead of mocking complex flow, verify the source code directly
        import inspect
        from RubberBand.src import options_execution

        source = inspect.getsource(options_execution.submit_spread_order)

        # Must have position_intent for proper Alpaca spread handling
        assert "position_intent" in source, "submit_spread_order missing position_intent"
        assert "buy_to_open" in source, "submit_spread_order missing buy_to_open"
        assert "sell_to_open" in source, "submit_spread_order missing sell_to_open"
        assert "ratio_qty" in source, "submit_spread_order missing ratio_qty"

    def test_close_spread_code_has_position_intent(self):
        """Verify production close_spread includes position_intent."""
        import inspect
        from RubberBand.src import options_execution

        source = inspect.getsource(options_execution.close_spread)

        # Must have closing intents
        assert "position_intent" in source, "close_spread missing position_intent"
        assert "sell_to_close" in source, "close_spread missing sell_to_close"
        assert "buy_to_close" in source, "close_spread missing buy_to_close"
