"""
BrokerClient - Unified Alpaca API Gateway.

This module provides a single entry point for ALL Alpaca API operations,
replacing scattered API calls across data.py, options_data.py, and
options_execution.py.

Design Principles:
- Single source of truth for broker communication
- Consistent error handling across all operations
- Easy to mock for testing
- Future-proof for additional broker support

Usage:
    from RubberBand.src.core.broker import BrokerClient
    from RubberBand.src.alpaca_creds import resolve_credentials

    base_url, key, secret = resolve_credentials()
    broker = BrokerClient(base_url, key, secret)

    # Market data
    bars = broker.get_bars(["AAPL", "MSFT"], "15Min")
    quote = broker.get_quote("AAPL")

    # Account operations
    account = broker.get_account()
    positions = broker.get_positions()

    # Order operations
    order = broker.submit_order("AAPL", 10, "buy", "market")
    broker.cancel_order(order["id"])

    # Options operations
    contracts = broker.get_option_contracts("AAPL", "2026-02-21", "call")
    spread_order = broker.submit_spread_order(long_symbol, short_symbol, 1)

    # Cleanup
    broker.close()
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import pandas as pd

from RubberBand.src.core.http_client import AlpacaHttpClient, AlpacaHttpError

# Configure module logger
logger = logging.getLogger(__name__)

# Timezone for market hours
ET = ZoneInfo("America/New_York")


class BrokerError(Exception):
    """Exception raised for broker operation errors."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.operation = operation
        self.details = details or {}


class BrokerClient:
    """
    Unified client for all Alpaca API operations.

    This class consolidates all broker communication into a single interface,
    providing consistent error handling, logging, and retry behavior.

    Args:
        base_url: Alpaca API base URL
        api_key: Alpaca API key
        api_secret: Alpaca API secret
        timeout: Default request timeout in seconds
        max_retries: Maximum retry attempts for failed requests

    Example:
        broker = BrokerClient(
            "https://paper-api.alpaca.markets",
            "your-key",
            "your-secret"
        )

        # Check if market is open
        if broker.is_market_open():
            positions = broker.get_positions()
            for pos in positions:
                print(f"{pos['symbol']}: {pos['qty']} shares")

        broker.close()
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        api_secret: str,
        timeout: int = 15,
        max_retries: int = 5,
    ):
        """Initialize the broker client."""
        self.base_url = base_url.rstrip("/")
        self._http = AlpacaHttpClient(
            base_url=base_url,
            api_key=api_key,
            api_secret=api_secret,
            timeout=timeout,
            max_retries=max_retries,
        )

        # Data API base URL (different from trading API)
        self._data_base_url = self._get_data_base_url(base_url)

    def _get_data_base_url(self, trading_url: str) -> str:
        """Derive data API URL from trading URL."""
        if "paper-api" in trading_url:
            return "https://data.alpaca.markets"
        elif "api.alpaca.markets" in trading_url:
            return "https://data.alpaca.markets"
        return "https://data.alpaca.markets"

    # =========================================================================
    # MARKET STATUS
    # =========================================================================

    def is_market_open(self) -> bool:
        """
        Check if the market is currently open.

        Returns:
            True if market is open, False otherwise
        """
        try:
            clock = self._http.get("/v2/clock")
            return clock.get("is_open", False)
        except AlpacaHttpError as e:
            logger.error(f"Failed to check market status: {e}")
            return False

    def get_clock(self) -> Dict[str, Any]:
        """
        Get market clock information.

        Returns:
            Dict with is_open, next_open, next_close timestamps
        """
        return self._http.get("/v2/clock")

    # =========================================================================
    # ACCOUNT OPERATIONS
    # =========================================================================

    def get_account(self) -> Dict[str, Any]:
        """
        Get account information.

        Returns:
            Dict with account details (equity, buying_power, etc.)
        """
        return self._http.get("/v2/account")

    def get_buying_power(self) -> float:
        """
        Get available buying power.

        Returns:
            Available buying power as float
        """
        account = self.get_account()
        return float(account.get("buying_power", 0))

    def get_equity(self) -> float:
        """
        Get current account equity.

        Returns:
            Account equity as float
        """
        account = self.get_account()
        return float(account.get("equity", 0))

    # =========================================================================
    # POSITION OPERATIONS
    # =========================================================================

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open positions.

        Returns:
            List of position dicts with symbol, qty, avg_entry_price, etc.
        """
        result = self._http.get("/v2/positions")
        return result if isinstance(result, list) else []

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific position by symbol.

        Args:
            symbol: Stock or option symbol

        Returns:
            Position dict or None if not found
        """
        try:
            return self._http.get(f"/v2/positions/{symbol}")
        except AlpacaHttpError as e:
            if e.status_code == 404:
                return None
            raise

    def get_stock_positions(self) -> List[Dict[str, Any]]:
        """
        Get only stock positions (not options).

        Returns:
            List of stock position dicts
        """
        positions = self.get_positions()
        # Options symbols are longer (OCC format: 21+ chars)
        return [p for p in positions if len(p.get("symbol", "")) <= 10]

    def get_option_positions(self) -> List[Dict[str, Any]]:
        """
        Get only option positions.

        Returns:
            List of option position dicts
        """
        positions = self.get_positions()
        # Options symbols are in OCC format (21+ chars)
        return [p for p in positions if len(p.get("symbol", "")) > 10]

    def close_position(self, symbol: str) -> Dict[str, Any]:
        """
        Close a position by symbol.

        Args:
            symbol: Stock or option symbol to close

        Returns:
            Order dict for the closing order
        """
        return self._http.delete(f"/v2/positions/{symbol}")

    def close_all_positions(self) -> List[Dict[str, Any]]:
        """
        Close all open positions.

        Returns:
            List of closing order dicts
        """
        result = self._http.delete("/v2/positions")
        return result if isinstance(result, list) else []

    # =========================================================================
    # ORDER OPERATIONS
    # =========================================================================

    def get_orders(
        self,
        status: str = "open",
        limit: int = 100,
        symbols: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get orders with optional filtering.

        Args:
            status: Order status filter (open, closed, all)
            limit: Maximum number of orders to return
            symbols: Filter by symbols (optional)

        Returns:
            List of order dicts
        """
        params = {"status": status, "limit": limit}
        if symbols:
            params["symbols"] = ",".join(symbols)

        result = self._http.get("/v2/orders", params=params)
        return result if isinstance(result, list) else []

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get a specific order by ID.

        Args:
            order_id: Alpaca order ID

        Returns:
            Order dict
        """
        return self._http.get(f"/v2/orders/{order_id}")

    def get_order_by_client_id(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an order by client order ID.

        Args:
            client_order_id: Client-specified order ID

        Returns:
            Order dict or None if not found
        """
        try:
            return self._http.get(
                "/v2/orders:by_client_order_id",
                params={"client_order_id": client_order_id},
            )
        except AlpacaHttpError as e:
            if e.status_code == 404:
                return None
            raise

    def submit_order(
        self,
        symbol: str,
        qty: Union[int, float],
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        extended_hours: bool = False,
    ) -> Dict[str, Any]:
        """
        Submit a simple order.

        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: "buy" or "sell"
            order_type: "market", "limit", "stop", "stop_limit"
            time_in_force: "day", "gtc", "ioc", "fok"
            limit_price: Limit price (required for limit orders)
            stop_price: Stop price (required for stop orders)
            client_order_id: Client-specified order ID
            extended_hours: Allow extended hours trading

        Returns:
            Order dict with order details
        """
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }

        if limit_price is not None:
            payload["limit_price"] = str(limit_price)
        if stop_price is not None:
            payload["stop_price"] = str(stop_price)
        if client_order_id:
            payload["client_order_id"] = client_order_id
        if extended_hours:
            payload["extended_hours"] = True

        return self._http.post("/v2/orders", json=payload)

    def submit_bracket_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        limit_price: float,
        take_profit_price: float,
        stop_loss_price: float,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a bracket order (entry + take profit + stop loss).

        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: "buy" or "sell"
            limit_price: Entry limit price
            take_profit_price: Take profit limit price
            stop_loss_price: Stop loss price
            time_in_force: Order duration
            client_order_id: Client-specified order ID

        Returns:
            Order dict with order details
        """
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": "limit",
            "time_in_force": time_in_force,
            "limit_price": str(limit_price),
            "order_class": "bracket",
            "take_profit": {"limit_price": str(take_profit_price)},
            "stop_loss": {"stop_price": str(stop_loss_price)},
        }

        if client_order_id:
            payload["client_order_id"] = client_order_id

        return self._http.post("/v2/orders", json=payload)

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order by ID.

        Args:
            order_id: Alpaca order ID

        Returns:
            True if cancelled successfully
        """
        try:
            self._http.delete(f"/v2/orders/{order_id}")
            return True
        except AlpacaHttpError as e:
            if e.status_code == 404:
                return False  # Already cancelled or filled
            raise

    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.

        Returns:
            Number of orders cancelled
        """
        result = self._http.delete("/v2/orders")
        if isinstance(result, list):
            return len(result)
        return 0

    # =========================================================================
    # MARKET DATA - STOCKS
    # =========================================================================

    def get_bars(
        self,
        symbols: List[str],
        timeframe: str = "15Min",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 1000,
        feed: str = "iex",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical bars for multiple symbols.

        Args:
            symbols: List of stock symbols
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)
            start: Start time ISO8601 (default: 10 days ago)
            end: End time ISO8601 (default: now)
            limit: Max bars per symbol
            feed: Data feed (iex, sip)

        Returns:
            Dict mapping symbol to DataFrame with OHLCV data
        """
        if not symbols:
            return {}

        # Default time range
        now = datetime.now(ET)
        if not end:
            end = now.isoformat()
        if not start:
            start = (now - timedelta(days=10)).isoformat()

        # Use data API for bars
        data_http = AlpacaHttpClient(
            base_url=self._data_base_url,
            api_key=self._http.api_key,
            api_secret=self._http.api_secret,
            timeout=30,
        )

        try:
            result: Dict[str, pd.DataFrame] = {}
            page_token = None

            while True:
                params: Dict[str, Any] = {
                    "symbols": ",".join(symbols),
                    "timeframe": timeframe,
                    "start": start,
                    "end": end,
                    "limit": min(limit, 10000),
                    "feed": feed,
                    "adjustment": "split",
                }
                if page_token:
                    params["page_token"] = page_token

                response = data_http.get("/v2/stocks/bars", params=params)
                bars_data = response.get("bars", {})

                # Process bars for each symbol
                for symbol, bars in bars_data.items():
                    if not bars:
                        continue

                    df = pd.DataFrame(bars)
                    if not df.empty:
                        df["t"] = pd.to_datetime(df["t"])
                        df = df.rename(
                            columns={
                                "t": "timestamp",
                                "o": "open",
                                "h": "high",
                                "l": "low",
                                "c": "close",
                                "v": "volume",
                                "n": "trade_count",
                                "vw": "vwap",
                            }
                        )
                        df = df.set_index("timestamp")

                        if symbol in result:
                            result[symbol] = pd.concat([result[symbol], df])
                        else:
                            result[symbol] = df

                # Check for more pages
                page_token = response.get("next_page_token")
                if not page_token:
                    break

            return result

        finally:
            data_http.close()

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest quote for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Quote dict with bid, ask, etc. or None
        """
        data_http = AlpacaHttpClient(
            base_url=self._data_base_url,
            api_key=self._http.api_key,
            api_secret=self._http.api_secret,
            timeout=10,
        )

        try:
            response = data_http.get(
                f"/v2/stocks/{symbol}/quotes/latest",
                params={"feed": "iex"},
            )
            return response.get("quote")
        except AlpacaHttpError:
            return None
        finally:
            data_http.close()

    # =========================================================================
    # OPTIONS DATA
    # =========================================================================

    def get_option_contracts(
        self,
        underlying: str,
        expiration_date: Optional[str] = None,
        option_type: Optional[str] = None,
        strike_price_gte: Optional[float] = None,
        strike_price_lte: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get option contracts for an underlying.

        Args:
            underlying: Underlying stock symbol
            expiration_date: Filter by expiration (YYYY-MM-DD)
            option_type: "call" or "put"
            strike_price_gte: Minimum strike price
            strike_price_lte: Maximum strike price
            limit: Max contracts to return

        Returns:
            List of option contract dicts
        """
        params: Dict[str, Any] = {
            "underlying_symbols": underlying,
            "limit": limit,
            "status": "active",
        }

        if expiration_date:
            params["expiration_date"] = expiration_date
        if option_type:
            params["type"] = option_type
        if strike_price_gte is not None:
            params["strike_price_gte"] = str(strike_price_gte)
        if strike_price_lte is not None:
            params["strike_price_lte"] = str(strike_price_lte)

        result = self._http.get("/v2/options/contracts", params=params)
        return result.get("option_contracts", []) if isinstance(result, dict) else []

    def get_option_quote(self, option_symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest quote for an option.

        Args:
            option_symbol: OCC option symbol

        Returns:
            Quote dict with bid, ask, etc. or None
        """
        data_http = AlpacaHttpClient(
            base_url=self._data_base_url,
            api_key=self._http.api_key,
            api_secret=self._http.api_secret,
            timeout=10,
        )

        try:
            response = data_http.get(
                "/v1beta1/options/quotes/latest",
                params={"symbols": option_symbol, "feed": "indicative"},
            )
            quotes = response.get("quotes", {})
            return quotes.get(option_symbol)
        except AlpacaHttpError:
            return None
        finally:
            data_http.close()

    def get_option_snapshots(
        self, option_symbols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get snapshots for multiple options.

        Args:
            option_symbols: List of OCC option symbols

        Returns:
            Dict mapping symbol to snapshot data
        """
        if not option_symbols:
            return {}

        data_http = AlpacaHttpClient(
            base_url=self._data_base_url,
            api_key=self._http.api_key,
            api_secret=self._http.api_secret,
            timeout=15,
        )

        try:
            response = data_http.get(
                "/v1beta1/options/snapshots",
                params={"symbols": ",".join(option_symbols), "feed": "indicative"},
            )
            return response.get("snapshots", {})
        except AlpacaHttpError:
            return {}
        finally:
            data_http.close()

    # =========================================================================
    # OPTIONS ORDERS
    # =========================================================================

    def submit_option_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = "limit",
        limit_price: Optional[float] = None,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a single-leg option order.

        Args:
            symbol: OCC option symbol
            qty: Number of contracts
            side: "buy" or "sell"
            order_type: "market" or "limit"
            limit_price: Limit price (required for limit orders)
            time_in_force: Order duration
            client_order_id: Client-specified order ID

        Returns:
            Order dict
        """
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }

        if limit_price is not None:
            payload["limit_price"] = str(limit_price)
        if client_order_id:
            payload["client_order_id"] = client_order_id

        return self._http.post("/v2/orders", json=payload)

    def submit_spread_order(
        self,
        long_symbol: str,
        short_symbol: str,
        qty: int,
        limit_price: float,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a debit spread order (buy long, sell short).

        Args:
            long_symbol: OCC symbol for long leg (buy)
            short_symbol: OCC symbol for short leg (sell)
            qty: Number of spreads
            limit_price: Net debit limit price
            time_in_force: Order duration
            client_order_id: Client-specified order ID

        Returns:
            Order dict
        """
        payload: Dict[str, Any] = {
            "order_class": "mleg",
            "time_in_force": time_in_force,
            "type": "limit",
            "limit_price": str(limit_price),
            "legs": [
                {"symbol": long_symbol, "qty": str(qty), "side": "buy"},
                {"symbol": short_symbol, "qty": str(qty), "side": "sell"},
            ],
        }

        if client_order_id:
            payload["client_order_id"] = client_order_id

        return self._http.post("/v2/orders", json=payload)

    def close_spread(
        self,
        long_symbol: str,
        short_symbol: str,
        qty: int,
        limit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Close a spread position (sell long, buy short).

        Args:
            long_symbol: OCC symbol for long leg (sell to close)
            short_symbol: OCC symbol for short leg (buy to close)
            qty: Number of spreads to close
            limit_price: Net credit limit price (optional, market if None)

        Returns:
            Order dict
        """
        payload: Dict[str, Any] = {
            "order_class": "mleg",
            "time_in_force": "day",
            "type": "limit" if limit_price else "market",
            "legs": [
                {"symbol": long_symbol, "qty": str(qty), "side": "sell"},
                {"symbol": short_symbol, "qty": str(qty), "side": "buy"},
            ],
        }

        if limit_price is not None:
            payload["limit_price"] = str(limit_price)

        return self._http.post("/v2/orders", json=payload)

    # =========================================================================
    # ACTIVITY / FILLS
    # =========================================================================

    def get_activities(
        self,
        activity_types: Optional[str] = None,
        date: Optional[str] = None,
        after: Optional[str] = None,
        until: Optional[str] = None,
        direction: str = "desc",
        page_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get account activities (fills, dividends, etc.).

        Args:
            activity_types: Filter by type (e.g., "FILL")
            date: Filter by date (YYYY-MM-DD)
            after: After this timestamp
            until: Until this timestamp
            direction: Sort direction ("asc" or "desc")
            page_size: Results per page

        Returns:
            List of activity dicts
        """
        params: Dict[str, Any] = {
            "direction": direction,
            "page_size": page_size,
        }

        if activity_types:
            params["activity_types"] = activity_types
        if date:
            params["date"] = date
        if after:
            params["after"] = after
        if until:
            params["until"] = until

        result = self._http.get("/v2/account/activities", params=params)
        return result if isinstance(result, list) else []

    def get_fills_today(self) -> List[Dict[str, Any]]:
        """
        Get all fills from today.

        Returns:
            List of fill activity dicts
        """
        today = datetime.now(ET).strftime("%Y-%m-%d")
        return self.get_activities(activity_types="FILL", date=today)

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def close(self) -> None:
        """Close the underlying HTTP client and release resources."""
        if self._http:
            self._http.close()

    def __enter__(self) -> "BrokerClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


def create_broker_from_env(
    timeout: int = 15,
    max_retries: int = 5,
) -> BrokerClient:
    """
    Create a BrokerClient using credentials from environment variables.

    Args:
        timeout: Default request timeout in seconds
        max_retries: Maximum retry attempts

    Returns:
        Configured BrokerClient instance

    Example:
        broker = create_broker_from_env()
        positions = broker.get_positions()
        broker.close()
    """
    from RubberBand.src.alpaca_creds import resolve_credentials

    base_url, api_key, api_secret = resolve_credentials()

    return BrokerClient(
        base_url=base_url,
        api_key=api_key,
        api_secret=api_secret,
        timeout=timeout,
        max_retries=max_retries,
    )
