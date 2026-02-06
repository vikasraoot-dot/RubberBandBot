"""
HTTP Client abstraction for Alpaca API.

Provides unified HTTP handling with:
- Connection pooling via requests.Session
- Retry logic with exponential backoff
- Rate limit handling (429 responses)
- Consistent timeout management
- Structured error responses

This module is designed as a foundation for the BrokerClient facade.
It does NOT break any existing code - existing modules continue to
work unchanged.

Usage:
    from RubberBand.src.core.http_client import AlpacaHttpClient
    from RubberBand.src.alpaca_creds import resolve_credentials

    base_url, key, secret = resolve_credentials()
    client = AlpacaHttpClient(base_url, key, secret)

    # Make requests
    account = client.get("/v2/account")
    positions = client.get("/v2/positions")

    # Submit order
    order = client.post("/v2/orders", json={"symbol": "AAPL", ...})
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure module logger
logger = logging.getLogger(__name__)

# HTTP status codes that should trigger retry
RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Default configuration
DEFAULT_TIMEOUT = 15  # seconds
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_FACTOR = 0.5  # seconds


class AlpacaHttpError(Exception):
    """Exception raised for Alpaca API errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body or {}

    def __str__(self) -> str:
        if self.status_code:
            return f"[HTTP {self.status_code}] {super().__str__()}"
        return super().__str__()


class AlpacaHttpClient:
    """
    HTTP client for Alpaca API with retry logic and connection pooling.

    This class provides a unified interface for making HTTP requests to
    the Alpaca API with:
    - Automatic retry on transient failures (429, 5xx)
    - Exponential backoff between retries
    - Connection pooling for performance
    - Consistent header and timeout management

    Args:
        base_url: Alpaca API base URL (e.g., "https://paper-api.alpaca.markets")
        api_key: Alpaca API key
        api_secret: Alpaca API secret
        timeout: Default timeout for requests in seconds (default: 15)
        max_retries: Maximum number of retry attempts (default: 5)
        backoff_factor: Base backoff time in seconds (default: 0.5)

    Example:
        client = AlpacaHttpClient(
            "https://paper-api.alpaca.markets",
            "your-api-key",
            "your-api-secret"
        )

        # GET request
        account = client.get("/v2/account")

        # POST request
        order = client.post("/v2/orders", json={
            "symbol": "AAPL",
            "qty": "10",
            "side": "buy",
            "type": "market",
            "time_in_force": "day"
        })

        # Clean up when done
        client.close()
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        api_secret: str,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    ):
        """Initialize the HTTP client with credentials and configuration."""
        if not base_url:
            raise ValueError("base_url is required")
        if not api_key:
            raise ValueError("api_key is required")
        if not api_secret:
            raise ValueError("api_secret is required")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        # Create session with connection pooling
        self._session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry configuration."""
        session = requests.Session()

        # Configure retry strategy for urllib3 (handles connection-level retries)
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=list(RETRY_STATUS_CODES),
            allowed_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            raise_on_status=False,  # We handle status ourselves
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10,
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _get_headers(self) -> Dict[str, str]:
        """Build request headers with authentication."""
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            path: API path (e.g., "/v2/account")
            params: Query parameters
            json: JSON body for POST/PUT requests
            timeout: Request timeout in seconds (overrides default)

        Returns:
            Dict containing the JSON response

        Raises:
            AlpacaHttpError: If request fails after all retries
        """
        url = f"{self.base_url}{path}"
        request_timeout = timeout or self.timeout

        attempt = 0
        last_error: Optional[Exception] = None

        while attempt <= self.max_retries:
            try:
                response = self._session.request(
                    method=method,
                    url=url,
                    headers=self._get_headers(),
                    params=params,
                    json=json,
                    timeout=request_timeout,
                )

                # Check for rate limiting or server errors
                if response.status_code in RETRY_STATUS_CODES:
                    if attempt < self.max_retries:
                        wait_time = self.backoff_factor * (2 ** attempt)
                        logger.warning(
                            f"Retryable status {response.status_code} on {method} {path}, "
                            f"waiting {wait_time:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                        )
                        time.sleep(wait_time)
                        attempt += 1
                        continue
                    else:
                        # Out of retries
                        raise AlpacaHttpError(
                            f"Max retries exceeded for {method} {path}",
                            status_code=response.status_code,
                            response_body=self._safe_json(response),
                        )

                # Check for other errors
                if response.status_code >= 400:
                    error_body = self._safe_json(response)
                    error_message = error_body.get("message", response.text[:200])
                    raise AlpacaHttpError(
                        f"{method} {path} failed: {error_message}",
                        status_code=response.status_code,
                        response_body=error_body,
                    )

                # Success
                return self._safe_json(response)

            except requests.Timeout as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    logger.warning(
                        f"Timeout on {method} {path}, waiting {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                    attempt += 1
                else:
                    raise AlpacaHttpError(
                        f"Request timeout after {self.max_retries} retries: {method} {path}",
                    ) from e

            except requests.ConnectionError as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    logger.warning(
                        f"Connection error on {method} {path}, waiting {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                    attempt += 1
                else:
                    raise AlpacaHttpError(
                        f"Connection failed after {self.max_retries} retries: {method} {path}",
                    ) from e

            except AlpacaHttpError:
                raise

            except Exception as e:
                logger.error(f"Unexpected error on {method} {path}: {e}")
                raise AlpacaHttpError(
                    f"Unexpected error: {method} {path}: {e}",
                ) from e

        # Should not reach here, but just in case
        raise AlpacaHttpError(
            f"Request failed after {self.max_retries} retries: {method} {path}",
        ) from last_error

    @staticmethod
    def _safe_json(response: requests.Response) -> Dict[str, Any]:
        """Safely parse JSON response, returning empty dict on failure."""
        try:
            if response.text:
                return response.json()
        except (ValueError, TypeError):
            pass
        return {}

    def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Make a GET request.

        Args:
            path: API path (e.g., "/v2/account")
            params: Query parameters
            timeout: Request timeout in seconds

        Returns:
            Dict containing the JSON response
        """
        return self._request("GET", path, params=params, timeout=timeout)

    def post(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Make a POST request.

        Args:
            path: API path (e.g., "/v2/orders")
            json: JSON body
            params: Query parameters
            timeout: Request timeout in seconds

        Returns:
            Dict containing the JSON response
        """
        return self._request("POST", path, params=params, json=json, timeout=timeout)

    def put(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Make a PUT request.

        Args:
            path: API path
            json: JSON body
            params: Query parameters
            timeout: Request timeout in seconds

        Returns:
            Dict containing the JSON response
        """
        return self._request("PUT", path, params=params, json=json, timeout=timeout)

    def delete(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Make a DELETE request.

        Args:
            path: API path (e.g., "/v2/orders/{order_id}")
            params: Query parameters
            timeout: Request timeout in seconds

        Returns:
            Dict containing the JSON response
        """
        return self._request("DELETE", path, params=params, timeout=timeout)

    def patch(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Make a PATCH request.

        Args:
            path: API path
            json: JSON body
            params: Query parameters
            timeout: Request timeout in seconds

        Returns:
            Dict containing the JSON response
        """
        return self._request("PATCH", path, params=params, json=json, timeout=timeout)

    def close(self) -> None:
        """Close the underlying session and release resources."""
        if self._session:
            self._session.close()

    def __enter__(self) -> "AlpacaHttpClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


def create_client_from_env(
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
) -> AlpacaHttpClient:
    """
    Create an AlpacaHttpClient using credentials from environment variables.

    This is a convenience function that combines credential resolution
    with client creation.

    Args:
        timeout: Default timeout for requests in seconds
        max_retries: Maximum number of retry attempts
        backoff_factor: Base backoff time in seconds

    Returns:
        Configured AlpacaHttpClient instance

    Example:
        client = create_client_from_env()
        account = client.get("/v2/account")
        client.close()
    """
    # Import here to avoid circular imports
    from RubberBand.src.alpaca_creds import resolve_credentials

    base_url, api_key, api_secret = resolve_credentials()

    return AlpacaHttpClient(
        base_url=base_url,
        api_key=api_key,
        api_secret=api_secret,
        timeout=timeout,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
    )
