"""
Core infrastructure modules for RubberBand trading bot.

This package contains the foundational abstractions:
- credentials: Unified Alpaca credential handling
- http_client: HTTP abstraction with retry logic and connection pooling
- broker: BrokerClient facade for Alpaca API

These modules are designed to be backward compatible - existing code
continues to work unchanged while new code can use the improved abstractions.
"""

from RubberBand.src.alpaca_creds import (
    resolve_credentials,
    get_headers,
    get_base_url,
    get_headers_from_env,
)

from RubberBand.src.core.http_client import (
    AlpacaHttpClient,
    AlpacaHttpError,
    create_client_from_env,
)

from RubberBand.src.core.broker import (
    BrokerClient,
    BrokerError,
    create_broker_from_env,
)

__all__ = [
    # Credentials
    "resolve_credentials",
    "get_headers",
    "get_base_url",
    "get_headers_from_env",
    # HTTP Client
    "AlpacaHttpClient",
    "AlpacaHttpError",
    "create_client_from_env",
    # Broker Client
    "BrokerClient",
    "BrokerError",
    "create_broker_from_env",
]
