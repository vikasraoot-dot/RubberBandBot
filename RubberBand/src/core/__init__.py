"""
Core infrastructure modules for RubberBand trading bot.

This package contains the foundational abstractions:
- credentials: Unified Alpaca credential handling
- http_client: HTTP abstraction with retry logic and connection pooling
- broker: (Phase 2) BrokerClient facade for Alpaca API

These modules are designed to be backward compatible - existing code
continues to work unchanged while new code can use the improved abstractions.
"""

from RubberBand.src.alpaca_creds import (
    resolve_credentials,
    get_headers,
    get_base_url,
    get_headers_from_env,
)

__all__ = [
    "resolve_credentials",
    "get_headers",
    "get_base_url",
    "get_headers_from_env",
]
