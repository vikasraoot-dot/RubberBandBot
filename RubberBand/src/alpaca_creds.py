"""
Alpaca Credentials Module: Unified credential handling for all Alpaca API interactions.

This module consolidates credential resolution, header building, and base URL handling
that was previously duplicated across data.py, options_data.py, and options_execution.py.

Usage:
    from RubberBand.src.alpaca_creds import resolve_credentials, get_headers, get_base_url

    # Get all credentials at once
    key, secret, base_url = resolve_credentials()

    # Build headers for API requests
    headers = get_headers(key, secret)

    # Or resolve individually
    base_url = get_base_url()
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple


# Default base URL for paper trading
DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"


def resolve_credentials(
    key: Optional[str] = None,
    secret: Optional[str] = None,
    base_url: Optional[str] = None
) -> Tuple[str, str, str]:
    """
    Resolve Alpaca API credentials from explicit args or environment variables.

    Credential resolution order (first non-empty wins):
    1. Explicit argument passed to function
    2. APCA_API_KEY_ID / APCA_API_SECRET_KEY (official Alpaca SDK names)
    3. ALPACA_KEY_ID / ALPACA_SECRET_KEY (common alternative names)

    Base URL resolution order:
    1. Explicit base_url argument
    2. APCA_API_BASE_URL environment variable
    3. APCA_BASE_URL environment variable
    4. ALPACA_BASE_URL environment variable
    5. Default: https://paper-api.alpaca.markets

    Args:
        key: Optional API key. If None, resolved from environment.
        secret: Optional API secret. If None, resolved from environment.
        base_url: Optional base URL. If None, resolved from environment.

    Returns:
        Tuple of (api_key, api_secret, base_url) with trailing slashes stripped from URL.

    Example:
        # Using environment variables
        key, secret, base = resolve_credentials()

        # Using explicit credentials
        key, secret, base = resolve_credentials(
            key="PKXXXXXXXX",
            secret="XXXXXXXX",
            base_url="https://api.alpaca.markets"
        )
    """
    resolved_key = _resolve_key(key)
    resolved_secret = _resolve_secret(secret)
    resolved_base_url = get_base_url(base_url)

    return resolved_key, resolved_secret, resolved_base_url


def _resolve_key(key: Optional[str] = None) -> str:
    """
    Resolve API key from explicit arg or environment.

    Args:
        key: Optional explicit API key.

    Returns:
        Resolved API key (may be empty string if not found).
    """
    if key:
        return key.strip()

    # Check environment variables in priority order
    env_key = (
        os.getenv("APCA_API_KEY_ID")
        or os.getenv("ALPACA_KEY_ID")
        or ""
    )
    return env_key.strip()


def _resolve_secret(secret: Optional[str] = None) -> str:
    """
    Resolve API secret from explicit arg or environment.

    Args:
        secret: Optional explicit API secret.

    Returns:
        Resolved API secret (may be empty string if not found).
    """
    if secret:
        return secret.strip()

    # Check environment variables in priority order
    env_secret = (
        os.getenv("APCA_API_SECRET_KEY")
        or os.getenv("ALPACA_SECRET_KEY")
        or ""
    )
    return env_secret.strip()


def get_base_url(base_url: Optional[str] = None) -> str:
    """
    Resolve Alpaca API base URL from explicit arg or environment.

    Resolution order:
    1. Explicit base_url argument
    2. APCA_API_BASE_URL environment variable
    3. APCA_BASE_URL environment variable
    4. ALPACA_BASE_URL environment variable
    5. Default: https://paper-api.alpaca.markets

    Args:
        base_url: Optional explicit base URL.

    Returns:
        Resolved base URL with trailing slashes stripped.

    Example:
        # From environment
        url = get_base_url()

        # Explicit override
        url = get_base_url("https://api.alpaca.markets")
    """
    resolved = (
        base_url
        or os.getenv("APCA_API_BASE_URL")
        or os.getenv("APCA_BASE_URL")
        or os.getenv("ALPACA_BASE_URL")
        or DEFAULT_BASE_URL
    )
    return resolved.rstrip("/")


def get_headers(key: str, secret: str) -> Dict[str, str]:
    """
    Build HTTP headers for Alpaca API requests.

    Args:
        key: Alpaca API key.
        secret: Alpaca API secret.

    Returns:
        Dictionary with required Alpaca authentication headers.

    Example:
        key, secret, _ = resolve_credentials()
        headers = get_headers(key, secret)
        response = requests.get(url, headers=headers)
    """
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Content-Type": "application/json",
    }


def get_headers_from_env() -> Dict[str, str]:
    """
    Convenience function to get headers using environment credentials.

    Returns:
        Dictionary with required Alpaca authentication headers.

    Example:
        headers = get_headers_from_env()
        response = requests.get(url, headers=headers)
    """
    key, secret, _ = resolve_credentials()
    return get_headers(key, secret)
