"""Shared utility helpers for the watchdog subsystem.

Provides Decimal conversion functions used across all watchdog modules.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any


def to_dec(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Number, string, or None to convert.

    Returns:
        Decimal representation, or Decimal("0") on failure.
    """
    if value is None:
        return Decimal("0")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal("0")


def dec_to_float(d: Decimal) -> float:
    """Round a Decimal to 2 places and return as float for JSON serialisation.

    Args:
        d: Decimal value.

    Returns:
        Float rounded to 2 decimal places.
    """
    return float(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
