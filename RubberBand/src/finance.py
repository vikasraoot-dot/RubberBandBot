from decimal import Decimal, ROUND_HALF_UP
from typing import Union, Any

# Standard precision for currency (2 decimal places)
CENTS = Decimal("0.01")

def to_decimal(value: Any) -> Decimal:
    """
    Convert any numeric input to a Decimal quantified to 2 decimal places.
    Handles floats, ints, strings, and existing Decimals.
    """
    if value is None:
        return Decimal("0.00")
        
    if isinstance(value, float):
        # Convert via string to avoid float precision artifacts
        # e.g. float(1.1) -> 1.1000000000000000888
        value = str(value)
        
    try:
        d = Decimal(value)
        return d.quantize(CENTS, rounding=ROUND_HALF_UP)
    except Exception as e:
        # Fallback or re-raise? 
        # For financial safety, we should strictly raise if conversion fails.
        raise ValueError(f"Cannot convert {value} to Decimal: {e}")

def money_add(a: Any, b: Any) -> Decimal:
    return to_decimal(a) + to_decimal(b)

def money_sub(a: Any, b: Any) -> Decimal:
    return to_decimal(a) - to_decimal(b)

def money_mul(a: Any, b: Any) -> Decimal:
    # Adding extra precision for intermediate multiplication might be needed,
    # but final result should be cents.
    res = to_decimal(a) * to_decimal(b)
    return res.quantize(CENTS, rounding=ROUND_HALF_UP)

def money_div(a: Any, b: Any) -> Decimal:
    if to_decimal(b) == 0:
        return Decimal("0.00")
    res = to_decimal(a) / to_decimal(b)
    return res.quantize(CENTS, rounding=ROUND_HALF_UP)

def safe_float(d: Decimal) -> float:
    """Convert back to float for APIs that require it (e.g. Alpaca args)."""
    return float(d)
