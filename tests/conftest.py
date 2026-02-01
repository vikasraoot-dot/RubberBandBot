import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
import sys
import os
from datetime import datetime, timedelta

# Add repo root to path
_test_dir = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_test_dir, ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from RubberBand.src.regime_manager import RegimeManager


def _create_vixy_df(closes: list, volumes: list = None, base_date: datetime = None) -> pd.DataFrame:
    """
    Helper to create a proper VIXY DataFrame for regime manager testing.

    Args:
        closes: List of closing prices (needs 25+ for valid SMA calculation)
        volumes: List of volumes (defaults to 1M per day)
        base_date: Starting date for index (defaults to 30 days ago)

    Returns:
        DataFrame with datetime index, close, and volume columns
    """
    n = len(closes)
    if volumes is None:
        volumes = [1_000_000] * n

    if base_date is None:
        base_date = datetime.now() - timedelta(days=n)

    dates = pd.date_range(start=base_date, periods=n, freq='D')

    df = pd.DataFrame({
        "open": closes,  # Simplified: open = close
        "high": [c * 1.02 for c in closes],
        "low": [c * 0.98 for c in closes],
        "close": closes,
        "volume": volumes,
    }, index=dates)

    return df


@pytest.fixture
def mock_vixy_calm():
    """
    Returns a DataFrame simulating CALM regime.

    CALM requires: Price < SMA_20 for 3 consecutive days.
    We create 25 days of data where the last 5 days are all below the SMA.
    """
    # Create stable prices around 40, then drop to 35 for last 5 days
    # SMA will be ~40, and closes at 35 will be below
    closes = [40.0] * 20 + [35.0, 35.0, 35.0, 35.0, 35.0]
    volumes = [1_000_000] * 25  # Normal volume (no panic)

    return _create_vixy_df(closes, volumes)


@pytest.fixture
def mock_vixy_normal():
    """
    Returns a DataFrame simulating NORMAL regime.

    NORMAL is the default when:
    - Not PANIC (no price spike with volume confirmation)
    - Not CALM (price not below SMA for 3+ days)
    """
    # Prices oscillating around SMA - not consistently below
    closes = [40.0] * 20 + [42.0, 38.0, 41.0, 39.0, 40.0]
    volumes = [1_000_000] * 25  # Normal volume

    return _create_vixy_df(closes, volumes)


@pytest.fixture
def mock_vixy_panic():
    """
    Returns a DataFrame simulating PANIC regime.

    PANIC requires: (Price > Upper Band OR Spike > +8%) AND (Vol > 1.5x Avg)
    We create data where the last day has a spike above upper band with high volume.
    """
    # Stable prices, then big spike on last day
    closes = [40.0] * 24 + [55.0]  # ~37.5% spike on last day
    # High volume on last day (3x average to exceed 1.5x threshold)
    volumes = [1_000_000] * 24 + [3_000_000]

    return _create_vixy_df(closes, volumes)


@pytest.fixture
def regime_manager():
    """Returns a RegimeManager instance with verbose=False"""
    return RegimeManager(verbose=False)
