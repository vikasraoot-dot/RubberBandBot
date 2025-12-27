import pytest
import pandas as pd
from unittest.mock import MagicMock
import sys
import os

# Add repo root to path
_test_dir = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_test_dir, ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from RubberBand.src.regime_manager import RegimeManager

@pytest.fixture
def mock_vixy_calm():
    """Returns a DataFrame simulating VIXY < 35 (CALM)"""
    df = pd.DataFrame({
        "close": [30.0, 32.0, 31.0]
    })
    return df

@pytest.fixture
def mock_vixy_normal():
    """Returns a DataFrame simulating VIXY 35-55 (NORMAL)"""
    df = pd.DataFrame({
        "close": [40.0, 42.0, 45.0]
    })
    return df

@pytest.fixture
def mock_vixy_panic():
    """Returns a DataFrame simulating VIXY > 55 (PANIC)"""
    df = pd.DataFrame({
        "close": [60.0, 58.0, 62.0]
    })
    return df

@pytest.fixture
def regime_manager():
    """Returns a RegimeManager instance with verbose=False"""
    return RegimeManager(verbose=False)
