import pytest
import sys
import os
from unittest.mock import MagicMock

# Create a mock for alpaca_trade_api before importing scripts that use it
# This prevents the need for real API credentials during testing
sys.modules["alpaca_trade_api"] = MagicMock()
sys.modules["alpaca_trade_api.rest"] = MagicMock()

def test_import_live_spreads_loop():
    """Verify live_spreads_loop.py imports without error"""
    try:
        from RubberBand.scripts import live_spreads_loop
        assert hasattr(live_spreads_loop, "try_spread_entry")
        assert hasattr(live_spreads_loop, "check_slope_filter") # Should be imported
    except ImportError as e:
        pytest.fail(f"Failed to import live_spreads_loop: {e}")

def test_import_live_paper_loop():
    """Verify live_paper_loop.py imports without error"""
    try:
        from RubberBand.scripts import live_paper_loop
        # live_paper_loop is a script, might not expose functions easily if not defined well,
        # but we know it has some functions if we looked at it.
        # It's mostly a script.
    except ImportError as e:
        pytest.fail(f"Failed to import live_paper_loop: {e}")
