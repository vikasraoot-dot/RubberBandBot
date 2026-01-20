
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure the src directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../RubberBand')))

from scripts.live_spreads_loop import try_spread_entry

def test_try_spread_entry_cfg_access():
    """
    Regression test for the 'name 'cfg' is not defined' error.
    Ensures try_spread_entry accepts 'cfg' and uses it for capital checks.
    """
    # Mock dependencies
    mock_logger = MagicMock()
    mock_registry = MagicMock()
    mock_registry.was_traded_today.return_value = False
    
    # Mock spread entry to return a valid spread so code proceeds to capital check
    with patch('scripts.live_spreads_loop.select_spread_contracts') as mock_select:
        mock_select.return_value = {
            "long": {"symbol": "AAPL_260120_C150"},
            "short": {"symbol": "AAPL_260120_C160"},
            "expiration": "2026-01-20",
            "atm_strike": 150,
            "otm_strike": 160
        }
        
        # Mock other dependencies that might be called
        with patch('scripts.live_spreads_loop.get_option_quote') as mock_quote:
            mock_quote.return_value = {"ask": 1.5, "bid": 1.0}
            
            with patch('scripts.live_spreads_loop.get_option_snapshot') as mock_snap:
                mock_snap.return_value = {}

                # Setup test data
                signal = {
                    "symbol": "AAPL",
                    "entry_price": 150.0,
                    "rsi": 30.0,
                    "atr": 2.0,
                    "entry_reason": "Test Signal"
                }
                
                spread_cfg = {
                    "dte": 3,
                    "min_dte": 0,
                    "max_debit": 5.0, # High debit to ensure we get to capital check if logic flows
                    "contracts": 1
                }
                
                # The 'cfg' dictionary that was missing
                cfg = {"max_capital": 100000}
                
                # ACT
                # This call should fail if the fix is not applied because try_spread_entry 
                # (in the buggy version) doesn't accept 'cfg' but tries to use it globally
                try:
                    # In the buggy version, this signature will mismatch or internal code will fail
                    # We try to pass cfg. If the function doesn't accept it, Python raises TypeError.
                    # If it accepts but doesn't use it (and uses global 'cfg' which is missing), it raises NameError.
                    
                    # NOTE: We can't easily simulate "missing global" in a unit test because test runners might
                    # inject things, but we can check if the function *signature* accepts it and *uses* it.
                    
                    try_spread_entry(signal, spread_cfg, mock_logger, mock_registry, dry_run=False, cfg=cfg)
                    
                except TypeError as e:
                    pytest.fail(f"Regression: Function signature does not accept 'cfg': {e}")
                except NameError as e:
                     pytest.fail(f"Regression: Function failed to access 'cfg' internally: {e}")
                except Exception as e:
                    # If it fails for other reasons (e.g. valid checks), that's fine for this test 
                    # as long as it's not NameError on 'cfg'.
                    # actually we want to ensure it DOESN'T fail with NameError
                    if "cfg" in str(e) and "not defined" in str(e):
                         pytest.fail(f"Regression: NameError 'cfg' detected: {e}")
                    pass

