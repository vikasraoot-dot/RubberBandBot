import pytest
import pandas as pd
import numpy as np
from RubberBand.scripts import scan_for_bot

# Mock functions if needed, but calculate_indicators is pure logic
from RubberBand.scripts.scan_for_bot import calculate_indicators

# ----------------------------------------------------------------------
# 1. Scanner Logic Tests
# ----------------------------------------------------------------------

@pytest.fixture
def mock_downtrend_df():
    """Stock in a downtrend (Price < SMA)"""
    # Create simple DF: Close declining
    # SMA-20 ~ 100
    prices = [100.0] * 15 + [95.0] * 4 + [90.0] # Last price 90. SMA > 90.
    # SMA20 of [100...95...90] is roughly (1500 + 380 + 90)/20 = 98.5
    # Price 90 < SMA 98.5 -> Downtrend
    
    df = pd.DataFrame({
        "close": prices,
        "high": [p + 1 for p in prices],
        "low": [p - 1 for p in prices],
        "volume": [1_000_000] * 20
    })
    return df

def test_scanner_sma_enabled(mock_downtrend_df):
    """
    Test 15M_OPT Profile behavior (SMA Enabled).
    Expect: in_uptrend = False because Price < SMA.
    """
    # SMA Period = 20 (Enabled)
    inds = calculate_indicators(mock_downtrend_df, sma_period=20)
    
    assert inds["price"] == 90.0
    assert inds["sma"] > 90.0
    assert inds["in_uptrend"] is False # Should fail trend check

def test_scanner_sma_disabled(mock_downtrend_df):
    """
    Test WK_STK/WK_OPT Profile behavior (SMA Disabled).
    Expect: in_uptrend = True even though Price < SMA.
    """
    # SMA Period = 0 (Disabled)
    inds = calculate_indicators(mock_downtrend_df, sma_period=0)
    
    assert inds["price"] == 90.0
    assert inds["sma"] == 0.0      # Should report 0
    assert inds["in_uptrend"] is True # Should bypass trend check

# ----------------------------------------------------------------------
# 2. Weekly Signal Logic Tests (Decimal vs Percent)
# ----------------------------------------------------------------------

def check_signal_logic(mean_dev_pct_val, threshold_int):
    """
    Replicates the logic used in live_weekly_options_loop.py and backtest.
    
    Logic:
    mean_dev_thresh = float(threshold_int) / 100.0
    is_stretched = mean_dev_pct_val < mean_dev_thresh
    """
    mean_dev_thresh = float(threshold_int) / 100.0
    is_stretched = mean_dev_pct_val < mean_dev_thresh
    return is_stretched

def test_signal_decimal_logic_match():
    """
    Verify that a -6% deviation (-0.06) triggers a -5% threshold (-5).
    """
    mean_dev = -0.06  # -6%
    threshold = -5    # -5%
    
    triggered = check_signal_logic(mean_dev, threshold)
    
    assert triggered is True # -0.06 < -0.05

def test_signal_decimal_logic_miss():
    """
    Verify that a -4% deviation (-0.04) does NOT trigger a -5% threshold.
    """
    mean_dev = -0.04  # -4%
    threshold = -5    # -5%
    
    triggered = check_signal_logic(mean_dev, threshold)
    
    assert triggered is False # -0.04 is NOT < -0.05

def test_signal_decimal_logic_int_bug_repro():
    """
    Demonstrate why the old code failed (Documenting the fix).
    Old logic: deviation (-0.06) < threshold (-5.0) -> False
    """
    mean_dev = -0.06
    threshold_raw = -5.0
    
    # OLD Buggy Logic
    buggy_trigger = mean_dev < threshold_raw
    assert buggy_trigger is False # -0.06 is > -5.0
    
    # NEW Correct Logic
    correct_trigger = mean_dev < (threshold_raw / 100.0)
    assert correct_trigger is True # -0.06 < -0.05
