
import sys
import os
import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock, patch
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RubberBand.src.options_data import is_options_trading_allowed
from RubberBand.scripts.live_spreads_loop import get_long_signals

ET = ZoneInfo("US/Eastern")
UTC = ZoneInfo("UTC")

# -------------------------------------------------------------------------
# Test 1: Time Cutoff (is_options_trading_allowed)
# -------------------------------------------------------------------------
def test_time_cutoff():
    # Helper to mock time
    def mock_time(hour, minute):
        # Create a time for TODAY but specific hour/minute
        now = datetime.now(ET)
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    # Test 9:29 (Pre-market) -> False
    with patch('RubberBand.src.options_data.datetime') as mock_dt:
        mock_dt.now.return_value = mock_time(9, 29)
        assert is_options_trading_allowed() == False, "9:29 should be closed"

    # Test 9:30 (Open) -> True
    with patch('RubberBand.src.options_data.datetime') as mock_dt:
        mock_dt.now.return_value = mock_time(9, 30)
        assert is_options_trading_allowed() == True, "9:30 should be open"

    # Test 14:29 (Before Cutoff) -> True
    with patch('RubberBand.src.options_data.datetime') as mock_dt:
        mock_dt.now.return_value = mock_time(14, 29)
        assert is_options_trading_allowed() == True, "14:29 should be open"

    # Test 14:30 (Cutoff Exact) -> False
    # Cutoff moved from 3:45 PM to 2:30 PM to avoid late-day illiquidity
    with patch('RubberBand.src.options_data.datetime') as mock_dt:
        mock_dt.now.return_value = mock_time(14, 30)
        assert is_options_trading_allowed() == False, "14:30 should be closed"

    # Test 14:31 (After Cutoff) -> False
    with patch('RubberBand.src.options_data.datetime') as mock_dt:
        mock_dt.now.return_value = mock_time(14, 31)
        assert is_options_trading_allowed() == False, "14:31 should be closed"

    # Test 15:45 (Well After Cutoff) -> False
    with patch('RubberBand.src.options_data.datetime') as mock_dt:
        mock_dt.now.return_value = mock_time(15, 45)
        assert is_options_trading_allowed() == False, "15:45 should be closed"

# -------------------------------------------------------------------------
# Test 2: Intra-bar Logic (get_long_signals)
# -------------------------------------------------------------------------
@patch("RubberBand.scripts.live_spreads_loop.fetch_latest_bars")
@patch("RubberBand.scripts.live_spreads_loop.attach_verifiers")
@patch("RubberBand.scripts.live_spreads_loop.check_slope_filter")
@patch("RubberBand.scripts.live_spreads_loop.check_bearish_bar_filter")  # NEW: Mock bearish bar filter
@patch("RubberBand.scripts.live_spreads_loop.datetime")  # PATCH DATETIME IN SCRIPT
def test_audit_logic_forming_bar(mock_dt_script, mock_bearish_filter, mock_check_slope, mock_verifiers, mock_fetch):
    # Setup Fixed Time
    fixed_now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    mock_dt_script.now.return_value = fixed_now

    # Setup Mocks
    mock_logger = MagicMock()
    mock_check_slope.return_value = (False, "")  # Don't skip due to slope
    mock_bearish_filter.return_value = (False, "")  # Don't skip due to bearish bar

    # CASE A: Forming Bar (< 15 mins old)
    # -----------------------------------
    # Last bar is 5 mins old
    last_ts = fixed_now - timedelta(minutes=5)

    # Create Mock DF (Needs > 20 rows)
    data = {
        "open": [99.0] * 30,  # Add open column
        "close": [100.0] * 30,
        "rsi": [50.0] * 30,
        "atr": [2.0] * 30,  # Add atr column
        "long_signal": [False] * 30,
        "rsi_oversold": [False] * 30,
        "ema_ok": [True] * 30,
        "touch": [False] * 30,
        "slope": [0.0] * 30,  # Add slope column
    }

    # Update last row to be the signal
    data["open"][-1] = 94.0
    data["close"][-1] = 95.0
    data["rsi"][-1] = 20.0
    data["long_signal"][-1] = True
    data["rsi_oversold"][-1] = True
    data["touch"][-1] = True

    # Create Index: [last_ts - 29*15m, ..., last_ts]
    times = [last_ts - timedelta(minutes=15 * i) for i in range(29, -1, -1)]
    df = pd.DataFrame(data, index=times)

    # Mocks return this DF
    mock_fetch.return_value = ({"TEST": df}, None)
    mock_verifiers.return_value = df

    # Run - pass empty regime_cfg dict
    confirmed, forming = get_long_signals(["TEST"], {}, mock_logger, regime_cfg={})

    # Assertions
    assert len(forming) == 1, f"Should detect 1 forming signal. Start: {fixed_now}, Last: {last_ts}"
    assert forming[0]["rsi"] == 20.0
    assert len(confirmed) == 0, "Should NOT have any confirmed signals (last bar was dropped)"

@patch("RubberBand.scripts.live_spreads_loop.fetch_latest_bars")
@patch("RubberBand.scripts.live_spreads_loop.attach_verifiers")
@patch("RubberBand.scripts.live_spreads_loop.check_slope_filter")
@patch("RubberBand.scripts.live_spreads_loop.check_bearish_bar_filter")  # NEW: Mock bearish bar filter
@patch("RubberBand.scripts.live_spreads_loop.get_daily_sma")  # Mock inner function
@patch("RubberBand.scripts.live_spreads_loop.datetime")  # PATCH DATETIME IN SCRIPT
def test_audit_logic_closed_bar(mock_dt_script, mock_get_sma, mock_bearish_filter, mock_check_slope, mock_verifiers, mock_fetch):
    # Setup Fixed Time
    fixed_now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    mock_dt_script.now.return_value = fixed_now

    # Setup Mocks
    mock_logger = MagicMock()
    mock_check_slope.return_value = (False, "")  # Don't skip due to slope
    mock_bearish_filter.return_value = (False, "")  # Don't skip due to bearish bar
    mock_get_sma.return_value = None  # Disable trend filter logic

    # CASE B: Closed Bar (> 15 mins old)
    # ----------------------------------
    # Last bar is 16 mins old (Closed)
    last_ts = fixed_now - timedelta(minutes=16)

    # Create Mock DF (Needs > 20 rows)
    data = {
        "open": [99.0] * 30,  # Add open column for bearish bar filter
        "close": [100.0] * 30,
        "rsi": [50.0] * 30,
        "atr": [2.0] * 30,  # Add atr column
        "long_signal": [False] * 30,
        "rsi_oversold": [False] * 30,
        "ema_ok": [True] * 30,
        "touch": [False] * 30,
        "kc_middle": [100.0] * 30,  # Add for slope calculation
    }
    # Update last row to be the signal
    data["open"][-1] = 94.0  # Bullish bar (close > open)
    data["close"][-1] = 95.0
    data["rsi"][-1] = 20.0
    data["atr"][-1] = 2.0
    data["long_signal"][-1] = True
    data["rsi_oversold"][-1] = True
    data["touch"][-1] = True

    # Create Index
    times = [last_ts - timedelta(minutes=15 * i) for i in range(29, -1, -1)]
    df = pd.DataFrame(data, index=times)

    # Mocks
    mock_fetch.return_value = ({"TEST": df}, None)
    mock_verifiers.return_value = df

    # Pass empty regime_cfg dict instead of relying on default None
    confirmed, forming = get_long_signals(["TEST"], {}, mock_logger, regime_cfg={})

    # DEBUG: Check for errors
    if mock_logger.error.called:
        print(f"\nLOGGER ERROR CALLED: {mock_logger.error.call_args}")

    # Assertions
    assert len(forming) == 0, "Should NOT have forming signal (bar is old)"
    assert len(confirmed) == 1, f"Should have 1 confirmed signal. Confirmed: {confirmed}"
    assert confirmed[0]["rsi"] == 20.0

if __name__ == "__main__":
    # Manually run if executed as script
    try:
        test_time_cutoff()
        print("test_time_cutoff passed")
    except AssertionError as e:
        print(f"test_time_cutoff failed: {e}")
        
    # We can't easily run the mocked ones without pytest runner structure or manual mock setup
    print("Run with `pytest tests/test_audit_cutoff.py` to verify all.")
