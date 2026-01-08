import pytest
import pandas as pd
from RubberBand.strategy import check_slope_filter

@pytest.fixture
def mock_df_flat():
    """Slope = 0.0% (Flat)"""
    return pd.DataFrame({
        "close": [100]*4,
        "kc_middle": [100.0, 100.0, 100.0, 100.0] 
    })

@pytest.fixture
def mock_df_crash():
    """Slope = -0.25% (Violent Crash)"""
    # 3-bar change needed: -0.75 on price 100
    # kc_middle: 100.75 -> 100.00
    return pd.DataFrame({
        "close": [100]*4,
        "kc_middle": [100.75, 100.50, 100.25, 100.00]
    })

@pytest.fixture
def mock_df_dip():
    """Slope = -0.15% (Moderate Dip)"""
    # 3-bar change needed: -0.45 on price 100
    return pd.DataFrame({
        "close": [100]*4,
        "kc_middle": [100.45, 100.30, 100.15, 100.00]
    })

def test_slope_calm_mode(mock_df_flat, mock_df_crash, mock_df_dip):
    """
    CALM Regime (Panic Buyer):
    - Expects Dip > 0.08
    - Flat (0.00) -> SKIP
    - Dip (-0.15) -> TRADE (It IS a dip)
    - Crash (-0.25) -> TRADE (It IS a dip)
    """
    regime_cfg = {"slope_threshold_pct": -0.08, "dead_knife_filter": False}
    
    # Flat
    skip, reason = check_slope_filter(mock_df_flat, regime_cfg)
    assert skip is True
    assert "Too_Flat" in reason
    
    # Dip (-0.15 < -0.08) -> IS STEEP ENOUGH -> Trade
    skip, reason = check_slope_filter(mock_df_dip, regime_cfg)
    assert skip is False
    
    # Crash (-0.25 < -0.08) -> IS STEEP ENOUGH -> Trade
    skip, reason = check_slope_filter(mock_df_crash, regime_cfg)
    assert skip is False

def test_slope_panic_mode(mock_df_flat, mock_df_crash, mock_df_dip):
    """
    PANIC Regime (Safety Mode):
    - Expects Crash NOT > 0.20
    - Logic: SKIP if Slope < -0.20 (Too Steep)
    
    - Flat (0.00) -> TRADE (Safe)
    - Dip (-0.15) -> TRADE (Safe, > -0.20)
    - Crash (-0.25) -> SKIP (Unsafe, < -0.20)
    """
    regime_cfg = {"slope_threshold_pct": -0.20, "dead_knife_filter": True}
    
    # Flat (-0.00 > -0.20) -> SAFE
    skip, reason = check_slope_filter(mock_df_flat, regime_cfg)
    assert skip is False
    
    # Dip (-0.15 > -0.20) -> SAFE
    skip, reason = check_slope_filter(mock_df_dip, regime_cfg)
    assert skip is False
    
    # Crash (-0.25 < -0.20) -> UNSAFE -> SKIP
    skip, reason = check_slope_filter(mock_df_crash, regime_cfg)
    assert skip is True
    assert "Safety_Knife_Filter" in reason


# ============================================================================
# Bearish Bar Filter Tests
# ============================================================================
from RubberBand.strategy import check_bearish_bar_filter


@pytest.fixture
def mock_df_bullish_bar():
    """Bar where close > open (bullish)"""
    return pd.DataFrame({
        "open": [100.0],
        "high": [105.0],
        "low": [99.0],
        "close": [104.0],  # Close above open
    })


@pytest.fixture
def mock_df_bearish_bar():
    """Bar where close < open (bearish)"""
    return pd.DataFrame({
        "open": [104.0],
        "high": [105.0],
        "low": [99.0],
        "close": [100.0],  # Close below open
    })


@pytest.fixture
def mock_df_doji_bar():
    """Bar where close == open (doji)"""
    return pd.DataFrame({
        "open": [100.0],
        "high": [105.0],
        "low": [99.0],
        "close": [100.0],  # Close equals open
    })


def test_bearish_bar_filter_disabled():
    """When filter is disabled, should never skip."""
    df = pd.DataFrame({
        "open": [104.0],
        "high": [105.0],
        "low": [99.0],
        "close": [100.0],  # Bearish bar
    })
    cfg = {"bearish_bar_filter": False}
    
    skip, reason = check_bearish_bar_filter(df, cfg)
    assert skip is False
    assert reason == ""


def test_bearish_bar_filter_bullish_bar(mock_df_bullish_bar):
    """Bullish bar (close > open) should NOT be skipped."""
    cfg = {"bearish_bar_filter": True}
    
    skip, reason = check_bearish_bar_filter(mock_df_bullish_bar, cfg)
    assert skip is False
    assert reason == ""


def test_bearish_bar_filter_bearish_bar(mock_df_bearish_bar):
    """Bearish bar (close < open) SHOULD be skipped."""
    cfg = {"bearish_bar_filter": True}
    
    skip, reason = check_bearish_bar_filter(mock_df_bearish_bar, cfg)
    assert skip is True
    assert "BearishBar_Filter" in reason
    assert "Close=100.00" in reason
    assert "Open=104.00" in reason


def test_bearish_bar_filter_doji(mock_df_doji_bar):
    """Doji bar (close == open) should NOT be skipped."""
    cfg = {"bearish_bar_filter": True}
    
    skip, reason = check_bearish_bar_filter(mock_df_doji_bar, cfg)
    assert skip is False
    assert reason == ""


def test_bearish_bar_filter_empty_df():
    """Empty DataFrame should not skip."""
    cfg = {"bearish_bar_filter": True}
    df = pd.DataFrame()
    
    skip, reason = check_bearish_bar_filter(df, cfg)
    assert skip is False


def test_bearish_bar_filter_missing_columns():
    """DataFrame missing open/close columns should not skip."""
    cfg = {"bearish_bar_filter": True}
    df = pd.DataFrame({"high": [105.0], "low": [99.0]})  # Missing open and close
    
    skip, reason = check_bearish_bar_filter(df, cfg)
    assert skip is False

