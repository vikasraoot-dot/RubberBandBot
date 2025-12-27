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
