import pytest
from unittest.mock import patch
from RubberBand.src.regime_manager import RegimeManager

def test_regime_calm(regime_manager, mock_vixy_calm):
    """Test VIXY < 35 triggers CALM regime"""
    with patch("RubberBand.src.regime_manager.fetch_latest_bars") as mock_fetch:
        # Mock return signature: (bars_map, failures)
        mock_fetch.return_value = ({"VIXY": mock_vixy_calm}, [])
        
        regime = regime_manager.update()
        
        assert regime == "CALM"
        cfg = regime_manager.get_config_overrides()
        assert cfg["slope_threshold_pct"] == -0.08
        assert cfg["dead_knife_filter"] is False

def test_regime_normal(regime_manager, mock_vixy_normal):
    """Test VIXY 35-55 triggers NORMAL regime"""
    with patch("RubberBand.src.regime_manager.fetch_latest_bars") as mock_fetch:
        mock_fetch.return_value = ({"VIXY": mock_vixy_normal}, [])
        
        regime = regime_manager.update()
        
        assert regime == "NORMAL"
        cfg = regime_manager.get_config_overrides()
        assert cfg["slope_threshold_pct"] == -0.12
        assert cfg["dead_knife_filter"] is False

def test_regime_panic(regime_manager, mock_vixy_panic):
    """Test VIXY > 55 triggers PANIC regime"""
    with patch("RubberBand.src.regime_manager.fetch_latest_bars") as mock_fetch:
        mock_fetch.return_value = ({"VIXY": mock_vixy_panic}, [])
        
        regime = regime_manager.update()
        
        assert regime == "PANIC"
        cfg = regime_manager.get_config_overrides()
        assert cfg["slope_threshold_pct"] == -0.20
        assert cfg["dead_knife_filter"] is True
