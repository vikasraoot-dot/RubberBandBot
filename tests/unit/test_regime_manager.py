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
        assert cfg["slope_threshold_pct"] == -0.20  # Golden Config: unified threshold
        assert cfg["dead_knife_filter"] is False

def test_regime_normal(regime_manager, mock_vixy_normal):
    """Test VIXY 35-55 triggers NORMAL regime"""
    with patch("RubberBand.src.regime_manager.fetch_latest_bars") as mock_fetch:
        mock_fetch.return_value = ({"VIXY": mock_vixy_normal}, [])
        
        regime = regime_manager.update()
        
        assert regime == "NORMAL"
        cfg = regime_manager.get_config_overrides()
        assert cfg["slope_threshold_pct"] == -0.20  # Golden Config: unified threshold
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


def test_intraday_panic_spike(regime_manager, mock_vixy_normal):
    """Test intraday VIXY spike triggers PANIC even when daily regime is NORMAL"""
    import pandas as pd
    from datetime import datetime, timedelta

    with patch("RubberBand.src.regime_manager.fetch_latest_bars") as mock_fetch:
        # First call: daily update returns NORMAL regime
        # (reference close = 40.0, upper_band ~44 based on normal data)
        mock_fetch.return_value = ({"VIXY": mock_vixy_normal}, [])

        daily_regime = regime_manager.update()
        assert daily_regime == "NORMAL"
        assert regime_manager._reference_close == 40.0  # Last close in mock_vixy_normal

        # Second call: intraday check with spiked price (+10%)
        # Create a 5-min bar with current price of 44.0 (+10% from 40.0)
        intraday_df = pd.DataFrame({
            "open": [43.5],
            "high": [44.5],
            "low": [43.0],
            "close": [44.0],  # +10% spike
            "volume": [100000],
        }, index=[datetime.now() - timedelta(minutes=5)])

        mock_fetch.return_value = ({"VIXY": intraday_df}, [])

        effective_regime = regime_manager.get_effective_regime()

        # Should trigger PANIC due to >8% intraday spike
        assert effective_regime == "PANIC"
        assert regime_manager._intraday_panic is True


def test_intraday_no_spike_returns_daily_regime(regime_manager, mock_vixy_normal):
    """Test that small intraday moves don't trigger PANIC"""
    import pandas as pd
    from datetime import datetime, timedelta

    with patch("RubberBand.src.regime_manager.fetch_latest_bars") as mock_fetch:
        # Daily update returns NORMAL regime
        mock_fetch.return_value = ({"VIXY": mock_vixy_normal}, [])

        daily_regime = regime_manager.update()
        assert daily_regime == "NORMAL"

        # Intraday check with small move (+3%)
        intraday_df = pd.DataFrame({
            "open": [41.0],
            "high": [41.5],
            "low": [40.5],
            "close": [41.2],  # +3% (below 8% threshold)
            "volume": [100000],
        }, index=[datetime.now() - timedelta(minutes=5)])

        mock_fetch.return_value = ({"VIXY": intraday_df}, [])

        effective_regime = regime_manager.get_effective_regime()

        # Should return NORMAL (no panic trigger)
        assert effective_regime == "NORMAL"
        assert regime_manager._intraday_panic is False


def test_intraday_panic_persists(regime_manager, mock_vixy_normal):
    """Test that once intraday PANIC triggers, it persists until daily reset"""
    import pandas as pd
    from datetime import datetime, timedelta

    with patch("RubberBand.src.regime_manager.fetch_latest_bars") as mock_fetch:
        # Daily update
        mock_fetch.return_value = ({"VIXY": mock_vixy_normal}, [])
        regime_manager.update()

        # Trigger intraday panic with spike
        spike_df = pd.DataFrame({
            "close": [44.0],  # +10% spike
        }, index=[datetime.now() - timedelta(minutes=5)])
        mock_fetch.return_value = ({"VIXY": spike_df}, [])

        regime1 = regime_manager.get_effective_regime()
        assert regime1 == "PANIC"

        # Even if VIXY drops back, panic should persist
        calm_df = pd.DataFrame({
            "close": [39.0],  # Back to normal
        }, index=[datetime.now() - timedelta(minutes=5)])
        mock_fetch.return_value = ({"VIXY": calm_df}, [])

        regime2 = regime_manager.get_effective_regime()
        assert regime2 == "PANIC"  # Still PANIC (persists)

        # Only daily update() resets intraday panic
        mock_fetch.return_value = ({"VIXY": mock_vixy_normal}, [])
        regime_manager.update()
        assert regime_manager._intraday_panic is False  # Reset
