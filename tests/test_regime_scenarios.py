
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

# Ensure we can import from src
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.regime_manager import RegimeManager

class TestRegimeManager(unittest.TestCase):
    
    def setUp(self):
        self.rm = RegimeManager(verbose=False)
        
    def create_mock_df(self, length=30, base_price=20.0, base_vol=100000):
        # Create a baseline dataframe
        dates = pd.date_range(end=pd.Timestamp.now(), periods=length, freq='B')
        data = {
            "close": [base_price] * length,
            "volume": [base_vol] * length
        }
        df = pd.DataFrame(data, index=dates)
        return df

    @patch('RubberBand.src.regime_manager.fetch_latest_bars')
    def test_scenario_a_spike(self, mock_fetch):
        # Scenario A: Price jumps +10% on 2x Volume -> Expect PANIC
        df = self.create_mock_df(base_price=20.0, base_vol=100000)
        
        # Last close was 20. A 10% jump is 22.
        last_date = df.index[-1]
        df.loc[last_date, 'close'] = 22.0 
        # Volume 2x
        df.loc[last_date, 'volume'] = 200000 
        
        mock_fetch.return_value = ({"VIXY": df}, None)
        
        regime = self.rm.update()
        self.assertEqual(regime, "PANIC", "Scenario A (Spike) should result in PANIC")

    @patch('RubberBand.src.regime_manager.fetch_latest_bars')
    def test_scenario_b_fakeout(self, mock_fetch):
        # Scenario B: Price jumps +10% on 0.5x Volume -> Expect NORMAL
        df = self.create_mock_df(base_price=20.0, base_vol=100000)
        
        last_date = df.index[-1]
        df.loc[last_date, 'close'] = 22.0 # +10%
        df.loc[last_date, 'volume'] = 50000 # 0.5x Avg
        
        mock_fetch.return_value = ({"VIXY": df}, None)
        
        regime = self.rm.update()
        self.assertEqual(regime, "NORMAL", "Scenario B (Fake-out) should result in NORMAL")

    @patch('RubberBand.src.regime_manager.fetch_latest_bars')
    def test_scenario_c_creep(self, mock_fetch):
        # Scenario C: Price slowly rises above SMA but stays inside Band -> Expect NORMAL
        df = self.create_mock_df(base_price=20.0, base_vol=100000)
        
        # Base 20, but oscillating slightly
        # We need std dev > 0 for bands to exist
        # Let's just set alternating values 19, 21, 19, 21...
        # SMA will be 20. Std Dev will be 1. Upper Band 22.
        df['close'] = [20.0 + (1 if i % 2 == 0 else -1) for i in range(len(df))]
        
        # Latest price 21.0
        # SMA 20. Upper Band 20 + 2*1 = 22.
        # Price 21 < 22. Not Panic.
        # Check Calm: Price 21 > SMA 20. Not Calm.
        
        last_date = df.index[-1]
        # Make sure last one is 21
        df.loc[last_date, 'close'] = 21.0
        
        mock_fetch.return_value = ({"VIXY": df}, None)
        
        regime = self.rm.update()
        self.assertEqual(regime, "NORMAL", "Scenario C (Creep) should result in NORMAL")

    @patch('RubberBand.src.regime_manager.fetch_latest_bars')
    def test_scenario_d_recovery(self, mock_fetch):
        # Scenario D: Price drops below SMA for 1 day -> Expect NORMAL (need 3 days for CALM)
        df = self.create_mock_df(base_price=20.0, base_vol=100000)
        
        # All 20. SMA is 20.
        # Set last day to 19.
        last_date = df.index[-1]
        df.loc[last_date, 'close'] = 19.0
        
        # Day -1: 19. SMA ~19.96. 19 < 19.96 (True).
        # Day -2: 20. SMA 20. 20 >= 20 (True). Not Below.
        # So Calm condition fails.
        
        mock_fetch.return_value = ({"VIXY": df}, None)
        
        regime = self.rm.update()
        self.assertEqual(regime, "NORMAL", "Scenario D (Recovery - 1 day) should result in NORMAL")

    @patch('RubberBand.src.regime_manager.fetch_latest_bars')
    def test_scenario_e_total_calm(self, mock_fetch):
        # Scenario E: Price below SMA for 3 consecutive days -> Expect CALM
        df = self.create_mock_df(base_price=20.0, base_vol=100000)
        
        # All 20.
        # Last 3 days 19.
        dates = df.index
        df.loc[dates[-1], 'close'] = 19.0
        df.loc[dates[-2], 'close'] = 19.0
        df.loc[dates[-3], 'close'] = 19.0
        
        # SMA will drop slowly but still be > 19.
        
        mock_fetch.return_value = ({"VIXY": df}, None)
        
        regime = self.rm.update()
        self.assertEqual(regime, "CALM", "Scenario E (Total Calm) should result in CALM")

if __name__ == '__main__':
    unittest.main()
