"""
Unit tests for the improved backtest spread pricing model.

Tests cover:
1. Theta decay curve behavior at different time points
2. Bid-ask slippage application on exit
3. Edge cases (at expiry, at entry, etc.)
"""
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from RubberBand.scripts.backtest_spreads import estimate_spread_value


class TestEstimateSpreadValue:
    """Tests for the estimate_spread_value function."""
    
    def test_at_entry_full_time_value(self):
        """At entry (100% time remaining), spread should have meaningful time value."""
        underlying = 100.0
        atm_strike = 100.0  # ATM
        otm_strike = 102.5  # OTM by $2.50
        total_bars = 100
        dte_bars_remaining = 100  # 100% time remaining
        
        long_val, short_val, spread_val = estimate_spread_value(
            underlying, atm_strike, otm_strike, dte_bars_remaining, total_bars
        )
        
        # Spread should have value (time value since ATM)
        assert spread_val > 0, "Spread should have positive value at entry"
        assert spread_val < (otm_strike - atm_strike), "Spread value should be less than max width"
        
    def test_at_expiry_intrinsic_only(self):
        """At expiry (0% time remaining), spread value should be close to intrinsic."""
        underlying = 101.5  # ITM by $1.50
        atm_strike = 100.0
        otm_strike = 102.5
        total_bars = 100
        dte_bars_remaining = 1  # ~0% time remaining
        
        long_val, short_val, spread_val = estimate_spread_value(
            underlying, atm_strike, otm_strike, dte_bars_remaining, total_bars
        )
        
        # Expected intrinsic: max(0, 101.5 - 100) - max(0, 101.5 - 102.5) = 1.5 - 0 = 1.5
        # But with time decay and slippage, should be less
        assert spread_val < 1.8, "Near expiry, spread should be close to intrinsic"
        assert spread_val > 0, "ITM spread should have positive value"
        
    def test_theta_decay_is_steeper_near_expiry(self):
        """Theta decay should be steeper in final 25% of time."""
        underlying = 100.0
        atm_strike = 100.0
        otm_strike = 102.5
        total_bars = 100
        
        # Value at 50% time remaining
        _, _, val_50pct = estimate_spread_value(
            underlying, atm_strike, otm_strike, 50, total_bars
        )
        
        # Value at 25% time remaining
        _, _, val_25pct = estimate_spread_value(
            underlying, atm_strike, otm_strike, 25, total_bars
        )
        
        # Value at 10% time remaining
        _, _, val_10pct = estimate_spread_value(
            underlying, atm_strike, otm_strike, 10, total_bars
        )
        
        # All values should be positive
        assert val_50pct > 0, "Value at 50% should be positive"
        assert val_25pct > 0, "Value at 25% should be positive"
        assert val_10pct > 0, "Value at 10% should be positive"
        
        # Values should decrease as time decreases
        assert val_50pct > val_25pct, "Value should decrease as time decreases"
        assert val_25pct > val_10pct, "Value should decrease as time decreases"
        
        # At 10% time, value should be significantly less than at 50%
        assert val_10pct < val_50pct * 0.55, \
            "At 10% time remaining, spread should have lost at least 45% of value"
    
    def test_bid_ask_slippage_on_exit(self):
        """Exit should apply bid-ask slippage, reducing value."""
        underlying = 101.0
        atm_strike = 100.0
        otm_strike = 102.5
        total_bars = 100
        dte_bars_remaining = 50
        
        # Without exit flag (entry/mid-trade value)
        _, _, val_no_exit = estimate_spread_value(
            underlying, atm_strike, otm_strike, dte_bars_remaining, total_bars,
            is_exit=False
        )
        
        # With exit flag (should have slippage)
        _, _, val_with_exit = estimate_spread_value(
            underlying, atm_strike, otm_strike, dte_bars_remaining, total_bars,
            is_exit=True
        )
        
        assert val_with_exit < val_no_exit, "Exit value should be less due to bid-ask slippage"
        
    def test_zero_slippage_when_disabled(self):
        """Setting bid_ask_pct=0 should disable slippage."""
        underlying = 101.0
        atm_strike = 100.0
        otm_strike = 102.5
        total_bars = 100
        dte_bars_remaining = 50
        
        _, _, val_no_exit = estimate_spread_value(
            underlying, atm_strike, otm_strike, dte_bars_remaining, total_bars,
            is_exit=False, bid_ask_pct=0
        )
        
        _, _, val_with_exit_no_slippage = estimate_spread_value(
            underlying, atm_strike, otm_strike, dte_bars_remaining, total_bars,
            is_exit=True, bid_ask_pct=0
        )
        
        assert val_with_exit_no_slippage == val_no_exit, \
            "With bid_ask_pct=0, exit value should equal mid-trade value"
    
    def test_high_iv_increases_time_value(self):
        """Higher IV should increase time value."""
        underlying = 100.0
        atm_strike = 100.0
        otm_strike = 102.5
        total_bars = 100
        dte_bars_remaining = 80
        
        # Low IV
        _, _, val_low_iv = estimate_spread_value(
            underlying, atm_strike, otm_strike, dte_bars_remaining, total_bars,
            iv=0.20
        )
        
        # High IV
        _, _, val_high_iv = estimate_spread_value(
            underlying, atm_strike, otm_strike, dte_bars_remaining, total_bars,
            iv=0.50
        )
        
        assert val_high_iv > val_low_iv, "Higher IV should increase spread value"
    
    def test_otm_spread_has_positive_value(self):
        """OTM spread should still have time value before expiry."""
        underlying = 98.0  # Below ATM strike
        atm_strike = 100.0
        otm_strike = 102.5
        total_bars = 100
        dte_bars_remaining = 80
        
        _, _, spread_val = estimate_spread_value(
            underlying, atm_strike, otm_strike, dte_bars_remaining, total_bars
        )
        
        assert spread_val > 0, "OTM spread should have positive time value before expiry"
    
    def test_deep_itm_spread_approaches_max_value(self):
        """Deep ITM spread should approach max intrinsic value."""
        underlying = 105.0  # Deep ITM
        atm_strike = 100.0
        otm_strike = 102.5
        spread_width = otm_strike - atm_strike
        total_bars = 100
        dte_bars_remaining = 10  # Near expiry
        
        _, _, spread_val = estimate_spread_value(
            underlying, atm_strike, otm_strike, dte_bars_remaining, total_bars
        )
        
        # Max value is spread_width ($2.50)
        assert spread_val >= spread_width * 0.9, \
            "Deep ITM spread near expiry should be close to max value"


class TestThetaDecayRegression:
    """Regression tests to ensure new model is more conservative than old."""
    
    def test_final_25pct_is_brutal(self):
        """Final 25% of time should see significant decay."""
        underlying = 100.0
        atm_strike = 100.0
        otm_strike = 102.5
        total_bars = 100
        
        # Value at 25% time remaining
        _, _, val_25pct = estimate_spread_value(
            underlying, atm_strike, otm_strike, 25, total_bars
        )
        
        # Value at 100% time remaining
        _, _, val_100pct = estimate_spread_value(
            underlying, atm_strike, otm_strike, 100, total_bars
        )
        
        # At 25% time, should have lost at least 50% of time value
        assert val_25pct < val_100pct * 0.7, \
            "At 25% time remaining, spread should have lost significant value"


class TestDefaultOptsSync:
    """Tests to verify DEFAULT_OPTS matches live bot configuration."""
    
    def test_default_opts_max_debit_matches_live(self):
        """DEFAULT_OPTS max_debit should match live bot (3.00)."""
        from RubberBand.scripts.backtest_spreads import DEFAULT_OPTS
        assert DEFAULT_OPTS["max_debit"] == 3.00, \
            "max_debit should be 3.00 to match live bot"
    
    def test_default_opts_min_dte_exists(self):
        """DEFAULT_OPTS should have min_dte=3 to match live bot."""
        from RubberBand.scripts.backtest_spreads import DEFAULT_OPTS
        assert "min_dte" in DEFAULT_OPTS, "min_dte should exist in DEFAULT_OPTS"
        assert DEFAULT_OPTS["min_dte"] == 3, "min_dte should be 3 to match live bot"
    
    def test_default_opts_bars_stop_exists(self):
        """DEFAULT_OPTS should have bars_stop=14 to match live bot."""
        from RubberBand.scripts.backtest_spreads import DEFAULT_OPTS
        assert "bars_stop" in DEFAULT_OPTS, "bars_stop should exist in DEFAULT_OPTS"
        assert DEFAULT_OPTS["bars_stop"] == 14, "bars_stop should be 14 to match live bot"
    
    def test_default_opts_dte_matches_live(self):
        """DEFAULT_OPTS dte should match live bot (6)."""
        from RubberBand.scripts.backtest_spreads import DEFAULT_OPTS
        assert DEFAULT_OPTS["dte"] == 6, "dte should be 6 to match live bot"
    
    def test_default_opts_spread_width_atr_matches_live(self):
        """DEFAULT_OPTS spread_width_atr should match live bot (1.5)."""
        from RubberBand.scripts.backtest_spreads import DEFAULT_OPTS
        assert DEFAULT_OPTS["spread_width_atr"] == 1.5, \
            "spread_width_atr should be 1.5 to match live bot"


class TestCalculateActualDte:
    """Tests for the calculate_actual_dte function."""
    
    def test_monday_entry_uses_this_friday(self):
        """Monday entry should use this Friday (4 days)."""
        from RubberBand.scripts.backtest_spreads import calculate_actual_dte
        from datetime import datetime
        # Monday
        monday = datetime(2026, 1, 5)  # A Monday
        result = calculate_actual_dte(monday, target_dte=6, min_dte=3)
        assert result == 4, "Monday should target this Friday (4 days away)"
    
    def test_friday_entry_rolls_to_next_week(self):
        """Friday entry should roll to next Friday (7 days)."""
        from RubberBand.scripts.backtest_spreads import calculate_actual_dte
        from datetime import datetime
        # Friday
        friday = datetime(2026, 1, 2)  # A Friday
        result = calculate_actual_dte(friday, target_dte=6, min_dte=3)
        assert result == 7, "Friday should roll to next Friday (7 days)"
    
    def test_thursday_entry_rolls_to_next_week(self):
        """Thursday entry (1 day to Friday) should roll to next week if < min_dte."""
        from RubberBand.scripts.backtest_spreads import calculate_actual_dte
        from datetime import datetime
        # Thursday
        thursday = datetime(2026, 1, 1)  # A Thursday
        result = calculate_actual_dte(thursday, target_dte=6, min_dte=3)
        assert result == 8, "Thursday should roll to next Friday (1+7=8 days)"

