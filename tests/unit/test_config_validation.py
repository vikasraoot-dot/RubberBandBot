"""
Unit tests for configuration file validation.

Verifies that config.yaml and config_weekly.yaml contain the correct values
after recent changes. These tests act as regression guards against accidental
config modifications that could impact live trading behavior.

Tests cover:
1. config.yaml: rsi_oversold=25, atr_mult_sl=1.5
2. config_weekly.yaml: loads correctly with all required sections
3. Config structure validation (required keys present)
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


def _load_yaml(filename: str) -> dict:
    """Load a YAML config file from the RubberBand directory."""
    import yaml

    config_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'RubberBand', filename
    )
    config_path = os.path.abspath(config_path)

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


class TestMainConfig:
    """Tests for config.yaml (15-minute bot configuration)."""

    def test_config_loads_without_error(self):
        """config.yaml should load without parsing errors."""
        cfg = _load_yaml('config.yaml')
        assert isinstance(cfg, dict), "Config should be a dict"
        assert len(cfg) > 0, "Config should not be empty"

    def test_rsi_oversold_is_25(self):
        """
        RSI oversold threshold should be 25 (optimized from 30).
        Comment in config: "RSI<25 = 3.6x more trades, same/better WR"
        """
        cfg = _load_yaml('config.yaml')
        rsi_oversold = cfg.get("filters", {}).get("rsi_oversold")
        assert rsi_oversold == 25, (
            f"filters.rsi_oversold should be 25, got {rsi_oversold}"
        )

    def test_atr_mult_sl_is_1_5(self):
        """
        ATR multiplier for stop loss should be 1.5.
        Comment in config: "matches live workflow env var SL_ATR"
        """
        cfg = _load_yaml('config.yaml')
        atr_mult_sl = cfg.get("brackets", {}).get("atr_mult_sl")
        assert atr_mult_sl == 1.5, (
            f"brackets.atr_mult_sl should be 1.5, got {atr_mult_sl}"
        )

    def test_take_profit_r_is_2_0(self):
        """Take profit R multiple should be 2.0."""
        cfg = _load_yaml('config.yaml')
        take_profit_r = cfg.get("brackets", {}).get("take_profit_r")
        assert take_profit_r == 2.0, (
            f"brackets.take_profit_r should be 2.0, got {take_profit_r}"
        )

    def test_brackets_enabled(self):
        """Brackets (stop loss / take profit) should be enabled."""
        cfg = _load_yaml('config.yaml')
        enabled = cfg.get("brackets", {}).get("enabled")
        assert enabled is True, "brackets.enabled should be True"

    def test_slope_threshold_is_negative(self):
        """Slope threshold should be negative (crash filter)."""
        cfg = _load_yaml('config.yaml')
        slope = cfg.get("slope_threshold")
        assert slope is not None, "slope_threshold should exist"
        assert slope < 0, f"slope_threshold should be negative, got {slope}"

    def test_feed_is_sip(self):
        """Data feed should be 'sip' (consolidated tape, free-tier 15-min delay)."""
        cfg = _load_yaml('config.yaml')
        feed = cfg.get("feed")
        assert feed == "sip", f"feed should be 'sip', got {feed}"

    def test_trend_filter_sma_period(self):
        """Trend filter SMA period should be 100 (All-Weather optimization)."""
        cfg = _load_yaml('config.yaml')
        sma_period = cfg.get("trend_filter", {}).get("sma_period")
        assert sma_period == 100, f"trend_filter.sma_period should be 100, got {sma_period}"

    def test_trend_filter_enabled(self):
        """Trend filter should be enabled for options."""
        cfg = _load_yaml('config.yaml')
        enabled = cfg.get("trend_filter", {}).get("enabled")
        assert enabled is True, "trend_filter.enabled should be True"

    def test_resilience_section_exists(self):
        """Resilience (dynamic circuit breaker) section should exist."""
        cfg = _load_yaml('config.yaml')
        resilience = cfg.get("resilience")
        assert resilience is not None, "resilience section should exist"
        assert resilience.get("enabled") is True, "resilience should be enabled"

    def test_entry_windows_exist(self):
        """Entry windows should be defined to avoid first 15m volatility."""
        cfg = _load_yaml('config.yaml')
        windows = cfg.get("entry_windows")
        assert windows is not None, "entry_windows should exist"
        assert len(windows) > 0, "entry_windows should have at least one window"

    def test_rth_only_enabled(self):
        """Regular trading hours only should be enabled."""
        cfg = _load_yaml('config.yaml')
        rth = cfg.get("rth_only")
        assert rth is True, "rth_only should be True"

    def test_required_top_level_keys(self):
        """All required top-level keys should be present."""
        cfg = _load_yaml('config.yaml')
        required_keys = [
            "intervals", "timezone", "feed", "keltner_length",
            "atr_length", "rsi_length", "filters", "brackets",
            "results_dir",
        ]
        for key in required_keys:
            assert key in cfg, f"Required key '{key}' missing from config.yaml"


class TestWeeklyConfig:
    """Tests for config_weekly.yaml (weekly bot configuration)."""

    def test_config_loads_without_error(self):
        """config_weekly.yaml should load without parsing errors."""
        cfg = _load_yaml('config_weekly.yaml')
        assert isinstance(cfg, dict), "Config should be a dict"
        assert len(cfg) > 0, "Config should not be empty"

    def test_timeframe_is_weekly(self):
        """Timeframe should be '1Week'."""
        cfg = _load_yaml('config_weekly.yaml')
        assert cfg.get("timeframe") == "1Week", "timeframe should be '1Week'"

    def test_weekly_rsi_oversold(self):
        """
        Weekly RSI oversold should be 40 (different from 15m which is 25).
        Weekly RSI rarely drops below 30.
        """
        cfg = _load_yaml('config_weekly.yaml')
        rsi_oversold = cfg.get("filters", {}).get("rsi_oversold")
        assert rsi_oversold == 40, (
            f"Weekly filters.rsi_oversold should be 40, got {rsi_oversold}"
        )

    def test_weekly_mean_deviation_threshold(self):
        """Mean deviation threshold should be -5 (5% below SMA)."""
        cfg = _load_yaml('config_weekly.yaml')
        threshold = cfg.get("filters", {}).get("mean_deviation_threshold")
        assert threshold == -5, (
            f"Weekly filters.mean_deviation_threshold should be -5, got {threshold}"
        )

    def test_weekly_atr_mult_sl(self):
        """Weekly ATR SL multiplier should be 1.5 (tighter than old 2.0)."""
        cfg = _load_yaml('config_weekly.yaml')
        atr_mult_sl = cfg.get("brackets", {}).get("atr_mult_sl")
        assert atr_mult_sl == 1.5, (
            f"Weekly brackets.atr_mult_sl should be 1.5, got {atr_mult_sl}"
        )

    def test_weekly_take_profit_r(self):
        """Weekly take profit R should be 1.5."""
        cfg = _load_yaml('config_weekly.yaml')
        tp_r = cfg.get("brackets", {}).get("take_profit_r")
        assert tp_r == 1.5, f"Weekly brackets.take_profit_r should be 1.5, got {tp_r}"

    def test_max_concurrent_positions(self):
        """Max concurrent positions should be defined and reasonable."""
        cfg = _load_yaml('config_weekly.yaml')
        max_pos = cfg.get("max_concurrent_positions")
        assert max_pos is not None, "max_concurrent_positions should exist"
        assert max_pos == 5, f"max_concurrent_positions should be 5, got {max_pos}"

    def test_max_notional_per_trade(self):
        """Max notional per trade should be defined."""
        cfg = _load_yaml('config_weekly.yaml')
        max_notional = cfg.get("max_notional_per_trade")
        assert max_notional is not None, "max_notional_per_trade should exist"
        assert max_notional == 2000, f"max_notional_per_trade should be 2000, got {max_notional}"

    def test_allow_shorts_disabled(self):
        """Weekly bot should be long-only."""
        cfg = _load_yaml('config_weekly.yaml')
        assert cfg.get("allow_shorts") is False, "allow_shorts should be False (long-only)"

    def test_indicator_settings_exist(self):
        """Indicator settings section should exist with keltner and rsi sub-sections."""
        cfg = _load_yaml('config_weekly.yaml')
        indicators = cfg.get("indicators")
        assert indicators is not None, "indicators section should exist"
        assert "keltner" in indicators, "indicators.keltner should exist"
        assert "rsi" in indicators, "indicators.rsi should exist"

    def test_keltner_settings(self):
        """Verify weekly keltner settings."""
        cfg = _load_yaml('config_weekly.yaml')
        keltner = cfg.get("indicators", {}).get("keltner", {})
        assert keltner.get("ema_period") == 10, "keltner.ema_period should be 10"
        assert keltner.get("atr_period") == 10, "keltner.atr_period should be 10"
        assert keltner.get("atr_multiplier") == 2.0, "keltner.atr_multiplier should be 2.0"

    def test_required_sections_present(self):
        """All required config sections should be present."""
        cfg = _load_yaml('config_weekly.yaml')
        required_sections = [
            "timeframe", "feed", "timezone",
            "max_notional_per_trade", "filters", "brackets",
            "indicators", "max_concurrent_positions",
        ]
        for section in required_sections:
            assert section in cfg, f"Required section '{section}' missing from config_weekly.yaml"


class TestConfigConsistency:
    """Cross-config consistency checks."""

    def test_both_configs_use_same_feed(self):
        """Both configs should use the same data feed."""
        main_cfg = _load_yaml('config.yaml')
        weekly_cfg = _load_yaml('config_weekly.yaml')

        assert main_cfg.get("feed") == weekly_cfg.get("feed"), (
            "Both configs should use the same data feed"
        )

    def test_both_configs_use_same_timezone(self):
        """Both configs should use the same timezone."""
        main_cfg = _load_yaml('config.yaml')
        weekly_cfg = _load_yaml('config_weekly.yaml')

        assert main_cfg.get("timezone") == weekly_cfg.get("timezone"), (
            "Both configs should use the same timezone"
        )

    def test_weekly_rsi_is_more_lenient(self):
        """
        Weekly RSI oversold (40) should be higher than 15m RSI oversold (25)
        because weekly RSI rarely drops as low.
        """
        main_cfg = _load_yaml('config.yaml')
        weekly_cfg = _load_yaml('config_weekly.yaml')

        main_rsi = main_cfg.get("filters", {}).get("rsi_oversold", 0)
        weekly_rsi = weekly_cfg.get("filters", {}).get("rsi_oversold", 0)

        assert weekly_rsi > main_rsi, (
            f"Weekly RSI oversold ({weekly_rsi}) should be higher than 15m ({main_rsi})"
        )
