"""
Tests for the weekly options probability filter module.

Covers config loading/validation, threshold filtering, composite scoring,
CallCandidate log dict, and core evaluation with mocked IV resolution.
"""
import math
import pytest
from unittest.mock import patch

from RubberBand.src.weekly_probability_filter import (
    WeeklyProbabilityFilterConfig,
    load_weekly_probability_filter_config,
    CallCandidate,
    compute_call_composite_score,
    apply_call_threshold_filters,
    evaluate_call_quality,
    _sigmoid,
)


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadWeeklyConfig:
    """Tests for load_weekly_probability_filter_config."""

    def test_empty_config_returns_defaults(self):
        config = load_weekly_probability_filter_config({})
        assert config.enabled is False
        assert config.mode == "shadow"
        assert config.min_breakeven_prob == 0.45
        assert config.min_intrinsic_ratio == 0.20
        assert config.min_expected_value == -2.0
        assert config.risk_free_rate == 0.045
        assert config.fallback_to_legacy is True

    def test_full_config(self):
        cfg = {
            "weekly_probability_filter": {
                "enabled": True,
                "mode": "filter",
                "min_breakeven_prob": 0.55,
                "min_intrinsic_ratio": 0.30,
                "min_expected_value": -1.0,
                "max_iv_rank": 80.0,
                "risk_free_rate": 0.05,
                "fallback_to_legacy": False,
                "w_breakeven": 0.50,
                "w_intrinsic": 0.20,
                "w_expected_value": 0.20,
                "w_iv_rank": 0.10,
            }
        }
        config = load_weekly_probability_filter_config(cfg)
        assert config.enabled is True
        assert config.mode == "filter"
        assert config.min_breakeven_prob == 0.55
        assert config.min_intrinsic_ratio == 0.30
        assert config.min_expected_value == -1.0
        assert config.max_iv_rank == 80.0
        assert config.risk_free_rate == 0.05
        assert config.fallback_to_legacy is False
        assert config.w_breakeven == 0.50

    def test_partial_config_uses_defaults(self):
        cfg = {"weekly_probability_filter": {"enabled": True}}
        config = load_weekly_probability_filter_config(cfg)
        assert config.enabled is True
        assert config.mode == "shadow"  # default
        assert config.min_breakeven_prob == 0.45  # default

    def test_invalid_mode_raises(self):
        cfg = {"weekly_probability_filter": {"mode": "invalid"}}
        with pytest.raises(ValueError, match="mode must be"):
            load_weekly_probability_filter_config(cfg)

    def test_breakeven_prob_out_of_range_raises(self):
        cfg = {"weekly_probability_filter": {"min_breakeven_prob": 1.5}}
        with pytest.raises(ValueError, match="min_breakeven_prob"):
            load_weekly_probability_filter_config(cfg)

    def test_breakeven_prob_negative_raises(self):
        cfg = {"weekly_probability_filter": {"min_breakeven_prob": -0.1}}
        with pytest.raises(ValueError, match="min_breakeven_prob"):
            load_weekly_probability_filter_config(cfg)

    def test_intrinsic_ratio_out_of_range_raises(self):
        cfg = {"weekly_probability_filter": {"min_intrinsic_ratio": 2.0}}
        with pytest.raises(ValueError, match="min_intrinsic_ratio"):
            load_weekly_probability_filter_config(cfg)

    def test_weights_sum_warning(self, caplog):
        cfg = {
            "weekly_probability_filter": {
                "w_breakeven": 0.80,
                "w_intrinsic": 0.80,
                "w_expected_value": 0.80,
                "w_iv_rank": 0.80,
            }
        }
        import logging
        with caplog.at_level(logging.WARNING):
            load_weekly_probability_filter_config(cfg)
        assert "weights sum" in caplog.text.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Threshold filters
# ──────────────────────────────────────────────────────────────────────────────

def _make_candidate(**overrides) -> CallCandidate:
    """Helper to create a valid CallCandidate with sensible defaults."""
    defaults = dict(
        underlying="AAPL",
        option_symbol="AAPL260401C00220000",
        strike=220.0,
        expiration="2026-04-01",
        dte=45,
        underlying_price=230.0,
        bid=11.50,
        ask=12.00,
        mid=11.75,
        premium=12.00,
        bsm_iv=0.28,
        breakeven_prob=0.55,
        itm_prob=0.72,
        intrinsic_value=10.0,
        intrinsic_ratio=0.833,
        time_value=2.0,
        bsm_fair_value=11.75,
        edge=-0.25,
        expected_value=1.50,
    )
    defaults.update(overrides)
    return CallCandidate(**defaults)


class TestCallThresholdFilters:
    """Tests for apply_call_threshold_filters."""

    def test_passing_candidate_not_rejected(self):
        candidate = _make_candidate()
        config = WeeklyProbabilityFilterConfig()
        apply_call_threshold_filters(candidate, config)
        assert candidate.rejected is False
        assert candidate.reject_reasons == []

    def test_low_breakeven_prob_rejected(self):
        candidate = _make_candidate(breakeven_prob=0.30)
        config = WeeklyProbabilityFilterConfig(min_breakeven_prob=0.45)
        apply_call_threshold_filters(candidate, config)
        assert candidate.rejected is True
        assert any("P_BE" in r for r in candidate.reject_reasons)

    def test_low_intrinsic_ratio_rejected(self):
        candidate = _make_candidate(intrinsic_ratio=0.10)
        config = WeeklyProbabilityFilterConfig(min_intrinsic_ratio=0.20)
        apply_call_threshold_filters(candidate, config)
        assert candidate.rejected is True
        assert any("INTR" in r for r in candidate.reject_reasons)

    def test_low_expected_value_rejected(self):
        candidate = _make_candidate(expected_value=-5.0)
        config = WeeklyProbabilityFilterConfig(min_expected_value=-2.0)
        apply_call_threshold_filters(candidate, config)
        assert candidate.rejected is True
        assert any("EV" in r for r in candidate.reject_reasons)

    def test_multiple_failures_all_logged(self):
        candidate = _make_candidate(
            breakeven_prob=0.20, intrinsic_ratio=0.05, expected_value=-10.0,
        )
        config = WeeklyProbabilityFilterConfig()
        apply_call_threshold_filters(candidate, config)
        assert candidate.rejected is True
        assert len(candidate.reject_reasons) == 3

    def test_exactly_at_threshold_passes(self):
        """Boundary: exactly at threshold should pass."""
        candidate = _make_candidate(
            breakeven_prob=0.45, intrinsic_ratio=0.20, expected_value=-2.0,
        )
        config = WeeklyProbabilityFilterConfig(
            min_breakeven_prob=0.45,
            min_intrinsic_ratio=0.20,
            min_expected_value=-2.0,
        )
        apply_call_threshold_filters(candidate, config)
        assert candidate.rejected is False

    def test_custom_thresholds(self):
        candidate = _make_candidate(breakeven_prob=0.60)
        config = WeeklyProbabilityFilterConfig(min_breakeven_prob=0.65)
        apply_call_threshold_filters(candidate, config)
        assert candidate.rejected is True


# ──────────────────────────────────────────────────────────────────────────────
# Composite scoring
# ──────────────────────────────────────────────────────────────────────────────

class TestCallCompositeScore:
    """Tests for compute_call_composite_score."""

    def test_perfect_candidate_scores_high(self):
        candidate = _make_candidate(
            breakeven_prob=0.90, intrinsic_ratio=0.90, expected_value=5.0,
        )
        config = WeeklyProbabilityFilterConfig()
        score = compute_call_composite_score(candidate, config, iv_rank_value=10.0)
        assert score > 0.70

    def test_poor_candidate_scores_low(self):
        candidate = _make_candidate(
            breakeven_prob=0.10, intrinsic_ratio=0.05, expected_value=-10.0,
        )
        config = WeeklyProbabilityFilterConfig()
        score = compute_call_composite_score(candidate, config, iv_rank_value=95.0)
        assert score < 0.30

    def test_score_in_0_1_range(self):
        candidate = _make_candidate()
        config = WeeklyProbabilityFilterConfig()
        score = compute_call_composite_score(candidate, config)
        assert 0.0 <= score <= 1.0

    def test_iv_rank_affects_score(self):
        candidate = _make_candidate()
        config = WeeklyProbabilityFilterConfig()
        score_low_iv = compute_call_composite_score(candidate, config, iv_rank_value=10.0)
        score_high_iv = compute_call_composite_score(candidate, config, iv_rank_value=90.0)
        assert score_low_iv > score_high_iv

    def test_no_iv_rank_uses_neutral(self):
        """No IV rank → neutral (0.5) for IV component."""
        candidate = _make_candidate()
        config = WeeklyProbabilityFilterConfig()
        score_none = compute_call_composite_score(candidate, config, iv_rank_value=None)
        score_50 = compute_call_composite_score(candidate, config, iv_rank_value=50.0)
        assert score_none == pytest.approx(score_50, abs=0.01)

    def test_higher_breakeven_prob_wins(self):
        config = WeeklyProbabilityFilterConfig()
        c1 = _make_candidate(breakeven_prob=0.40)
        c2 = _make_candidate(breakeven_prob=0.70)
        s1 = compute_call_composite_score(c1, config)
        s2 = compute_call_composite_score(c2, config)
        assert s2 > s1

    def test_higher_intrinsic_ratio_wins(self):
        config = WeeklyProbabilityFilterConfig()
        c1 = _make_candidate(intrinsic_ratio=0.10)
        c2 = _make_candidate(intrinsic_ratio=0.80)
        s1 = compute_call_composite_score(c1, config)
        s2 = compute_call_composite_score(c2, config)
        assert s2 > s1

    def test_zero_premium_uses_fallback_denom(self):
        """premium=0 in EV sigmoid should not crash."""
        candidate = _make_candidate(premium=0.0)
        config = WeeklyProbabilityFilterConfig()
        score = compute_call_composite_score(candidate, config)
        assert math.isfinite(score)


# ──────────────────────────────────────────────────────────────────────────────
# Sigmoid
# ──────────────────────────────────────────────────────────────────────────────

class TestSigmoid:
    """Tests for the _sigmoid helper."""

    def test_zero_returns_half(self):
        assert _sigmoid(0) == pytest.approx(0.5)

    def test_large_positive(self):
        assert _sigmoid(600) == 1.0

    def test_large_negative(self):
        assert _sigmoid(-600) == 0.0

    def test_positive_above_half(self):
        assert _sigmoid(2.0) > 0.5

    def test_negative_below_half(self):
        assert _sigmoid(-2.0) < 0.5


# ──────────────────────────────────────────────────────────────────────────────
# CallCandidate
# ──────────────────────────────────────────────────────────────────────────────

class TestCallCandidate:
    """Tests for CallCandidate dataclass."""

    def test_to_log_dict_has_all_keys(self):
        candidate = _make_candidate()
        d = candidate.to_log_dict()
        expected_keys = [
            "wpf_underlying", "wpf_option_symbol", "wpf_strike",
            "wpf_expiration", "wpf_dte", "wpf_underlying_price",
            "wpf_bid", "wpf_ask", "wpf_mid", "wpf_premium",
            "wpf_bsm_iv", "wpf_breakeven_prob", "wpf_itm_prob",
            "wpf_intrinsic_value", "wpf_intrinsic_ratio", "wpf_time_value",
            "wpf_bsm_fair_value", "wpf_edge", "wpf_expected_value",
            "wpf_delta", "wpf_theta", "wpf_gamma", "wpf_vega",
            "wpf_composite_score", "wpf_rejected", "wpf_reject_reasons",
        ]
        for key in expected_keys:
            assert key in d, f"Missing key: {key}"

    def test_to_log_dict_values_are_rounded(self):
        candidate = _make_candidate(breakeven_prob=0.12345678)
        d = candidate.to_log_dict()
        assert d["wpf_breakeven_prob"] == round(0.12345678, 4)

    def test_wpf_prefix_no_collision(self):
        """All keys start with wpf_ (not pf_)."""
        candidate = _make_candidate()
        d = candidate.to_log_dict()
        for key in d:
            assert key.startswith("wpf_"), f"Key {key} missing wpf_ prefix"


# ──────────────────────────────────────────────────────────────────────────────
# evaluate_call_quality (core evaluation)
# ──────────────────────────────────────────────────────────────────────────────

class TestEvaluateCallQuality:
    """Tests for the core evaluate_call_quality function."""

    def test_valid_itm_call_passes(self):
        """Valid ITM call with good IV should produce a non-rejected candidate."""
        config = WeeklyProbabilityFilterConfig(
            min_breakeven_prob=0.30,
            min_intrinsic_ratio=0.10,
            min_expected_value=-10.0,
        )
        result = evaluate_call_quality(
            underlying="AAPL",
            option_symbol="AAPL260401C00220000",
            strike=220.0,
            premium=15.0,
            underlying_price=230.0,
            expiration="2026-04-01",
            dte=45,
            iv=0.28,
            config=config,
        )
        assert isinstance(result, CallCandidate)
        assert result.rejected is False
        assert result.composite_score > 0
        assert math.isfinite(result.breakeven_prob)
        assert math.isfinite(result.itm_prob)

    def test_otm_call_zero_intrinsic(self):
        """OTM call should have intrinsic_ratio = 0."""
        config = WeeklyProbabilityFilterConfig(min_intrinsic_ratio=0.0)
        result = evaluate_call_quality(
            underlying="AAPL",
            option_symbol="AAPL260401C00240000",
            strike=240.0,
            premium=3.0,
            underlying_price=230.0,
            expiration="2026-04-01",
            dte=45,
            iv=0.28,
            config=config,
        )
        assert result.intrinsic_ratio == pytest.approx(0.0, abs=0.01)

    def test_no_valid_iv_rejected(self):
        """IV=0 and BSM solver also fails → rejected with 'no_valid_iv'."""
        config = WeeklyProbabilityFilterConfig()
        with patch(
            "RubberBand.src.weekly_probability_filter._try_compute_iv",
            return_value=float("nan"),
        ):
            result = evaluate_call_quality(
                underlying="AAPL",
                option_symbol="AAPL260401C00220000",
                strike=220.0,
                premium=15.0,
                underlying_price=230.0,
                expiration="2026-04-01",
                dte=45,
                iv=0.0,
                config=config,
                snapshot=None,
            )
        assert result.rejected is True
        assert "no_valid_iv" in result.reject_reasons

    def test_iv_zero_falls_back_to_bsm_solver(self):
        """IV=0 but BSM solver can compute from premium → uses computed IV."""
        config = WeeklyProbabilityFilterConfig(
            min_breakeven_prob=0.0,
            min_intrinsic_ratio=0.0,
            min_expected_value=-100.0,
        )
        result = evaluate_call_quality(
            underlying="AAPL",
            option_symbol="AAPL260401C00220000",
            strike=220.0,
            premium=15.0,
            underlying_price=230.0,
            expiration="2026-04-01",
            dte=45,
            iv=0.0,
            config=config,
            snapshot=None,
        )
        # BSM solver should have recovered a valid IV from the premium
        assert result.rejected is False
        assert result.bsm_iv > 0.01

    def test_alpaca_iv_used_when_available(self):
        """When snapshot has valid IV, it should be used."""
        config = WeeklyProbabilityFilterConfig(
            min_breakeven_prob=0.0,
            min_intrinsic_ratio=0.0,
            min_expected_value=-100.0,
        )
        result = evaluate_call_quality(
            underlying="AAPL",
            option_symbol="AAPL260401C00220000",
            strike=220.0,
            premium=15.0,
            underlying_price=230.0,
            expiration="2026-04-01",
            dte=45,
            iv=0.35,
            config=config,
        )
        assert result.bsm_iv == pytest.approx(0.35, abs=0.01)

    @patch("RubberBand.src.weekly_probability_filter._try_compute_iv")
    def test_bsm_iv_fallback_when_alpaca_zero(self, mock_compute_iv):
        """When snapshot IV is 0, fall back to BSM compute_iv."""
        mock_compute_iv.return_value = 0.30
        config = WeeklyProbabilityFilterConfig(
            min_breakeven_prob=0.0,
            min_intrinsic_ratio=0.0,
            min_expected_value=-100.0,
        )
        result = evaluate_call_quality(
            underlying="AAPL",
            option_symbol="AAPL260401C00220000",
            strike=220.0,
            premium=15.0,
            underlying_price=230.0,
            expiration="2026-04-01",
            dte=45,
            iv=0.0,
            config=config,
            snapshot={"bid": 14.5, "ask": 15.5, "mid": 15.0, "iv": 0.0},
        )
        mock_compute_iv.assert_called_once()
        assert result.bsm_iv == pytest.approx(0.30, abs=0.01)

    def test_greeks_populated_from_snapshot(self):
        """Greeks should be carried from snapshot dict."""
        config = WeeklyProbabilityFilterConfig(
            min_breakeven_prob=0.0,
            min_intrinsic_ratio=0.0,
            min_expected_value=-100.0,
        )
        snapshot = {
            "bid": 14.5, "ask": 15.5, "mid": 15.0,
            "iv": 0.28, "delta": 0.65, "theta": -0.15,
            "gamma": 0.03, "vega": 0.45,
        }
        result = evaluate_call_quality(
            underlying="AAPL",
            option_symbol="AAPL260401C00220000",
            strike=220.0,
            premium=15.0,
            underlying_price=230.0,
            expiration="2026-04-01",
            dte=45,
            iv=0.28,
            config=config,
            snapshot=snapshot,
        )
        assert result.delta == pytest.approx(0.65, abs=0.01)
        assert result.theta == pytest.approx(-0.15, abs=0.01)
        assert result.gamma == pytest.approx(0.03, abs=0.01)
        assert result.vega == pytest.approx(0.45, abs=0.01)

    def test_zero_premium_rejected(self):
        """Zero premium → BSM returns invalid → rejected."""
        config = WeeklyProbabilityFilterConfig()
        result = evaluate_call_quality(
            underlying="AAPL",
            option_symbol="AAPL260401C00220000",
            strike=220.0,
            premium=0.0,
            underlying_price=230.0,
            expiration="2026-04-01",
            dte=45,
            iv=0.28,
            config=config,
        )
        assert result.rejected is True

    def test_expired_option_rejected(self):
        """DTE=0 → T=0 → BSM invalid → rejected."""
        config = WeeklyProbabilityFilterConfig()
        result = evaluate_call_quality(
            underlying="AAPL",
            option_symbol="AAPL260401C00220000",
            strike=220.0,
            premium=15.0,
            underlying_price=230.0,
            expiration="2026-02-19",
            dte=0,
            iv=0.28,
            config=config,
        )
        assert result.rejected is True

    def test_threshold_rejection_in_evaluate(self):
        """Low breakeven prob triggers rejection via threshold filters."""
        config = WeeklyProbabilityFilterConfig(min_breakeven_prob=0.99)
        result = evaluate_call_quality(
            underlying="AAPL",
            option_symbol="AAPL260401C00220000",
            strike=220.0,
            premium=15.0,
            underlying_price=230.0,
            expiration="2026-04-01",
            dte=45,
            iv=0.28,
            config=config,
        )
        assert result.rejected is True
        assert any("P_BE" in r for r in result.reject_reasons)

    def test_composite_score_populated(self):
        """Composite score should be a finite number > 0."""
        config = WeeklyProbabilityFilterConfig(
            min_breakeven_prob=0.0,
            min_intrinsic_ratio=0.0,
            min_expected_value=-100.0,
        )
        result = evaluate_call_quality(
            underlying="AAPL",
            option_symbol="AAPL260401C00220000",
            strike=220.0,
            premium=15.0,
            underlying_price=230.0,
            expiration="2026-04-01",
            dte=45,
            iv=0.28,
            config=config,
        )
        assert math.isfinite(result.composite_score)
        assert result.composite_score > 0

    def test_snapshot_quotes_override_premium(self):
        """Snapshot bid/ask/mid should populate candidate fields."""
        config = WeeklyProbabilityFilterConfig(
            min_breakeven_prob=0.0,
            min_intrinsic_ratio=0.0,
            min_expected_value=-100.0,
        )
        snapshot = {
            "bid": 14.50, "ask": 15.50, "mid": 15.00,
            "iv": 0.28,
        }
        result = evaluate_call_quality(
            underlying="AAPL",
            option_symbol="AAPL260401C00220000",
            strike=220.0,
            premium=15.50,
            underlying_price=230.0,
            expiration="2026-04-01",
            dte=45,
            iv=0.28,
            config=config,
            snapshot=snapshot,
        )
        assert result.bid == pytest.approx(14.50, abs=0.01)
        assert result.ask == pytest.approx(15.50, abs=0.01)
        assert result.mid == pytest.approx(15.00, abs=0.01)


# ──────────────────────────────────────────────────────────────────────────────
# Fail-open behavior
# ──────────────────────────────────────────────────────────────────────────────

class TestFailOpen:
    """Test that invalid inputs produce rejected candidates, not exceptions."""

    def test_negative_dte_rejected(self):
        config = WeeklyProbabilityFilterConfig()
        result = evaluate_call_quality(
            underlying="AAPL",
            option_symbol="AAPL260401C00220000",
            strike=220.0,
            premium=15.0,
            underlying_price=230.0,
            expiration="2026-04-01",
            dte=-5,
            iv=0.28,
            config=config,
        )
        assert result.rejected is True

    def test_negative_strike_rejected(self):
        config = WeeklyProbabilityFilterConfig()
        result = evaluate_call_quality(
            underlying="AAPL",
            option_symbol="AAPL260401C00220000",
            strike=-220.0,
            premium=15.0,
            underlying_price=230.0,
            expiration="2026-04-01",
            dte=45,
            iv=0.28,
            config=config,
        )
        assert result.rejected is True


# ──────────────────────────────────────────────────────────────────────────────
# IV rank enforcement
# ──────────────────────────────────────────────────────────────────────────────

class TestIVRankEnforcement:
    """Tests that max_iv_rank is enforced as a hard gate."""

    def test_iv_rank_above_max_rejected(self):
        """IV rank above max_iv_rank should reject the candidate."""
        config = WeeklyProbabilityFilterConfig(
            max_iv_rank=80.0,
            min_breakeven_prob=0.0,
            min_intrinsic_ratio=0.0,
            min_expected_value=-100.0,
        )
        result = evaluate_call_quality(
            underlying="AAPL",
            option_symbol="AAPL260401C00220000",
            strike=220.0,
            premium=15.0,
            underlying_price=230.0,
            expiration="2026-04-01",
            dte=45,
            iv=0.28,
            config=config,
            iv_rank_value=85.0,
        )
        assert result.rejected is True
        assert any("IV_RANK" in r for r in result.reject_reasons)

    def test_iv_rank_below_max_passes(self):
        """IV rank below max_iv_rank should not trigger rejection."""
        config = WeeklyProbabilityFilterConfig(
            max_iv_rank=80.0,
            min_breakeven_prob=0.0,
            min_intrinsic_ratio=0.0,
            min_expected_value=-100.0,
        )
        result = evaluate_call_quality(
            underlying="AAPL",
            option_symbol="AAPL260401C00220000",
            strike=220.0,
            premium=15.0,
            underlying_price=230.0,
            expiration="2026-04-01",
            dte=45,
            iv=0.28,
            config=config,
            iv_rank_value=50.0,
        )
        assert result.rejected is False

    def test_iv_rank_none_no_rejection(self):
        """No IV rank → no rejection from IV rank check."""
        config = WeeklyProbabilityFilterConfig(
            max_iv_rank=80.0,
            min_breakeven_prob=0.0,
            min_intrinsic_ratio=0.0,
            min_expected_value=-100.0,
        )
        result = evaluate_call_quality(
            underlying="AAPL",
            option_symbol="AAPL260401C00220000",
            strike=220.0,
            premium=15.0,
            underlying_price=230.0,
            expiration="2026-04-01",
            dte=45,
            iv=0.28,
            config=config,
            iv_rank_value=None,
        )
        assert result.rejected is False
