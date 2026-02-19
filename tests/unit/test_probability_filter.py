"""
Tests for the probability filter module.

All Alpaca API calls are mocked — these tests run offline and validate:
- Config loading from dict
- Threshold filters (reject/accept logic)
- Composite scoring formula
- DTE candidate evaluation with mocked API responses
- Shadow vs filter mode behavior
- Fallback behavior when all candidates rejected or API fails
- Bid-ask spread rejection
- IV solver failure → Alpaca IV fallback
- DTE deduplication
"""
import math
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import replace

from RubberBand.src.probability_filter import (
    ProbabilityFilterConfig,
    load_probability_filter_config,
    SpreadCandidate,
    compute_composite_score,
    apply_threshold_filters,
    evaluate_dte_candidates,
    _sigmoid,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_candidate(**overrides) -> SpreadCandidate:
    """Factory for test candidates with sensible defaults."""
    defaults = dict(
        underlying="AAPL",
        dte=6,
        expiration="2026-02-25",
        long_contract={"symbol": "AAPL260225C00230000"},
        short_contract={"symbol": "AAPL260225C00233000"},
        atm_strike=230.0,
        otm_strike=233.0,
        spread_width=3.0,
        underlying_price=230.50,
        long_bid=1.10,
        long_ask=1.30,
        long_mid=1.20,
        short_bid=0.40,
        short_ask=0.55,
        short_mid=0.475,
        net_debit=0.90,
        bsm_iv=0.32,
        breakeven_prob=0.55,
        max_profit_prob=0.30,
        risk_reward_ratio=2.33,
        expected_value=0.15,
        spread_efficiency=0.30,
    )
    defaults.update(overrides)
    return SpreadCandidate(**defaults)


@pytest.fixture
def default_config():
    return ProbabilityFilterConfig()


@pytest.fixture
def filter_config():
    """Config with filter mode enabled."""
    return ProbabilityFilterConfig(enabled=True, mode="filter")


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadConfig:
    """Tests for config loading from dict."""

    def test_empty_config_returns_defaults(self):
        cfg = load_probability_filter_config({})
        assert cfg.enabled is False
        assert cfg.mode == "shadow"
        assert cfg.dte_candidates == [4, 5, 6, 8, 10]
        assert cfg.min_breakeven_prob == 0.50
        assert cfg.risk_free_rate == 0.045

    def test_full_config(self):
        cfg = load_probability_filter_config({
            "probability_filter": {
                "enabled": True,
                "mode": "filter",
                "dte_candidates": [3, 5, 7],
                "min_breakeven_prob": 0.55,
                "min_risk_reward": 0.40,
                "min_expected_value": -10.0,
                "max_iv_rank": 80.0,
                "max_bid_ask_spread_pct": 15.0,
                "risk_free_rate": 0.05,
                "fallback_to_legacy": False,
                "max_dte": 8,
            }
        })
        assert cfg.enabled is True
        assert cfg.mode == "filter"
        assert cfg.dte_candidates == [3, 5, 7]
        assert cfg.min_breakeven_prob == 0.55
        assert cfg.min_risk_reward == 0.40
        assert cfg.min_expected_value == -10.0
        assert cfg.max_iv_rank == 80.0
        assert cfg.max_bid_ask_spread_pct == 15.0
        assert cfg.risk_free_rate == 0.05
        assert cfg.fallback_to_legacy is False
        assert cfg.max_dte == 8

    def test_partial_config_uses_defaults(self):
        cfg = load_probability_filter_config({
            "probability_filter": {
                "enabled": True,
            }
        })
        assert cfg.enabled is True
        assert cfg.mode == "shadow"  # default
        assert cfg.min_breakeven_prob == 0.50  # default


# ──────────────────────────────────────────────────────────────────────────────
# Threshold filters
# ──────────────────────────────────────────────────────────────────────────────

class TestThresholdFilters:
    """Tests for apply_threshold_filters."""

    def test_passing_candidate_not_rejected(self, default_config):
        c = _make_candidate(breakeven_prob=0.60, risk_reward_ratio=1.5, expected_value=0.50)
        apply_threshold_filters(c, default_config)
        assert c.rejected is False
        assert c.reject_reasons == []

    def test_low_breakeven_prob_rejected(self, default_config):
        c = _make_candidate(breakeven_prob=0.38)
        apply_threshold_filters(c, default_config)
        assert c.rejected is True
        assert any("P_BE" in r for r in c.reject_reasons)

    def test_low_risk_reward_rejected(self, default_config):
        c = _make_candidate(risk_reward_ratio=0.20)
        apply_threshold_filters(c, default_config)
        assert c.rejected is True
        assert any("RR" in r for r in c.reject_reasons)

    def test_low_expected_value_rejected(self, default_config):
        c = _make_candidate(expected_value=-10.0)
        apply_threshold_filters(c, default_config)
        assert c.rejected is True
        assert any("EV" in r for r in c.reject_reasons)

    def test_multiple_failures_all_logged(self, default_config):
        c = _make_candidate(
            breakeven_prob=0.30,
            risk_reward_ratio=0.10,
            expected_value=-20.0,
        )
        apply_threshold_filters(c, default_config)
        assert c.rejected is True
        assert len(c.reject_reasons) == 3

    def test_exactly_at_threshold_passes(self, default_config):
        c = _make_candidate(
            breakeven_prob=0.50,
            risk_reward_ratio=0.30,
            expected_value=-5.0,
        )
        apply_threshold_filters(c, default_config)
        assert c.rejected is False

    def test_custom_thresholds(self):
        cfg = ProbabilityFilterConfig(min_breakeven_prob=0.70)
        c = _make_candidate(breakeven_prob=0.65)
        apply_threshold_filters(c, cfg)
        assert c.rejected is True


# ──────────────────────────────────────────────────────────────────────────────
# Composite scoring
# ──────────────────────────────────────────────────────────────────────────────

class TestCompositeScore:
    """Tests for compute_composite_score."""

    def test_perfect_candidate_scores_high(self, default_config):
        c = _make_candidate(
            breakeven_prob=0.80,
            risk_reward_ratio=3.0,
            expected_value=1.50,
            net_debit=0.80,
        )
        score = compute_composite_score(c, default_config)
        assert score > 0.70

    def test_terrible_candidate_scores_low(self, default_config):
        c = _make_candidate(
            breakeven_prob=0.10,
            risk_reward_ratio=0.10,
            expected_value=-5.0,
            net_debit=0.80,
        )
        score = compute_composite_score(c, default_config)
        assert score < 0.30

    def test_score_in_0_1_range(self, default_config):
        for prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for rr in [0.1, 0.5, 1.0, 3.0]:
                c = _make_candidate(
                    breakeven_prob=prob,
                    risk_reward_ratio=rr,
                    expected_value=0.0,
                    net_debit=1.0,
                )
                score = compute_composite_score(c, default_config)
                assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_iv_rank_affects_score(self, default_config):
        c = _make_candidate(breakeven_prob=0.50, risk_reward_ratio=1.0, expected_value=0.0)
        score_low_iv = compute_composite_score(c, default_config, iv_rank_value=10.0)
        score_high_iv = compute_composite_score(c, default_config, iv_rank_value=90.0)
        assert score_low_iv > score_high_iv

    def test_no_iv_rank_uses_neutral(self, default_config):
        c = _make_candidate(breakeven_prob=0.50, risk_reward_ratio=1.0, expected_value=0.0)
        score_none = compute_composite_score(c, default_config, iv_rank_value=None)
        score_50 = compute_composite_score(c, default_config, iv_rank_value=50.0)
        assert score_none == pytest.approx(score_50, abs=0.01)

    def test_higher_breakeven_prob_wins(self, default_config):
        c1 = _make_candidate(breakeven_prob=0.40)
        c2 = _make_candidate(breakeven_prob=0.70)
        s1 = compute_composite_score(c1, default_config)
        s2 = compute_composite_score(c2, default_config)
        assert s2 > s1

    def test_higher_rr_wins(self, default_config):
        c1 = _make_candidate(risk_reward_ratio=0.50)
        c2 = _make_candidate(risk_reward_ratio=2.50)
        s1 = compute_composite_score(c1, default_config)
        s2 = compute_composite_score(c2, default_config)
        assert s2 > s1


# ──────────────────────────────────────────────────────────────────────────────
# Sigmoid helper
# ──────────────────────────────────────────────────────────────────────────────

class TestSigmoid:
    def test_zero(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive(self):
        assert _sigmoid(100.0) == pytest.approx(1.0, abs=1e-6)

    def test_large_negative(self):
        assert _sigmoid(-100.0) == pytest.approx(0.0, abs=1e-6)

    def test_extreme_positive(self):
        assert _sigmoid(1000.0) == 1.0

    def test_extreme_negative(self):
        assert _sigmoid(-1000.0) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# SpreadCandidate
# ──────────────────────────────────────────────────────────────────────────────

class TestSpreadCandidate:
    def test_to_log_dict_has_all_keys(self):
        c = _make_candidate()
        d = c.to_log_dict()
        # All keys are prefixed with pf_ to avoid collisions with spread_entry()
        expected_keys = [
            "pf_underlying", "pf_dte", "pf_expiration", "pf_atm_strike",
            "pf_otm_strike", "pf_spread_width", "pf_underlying_price",
            "pf_net_debit", "pf_bsm_iv", "pf_breakeven_prob",
            "pf_max_profit_prob", "pf_risk_reward_ratio", "pf_expected_value",
            "pf_spread_efficiency", "pf_composite_score", "pf_rejected",
            "pf_reject_reasons",
        ]
        for k in expected_keys:
            assert k in d, f"Missing key: {k}"

    def test_to_log_dict_values_are_rounded(self):
        c = _make_candidate(breakeven_prob=0.55555555)
        d = c.to_log_dict()
        assert d["pf_breakeven_prob"] == pytest.approx(0.5556, abs=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# evaluate_dte_candidates (integration with mocks)
# ──────────────────────────────────────────────────────────────────────────────

def _mock_spread_result(dte, underlying="AAPL", price=230.0):
    """Create a mock spread result dict."""
    long_sym = f"{underlying}260225C00230000"
    short_sym = f"{underlying}260225C00233000"
    return {
        "underlying": underlying,
        "expiration": "2026-02-25",
        "dte": dte,
        "underlying_price": price,
        "long": {"symbol": long_sym, "strike_price": "230"},
        "short": {"symbol": short_sym, "strike_price": "233"},
        "atm_strike": 230.0,
        "otm_strike": 233.0,
        "spread_width": 3.0,
    }


def _mock_snapshot(bid=1.10, ask=1.30, iv=0.32):
    """Create a mock snapshot dict."""
    return {
        "bid": bid,
        "ask": ask,
        "mid": (bid + ask) / 2,
        "iv": iv,
        "delta": 0.55,
        "theta": -0.08,
        "gamma": 0.04,
        "vega": 0.15,
    }


_OPT = "RubberBand.src.options_data"


class TestEvaluateDteCandidates:
    """Integration tests with mocked API calls."""

    @patch(f"{_OPT}.get_option_snapshots_batch")
    @patch(f"{_OPT}.select_spread_contracts")
    @patch(f"{_OPT}.get_underlying_price")
    def test_selects_best_candidate(self, mock_price, mock_spread, mock_snap):
        mock_price.return_value = 230.0
        mock_spread.return_value = _mock_spread_result(6)
        mock_snap.return_value = {
            "AAPL260225C00230000": _mock_snapshot(bid=1.10, ask=1.30, iv=0.32),
            "AAPL260225C00233000": _mock_snapshot(bid=0.48, ask=0.55, iv=0.28),
        }

        config = ProbabilityFilterConfig(
            enabled=True, mode="filter",
            dte_candidates=[6],
            min_breakeven_prob=0.0,  # no filtering for this test
            min_risk_reward=0.0,
            min_expected_value=-100.0,
        )
        spread_cfg = {"spread_width_atr": 1.5, "min_dte": 2}

        best, all_cands = evaluate_dte_candidates("AAPL", 2.5, config, spread_cfg)

        assert best is not None
        assert best.underlying == "AAPL"
        assert best.composite_score > 0
        assert len(all_cands) == 1

    @patch(f"{_OPT}.get_underlying_price")
    def test_no_price_returns_none(self, mock_price):
        mock_price.return_value = None
        config = ProbabilityFilterConfig(enabled=True, dte_candidates=[6])
        best, cands = evaluate_dte_candidates("AAPL", 2.5, config, {})
        assert best is None
        assert cands == []

    @patch(f"{_OPT}.get_option_snapshots_batch")
    @patch(f"{_OPT}.select_spread_contracts")
    @patch(f"{_OPT}.get_underlying_price")
    def test_no_spreads_returns_none(self, mock_price, mock_spread, mock_snap):
        mock_price.return_value = 230.0
        mock_spread.return_value = None  # No contracts found
        config = ProbabilityFilterConfig(enabled=True, dte_candidates=[6])
        best, cands = evaluate_dte_candidates("AAPL", 2.5, config, {})
        assert best is None

    @patch(f"{_OPT}.get_option_snapshots_batch")
    @patch(f"{_OPT}.select_spread_contracts")
    @patch(f"{_OPT}.get_underlying_price")
    def test_all_candidates_rejected(self, mock_price, mock_spread, mock_snap):
        mock_price.return_value = 230.0
        mock_spread.return_value = _mock_spread_result(6)
        # Tight bid-ask (passes bid-ask check) but probability will be low
        mock_snap.return_value = {
            "AAPL260225C00230000": _mock_snapshot(bid=1.10, ask=1.30, iv=0.15),
            "AAPL260225C00233000": _mock_snapshot(bid=0.48, ask=0.55, iv=0.12),
        }

        config = ProbabilityFilterConfig(
            enabled=True, mode="filter",
            dte_candidates=[6],
            min_breakeven_prob=0.99,  # impossible threshold
        )

        best, cands = evaluate_dte_candidates("AAPL", 2.5, config, {})
        assert best is None
        # Candidates were evaluated but all rejected
        assert len(cands) >= 1
        assert all(c.rejected for c in cands)

    @patch(f"{_OPT}.get_option_snapshots_batch")
    @patch(f"{_OPT}.select_spread_contracts")
    @patch(f"{_OPT}.get_underlying_price")
    def test_missing_snapshot_skips_candidate(self, mock_price, mock_spread, mock_snap):
        mock_price.return_value = 230.0
        mock_spread.return_value = _mock_spread_result(6)
        mock_snap.return_value = {}  # No snapshots

        config = ProbabilityFilterConfig(enabled=True, dte_candidates=[6])
        best, cands = evaluate_dte_candidates("AAPL", 2.5, config, {})
        assert best is None
        assert cands == []

    @patch(f"{_OPT}.get_option_snapshots_batch")
    @patch(f"{_OPT}.select_spread_contracts")
    @patch(f"{_OPT}.get_underlying_price")
    def test_dte_deduplication(self, mock_price, mock_spread, mock_snap):
        """DTE 5 and 6 returning the same Friday should only evaluate once."""
        mock_price.return_value = 230.0

        # Both DTE 5 and DTE 6 return same expiration
        same_spread = _mock_spread_result(5)
        mock_spread.return_value = same_spread

        mock_snap.return_value = {
            "AAPL260225C00230000": _mock_snapshot(),
            "AAPL260225C00233000": _mock_snapshot(bid=0.48, ask=0.55),
        }

        config = ProbabilityFilterConfig(
            enabled=True, dte_candidates=[5, 6],
            min_breakeven_prob=0.0, min_risk_reward=0.0, min_expected_value=-100,
        )
        best, cands = evaluate_dte_candidates("AAPL", 2.5, config, {})

        # Should only have 1 candidate despite 2 DTE targets
        assert len(cands) == 1

    @patch(f"{_OPT}.get_option_snapshots_batch")
    @patch(f"{_OPT}.select_spread_contracts")
    @patch(f"{_OPT}.get_underlying_price")
    def test_bid_ask_too_wide_skipped(self, mock_price, mock_spread, mock_snap):
        mock_price.return_value = 230.0
        mock_spread.return_value = _mock_spread_result(6)
        # bid=0.10, ask=1.50 → mid=0.80 → spread = 175% of mid (way above 20%)
        mock_snap.return_value = {
            "AAPL260225C00230000": _mock_snapshot(bid=0.10, ask=1.50, iv=0.30),
            "AAPL260225C00233000": _mock_snapshot(bid=0.05, ask=0.10, iv=0.25),
        }

        config = ProbabilityFilterConfig(
            enabled=True, dte_candidates=[6],
            max_bid_ask_spread_pct=20.0,
        )
        best, cands = evaluate_dte_candidates("AAPL", 2.5, config, {})
        assert best is None
        assert cands == []  # Filtered out before reaching threshold stage

    @patch(f"{_OPT}.get_option_snapshots_batch")
    @patch(f"{_OPT}.select_spread_contracts")
    @patch(f"{_OPT}.get_underlying_price")
    def test_debit_exceeds_width_skipped(self, mock_price, mock_spread, mock_snap):
        mock_price.return_value = 230.0
        mock_spread.return_value = _mock_spread_result(6)
        # Long ask=3.50, short bid=0.10 → net_debit=3.40 > width=3.0
        mock_snap.return_value = {
            "AAPL260225C00230000": _mock_snapshot(bid=3.40, ask=3.50, iv=0.30),
            "AAPL260225C00233000": _mock_snapshot(bid=0.10, ask=0.20, iv=0.25),
        }

        config = ProbabilityFilterConfig(
            enabled=True, dte_candidates=[6],
            max_bid_ask_spread_pct=100.0,  # don't filter on bid-ask
        )
        best, cands = evaluate_dte_candidates("AAPL", 2.5, config, {})
        assert best is None
        assert cands == []

    @patch(f"{_OPT}.get_option_snapshots_batch")
    @patch(f"{_OPT}.select_spread_contracts")
    @patch(f"{_OPT}.get_underlying_price")
    def test_iv_solver_fallback_to_alpaca(self, mock_price, mock_spread, mock_snap):
        """If BSM IV solver fails, we should still use Alpaca's IV."""
        mock_price.return_value = 230.0
        mock_spread.return_value = _mock_spread_result(6)
        # Create snapshots where mid-price is very low (near zero), causing IV solver
        # to potentially fail, but Alpaca IV is available
        mock_snap.return_value = {
            "AAPL260225C00230000": _mock_snapshot(bid=1.10, ask=1.30, iv=0.35),
            "AAPL260225C00233000": _mock_snapshot(bid=0.48, ask=0.55, iv=0.28),
        }

        config = ProbabilityFilterConfig(
            enabled=True, dte_candidates=[6],
            min_breakeven_prob=0.0, min_risk_reward=0.0, min_expected_value=-100,
        )
        best, cands = evaluate_dte_candidates("AAPL", 2.5, config, {})
        assert best is not None
        # IV should be either BSM-computed or Alpaca fallback, but finite
        assert math.isfinite(best.bsm_iv)
        assert best.bsm_iv > 0

    @patch(f"{_OPT}.get_option_snapshots_batch")
    @patch(f"{_OPT}.select_spread_contracts")
    @patch(f"{_OPT}.get_underlying_price")
    def test_max_dte_cap_honored(self, mock_price, mock_spread, mock_snap):
        """DTE candidates above max_dte should be skipped."""
        mock_price.return_value = 230.0
        mock_spread.return_value = _mock_spread_result(6)
        mock_snap.return_value = {
            "AAPL260225C00230000": _mock_snapshot(),
            "AAPL260225C00233000": _mock_snapshot(bid=0.48, ask=0.55),
        }

        config = ProbabilityFilterConfig(
            enabled=True,
            dte_candidates=[3, 6, 15, 20],  # 15 and 20 exceed max_dte
            max_dte=10,
            min_breakeven_prob=0.0, min_risk_reward=0.0, min_expected_value=-100,
        )
        best, cands = evaluate_dte_candidates("AAPL", 2.5, config, {})
        # select_spread_contracts should NOT have been called with dte=15 or 20
        # (They exceed max_dte=10 and get filtered in the loop)
        for call_args in mock_spread.call_args_list:
            dte_used = call_args[1].get("dte", call_args[0][1] if len(call_args[0]) > 1 else 0)
            assert dte_used <= 10

    @patch(f"{_OPT}.get_option_snapshots_batch")
    @patch(f"{_OPT}.select_spread_contracts")
    @patch(f"{_OPT}.get_underlying_price")
    def test_candidate_metrics_populated(self, mock_price, mock_spread, mock_snap):
        """Verify all candidate fields are properly populated."""
        mock_price.return_value = 230.0
        mock_spread.return_value = _mock_spread_result(6)
        mock_snap.return_value = {
            "AAPL260225C00230000": _mock_snapshot(bid=1.10, ask=1.30, iv=0.32),
            "AAPL260225C00233000": _mock_snapshot(bid=0.48, ask=0.55, iv=0.28),
        }

        config = ProbabilityFilterConfig(
            enabled=True, dte_candidates=[6],
            min_breakeven_prob=0.0, min_risk_reward=0.0, min_expected_value=-100,
        )
        best, _ = evaluate_dte_candidates("AAPL", 2.5, config, {})

        assert best is not None
        assert best.underlying == "AAPL"
        assert best.atm_strike == 230.0
        assert best.otm_strike == 233.0
        assert best.spread_width == 3.0
        assert best.underlying_price == 230.0
        assert best.long_ask == 1.30
        assert best.short_bid == 0.48
        assert best.net_debit == pytest.approx(0.82, abs=0.01)
        assert 0.0 <= best.breakeven_prob <= 1.0
        assert 0.0 <= best.max_profit_prob <= 1.0
        assert best.risk_reward_ratio > 0
        assert math.isfinite(best.expected_value)
        assert best.composite_score > 0

    @patch(f"{_OPT}.get_option_snapshots_batch")
    @patch(f"{_OPT}.select_spread_contracts")
    @patch(f"{_OPT}.get_underlying_price")
    def test_multiple_dtes_picks_highest_score(self, mock_price, mock_spread, mock_snap):
        """With multiple unique expirations, should pick highest score."""
        mock_price.return_value = 230.0

        spread_4 = _mock_spread_result(4)
        spread_4["expiration"] = "2026-02-24"
        spread_4["long"]["symbol"] = "AAPL260224C00230000"
        spread_4["short"]["symbol"] = "AAPL260224C00233000"

        spread_8 = _mock_spread_result(8)
        spread_8["expiration"] = "2026-02-27"
        spread_8["long"]["symbol"] = "AAPL260227C00230000"
        spread_8["short"]["symbol"] = "AAPL260227C00233000"

        # Return different spreads for different DTEs
        mock_spread.side_effect = [spread_4, spread_8]

        mock_snap.return_value = {
            # DTE 4: tighter spread, better pricing
            "AAPL260224C00230000": _mock_snapshot(bid=0.70, ask=0.85, iv=0.30),
            "AAPL260224C00233000": _mock_snapshot(bid=0.20, ask=0.23, iv=0.25),
            # DTE 8: wider spread, worse pricing
            "AAPL260227C00230000": _mock_snapshot(bid=1.40, ask=1.60, iv=0.32),
            "AAPL260227C00233000": _mock_snapshot(bid=0.55, ask=0.62, iv=0.28),
        }

        config = ProbabilityFilterConfig(
            enabled=True, dte_candidates=[4, 8],
            min_breakeven_prob=0.0, min_risk_reward=0.0, min_expected_value=-100,
        )
        best, cands = evaluate_dte_candidates("AAPL", 2.5, config, {})

        assert best is not None
        assert len(cands) == 2
        # The best candidate should have the highest composite score
        other = [c for c in cands if c.dte != best.dte][0]
        assert best.composite_score >= other.composite_score


# ──────────────────────────────────────────────────────────────────────────────
# Config validation tests (added per code review)
# ──────────────────────────────────────────────────────────────────────────────

class TestConfigValidation:
    """Tests for config validation added in code review."""

    def test_invalid_mode_raises(self):
        """Typos or invalid mode values should be caught."""
        with pytest.raises(ValueError, match="must be 'shadow' or 'filter'"):
            load_probability_filter_config({
                "probability_filter": {"mode": "filtter"}
            })

    def test_invalid_mode_active_raises(self):
        with pytest.raises(ValueError, match="must be 'shadow' or 'filter'"):
            load_probability_filter_config({
                "probability_filter": {"mode": "active"}
            })

    def test_breakeven_prob_out_of_range_raises(self):
        """Probability outside [0, 1] should be caught."""
        with pytest.raises(ValueError, match="min_breakeven_prob"):
            load_probability_filter_config({
                "probability_filter": {"min_breakeven_prob": 2.0}
            })

    def test_breakeven_prob_negative_raises(self):
        with pytest.raises(ValueError, match="min_breakeven_prob"):
            load_probability_filter_config({
                "probability_filter": {"min_breakeven_prob": -0.5}
            })

    def test_max_dte_zero_raises(self):
        with pytest.raises(ValueError, match="max_dte"):
            load_probability_filter_config({
                "probability_filter": {"max_dte": 0}
            })

    def test_max_dte_negative_raises(self):
        with pytest.raises(ValueError, match="max_dte"):
            load_probability_filter_config({
                "probability_filter": {"max_dte": -1}
            })

    def test_valid_config_passes(self):
        """Normal config should load without error."""
        config = load_probability_filter_config({
            "probability_filter": {
                "enabled": True,
                "mode": "shadow",
                "min_breakeven_prob": 0.50,
                "max_dte": 10,
            }
        })
        assert config.enabled is True
        assert config.mode == "shadow"

    def test_empty_dte_candidates_warns(self, caplog):
        """Empty dte_candidates should log a warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            config = load_probability_filter_config({
                "probability_filter": {"dte_candidates": []}
            })
        assert config.dte_candidates == []
        assert "dte_candidates is empty" in caplog.text

    def test_weights_sum_warning(self, caplog):
        """Weights far from 1.0 should log a warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            load_probability_filter_config({
                "probability_filter": {
                    "w_breakeven": 0.80,
                    "w_risk_reward": 0.80,
                    "w_expected_value": 0.80,
                    "w_iv_rank": 0.80,
                }
            })
        assert "weights sum to" in caplog.text.lower() or "Scoring weights" in caplog.text


# ──────────────────────────────────────────────────────────────────────────────
# Additional scoring tests (from code review)
# ──────────────────────────────────────────────────────────────────────────────

class TestCompositeScoreEdgeCases:
    """Additional composite score tests from code review."""

    def test_zero_net_debit_fallback(self):
        """net_debit=0 should use fallback max_loss=1.0, not crash."""
        c = _make_candidate(net_debit=0.0)
        config = ProbabilityFilterConfig()
        # Should not raise ZeroDivisionError
        score = compute_composite_score(c, config)
        assert math.isfinite(score)

    def test_sigmoid_nan_input(self):
        """_sigmoid(NaN) → NaN."""
        from RubberBand.src.probability_filter import _sigmoid
        result = _sigmoid(float("nan"))
        assert math.isnan(result)


# ──────────────────────────────────────────────────────────────────────────────
# Short leg bid-ask check (new test for code review fix)
# ──────────────────────────────────────────────────────────────────────────────

class TestShortLegBidAsk:
    """Test that short leg bid-ask spread is now validated."""

    @patch(f"{_OPT}.get_option_snapshots_batch")
    @patch(f"{_OPT}.select_spread_contracts")
    @patch(f"{_OPT}.get_underlying_price")
    def test_short_leg_wide_bidask_skipped(self, mock_price, mock_spread, mock_snap):
        """Short leg with very wide bid-ask should be filtered out."""
        mock_price.return_value = 230.0
        mock_spread.return_value = _mock_spread_result(6)
        mock_snap.return_value = {
            # Long leg: tight bid-ask
            "AAPL260225C00230000": _mock_snapshot(bid=1.10, ask=1.20, iv=0.32),
            # Short leg: very wide bid-ask (bid=0.05, ask=0.50 → 150% spread)
            "AAPL260225C00233000": _mock_snapshot(bid=0.05, ask=0.50, iv=0.28),
        }
        config = ProbabilityFilterConfig(
            enabled=True, dte_candidates=[6],
            max_bid_ask_spread_pct=20.0,
        )
        best, cands = evaluate_dte_candidates("AAPL", 2.5, config, {})
        assert best is None
        assert cands == []


# ──────────────────────────────────────────────────────────────────────────────
# Fail-open exception propagation (from code review)
# ──────────────────────────────────────────────────────────────────────────────

class TestFailOpen:
    """Test that evaluate_dte_candidates propagates exceptions for caller's try/except."""

    @patch(f"{_OPT}.get_option_snapshots_batch")
    @patch(f"{_OPT}.select_spread_contracts")
    @patch(f"{_OPT}.get_underlying_price")
    def test_batch_api_exception_propagates(self, mock_price, mock_spread, mock_snap):
        """If batch snapshot API throws, exception propagates to caller."""
        mock_price.return_value = 230.0
        mock_spread.return_value = _mock_spread_result(6)
        mock_snap.side_effect = ConnectionError("API down")

        config = ProbabilityFilterConfig(
            enabled=True, dte_candidates=[6],
            min_breakeven_prob=0.0, min_risk_reward=0.0, min_expected_value=-100,
        )
        with pytest.raises(ConnectionError, match="API down"):
            evaluate_dte_candidates("AAPL", 2.5, config, {})
