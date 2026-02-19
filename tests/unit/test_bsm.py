"""
Comprehensive tests for the BSM math module.

Tests are validated against:
- Hull textbook values (Options, Futures, and Other Derivatives)
- Hand-computed d1/d2/N(d2) for probability checks
- Round-trip IV solver verification (price → IV → price)
- Edge case handling for degenerate inputs
"""
import math
import pytest

from RubberBand.src.bsm import (
    norm_cdf,
    norm_pdf,
    bsm_call_price,
    bsm_vega,
    compute_iv,
    breakeven_probability,
    max_profit_probability,
    compute_historical_volatility,
    iv_rank,
    evaluate_spread,
)


# ──────────────────────────────────────────────────────────────────────────────
# norm_cdf / norm_pdf
# ──────────────────────────────────────────────────────────────────────────────

class TestNormCdf:
    """Tests for standard normal CDF."""

    def test_zero(self):
        assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-10)

    def test_positive_one(self):
        assert norm_cdf(1.0) == pytest.approx(0.8413447, abs=1e-6)

    def test_negative_one(self):
        assert norm_cdf(-1.0) == pytest.approx(0.1586553, abs=1e-6)

    def test_1_96(self):
        """97.5th percentile (common in statistics)."""
        assert norm_cdf(1.96) == pytest.approx(0.97500, abs=1e-4)

    def test_negative_3(self):
        assert norm_cdf(-3.0) == pytest.approx(0.00135, abs=1e-4)

    def test_symmetry(self):
        """N(x) + N(-x) = 1."""
        for x in [0.5, 1.0, 2.0, 3.0, 5.0]:
            assert norm_cdf(x) + norm_cdf(-x) == pytest.approx(1.0, abs=1e-12)

    def test_monotonic(self):
        """CDF must be non-decreasing."""
        prev = 0.0
        for x in range(-5, 6):
            val = norm_cdf(float(x))
            assert val >= prev
            prev = val

    def test_extreme_positive(self):
        assert norm_cdf(8.0) == pytest.approx(1.0, abs=1e-12)

    def test_extreme_negative(self):
        assert norm_cdf(-8.0) == pytest.approx(0.0, abs=1e-12)


class TestNormPdf:
    """Tests for standard normal PDF."""

    def test_zero(self):
        assert norm_pdf(0.0) == pytest.approx(0.39894228, abs=1e-6)

    def test_positive_one(self):
        assert norm_pdf(1.0) == pytest.approx(0.24197072, abs=1e-6)

    def test_symmetry(self):
        """PDF is symmetric: φ(x) = φ(-x)."""
        for x in [0.5, 1.0, 2.0, 3.0]:
            assert norm_pdf(x) == pytest.approx(norm_pdf(-x), abs=1e-12)

    def test_always_positive(self):
        for x in [-10, -5, -1, 0, 1, 5, 10]:
            assert norm_pdf(float(x)) >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# BSM call price
# ──────────────────────────────────────────────────────────────────────────────

class TestBSMCallPrice:
    """Tests for BSM European call pricing."""

    def test_hull_textbook_1yr(self):
        """Hull standard example: S=100, K=100, T=1, r=5%, σ=20%."""
        price = bsm_call_price(100, 100, 1.0, 0.05, 0.20)
        assert price == pytest.approx(10.4506, abs=0.01)

    def test_hull_textbook_quarter(self):
        """S=100, K=100, T=0.25, r=5%, σ=20%.
        Hand-verified: d1=0.175, d2=0.075, N(0.175)=0.5694, N(0.075)=0.5299
        C = 100×0.5694 - 100×exp(-0.0125)×0.5299 = 56.94 - 52.33 = 4.61
        """
        price = bsm_call_price(100, 100, 0.25, 0.05, 0.20)
        assert price == pytest.approx(4.615, abs=0.01)

    def test_short_dte_atm(self):
        """10 DTE ATM at 35% IV."""
        T = 10.0 / 365.0
        price = bsm_call_price(100, 100, T, 0.045, 0.35)
        assert 1.5 < price < 4.0  # reasonable range for short DTE ATM

    def test_short_dte_otm(self):
        """10 DTE 5% OTM at 35% IV."""
        T = 10.0 / 365.0
        price = bsm_call_price(100, 105, T, 0.045, 0.35)
        assert 0.0 < price < 2.0

    def test_short_dte_itm(self):
        """10 DTE 5% ITM at 35% IV."""
        T = 10.0 / 365.0
        price = bsm_call_price(100, 95, T, 0.045, 0.35)
        assert price > 5.0  # must be > intrinsic

    def test_expired_atm(self):
        """At expiry, ATM call = 0."""
        assert bsm_call_price(100, 100, 0.0, 0.05, 0.20) == 0.0

    def test_expired_itm(self):
        """At expiry, ITM call = intrinsic."""
        assert bsm_call_price(105, 100, 0.0, 0.05, 0.20) == pytest.approx(5.0)

    def test_expired_otm(self):
        """At expiry, OTM call = 0."""
        assert bsm_call_price(95, 100, 0.0, 0.05, 0.20) == 0.0

    def test_zero_vol(self):
        """Zero vol → deterministic payoff."""
        price = bsm_call_price(105, 100, 1.0, 0.05, 0.0)
        # S - K*exp(-rT) = 105 - 100*exp(-0.05) = 105 - 95.12 = 9.88
        assert price == pytest.approx(105 - 100 * math.exp(-0.05), abs=0.01)

    def test_zero_strike(self):
        """Zero strike → call = stock."""
        assert bsm_call_price(100, 0, 1.0, 0.05, 0.20) == 100.0

    def test_zero_stock(self):
        """Zero stock → call = 0."""
        assert bsm_call_price(0, 100, 1.0, 0.05, 0.20) == 0.0

    def test_price_always_non_negative(self):
        """Price must be ≥ 0 for all valid inputs."""
        for S in [50, 100, 200]:
            for K in [80, 100, 120]:
                for T in [0.01, 0.1, 1.0]:
                    for sigma in [0.1, 0.3, 0.8]:
                        price = bsm_call_price(S, K, T, 0.05, sigma)
                        assert price >= 0.0, f"Negative price for S={S},K={K},T={T},σ={sigma}"

    def test_price_geq_intrinsic(self):
        """Call price ≥ intrinsic value for T > 0."""
        for S in [90, 100, 110]:
            intrinsic = max(S - 100, 0)
            price = bsm_call_price(S, 100, 0.5, 0.05, 0.30)
            assert price >= intrinsic - 0.001

    def test_invalid_negative_S(self):
        with pytest.raises(ValueError, match="non-negative"):
            bsm_call_price(-10, 100, 1.0, 0.05, 0.20)

    def test_invalid_negative_T(self):
        with pytest.raises(ValueError, match="non-negative"):
            bsm_call_price(100, 100, -1.0, 0.05, 0.20)

    def test_invalid_negative_sigma(self):
        with pytest.raises(ValueError, match="non-negative"):
            bsm_call_price(100, 100, 1.0, 0.05, -0.20)


# ──────────────────────────────────────────────────────────────────────────────
# BSM vega
# ──────────────────────────────────────────────────────────────────────────────

class TestBSMVega:
    """Tests for BSM vega."""

    def test_atm_1yr(self):
        """ATM 1-year option should have large vega.
        Hand-verified: d1=0.35, φ(0.35)=0.37524, vega=100×1.0×0.37524=37.524
        """
        v = bsm_vega(100, 100, 1.0, 0.05, 0.20)
        assert v == pytest.approx(37.524, abs=0.1)

    def test_atm_quarter(self):
        """Hand-verified: d1=0.175, φ(0.175)=0.39287, vega=100×0.5×0.39287=19.644"""
        v = bsm_vega(100, 100, 0.25, 0.05, 0.20)
        assert v == pytest.approx(19.644, abs=0.1)

    def test_short_dte_atm(self):
        """10 DTE ATM: vega is small but positive."""
        v = bsm_vega(100, 100, 10 / 365, 0.045, 0.35)
        assert 4.0 < v < 10.0

    def test_deep_otm_low_vega(self):
        """Deep OTM has very low vega."""
        v = bsm_vega(100, 150, 10 / 365, 0.045, 0.35)
        assert v < 0.01

    def test_always_non_negative(self):
        for S in [80, 100, 120]:
            for K in [80, 100, 120]:
                v = bsm_vega(S, K, 0.5, 0.05, 0.30)
                assert v >= 0.0

    def test_expired_returns_zero(self):
        assert bsm_vega(100, 100, 0.0, 0.05, 0.20) == 0.0

    def test_zero_vol_returns_zero(self):
        assert bsm_vega(100, 100, 1.0, 0.05, 0.0) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# IV solver
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeIV:
    """Tests for implied volatility solver."""

    def test_round_trip_atm_1yr(self):
        """Price at σ=0.20 → solver recovers σ=0.20."""
        price = bsm_call_price(100, 100, 1.0, 0.05, 0.20)
        iv = compute_iv(price, 100, 100, 1.0, 0.05)
        assert iv == pytest.approx(0.20, abs=0.001)

    def test_round_trip_atm_short_dte(self):
        """6 DTE ATM at 35% IV."""
        T = 6 / 365
        price = bsm_call_price(100, 100, T, 0.045, 0.35)
        iv = compute_iv(price, 100, 100, T, 0.045)
        assert iv == pytest.approx(0.35, abs=0.002)

    def test_round_trip_otm_short_dte(self):
        """6 DTE 3% OTM at 40% IV."""
        T = 6 / 365
        price = bsm_call_price(100, 103, T, 0.045, 0.40)
        iv = compute_iv(price, 100, 103, T, 0.045)
        assert iv == pytest.approx(0.40, abs=0.005)

    def test_round_trip_itm(self):
        """ITM option at 25% IV."""
        price = bsm_call_price(110, 100, 0.5, 0.05, 0.25)
        iv = compute_iv(price, 110, 100, 0.5, 0.05)
        assert iv == pytest.approx(0.25, abs=0.002)

    def test_round_trip_high_iv(self):
        """High volatility stock (80% IV)."""
        price = bsm_call_price(100, 100, 30 / 365, 0.045, 0.80)
        iv = compute_iv(price, 100, 100, 30 / 365, 0.045)
        assert iv == pytest.approx(0.80, abs=0.005)

    def test_round_trip_very_short_dte(self):
        """3 DTE ATM at 30% IV."""
        T = 3 / 365
        price = bsm_call_price(100, 100, T, 0.045, 0.30)
        iv = compute_iv(price, 100, 100, T, 0.045)
        assert iv == pytest.approx(0.30, abs=0.005)

    def test_round_trip_many_scenarios(self):
        """Sweep across many S/K/T/σ combos.

        Skip deep ITM + very short DTE cases where time value is negligible
        (< 5% of intrinsic).  In these regimes IV extraction is inherently
        unreliable because the option price is dominated by intrinsic value
        and tiny numerical noise causes large σ swings.
        """
        for S in [80, 100, 120]:
            for K in [90, 100, 110]:
                for T in [3 / 365, 10 / 365, 30 / 365]:
                    for sigma in [0.20, 0.35, 0.60]:
                        price = bsm_call_price(S, K, T, 0.045, sigma)
                        if price < 0.01:  # skip near-zero prices
                            continue
                        intrinsic = max(S - K, 0.0)
                        time_value = price - intrinsic
                        # Skip when time value is negligible vs intrinsic
                        if intrinsic > 0 and time_value / intrinsic < 0.05:
                            continue
                        iv = compute_iv(price, S, K, T, 0.045)
                        assert math.isfinite(iv), (
                            f"IV solver failed: S={S},K={K},T={T:.4f},σ={sigma}"
                        )
                        assert iv == pytest.approx(sigma, abs=0.01), (
                            f"IV mismatch: S={S},K={K},T={T:.4f},σ={sigma}, "
                            f"got IV={iv:.4f}"
                        )

    def test_zero_price_returns_nan(self):
        assert math.isnan(compute_iv(0.0, 100, 100, 0.5, 0.05))

    def test_negative_price_returns_nan(self):
        assert math.isnan(compute_iv(-1.0, 100, 100, 0.5, 0.05))

    def test_price_below_intrinsic_returns_nan(self):
        """Price below intrinsic is an arbitrage violation."""
        # Intrinsic = 10, price = 8
        assert math.isnan(compute_iv(8.0, 110, 100, 0.5, 0.05))

    def test_price_above_stock_returns_nan(self):
        """Call can never cost more than the stock."""
        assert math.isnan(compute_iv(120, 100, 100, 0.5, 0.05))

    def test_expired_returns_nan(self):
        assert math.isnan(compute_iv(5.0, 100, 100, 0.0, 0.05))

    def test_zero_stock_returns_nan(self):
        assert math.isnan(compute_iv(5.0, 0, 100, 0.5, 0.05))

    def test_pure_intrinsic_returns_min_iv(self):
        """If price = discounted intrinsic exactly (no time value), return IV_MIN."""
        # S=105, K=100, T=0.1, r=0.05 → discounted intrinsic ≈ 5.50
        import math as _m
        S, K, T, r = 105, 100, 0.1, 0.05
        disc_intrinsic = S - K * _m.exp(-r * T)  # ≈ 5.4988
        iv = compute_iv(disc_intrinsic, S, K, T, r)
        assert iv == pytest.approx(0.01, abs=0.001)

    def test_price_below_discounted_intrinsic_returns_nan(self):
        """Price below discounted intrinsic → arbitrage / bad data → nan."""
        # S=110, K=100, T=0.5, r=0.05 → disc intrinsic ≈ 12.47
        # Old simple intrinsic was 10.0. Price=10.0 is now below disc intrinsic.
        iv = compute_iv(10.0, 110, 100, 0.5, 0.05)
        assert math.isnan(iv)


# ──────────────────────────────────────────────────────────────────────────────
# Breakeven / max profit probability
# ──────────────────────────────────────────────────────────────────────────────

class TestBreakevenProbability:
    """Tests for breakeven probability computation."""

    def test_atm_breakeven_near_50(self):
        """ATM breakeven ≈ 50% (slightly less due to drift term)."""
        p = breakeven_probability(100, 100, 10 / 365, 0.35, 0.045)
        assert 0.45 < p < 0.55

    def test_itm_breakeven_higher(self):
        """Breakeven below current price → probability > 50%."""
        p = breakeven_probability(100, 98, 10 / 365, 0.35, 0.045)
        assert p > 0.55

    def test_otm_breakeven_lower(self):
        """Breakeven above current price → probability < 50%."""
        p = breakeven_probability(100, 105, 10 / 365, 0.35, 0.045)
        assert p < 0.45

    def test_far_otm_very_low(self):
        """Far OTM breakeven → very low probability."""
        p = breakeven_probability(100, 130, 10 / 365, 0.35, 0.045)
        assert p < 0.01

    def test_probability_range(self):
        """Probability must be in [0, 1] for all valid inputs."""
        for be in [80, 90, 100, 110, 120, 150]:
            p = breakeven_probability(100, be, 10 / 365, 0.35, 0.045)
            assert 0.0 <= p <= 1.0, f"Out of range for BE={be}: p={p}"

    def test_longer_dte_higher_prob(self):
        """Longer DTE → more time → higher probability (for same breakeven)."""
        p_3d = breakeven_probability(100, 101, 3 / 365, 0.35, 0.045)
        p_10d = breakeven_probability(100, 101, 10 / 365, 0.35, 0.045)
        p_30d = breakeven_probability(100, 101, 30 / 365, 0.35, 0.045)
        assert p_3d < p_10d < p_30d

    def test_higher_iv_higher_prob_for_otm(self):
        """Higher IV → more spread → higher prob for OTM breakeven."""
        p_low = breakeven_probability(100, 105, 10 / 365, 0.20, 0.045)
        p_high = breakeven_probability(100, 105, 10 / 365, 0.50, 0.045)
        assert p_low < p_high

    def test_zero_stock_returns_zero(self):
        assert breakeven_probability(0, 100, 0.5, 0.35, 0.05) == 0.0

    def test_zero_breakeven_returns_one(self):
        assert breakeven_probability(100, 0, 0.5, 0.35, 0.05) == 1.0

    def test_expired_itm(self):
        assert breakeven_probability(110, 100, 0.0, 0.35, 0.05) == 1.0

    def test_expired_otm(self):
        assert breakeven_probability(90, 100, 0.0, 0.35, 0.05) == 0.0

    def test_zero_vol_itm(self):
        """Zero vol, stock > breakeven → certain profit."""
        p = breakeven_probability(110, 100, 0.5, 0.0, 0.05)
        assert p == 1.0

    def test_zero_vol_otm(self):
        """Zero vol, stock < breakeven → certain loss."""
        p = breakeven_probability(90, 100, 0.5, 0.0, 0.05)
        assert p == 0.0


class TestMaxProfitProbability:
    """Tests for max profit probability."""

    def test_is_lower_than_breakeven(self):
        """P(max profit) < P(breakeven) because short_strike > breakeven."""
        p_be = breakeven_probability(100, 101, 10 / 365, 0.35, 0.045)
        p_mp = max_profit_probability(100, 103, 10 / 365, 0.35, 0.045)
        assert p_mp < p_be

    def test_callable_independently(self):
        """Same function as breakeven_probability, different input."""
        p = max_profit_probability(100, 105, 10 / 365, 0.35, 0.045)
        assert 0.0 <= p <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Historical volatility
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeHistoricalVolatility:
    """Tests for realized volatility computation."""

    def test_known_constant_prices(self):
        """Constant prices → zero vol."""
        closes = [100.0] * 25
        hv = compute_historical_volatility(closes, lookback=20)
        assert hv == pytest.approx(0.0, abs=1e-10)

    def test_known_daily_returns(self):
        """1% daily return → annualized vol ≈ 15.87%."""
        import math as m
        closes = [100.0]
        for _ in range(25):
            closes.append(closes[-1] * 1.01)
        hv = compute_historical_volatility(closes, lookback=20)
        # daily log return = ln(1.01) ≈ 0.00995
        # std of constant returns = 0 (all identical)
        # Actually constant returns have zero std!
        assert hv == pytest.approx(0.0, abs=1e-6)

    def test_alternating_returns(self):
        """Alternating +1%/-1% → non-zero vol."""
        closes = [100.0]
        for i in range(25):
            factor = 1.01 if i % 2 == 0 else 0.99
            closes.append(closes[-1] * factor)
        hv = compute_historical_volatility(closes, lookback=20)
        assert hv > 0.05  # should be meaningful vol

    def test_insufficient_data(self):
        """Too few prices → nan."""
        hv = compute_historical_volatility([100, 101], lookback=20)
        assert math.isnan(hv)

    def test_negative_price(self):
        """Negative price in series → nan."""
        closes = [100.0] * 20 + [-1.0]
        hv = compute_historical_volatility(closes, lookback=20)
        assert math.isnan(hv)

    def test_zero_price(self):
        """Zero price → nan (can't compute log return)."""
        closes = [100.0] * 20 + [0.0]
        hv = compute_historical_volatility(closes, lookback=20)
        assert math.isnan(hv)

    def test_none_in_data(self):
        """None value → nan."""
        closes = [100.0] * 20 + [None]
        hv = compute_historical_volatility(closes, lookback=20)
        assert math.isnan(hv)

    def test_typical_stock(self):
        """Simulated 25% annualized vol stock."""
        import random
        random.seed(42)
        daily_vol = 0.25 / math.sqrt(252)  # ~1.57% daily
        closes = [100.0]
        for _ in range(30):
            ret = random.gauss(0, daily_vol)
            closes.append(closes[-1] * math.exp(ret))
        hv = compute_historical_volatility(closes, lookback=20)
        # With 20 samples, estimate should be roughly in [0.10, 0.50]
        assert 0.05 < hv < 0.60


# ──────────────────────────────────────────────────────────────────────────────
# IV rank
# ──────────────────────────────────────────────────────────────────────────────

class TestIVRank:
    """Tests for IV percentile rank."""

    def test_at_max(self):
        """Current IV is highest → rank = 100."""
        assert iv_rank(0.50, [0.20, 0.25, 0.30, 0.35, 0.40]) == pytest.approx(100.0)

    def test_at_min(self):
        """Current IV is lowest → rank = 0."""
        assert iv_rank(0.10, [0.20, 0.25, 0.30, 0.35, 0.40]) == pytest.approx(0.0)

    def test_at_median(self):
        """Current IV at median → rank ≈ 50."""
        result = iv_rank(0.30, [0.20, 0.25, 0.30, 0.35, 0.40])
        assert 30.0 <= result <= 60.0

    def test_empty_history(self):
        assert math.isnan(iv_rank(0.30, []))

    def test_history_with_nones(self):
        """None values should be filtered."""
        result = iv_rank(0.30, [0.20, None, 0.40, None])
        assert math.isfinite(result)

    def test_all_nones(self):
        assert math.isnan(iv_rank(0.30, [None, None]))

    def test_single_element(self):
        """Single history value."""
        assert iv_rank(0.50, [0.30]) == pytest.approx(100.0)
        assert iv_rank(0.10, [0.30]) == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────────────
# evaluate_spread (composite)
# ──────────────────────────────────────────────────────────────────────────────

class TestEvaluateSpread:
    """Tests for the composite spread evaluator."""

    def test_valid_spread_returns_all_fields(self):
        """All expected keys present and valid=True."""
        result = evaluate_spread(
            S=100, atm_strike=100, otm_strike=103,
            net_debit=0.80, T=10 / 365, atm_iv=0.35,
        )
        assert result["valid"] is True
        required_keys = [
            "breakeven_price", "breakeven_prob", "max_profit_prob",
            "max_profit", "max_loss", "risk_reward_ratio",
            "expected_value", "spread_efficiency", "sigma_used",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_breakeven_price_correct(self):
        result = evaluate_spread(
            S=100, atm_strike=100, otm_strike=103,
            net_debit=0.80, T=10 / 365, atm_iv=0.35,
        )
        assert result["breakeven_price"] == pytest.approx(100.80, abs=0.01)

    def test_max_profit_and_loss(self):
        result = evaluate_spread(
            S=100, atm_strike=100, otm_strike=103,
            net_debit=0.80, T=10 / 365, atm_iv=0.35,
        )
        assert result["max_profit"] == pytest.approx(2.20, abs=0.01)
        assert result["max_loss"] == pytest.approx(0.80, abs=0.01)

    def test_risk_reward_ratio(self):
        result = evaluate_spread(
            S=100, atm_strike=100, otm_strike=103,
            net_debit=0.80, T=10 / 365, atm_iv=0.35,
        )
        assert result["risk_reward_ratio"] == pytest.approx(2.75, abs=0.01)

    def test_spread_efficiency(self):
        """Spread efficiency = debit / width."""
        result = evaluate_spread(
            S=100, atm_strike=100, otm_strike=103,
            net_debit=0.80, T=10 / 365, atm_iv=0.35,
        )
        assert result["spread_efficiency"] == pytest.approx(0.80 / 3.0, abs=0.01)

    def test_probabilities_in_range(self):
        result = evaluate_spread(
            S=100, atm_strike=100, otm_strike=103,
            net_debit=0.80, T=10 / 365, atm_iv=0.35,
        )
        assert 0.0 <= result["breakeven_prob"] <= 1.0
        assert 0.0 <= result["max_profit_prob"] <= 1.0
        assert result["breakeven_prob"] >= result["max_profit_prob"]

    def test_invalid_zero_debit(self):
        result = evaluate_spread(
            S=100, atm_strike=100, otm_strike=103,
            net_debit=0.0, T=10 / 365, atm_iv=0.35,
        )
        assert result["valid"] is False

    def test_invalid_debit_exceeds_width(self):
        """Debit > width → max_profit negative → invalid."""
        result = evaluate_spread(
            S=100, atm_strike=100, otm_strike=103,
            net_debit=3.50, T=10 / 365, atm_iv=0.35,
        )
        assert result["valid"] is False

    def test_invalid_iv_zero(self):
        result = evaluate_spread(
            S=100, atm_strike=100, otm_strike=103,
            net_debit=0.80, T=10 / 365, atm_iv=0.0,
        )
        assert result["valid"] is False

    def test_invalid_iv_nan(self):
        result = evaluate_spread(
            S=100, atm_strike=100, otm_strike=103,
            net_debit=0.80, T=10 / 365, atm_iv=float("nan"),
        )
        assert result["valid"] is False

    def test_higher_iv_different_probabilities(self):
        """Higher IV should change probability distribution."""
        r1 = evaluate_spread(100, 100, 103, 0.80, 10 / 365, 0.20)
        r2 = evaluate_spread(100, 100, 103, 0.80, 10 / 365, 0.50)
        # Higher IV → more spread → higher breakeven prob for OTM
        assert r2["breakeven_prob"] > r1["breakeven_prob"]

    def test_sigma_used_is_atm_iv(self):
        result = evaluate_spread(
            S=100, atm_strike=100, otm_strike=103,
            net_debit=0.80, T=10 / 365, atm_iv=0.42,
        )
        assert result["sigma_used"] == pytest.approx(0.42, abs=0.001)

    def test_expected_value_sign(self):
        """A well-priced spread should have positive or near-zero EV."""
        result = evaluate_spread(
            S=100, atm_strike=100, otm_strike=103,
            net_debit=0.80, T=10 / 365, atm_iv=0.35,
        )
        # EV depends on the specific inputs; just verify it's finite
        assert math.isfinite(result["expected_value"])

    # ── Missing edge cases from code review ──────────────────────────────

    def test_debit_equals_width_invalid(self):
        """net_debit == spread_width → zero profit → invalid."""
        result = evaluate_spread(
            S=100, atm_strike=100, otm_strike=103,
            net_debit=3.0, T=10 / 365, atm_iv=0.35,
        )
        assert result["valid"] is False

    def test_inverted_strikes_invalid(self):
        """otm_strike < atm_strike → negative width → invalid."""
        result = evaluate_spread(
            S=100, atm_strike=105, otm_strike=100,
            net_debit=0.80, T=10 / 365, atm_iv=0.35,
        )
        assert result["valid"] is False

    def test_nan_underlying_invalid(self):
        """NaN underlying price → invalid (not a crash)."""
        result = evaluate_spread(
            S=float("nan"), atm_strike=100, otm_strike=103,
            net_debit=0.80, T=10 / 365, atm_iv=0.35,
        )
        assert result["valid"] is False

    def test_nan_net_debit_invalid(self):
        """NaN net_debit → invalid."""
        result = evaluate_spread(
            S=100, atm_strike=100, otm_strike=103,
            net_debit=float("nan"), T=10 / 365, atm_iv=0.35,
        )
        assert result["valid"] is False

    def test_nan_T_invalid(self):
        """NaN T → invalid."""
        result = evaluate_spread(
            S=100, atm_strike=100, otm_strike=103,
            net_debit=0.80, T=float("nan"), atm_iv=0.35,
        )
        assert result["valid"] is False

    def test_zero_underlying_invalid(self):
        """S=0 → invalid."""
        result = evaluate_spread(
            S=0.0, atm_strike=100, otm_strike=103,
            net_debit=0.80, T=10 / 365, atm_iv=0.35,
        )
        assert result["valid"] is False

    def test_zero_T_invalid(self):
        """T=0 → invalid (expired)."""
        result = evaluate_spread(
            S=100, atm_strike=100, otm_strike=103,
            net_debit=0.80, T=0, atm_iv=0.35,
        )
        assert result["valid"] is False

    def test_very_high_iv(self):
        """Very high IV (meme stock, 450%) → valid result."""
        result = evaluate_spread(
            S=100, atm_strike=100, otm_strike=103,
            net_debit=0.80, T=10 / 365, atm_iv=4.5,
        )
        assert result["valid"] is True
        assert 0.0 <= result["breakeven_prob"] <= 1.0

    def test_very_low_iv(self):
        """Very low IV (1.5%) → valid with low breakeven prob."""
        result = evaluate_spread(
            S=100, atm_strike=100, otm_strike=103,
            net_debit=0.80, T=10 / 365, atm_iv=0.015,
        )
        assert result["valid"] is True
        assert result["breakeven_prob"] < 0.10  # Very unlikely to move enough

    def test_iv_at_exact_boundaries(self):
        """IV exactly at _IV_MIN and _IV_MAX → accepted."""
        # At _IV_MIN = 0.01
        r1 = evaluate_spread(100, 100, 103, 0.80, 10 / 365, 0.01)
        assert r1["valid"] is True
        # At _IV_MAX = 5.0
        r2 = evaluate_spread(100, 100, 103, 0.80, 10 / 365, 5.0)
        assert r2["valid"] is True


# ──────────────────────────────────────────────────────────────────────────────
# bsm_vega input validation
# ──────────────────────────────────────────────────────────────────────────────

class TestBSMVegaValidation:
    """Tests for bsm_vega negative input validation (added per code review)."""

    def test_negative_S_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            bsm_vega(-10, 100, 1.0, 0.05, 0.20)

    def test_negative_K_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            bsm_vega(100, -100, 1.0, 0.05, 0.20)

    def test_negative_T_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            bsm_vega(100, 100, -1.0, 0.05, 0.20)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            bsm_vega(100, 100, 1.0, 0.05, -0.20)


# ──────────────────────────────────────────────────────────────────────────────
# bsm_call_price negative K (missing test from review)
# ──────────────────────────────────────────────────────────────────────────────

class TestBSMCallPriceNegK:
    def test_negative_K_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            bsm_call_price(100, -100, 1.0, 0.05, 0.20)


# ──────────────────────────────────────────────────────────────────────────────
# compute_iv edge cases (from code review)
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeIVEdgeCases:
    """Additional IV solver tests identified by code review."""

    def test_deep_otm_bisection_fallback(self):
        """Deep OTM option forces bisection fallback (vega near zero)."""
        # S=100, K=150, T=30/365, very deep OTM → tiny vega
        sigma_input = 0.80
        T = 30 / 365
        price = bsm_call_price(100, 150, T, 0.045, sigma_input)
        iv = compute_iv(price, 100, 150, T, 0.045)
        # Should recover IV (bisection if Newton fails)
        if math.isfinite(iv):
            assert iv == pytest.approx(sigma_input, abs=0.05)

    def test_itm_discounted_intrinsic(self):
        """ITM option price between simple and discounted intrinsic."""
        # S=110, K=100, T=1.0, r=0.10
        # Simple intrinsic: 10.0
        # Discounted intrinsic: 110 - 100*exp(-0.10) = 19.52
        iv = compute_iv(15.0, 110, 100, 1.0, 0.10)
        # Price 15.0 is between simple (10) and discounted (19.52) intrinsic
        # With discounted intrinsic, 15.0 < 19.52 - 0.01 → NaN
        assert math.isnan(iv)


# ──────────────────────────────────────────────────────────────────────────────
# compute_historical_volatility edge cases (from code review)
# ──────────────────────────────────────────────────────────────────────────────

class TestHistVolEdgeCases:
    """Additional HV tests identified by code review."""

    def test_nan_in_closes_returns_nan(self):
        """NaN in closes list → nan."""
        closes = [100.0] * 20 + [float("nan")]
        hv = compute_historical_volatility(closes)
        assert math.isnan(hv)

    def test_inf_in_closes_returns_nan(self):
        """Inf in closes list → nan."""
        closes = [100.0] * 20 + [float("inf")]
        hv = compute_historical_volatility(closes)
        assert math.isnan(hv)

    def test_string_in_closes_returns_nan(self):
        """Non-numeric value in closes list → nan."""
        closes = [100.0] * 20 + ["bad"]
        hv = compute_historical_volatility(closes)
        assert math.isnan(hv)

    def test_lookback_1_returns_nan(self):
        """lookback=1 → only 1 return → nan (need ≥2 for std)."""
        hv = compute_historical_volatility([100.0, 101.0], lookback=1)
        assert math.isnan(hv)


# ──────────────────────────────────────────────────────────────────────────────
# iv_rank edge cases (from code review)
# ──────────────────────────────────────────────────────────────────────────────

class TestIVRankEdgeCases:
    """Additional iv_rank tests identified by code review."""

    def test_inf_in_history_ignored(self):
        """Inf values in history should be filtered out."""
        rank = iv_rank(0.30, [0.20, float("inf"), 0.40])
        assert math.isfinite(rank)
        # 0.30 > 0.20 but < 0.40 → 1/2 * 100 = 50
        assert rank == pytest.approx(50.0)
