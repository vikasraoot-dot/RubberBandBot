"""
Black-Scholes-Merton pricing, IV inversion, and probability analysis.

Pure math module — zero I/O, zero external dependencies beyond ``math``.
All functions use float arithmetic (not Decimal) because these are
analytical estimates for trade filtering, not financial transactions.

Convention:
    T  = calendar days / 365  (BSM continuous-time assumption)
    σ  = annualized implied volatility (e.g. 0.35 = 35%)
    r  = annualized risk-free rate (e.g. 0.045 = 4.5%)

Author: Claude Opus 4.6 (cross-validated by 3 independent analysis agents)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Normal distribution helpers
# ──────────────────────────────────────────────────────────────────────────────

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_TRADING_DAYS_PER_YEAR = 252


def norm_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function.

    Uses the identity Φ(x) = 0.5 × (1 + erf(x / √2)).
    This is *exact* (not an approximation) — ``math.erf`` computes the
    error function to full machine precision.

    Args:
        x: Input value (any real number).

    Returns:
        P(Z ≤ x) where Z ~ N(0, 1), in [0, 1].
    """
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def norm_pdf(x: float) -> float:
    """
    Standard normal probability density function.

    Formula: φ(x) = (1 / √(2π)) × exp(-x² / 2).

    Args:
        x: Input value (any real number).

    Returns:
        Density at *x* (always ≥ 0).
    """
    return math.exp(-0.5 * x * x) / _SQRT_2PI


# ──────────────────────────────────────────────────────────────────────────────
# Black-Scholes-Merton pricing
# ──────────────────────────────────────────────────────────────────────────────

# Minimum thresholds to avoid division by zero / degenerate math
_T_MIN = 1e-10
_SIGMA_MIN = 1e-10


def bsm_call_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Black-Scholes European call option price.

    C = S × N(d1) − K × e^{−rT} × N(d2)

    where:
        d1 = [ln(S/K) + (r + σ²/2) × T] / (σ √T)
        d2 = d1 − σ √T

    Args:
        S:     Current underlying price (> 0).
        K:     Strike price (> 0).
        T:     Time to expiration in years (≥ 0).  Use DTE/365.
        r:     Risk-free rate (annualized).
        sigma: Volatility (annualized).

    Returns:
        Theoretical call price (≥ 0).

    Raises:
        ValueError: If S < 0, K < 0, T < 0, or sigma < 0.
    """
    if S < 0 or K < 0 or T < 0 or sigma < 0:
        raise ValueError(
            f"Invalid BSM inputs: S={S}, K={K}, T={T}, sigma={sigma}. "
            f"All must be non-negative."
        )

    # Edge: expired option → intrinsic value
    if T <= _T_MIN:
        return max(S - K, 0.0)

    # Edge: zero volatility → deterministic payoff
    if sigma <= _SIGMA_MIN:
        return max(S - K * math.exp(-r * T), 0.0)

    # Edge: worthless underlying
    if S <= 0.0:
        return 0.0

    # Edge: zero strike
    if K <= 0.0:
        return S

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return max(price, 0.0)


def bsm_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    BSM vega: ∂C/∂σ = S × √T × φ(d1).

    Measures the sensitivity of the call price to a 1.0 (100 pp) change
    in volatility.

    Args:
        S:     Current underlying price.
        K:     Strike price.
        T:     Time to expiration in years.
        r:     Risk-free rate.
        sigma: Volatility.

    Returns:
        Vega (≥ 0).  Returns 0 for degenerate inputs.

    Raises:
        ValueError: If S < 0, K < 0, T < 0, or sigma < 0.
    """
    if S < 0 or K < 0 or T < 0 or sigma < 0:
        raise ValueError(
            f"Invalid vega inputs: S={S}, K={K}, T={T}, sigma={sigma}. "
            f"All must be non-negative."
        )
    if T <= _T_MIN or sigma <= _SIGMA_MIN or S <= 0.0 or K <= 0.0:
        return 0.0

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    return S * sqrt_T * norm_pdf(d1)


# ──────────────────────────────────────────────────────────────────────────────
# Implied volatility solver
# ──────────────────────────────────────────────────────────────────────────────

# Solver constants
_IV_MIN = 0.01     # 1 % annualized floor
_IV_MAX = 5.0      # 500 % annualized ceiling (meme stocks)
_IV_MAX_NEWTON = 50
_IV_MAX_BISECT = 100
_IV_TOLERANCE = 1e-6   # convergence in price dollars
_IV_MAX_STEP = 0.5     # max σ change per Newton iteration (damping)


def compute_iv(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
) -> float:
    """
    Compute implied volatility via Newton-Raphson with bisection fallback.

    Solves  BSM(S, K, T, r, σ) = market_price  for σ.

    Algorithm:
        1. Validate inputs.
        2. Initial guess via Brenner-Subrahmanyam (1988).
        3. Damped Newton-Raphson (up to 50 iterations).
        4. If Newton fails, bisection on [0.01, 5.0] (up to 100 iterations).

    Args:
        market_price: Observed option mid-price.
        S:            Underlying spot price.
        K:            Strike price.
        T:            Time to expiration in years.
        r:            Risk-free rate.

    Returns:
        Annualized implied volatility, or ``nan`` if the solver cannot
        converge (bad data, price < intrinsic, T ≤ 0, etc.).
    """
    # ── Input validation ────────────────────────────────────────────────
    if market_price <= 0.0 or S <= 0.0 or K <= 0.0:
        return float("nan")

    if T <= _T_MIN:
        return float("nan")

    # Use discounted intrinsic: max(S - K*exp(-rT), 0) for correct BSM lower bound
    intrinsic = max(S - K * math.exp(-r * T), 0.0)

    # Price below intrinsic minus small tolerance → arbitrage / bad data
    if market_price < intrinsic - 0.01:
        return float("nan")

    # Call price can never exceed stock price
    if market_price > S:
        return float("nan")

    # Pure intrinsic (no time value) → effectively zero IV
    if market_price <= intrinsic + 0.001:
        return _IV_MIN

    # ── Initial guess (Brenner-Subrahmanyam 1988) ───────────────────────
    # Use time value (not full price) for ITM options — the B-S formula
    # is derived for ATM where price ≈ time value.
    sqrt_T = math.sqrt(T)
    time_value = max(market_price - intrinsic, 0.01)
    sigma = time_value * _SQRT_2PI / (S * sqrt_T)
    sigma = max(_IV_MIN, min(_IV_MAX, sigma))

    # ── Newton-Raphson with damping ─────────────────────────────────────
    for _ in range(_IV_MAX_NEWTON):
        price = bsm_call_price(S, K, T, r, sigma)
        vega = bsm_vega(S, K, T, r, sigma)

        price_error = price - market_price

        if abs(price_error) < _IV_TOLERANCE:
            return sigma

        # Vega too small → Newton step is unreliable, switch to bisection
        if vega < 1e-10:
            break

        step = price_error / vega
        step = max(-_IV_MAX_STEP, min(_IV_MAX_STEP, step))

        sigma_new = sigma - step
        sigma_new = max(_IV_MIN, min(_IV_MAX, sigma_new))

        if abs(sigma_new - sigma) < 1e-10:
            return sigma  # converged (σ not changing)

        sigma = sigma_new

    # ── Bisection fallback (guaranteed convergence) ─────────────────────
    lo, hi = _IV_MIN, _IV_MAX
    for _ in range(_IV_MAX_BISECT):
        mid = (lo + hi) / 2.0
        price = bsm_call_price(S, K, T, r, mid)

        if abs(price - market_price) < _IV_TOLERANCE:
            return mid

        if price > market_price:
            hi = mid
        else:
            lo = mid

        if (hi - lo) < 1e-8:
            return mid

    # Complete failure
    return float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Probability functions
# ──────────────────────────────────────────────────────────────────────────────

def breakeven_probability(
    S: float,
    breakeven_price: float,
    T: float,
    sigma: float,
    r: float,
) -> float:
    """
    Probability that S_T > *breakeven_price* at expiration (risk-neutral).

    Under GBM:
        P(S_T > B) = N(d2)

    where:
        d2 = [ln(S/B) + (r − σ²/2) × T] / (σ √T)

    For short-dated options (3-10 DTE) the drift term (r − σ²/2)×T is
    negligible, so risk-neutral ≈ physical probability.

    Args:
        S:               Current underlying price.
        breakeven_price: Breakeven level (for bull call spread = ATM strike + net debit).
        T:               Time to expiration in years.
        sigma:           Annualized implied volatility.
        r:               Risk-free rate.

    Returns:
        Probability in [0, 1].
    """
    if S <= 0.0:
        return 0.0
    if breakeven_price <= 0.0:
        return 1.0
    if T <= _T_MIN:
        return 1.0 if S > breakeven_price else 0.0
    if sigma <= _SIGMA_MIN:
        forward = S * math.exp(r * T)
        return 1.0 if forward > breakeven_price else 0.0

    sqrt_T = math.sqrt(T)
    d2 = (
        math.log(S / breakeven_price) + (r - 0.5 * sigma * sigma) * T
    ) / (sigma * sqrt_T)

    return norm_cdf(d2)


def max_profit_probability(
    S: float,
    short_strike: float,
    T: float,
    sigma: float,
    r: float,
) -> float:
    """
    Probability of max profit: P(S_T > short_strike) at expiration.

    Identical to :func:`breakeven_probability` with B = short_strike.

    Args:
        S:            Current underlying price.
        short_strike: Strike of the sold (OTM) call.
        T:            Time to expiration in years.
        sigma:        Annualized implied volatility.
        r:            Risk-free rate.

    Returns:
        Probability in [0, 1].
    """
    return breakeven_probability(S, short_strike, T, sigma, r)


# ──────────────────────────────────────────────────────────────────────────────
# Historical volatility
# ──────────────────────────────────────────────────────────────────────────────

def compute_historical_volatility(
    closes: List[float],
    lookback: int = 20,
) -> float:
    """
    Annualized realized volatility from daily closing prices.

    Steps:
        1. Compute log-returns: r_i = ln(close_i / close_{i-1})
        2. Sample std of the most recent *lookback* returns.
        3. Annualize: σ = std × √252.

    Uses √252 (trading days) because we only observe returns on trading
    days.  This is directly comparable to BSM IV (which uses √365
    internally but is market-quoted in a way that matches √252 HV).

    Args:
        closes:   Closing prices, oldest first.  Needs ≥ lookback + 1 values.
        lookback: Number of returns to use (default 20 ≈ 1 month).

    Returns:
        Annualized HV, or ``nan`` if insufficient / invalid data.
    """
    if len(closes) < lookback + 1:
        return float("nan")

    recent = closes[-(lookback + 1):]

    # All prices must be finite positive numbers
    for p in recent:
        if p is None or not isinstance(p, (int, float)) or not math.isfinite(p) or p <= 0:
            return float("nan")

    log_returns = [
        math.log(recent[i] / recent[i - 1])
        for i in range(1, len(recent))
    ]

    n = len(log_returns)
    if n < 2:
        return float("nan")

    mean_r = sum(log_returns) / n
    variance = sum((r - mean_r) ** 2 for r in log_returns) / (n - 1)

    if variance < 0:
        return float("nan")

    return math.sqrt(variance) * math.sqrt(_TRADING_DAYS_PER_YEAR)


# ──────────────────────────────────────────────────────────────────────────────
# IV Rank
# ──────────────────────────────────────────────────────────────────────────────

def iv_rank(current_iv: float, iv_history: List[float]) -> float:
    """
    IV percentile rank relative to historical IV observations.

    iv_rank = (count of IVs below current) / total × 100

    - 90 → current IV higher than 90 % of history (expensive options).
    - 10 → current IV lower than 90 % of history (cheap options).

    Args:
        current_iv: Current implied volatility.
        iv_history: Historical IV values (e.g. 252 daily observations).

    Returns:
        Percentile in [0, 100], or ``nan`` if history is empty.
    """
    if not iv_history:
        return float("nan")

    valid = [iv for iv in iv_history if iv is not None and math.isfinite(iv)]
    if not valid:
        return float("nan")

    count_below = sum(1 for iv in valid if iv < current_iv)
    return (count_below / len(valid)) * 100.0


# ──────────────────────────────────────────────────────────────────────────────
# Composite spread evaluator
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_spread(
    S: float,
    atm_strike: float,
    otm_strike: float,
    net_debit: float,
    T: float,
    atm_iv: float,
    r: float = 0.045,
) -> Dict[str, Any]:
    """
    Evaluate a bull call spread using probability metrics.

    Computes breakeven probability, max-profit probability, risk-reward
    ratio, expected value, and spread efficiency.

    Args:
        S:          Current underlying price.
        atm_strike: Long (ATM) leg strike.
        otm_strike: Short (OTM) leg strike.
        net_debit:  Net debit paid per share.
        T:          Time to expiration in years (DTE / 365).
        atm_iv:     Implied volatility of the ATM (long) leg.
        r:          Risk-free rate.

    Returns:
        Dict with keys:
            breakeven_price, breakeven_prob, max_profit_prob,
            max_profit, max_loss, risk_reward_ratio, expected_value,
            spread_efficiency, sigma_used, valid
    """
    # Guard against NaN inputs (NaN <= 0 is False, so explicit check needed)
    if (
        not math.isfinite(S) or not math.isfinite(atm_strike)
        or not math.isfinite(otm_strike) or not math.isfinite(net_debit)
        or not math.isfinite(T) or not math.isfinite(atm_iv)
        or S <= 0 or T <= _T_MIN
    ):
        return {
            "breakeven_price": float("nan"),
            "breakeven_prob": float("nan"),
            "max_profit_prob": float("nan"),
            "max_profit": float("nan"),
            "max_loss": float("nan"),
            "risk_reward_ratio": float("nan"),
            "expected_value": float("nan"),
            "spread_efficiency": float("nan"),
            "sigma_used": atm_iv,
            "valid": False,
        }

    spread_width = otm_strike - atm_strike
    breakeven_price = atm_strike + net_debit
    max_profit = spread_width - net_debit
    max_loss = net_debit

    # Guard against invalid spread construction
    if spread_width <= 0 or net_debit <= 0 or max_profit <= 0:
        return {
            "breakeven_price": breakeven_price,
            "breakeven_prob": float("nan"),
            "max_profit_prob": float("nan"),
            "max_profit": max_profit,
            "max_loss": max_loss,
            "risk_reward_ratio": float("nan"),
            "expected_value": float("nan"),
            "spread_efficiency": net_debit / spread_width if spread_width > 0 else 1.0,
            "sigma_used": atm_iv,
            "valid": False,
        }

    sigma = atm_iv

    # Reject clearly invalid IV
    if not math.isfinite(sigma) or sigma < _IV_MIN or sigma > _IV_MAX:
        return {
            "breakeven_price": breakeven_price,
            "breakeven_prob": float("nan"),
            "max_profit_prob": float("nan"),
            "max_profit": max_profit,
            "max_loss": max_loss,
            "risk_reward_ratio": max_profit / max_loss if max_loss > 0 else float("inf"),
            "expected_value": float("nan"),
            "spread_efficiency": net_debit / spread_width,
            "sigma_used": sigma,
            "valid": False,
        }

    p_breakeven = breakeven_probability(S, breakeven_price, T, sigma, r)
    p_max_profit = max_profit_probability(S, otm_strike, T, sigma, r)

    # Expected value (3-state approximation):
    #   State 1: S_T < breakeven → lose max_loss          (prob = 1 − p_breakeven)
    #   State 2: breakeven < S_T < short_strike → partial  (prob = p_breakeven − p_max_profit)
    #   State 3: S_T > short_strike → win max_profit       (prob = p_max_profit)
    #
    # Average partial profit ≈ max_profit / 2  (linear interpolation)
    p_between = max(0.0, p_breakeven - p_max_profit)
    avg_partial = max_profit / 2.0

    expected_value = (
        p_max_profit * max_profit
        + p_between * avg_partial
        - (1.0 - p_breakeven) * max_loss
    )

    risk_reward = max_profit / max_loss if max_loss > 0 else float("inf")
    spread_efficiency = net_debit / spread_width

    return {
        "breakeven_price": round(breakeven_price, 4),
        "breakeven_prob": round(p_breakeven, 4),
        "max_profit_prob": round(p_max_profit, 4),
        "max_profit": round(max_profit, 4),
        "max_loss": round(max_loss, 4),
        "risk_reward_ratio": round(risk_reward, 4),
        "expected_value": round(expected_value, 4),
        "spread_efficiency": round(spread_efficiency, 4),
        "sigma_used": round(sigma, 4),
        "valid": True,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Composite single-call evaluator
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_single_call(
    S: float,
    K: float,
    premium: float,
    T: float,
    iv: float,
    r: float = 0.045,
) -> Dict[str, Any]:
    """
    Evaluate a single long call option using BSM probability metrics.

    Computes breakeven probability, ITM probability, intrinsic ratio,
    BSM fair-value edge, and scenario-based expected value.

    Args:
        S:       Current underlying price (must be > 0).
        K:       Strike price (must be > 0).
        premium: Premium paid per share — typically the ask price (must be > 0).
        T:       Time to expiration in years (DTE / 365, must be > 0).
        iv:      Annualized implied volatility (must be in [IV_MIN, IV_MAX]).
        r:       Risk-free rate (annualized, default 0.045).

    Returns:
        Dict with keys:
            breakeven_price, breakeven_prob, itm_prob,
            intrinsic_value, intrinsic_ratio, time_value,
            bsm_fair_value, edge, expected_value,
            sigma_used, valid
    """
    _invalid = {
        "breakeven_price": float("nan"),
        "breakeven_prob": float("nan"),
        "itm_prob": float("nan"),
        "intrinsic_value": float("nan"),
        "intrinsic_ratio": float("nan"),
        "time_value": float("nan"),
        "bsm_fair_value": float("nan"),
        "edge": float("nan"),
        "expected_value": float("nan"),
        "sigma_used": iv if isinstance(iv, (int, float)) else float("nan"),
        "valid": False,
    }

    # Guard against NaN / non-finite inputs
    if (
        not math.isfinite(S) or not math.isfinite(K)
        or not math.isfinite(premium) or not math.isfinite(T)
        or not math.isfinite(iv)
        or S <= 0 or K <= 0 or premium <= 0 or T <= _T_MIN
    ):
        return _invalid

    # Reject clearly invalid IV
    if iv < _IV_MIN or iv > _IV_MAX:
        return _invalid

    # ── Derived values ────────────────────────────────────────────────
    breakeven_price = K + premium
    intrinsic_value = max(S - K, 0.0)
    time_value = max(premium - intrinsic_value, 0.0)  # Clamp ≥ 0 (stale data guard)
    intrinsic_ratio = min(intrinsic_value / premium, 1.0)  # Cap at 1.0

    # ── BSM fair value and edge ───────────────────────────────────────
    bsm_fair = bsm_call_price(S, K, T, r, iv)
    edge = bsm_fair - premium  # negative = overpaying (typical when buying at ask)

    # ── Probabilities ─────────────────────────────────────────────────
    p_breakeven = breakeven_probability(S, breakeven_price, T, iv, r)
    p_itm = breakeven_probability(S, K, T, iv, r)

    # ── Scenario-based Expected Value (3-zone model) ──────────────────
    #
    #  Zone 1: S_T < K (total loss)
    #    payoff = -premium,  prob = 1 - p_itm
    #
    #  Zone 2: K < S_T < breakeven (partial loss — ITM but below breakeven)
    #    avg payoff ≈ -time_value / 2,  prob = p_itm - p_breakeven
    #
    #  Zone 3: S_T > breakeven (profit)
    #    For bounded EV with unlimited upside, cap at 1-sigma expected move.
    #    expected_move = S * iv * sqrt(T)
    #    profit_at_1sigma = max(S + expected_move - K, 0) - premium
    #    avg profit ≈ profit_at_1sigma / 2,  prob = p_breakeven
    #
    sqrt_T = math.sqrt(T)
    expected_move = S * iv * sqrt_T
    profit_at_1sigma = max(S + expected_move - K, 0.0) - premium
    avg_profit = max(profit_at_1sigma / 2.0, 0.0)

    p_total_loss = max(1.0 - p_itm, 0.0)
    p_partial = max(p_itm - p_breakeven, 0.0)

    expected_value = (
        p_total_loss * (-premium)
        + p_partial * (-time_value / 2.0)
        + p_breakeven * avg_profit
    )

    return {
        "breakeven_price": round(breakeven_price, 4),
        "breakeven_prob": round(p_breakeven, 4),
        "itm_prob": round(p_itm, 4),
        "intrinsic_value": round(intrinsic_value, 4),
        "intrinsic_ratio": round(intrinsic_ratio, 4),
        "time_value": round(time_value, 4),
        "bsm_fair_value": round(bsm_fair, 4),
        "edge": round(edge, 4),
        "expected_value": round(expected_value, 4),
        "sigma_used": round(iv, 4),
        "valid": True,
    }
