"""
BSM probability filter for weekly options (single ITM call) entry decisions.

Evaluates a single ITM call option using BSM-computed breakeven probability,
intrinsic ratio, expected value, and IV metrics.  Scores the candidate and
optionally rejects trades that fail quality thresholds.

Operates in two modes controlled by config:
    * **shadow** — compute & log metrics but never reject (data-collection phase).
    * **filter** — actually reject trades that fail threshold checks.

Feature-flagged (``enabled: false`` by default) so the live bot is
completely unaffected until explicitly turned on.

Author: Claude Opus 4.6
"""
from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from RubberBand.src.bsm import compute_iv, evaluate_single_call

logger = logging.getLogger(__name__)

# Ensure this module's logger never writes to stderr (which crashes
# PowerShell-based GitHub Actions runners via NativeCommandError).
# If no handler has been configured by the host script we attach a
# stdout handler so that WARNING+ messages go to stdout instead of
# Python's default stderr lastResort handler.
if not logging.root.handlers and not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WeeklyProbabilityFilterConfig:
    """
    Configuration for the weekly options probability filter.

    All thresholds are configurable via ``config_weekly.yaml`` under the
    ``weekly_probability_filter:`` key.
    """
    enabled: bool = False
    mode: str = "shadow"                        # "shadow" or "filter"
    min_breakeven_prob: float = 0.45            # P(S_T > K + premium) minimum
    min_intrinsic_ratio: float = 0.20           # At least 20% intrinsic
    min_expected_value: float = -2.0            # Per share, dollars
    max_iv_rank: float = 90.0                   # Skip if IV rank > 90th pctile
    risk_free_rate: float = 0.045
    fallback_to_legacy: bool = True

    # Scoring weights (should sum to ~1.0)
    w_breakeven: float = 0.40
    w_intrinsic: float = 0.25
    w_expected_value: float = 0.25
    w_iv_rank: float = 0.10


def load_weekly_probability_filter_config(
    cfg: Dict[str, Any],
) -> WeeklyProbabilityFilterConfig:
    """
    Load weekly probability filter config from the YAML config dict.

    Args:
        cfg: Full config dict (from ``config_weekly.yaml``).

    Returns:
        Populated WeeklyProbabilityFilterConfig.

    Raises:
        ValueError: If mode or threshold values are invalid.
    """
    section = cfg.get("weekly_probability_filter", {})
    if not section:
        return WeeklyProbabilityFilterConfig()

    config = WeeklyProbabilityFilterConfig(
        enabled=section.get("enabled", False),
        mode=section.get("mode", "shadow"),
        min_breakeven_prob=float(section.get("min_breakeven_prob", 0.45)),
        min_intrinsic_ratio=float(section.get("min_intrinsic_ratio", 0.20)),
        min_expected_value=float(section.get("min_expected_value", -2.0)),
        max_iv_rank=float(section.get("max_iv_rank", 90.0)),
        risk_free_rate=float(section.get("risk_free_rate", 0.045)),
        fallback_to_legacy=section.get("fallback_to_legacy", True),
        w_breakeven=float(section.get("w_breakeven", 0.40)),
        w_intrinsic=float(section.get("w_intrinsic", 0.25)),
        w_expected_value=float(section.get("w_expected_value", 0.25)),
        w_iv_rank=float(section.get("w_iv_rank", 0.10)),
    )

    # ── Validation ────────────────────────────────────────────────────
    if config.mode not in ("shadow", "filter"):
        raise ValueError(
            f"weekly_probability_filter.mode must be 'shadow' or 'filter', "
            f"got {config.mode!r}"
        )

    if not (0.0 <= config.min_breakeven_prob <= 1.0):
        raise ValueError(
            f"min_breakeven_prob must be in [0, 1], "
            f"got {config.min_breakeven_prob}"
        )

    if not (0.0 <= config.min_intrinsic_ratio <= 1.0):
        raise ValueError(
            f"min_intrinsic_ratio must be in [0, 1], "
            f"got {config.min_intrinsic_ratio}"
        )

    # Warn if scoring weights don't sum to ~1.0
    weight_sum = (
        config.w_breakeven + config.w_intrinsic
        + config.w_expected_value + config.w_iv_rank
    )
    if abs(weight_sum - 1.0) > 0.05:
        logger.warning(
            "Weekly probability filter scoring weights sum to %.2f "
            "(expected ~1.0): w_breakeven=%.2f, w_intrinsic=%.2f, "
            "w_expected_value=%.2f, w_iv_rank=%.2f",
            weight_sum,
            config.w_breakeven,
            config.w_intrinsic,
            config.w_expected_value,
            config.w_iv_rank,
        )

    return config


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CallCandidate:
    """
    Evaluated single call option with all computed metrics.

    All log-dict keys are prefixed with ``wpf_`` (weekly probability filter)
    to avoid collision with the spread filter's ``pf_`` namespace.
    """
    # Identification
    underlying: str
    option_symbol: str
    strike: float
    expiration: str
    dte: int
    underlying_price: float

    # Pricing
    bid: float
    ask: float
    mid: float
    premium: float              # = ask (what we pay per share)

    # BSM-computed metrics
    bsm_iv: float
    breakeven_prob: float
    itm_prob: float
    intrinsic_value: float
    intrinsic_ratio: float
    time_value: float
    bsm_fair_value: float
    edge: float
    expected_value: float

    # Greeks from snapshot (informational, not used in scoring)
    delta: float = 0.0
    theta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0

    # Scoring
    composite_score: float = 0.0

    # Rejection tracking
    rejected: bool = False
    reject_reasons: List[str] = field(default_factory=list)

    def to_log_dict(self) -> Dict[str, Any]:
        """
        Flatten to dict for JSON logging.

        All keys are prefixed with ``wpf_`` to avoid collisions when
        unpacked into log calls alongside other kwargs.
        """
        return {
            "wpf_underlying": self.underlying,
            "wpf_option_symbol": self.option_symbol,
            "wpf_strike": round(self.strike, 2),
            "wpf_expiration": self.expiration,
            "wpf_dte": self.dte,
            "wpf_underlying_price": round(self.underlying_price, 2),
            "wpf_bid": round(self.bid, 4),
            "wpf_ask": round(self.ask, 4),
            "wpf_mid": round(self.mid, 4),
            "wpf_premium": round(self.premium, 4),
            "wpf_bsm_iv": round(self.bsm_iv, 4),
            "wpf_breakeven_prob": round(self.breakeven_prob, 4),
            "wpf_itm_prob": round(self.itm_prob, 4),
            "wpf_intrinsic_value": round(self.intrinsic_value, 4),
            "wpf_intrinsic_ratio": round(self.intrinsic_ratio, 4),
            "wpf_time_value": round(self.time_value, 4),
            "wpf_bsm_fair_value": round(self.bsm_fair_value, 4),
            "wpf_edge": round(self.edge, 4),
            "wpf_expected_value": round(self.expected_value, 4),
            "wpf_delta": round(self.delta, 4),
            "wpf_theta": round(self.theta, 4),
            "wpf_gamma": round(self.gamma, 4),
            "wpf_vega": round(self.vega, 4),
            "wpf_composite_score": round(self.composite_score, 4),
            "wpf_rejected": self.rejected,
            "wpf_reject_reasons": self.reject_reasons,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Sigmoid squash, maps (-inf, inf) -> (0, 1)."""
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def compute_call_composite_score(
    candidate: CallCandidate,
    config: WeeklyProbabilityFilterConfig,
    iv_rank_value: Optional[float] = None,
) -> float:
    """
    Compute composite score for a single call candidate.

    Formula::

        score = w_be * breakeven_prob
              + w_intr * min(intrinsic_ratio, 1.0)
              + w_ev * sigmoid(expected_value / premium)
              + w_iv * (1 - iv_rank / 100)    [or 0.5 if unavailable]

    Args:
        candidate:     The evaluated call candidate.
        config:        Filter configuration (weights).
        iv_rank_value: IV percentile rank [0, 100] or None.

    Returns:
        Composite score in [0, 1] (higher = better).
    """
    # Breakeven probability component
    be_score = candidate.breakeven_prob

    # Intrinsic ratio component (higher = safer, less theta exposure)
    intr_score = min(candidate.intrinsic_ratio, 1.0)

    # Expected value component (sigmoid-squashed, normalized by premium)
    denom = candidate.premium if candidate.premium > 0 else 1.0
    ev_score = _sigmoid(candidate.expected_value / denom)

    # IV rank component (prefer lower IV rank = cheaper options)
    if iv_rank_value is not None and math.isfinite(iv_rank_value):
        iv_score = 1.0 - iv_rank_value / 100.0
    else:
        iv_score = 0.5  # neutral when unavailable

    score = (
        config.w_breakeven * be_score
        + config.w_intrinsic * intr_score
        + config.w_expected_value * ev_score
        + config.w_iv_rank * iv_score
    )

    return max(0.0, min(1.0, score))


# ──────────────────────────────────────────────────────────────────────────────
# Threshold checks
# ──────────────────────────────────────────────────────────────────────────────

def apply_call_threshold_filters(
    candidate: CallCandidate,
    config: WeeklyProbabilityFilterConfig,
) -> None:
    """
    Apply threshold filters to a call candidate.

    Mutates ``candidate.rejected`` and ``candidate.reject_reasons`` in place.

    Args:
        candidate: The call candidate to evaluate.
        config:    Filter configuration (thresholds).
    """
    reasons: List[str] = []

    if candidate.breakeven_prob < config.min_breakeven_prob:
        reasons.append(
            f"P_BE={candidate.breakeven_prob:.2f}<{config.min_breakeven_prob:.2f}"
        )

    if candidate.intrinsic_ratio < config.min_intrinsic_ratio:
        reasons.append(
            f"INTR={candidate.intrinsic_ratio:.2f}<{config.min_intrinsic_ratio:.2f}"
        )

    if candidate.expected_value < config.min_expected_value:
        reasons.append(
            f"EV={candidate.expected_value:.2f}<{config.min_expected_value:.2f}"
        )

    if reasons:
        candidate.rejected = True
        candidate.reject_reasons = reasons


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_call_quality(
    underlying: str,
    option_symbol: str,
    strike: float,
    premium: float,
    underlying_price: float,
    expiration: str,
    dte: int,
    iv: float,
    config: WeeklyProbabilityFilterConfig,
    snapshot: Optional[Dict[str, Any]] = None,
    iv_rank_value: Optional[float] = None,
) -> CallCandidate:
    """
    Evaluate a single ITM call option for quality.

    Called from ``try_weekly_option_entry()`` after contract selection and
    quote validation, but before premium cost check and order submission.

    IV resolution: uses Alpaca snapshot IV as primary.  Falls back to BSM
    ``compute_iv()`` if snapshot IV is invalid.  Rejects the candidate
    with reason ``"no_valid_iv"`` if both fail.

    Args:
        underlying:       Stock symbol (e.g. ``"AAPL"``).
        option_symbol:    OCC option symbol.
        strike:           Option strike price.
        premium:          Ask price per share (what we pay).
        underlying_price: Current underlying price.
        expiration:       Expiration date string (``YYYY-MM-DD``).
        dte:              Days to expiration.
        iv:               Implied volatility (from snapshot or 0 if unavailable).
        config:           Filter configuration.
        snapshot:         Option snapshot dict with greeks (optional).
        iv_rank_value:    IV percentile rank (optional).

    Returns:
        Populated CallCandidate with metrics, score, and rejection status.
    """
    # Extract quote info from snapshot or use premium
    bid = 0.0
    ask = premium
    mid = premium
    if snapshot:
        bid = float(snapshot.get("bid", 0) or 0)
        ask_snap = float(snapshot.get("ask", 0) or 0)
        if ask_snap > 0:
            ask = ask_snap
        mid_snap = float(snapshot.get("mid", 0) or 0)
        if mid_snap > 0:
            mid = mid_snap
        elif bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0

    # ── IV resolution ─────────────────────────────────────────────────
    resolved_iv = iv
    if not (isinstance(resolved_iv, (int, float))
            and math.isfinite(resolved_iv) and resolved_iv > 0.005):
        # Try computing IV ourselves from mid price
        resolved_iv = _try_compute_iv(mid, underlying_price, strike, dte, config.risk_free_rate)

    if not (isinstance(resolved_iv, (int, float))
            and math.isfinite(resolved_iv) and resolved_iv > 0.005):
        # Both failed — return rejected candidate
        return _make_rejected_candidate(
            underlying, option_symbol, strike, expiration, dte,
            underlying_price, bid, ask, mid, premium,
            reject_reasons=["no_valid_iv"],
        )

    # ── BSM evaluation ────────────────────────────────────────────────
    T = dte / 365.0
    result = evaluate_single_call(
        S=underlying_price,
        K=strike,
        premium=premium,
        T=T,
        iv=resolved_iv,
        r=config.risk_free_rate,
    )

    if not result.get("valid", False):
        return _make_rejected_candidate(
            underlying, option_symbol, strike, expiration, dte,
            underlying_price, bid, ask, mid, premium,
            reject_reasons=["bsm_invalid"],
        )

    # ── Build candidate ───────────────────────────────────────────────
    candidate = CallCandidate(
        underlying=underlying,
        option_symbol=option_symbol,
        strike=strike,
        expiration=expiration,
        dte=dte,
        underlying_price=underlying_price,
        bid=bid,
        ask=ask,
        mid=mid,
        premium=premium,
        bsm_iv=resolved_iv,
        breakeven_prob=result["breakeven_prob"],
        itm_prob=result["itm_prob"],
        intrinsic_value=result["intrinsic_value"],
        intrinsic_ratio=result["intrinsic_ratio"],
        time_value=result["time_value"],
        bsm_fair_value=result["bsm_fair_value"],
        edge=result["edge"],
        expected_value=result["expected_value"],
    )

    # Populate greeks from snapshot if available
    if snapshot:
        candidate.delta = float(snapshot.get("delta", 0) or 0)
        candidate.theta = float(snapshot.get("theta", 0) or 0)
        candidate.gamma = float(snapshot.get("gamma", 0) or 0)
        candidate.vega = float(snapshot.get("vega", 0) or 0)

    # ── Apply filters + scoring ───────────────────────────────────────
    apply_call_threshold_filters(candidate, config)

    # IV rank hard gate (checked here because iv_rank_value is not
    # on the candidate dataclass — it's an optional caller-provided value)
    if (iv_rank_value is not None
            and math.isfinite(iv_rank_value)
            and iv_rank_value > config.max_iv_rank):
        candidate.rejected = True
        candidate.reject_reasons.append(
            f"IV_RANK={iv_rank_value:.0f}>{config.max_iv_rank:.0f}"
        )

    candidate.composite_score = compute_call_composite_score(
        candidate, config, iv_rank_value,
    )

    return candidate


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _try_compute_iv(
    mid_price: float,
    S: float,
    K: float,
    dte: int,
    r: float,
) -> float:
    """
    Attempt to compute IV from mid price using BSM Newton-Raphson solver.

    Returns computed IV or NaN if solver fails.
    """
    try:
        T = dte / 365.0
        if T <= 0 or mid_price <= 0 or S <= 0 or K <= 0:
            return float("nan")
        return compute_iv(mid_price, S, K, T, r)
    except Exception as e:
        logger.warning("IV solver failed: %s (mid=%.2f, S=%.2f, K=%.2f, DTE=%d)",
                        e, mid_price, S, K, dte)
        return float("nan")


def _make_rejected_candidate(
    underlying: str,
    option_symbol: str,
    strike: float,
    expiration: str,
    dte: int,
    underlying_price: float,
    bid: float,
    ask: float,
    mid: float,
    premium: float,
    reject_reasons: List[str],
) -> CallCandidate:
    """Create a rejected CallCandidate with NaN metrics."""
    nan = float("nan")
    return CallCandidate(
        underlying=underlying,
        option_symbol=option_symbol,
        strike=strike,
        expiration=expiration,
        dte=dte,
        underlying_price=underlying_price,
        bid=bid,
        ask=ask,
        mid=mid,
        premium=premium,
        bsm_iv=nan,
        breakeven_prob=nan,
        itm_prob=nan,
        intrinsic_value=nan,
        intrinsic_ratio=nan,
        time_value=nan,
        bsm_fair_value=nan,
        edge=nan,
        expected_value=nan,
        rejected=True,
        reject_reasons=reject_reasons,
    )
