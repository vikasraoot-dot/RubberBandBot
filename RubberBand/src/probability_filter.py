"""
Probability-based pre-filter for bull call spread entry decisions.

Evaluates multiple DTE candidates using BSM-computed breakeven probability,
risk-reward ratio, expected value, and IV metrics.  Selects the
expiration with the highest composite score, or rejects the trade entirely
if no candidate meets threshold requirements.

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
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

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

ET = ZoneInfo("US/Eastern")


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ProbabilityFilterConfig:
    """
    Configuration for the probability filter.

    All thresholds are configurable via ``config.yaml`` under the
    ``probability_filter:`` key.
    """
    enabled: bool = False
    mode: str = "shadow"                        # "shadow" or "filter"
    dte_candidates: List[int] = field(
        default_factory=lambda: [4, 5, 6, 8, 10]
    )
    min_breakeven_prob: float = 0.50
    min_risk_reward: float = 0.30
    min_expected_value: float = -5.0            # Per contract, dollars
    max_iv_rank: float = 90.0
    max_bid_ask_spread_pct: float = 20.0        # Reject if bid-ask > 20% of mid
    risk_free_rate: float = 0.045
    fallback_to_legacy: bool = True
    max_dte: int = 10                           # Hard cap

    # Scoring weights (must sum to ~1.0)
    w_breakeven: float = 0.40
    w_risk_reward: float = 0.25
    w_expected_value: float = 0.25
    w_iv_rank: float = 0.10


def load_probability_filter_config(
    cfg: Dict[str, Any],
) -> ProbabilityFilterConfig:
    """
    Load :class:`ProbabilityFilterConfig` from the bot's config dict.

    Validates mode, threshold ranges, and scoring weights on load.

    Args:
        cfg: Full config dict (from config.yaml).

    Returns:
        Populated config, or defaults if the section is missing.

    Raises:
        ValueError: If ``mode`` is not "shadow" or "filter", or if
            ``min_breakeven_prob`` is outside [0, 1], or ``max_dte`` < 1.
    """
    section = cfg.get("probability_filter", {})
    if not section:
        return ProbabilityFilterConfig()

    config = ProbabilityFilterConfig(
        enabled=bool(section.get("enabled", False)),
        mode=str(section.get("mode", "shadow")),
        dte_candidates=list(section.get("dte_candidates", [4, 5, 6, 8, 10])),
        min_breakeven_prob=float(section.get("min_breakeven_prob", 0.50)),
        min_risk_reward=float(section.get("min_risk_reward", 0.30)),
        min_expected_value=float(section.get("min_expected_value", -5.0)),
        max_iv_rank=float(section.get("max_iv_rank", 90.0)),
        max_bid_ask_spread_pct=float(section.get("max_bid_ask_spread_pct", 20.0)),
        risk_free_rate=float(section.get("risk_free_rate", 0.045)),
        fallback_to_legacy=bool(section.get("fallback_to_legacy", True)),
        max_dte=int(section.get("max_dte", 10)),
        w_breakeven=float(section.get("w_breakeven", 0.40)),
        w_risk_reward=float(section.get("w_risk_reward", 0.25)),
        w_expected_value=float(section.get("w_expected_value", 0.25)),
        w_iv_rank=float(section.get("w_iv_rank", 0.10)),
    )

    # ── Validation ────────────────────────────────────────────────────
    if config.mode not in ("shadow", "filter"):
        raise ValueError(
            f"probability_filter.mode must be 'shadow' or 'filter', "
            f"got {config.mode!r}"
        )

    if not (0.0 <= config.min_breakeven_prob <= 1.0):
        raise ValueError(
            f"min_breakeven_prob must be in [0, 1], got {config.min_breakeven_prob}"
        )

    if config.max_dte < 1:
        raise ValueError(f"max_dte must be >= 1, got {config.max_dte}")

    if not config.dte_candidates:
        logger.warning(
            "[prob_filter] dte_candidates is empty — no expirations to evaluate"
        )

    # Warn if scoring weights are far from 1.0
    weight_sum = (
        config.w_breakeven + config.w_risk_reward
        + config.w_expected_value + config.w_iv_rank
    )
    if abs(weight_sum - 1.0) > 0.05:
        logger.warning(
            "[prob_filter] Scoring weights sum to %.3f (expected ~1.0): "
            "w_breakeven=%.2f w_risk_reward=%.2f w_expected_value=%.2f w_iv_rank=%.2f",
            weight_sum, config.w_breakeven, config.w_risk_reward,
            config.w_expected_value, config.w_iv_rank,
        )

    return config


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SpreadCandidate:
    """One evaluated DTE candidate with all computed metrics."""

    # Identification
    underlying: str
    dte: int
    expiration: str

    # Contract details (from select_spread_contracts)
    long_contract: Dict[str, Any]
    short_contract: Dict[str, Any]
    atm_strike: float
    otm_strike: float
    spread_width: float
    underlying_price: float

    # Pricing from batch snapshot
    long_bid: float
    long_ask: float
    long_mid: float
    short_bid: float
    short_ask: float
    short_mid: float
    net_debit: float  # long_ask - short_bid (worst fill)

    # BSM-computed metrics
    bsm_iv: float               # ATM IV from our solver (or Alpaca fallback)
    breakeven_prob: float
    max_profit_prob: float
    risk_reward_ratio: float
    expected_value: float        # Per-share EV
    spread_efficiency: float     # debit / width

    # Greeks from batch snapshot (avoids redundant API calls)
    long_iv: float = 0.0
    long_delta: float = 0.0
    long_theta: float = 0.0
    short_iv: float = 0.0
    short_delta: float = 0.0
    short_theta: float = 0.0

    # Scoring
    composite_score: float = 0.0

    # Rejection tracking
    rejected: bool = False
    reject_reasons: List[str] = field(default_factory=list)

    def to_log_dict(self) -> Dict[str, Any]:
        """Flatten to a dict suitable for JSON logging.

        All keys are prefixed with ``pf_`` to avoid collisions when
        unpacked into ``spread_entry(**kw)`` which has its own
        ``underlying``, ``dte``, ``expiration``, etc. parameters.
        """
        return {
            "pf_underlying": self.underlying,
            "pf_dte": self.dte,
            "pf_expiration": self.expiration,
            "pf_atm_strike": self.atm_strike,
            "pf_otm_strike": self.otm_strike,
            "pf_spread_width": round(self.spread_width, 2),
            "pf_underlying_price": round(self.underlying_price, 2),
            "pf_net_debit": round(self.net_debit, 4),
            "pf_bsm_iv": round(self.bsm_iv, 4),
            "pf_breakeven_prob": round(self.breakeven_prob, 4),
            "pf_max_profit_prob": round(self.max_profit_prob, 4),
            "pf_risk_reward_ratio": round(self.risk_reward_ratio, 4),
            "pf_expected_value": round(self.expected_value, 4),
            "pf_spread_efficiency": round(self.spread_efficiency, 4),
            "pf_composite_score": round(self.composite_score, 4),
            "pf_rejected": self.rejected,
            "pf_reject_reasons": self.reject_reasons,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Sigmoid squash, maps (-∞, ∞) → (0, 1)."""
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def compute_composite_score(
    candidate: SpreadCandidate,
    config: ProbabilityFilterConfig,
    iv_rank_value: Optional[float] = None,
) -> float:
    """
    Compute composite score for a spread candidate.

    Formula::

        score = w_be * breakeven_prob
              + w_rr * min(risk_reward / 2.0, 1.0)
              + w_ev * sigmoid(expected_value / max_loss)
              + w_iv * (1 - iv_rank / 100)    [or 0.5 if unavailable]

    Args:
        candidate:     The evaluated spread candidate.
        config:        Filter configuration (weights).
        iv_rank_value: IV percentile rank [0, 100] or None.

    Returns:
        Composite score in [0, 1] (higher = better).
    """
    # Breakeven probability component
    be_score = candidate.breakeven_prob

    # Risk-reward component (capped at 1.0 when R:R ≥ 2.0)
    rr_score = min(candidate.risk_reward_ratio / 2.0, 1.0)

    # Expected value component (sigmoid-squashed)
    max_loss = candidate.net_debit if candidate.net_debit > 0 else 1.0
    ev_score = _sigmoid(candidate.expected_value / max_loss)

    # IV rank component (prefer lower IV rank = cheaper options)
    if iv_rank_value is not None and math.isfinite(iv_rank_value):
        iv_score = 1.0 - iv_rank_value / 100.0
    else:
        iv_score = 0.5  # neutral when unavailable

    score = (
        config.w_breakeven * be_score
        + config.w_risk_reward * rr_score
        + config.w_expected_value * ev_score
        + config.w_iv_rank * iv_score
    )

    return max(0.0, min(1.0, score))


# ──────────────────────────────────────────────────────────────────────────────
# Threshold checks
# ──────────────────────────────────────────────────────────────────────────────

def apply_threshold_filters(
    candidate: SpreadCandidate,
    config: ProbabilityFilterConfig,
) -> None:
    """
    Apply threshold filters to a candidate.  Mutates ``candidate.rejected``
    and ``candidate.reject_reasons`` in place.

    Args:
        candidate: The spread candidate to evaluate.
        config:    Filter configuration (thresholds).
    """
    reasons: List[str] = []

    if candidate.breakeven_prob < config.min_breakeven_prob:
        reasons.append(
            f"P_BE={candidate.breakeven_prob:.2f}<{config.min_breakeven_prob:.2f}"
        )

    if candidate.risk_reward_ratio < config.min_risk_reward:
        reasons.append(
            f"RR={candidate.risk_reward_ratio:.2f}<{config.min_risk_reward:.2f}"
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

def evaluate_dte_candidates(
    underlying: str,
    signal_atr: float,
    config: ProbabilityFilterConfig,
    spread_cfg: Dict[str, Any],
) -> Tuple[Optional[SpreadCandidate], List[SpreadCandidate]]:
    """
    Evaluate multiple DTE candidates for a bull call spread.

    Steps:
        1. Fetch underlying price once.
        2. For each DTE in ``config.dte_candidates`` (capped at ``max_dte``):
            a. Call ``select_spread_contracts()`` with pre-fetched price.
            b. Deduplicate by actual expiration date.
        3. Batch-fetch option snapshots for all viable leg symbols.
        4. For each viable spread:
            a. Validate bid-ask spread.
            b. Compute IV from mid-price (fallback to Alpaca IV).
            c. Compute BSM probability metrics.
            d. Apply threshold filters.
            e. Compute composite score.
        5. Return best candidate (highest score) or None.

    Args:
        underlying:  Stock symbol (e.g. "AAPL").
        signal_atr:  ATR from the entry signal.
        config:      Probability filter configuration.
        spread_cfg:  Spread configuration (dte, spread_width_atr, etc.).

    Returns:
        Tuple of (best_candidate_or_None, all_candidates_list).
    """
    # Lazy imports to avoid circular dependencies and keep zero overhead
    # when the filter is disabled
    from RubberBand.src.options_data import (
        get_underlying_price,
        select_spread_contracts,
        get_option_snapshots_batch,
    )
    from RubberBand.src.bsm import (
        compute_iv,
        evaluate_spread,
    )

    # Step 1: Fetch underlying price ONCE
    price = get_underlying_price(underlying)
    if price is None:
        logger.warning("[prob_filter] Cannot get price for %s", underlying)
        return None, []

    spread_width_atr = spread_cfg.get("spread_width_atr", 1.5)
    min_dte = spread_cfg.get("min_dte", 2)

    # Step 2: Collect spread candidates from each DTE, dedup by expiration
    seen_expirations: set = set()
    raw_spreads: List[Tuple[int, Dict[str, Any]]] = []

    for dte in config.dte_candidates:
        if dte > config.max_dte:
            continue

        spread = select_spread_contracts(
            underlying,
            dte=dte,
            spread_width_atr=spread_width_atr,
            atr=signal_atr,
            min_dte=min_dte,
            underlying_price=price,
        )
        if not spread:
            continue

        exp = spread.get("expiration", "")
        if exp in seen_expirations:
            continue  # Same Friday hit from different DTE targets
        seen_expirations.add(exp)
        raw_spreads.append((dte, spread))

    if not raw_spreads:
        logger.info("[prob_filter] No viable spreads found for %s", underlying)
        return None, []

    # Step 3: Collect all leg symbols and batch-fetch snapshots
    all_symbols: List[str] = []
    for _, sp in raw_spreads:
        long_sym = sp["long"].get("symbol", "")
        short_sym = sp["short"].get("symbol", "")
        if long_sym:
            all_symbols.append(long_sym)
        if short_sym:
            all_symbols.append(short_sym)

    snapshots = get_option_snapshots_batch(all_symbols)

    # Step 4: Evaluate each spread
    candidates: List[SpreadCandidate] = []

    for dte_target, sp in raw_spreads:
        long_sym = sp["long"].get("symbol", "")
        short_sym = sp["short"].get("symbol", "")
        long_snap = snapshots.get(long_sym)
        short_snap = snapshots.get(short_sym)

        # Skip if no snapshot data for either leg
        if not long_snap or not short_snap:
            logger.debug(
                "[prob_filter] Missing snapshot for %s legs (%s/%s)",
                underlying, long_sym, short_sym,
            )
            continue

        long_bid = long_snap["bid"]
        long_ask = long_snap["ask"]
        long_mid = long_snap["mid"]
        short_bid = short_snap["bid"]
        short_ask = short_snap["ask"]
        short_mid = short_snap["mid"]

        # 4a: Validate bid-ask spread on BOTH legs
        if long_mid > 0:
            ba_spread_pct = (long_ask - long_bid) / long_mid * 100
            if ba_spread_pct > config.max_bid_ask_spread_pct:
                logger.debug(
                    "[prob_filter] %s DTE %d: long bid-ask too wide (%.1f%% > %.1f%%)",
                    underlying, dte_target, ba_spread_pct, config.max_bid_ask_spread_pct,
                )
                continue

        if short_mid > 0:
            short_ba_pct = (short_ask - short_bid) / short_mid * 100
            if short_ba_pct > config.max_bid_ask_spread_pct:
                logger.debug(
                    "[prob_filter] %s DTE %d: short bid-ask too wide (%.1f%% > %.1f%%)",
                    underlying, dte_target, short_ba_pct, config.max_bid_ask_spread_pct,
                )
                continue

        # Net debit = long ask - short bid (worst fill)
        net_debit = long_ask - short_bid
        if net_debit <= 0:
            continue

        atm_strike = sp["atm_strike"]
        otm_strike = sp["otm_strike"]
        spread_width = sp["spread_width"]

        # Skip if debit exceeds width (guaranteed loss)
        if net_debit >= spread_width:
            continue

        # Compute actual DTE from expiration
        exp_str = sp.get("expiration", "")
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            today = datetime.now(ET).date()
            actual_dte = (exp_date - today).days
        except (ValueError, TypeError):
            actual_dte = dte_target

        T = actual_dte / 365.0
        if T <= 0:
            continue

        # 4b: Compute IV from mid-price via BSM solver
        atm_iv = float("nan")
        alpaca_iv = long_snap.get("iv", 0)

        if long_mid > 0 and T > 0:
            atm_iv = compute_iv(long_mid, price, atm_strike, T, config.risk_free_rate)

        # Fallback to Alpaca IV if solver fails
        if not math.isfinite(atm_iv) or atm_iv <= 0:
            if alpaca_iv > 0:
                atm_iv = alpaca_iv
                logger.debug(
                    "[prob_filter] %s DTE %d: BSM IV failed, using Alpaca IV=%.4f",
                    underlying, actual_dte, alpaca_iv,
                )
            else:
                logger.debug(
                    "[prob_filter] %s DTE %d: No valid IV available, skipping",
                    underlying, actual_dte,
                )
                continue

        # 4c: Compute BSM probability metrics
        spread_eval = evaluate_spread(
            S=price,
            atm_strike=atm_strike,
            otm_strike=otm_strike,
            net_debit=net_debit,
            T=T,
            atm_iv=atm_iv,
            r=config.risk_free_rate,
        )

        if not spread_eval.get("valid", False):
            continue

        candidate = SpreadCandidate(
            underlying=underlying,
            dte=actual_dte,
            expiration=exp_str,
            long_contract=sp["long"],
            short_contract=sp["short"],
            atm_strike=atm_strike,
            otm_strike=otm_strike,
            spread_width=spread_width,
            underlying_price=price,
            long_bid=long_bid,
            long_ask=long_ask,
            long_mid=long_mid,
            short_bid=short_bid,
            short_ask=short_ask,
            short_mid=short_mid,
            net_debit=net_debit,
            bsm_iv=atm_iv,
            breakeven_prob=spread_eval["breakeven_prob"],
            max_profit_prob=spread_eval["max_profit_prob"],
            risk_reward_ratio=spread_eval["risk_reward_ratio"],
            expected_value=spread_eval["expected_value"],
            spread_efficiency=spread_eval["spread_efficiency"],
            # Carry greeks from batch snapshot
            long_iv=long_snap.get("iv", 0),
            long_delta=long_snap.get("delta", 0),
            long_theta=long_snap.get("theta", 0),
            short_iv=short_snap.get("iv", 0),
            short_delta=short_snap.get("delta", 0),
            short_theta=short_snap.get("theta", 0),
        )

        # 4d: Apply threshold filters
        apply_threshold_filters(candidate, config)

        # 4e: Compute composite score
        candidate.composite_score = compute_composite_score(candidate, config)

        candidates.append(candidate)

    # Step 5: Select best non-rejected candidate
    if not candidates:
        logger.info("[prob_filter] No valid candidates for %s", underlying)
        return None, []

    # Log all candidates for analysis
    for c in candidates:
        logger.info(
            "[prob_filter] %s DTE=%d score=%.3f P_BE=%.2f RR=%.2f EV=%.2f %s",
            c.underlying, c.dte, c.composite_score,
            c.breakeven_prob, c.risk_reward_ratio, c.expected_value,
            "REJECTED:" + ",".join(c.reject_reasons) if c.rejected else "OK",
        )

    viable = [c for c in candidates if not c.rejected]

    if not viable:
        # All rejected — log why
        all_reasons = []
        for c in candidates:
            all_reasons.extend(c.reject_reasons)
        logger.info(
            "[prob_filter] All %d candidates rejected for %s: %s",
            len(candidates), underlying, "; ".join(set(all_reasons)),
        )
        return None, candidates

    # Pick highest composite score
    best = max(viable, key=lambda c: c.composite_score)

    logger.info(
        "[prob_filter] SELECTED %s DTE=%d score=%.3f P_BE=%.2f RR=%.2f EV=%.2f",
        best.underlying, best.dte, best.composite_score,
        best.breakeven_prob, best.risk_reward_ratio, best.expected_value,
    )

    return best, candidates
