"""
XGBoost ML Signal Gate — Phase 3B of the AI Watchdog system.

Loads a trained XGBoost model and scores incoming trade signals.
Signals below the confidence threshold are rejected (logged as ML_GATE_SKIP).

Fail-open: if the model file is missing or loading fails, all signals are
approved so existing trading logic is unaffected.
"""
from __future__ import annotations

import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Module-level model cache to avoid reloading on every call
_cached_model = None
_cached_model_path: str = ""

# Default model location (relative to repo root)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_MODEL_PATH = os.path.join(_REPO_ROOT, "rubberband_xgb.json")

# Feature columns expected by the model (must match train_model.py)
FEATURE_COLS = [
    "rsi", "rsi_5", "adx", "macd_hist",
    "atr_pct", "dist_lower", "sma50_slope",
    "vol_rel", "hour",
]


def _load_model(model_path: str):
    """
    Load XGBoost model from disk, using module-level cache.

    Args:
        model_path: Path to the .json model file.

    Returns:
        Loaded XGBClassifier, or None if loading fails.
    """
    global _cached_model, _cached_model_path

    # Return cached if same path
    if _cached_model is not None and _cached_model_path == model_path:
        return _cached_model

    if not os.path.isfile(model_path):
        logger.warning("ML gate model not found at %s — running in fail-open mode", model_path)
        return None

    try:
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        _cached_model = model
        _cached_model_path = model_path
        logger.info("ML gate model loaded from %s", model_path)
        return model
    except ImportError:
        logger.warning("xgboost package not installed — ML gate disabled (fail-open)")
        return None
    except Exception as exc:
        logger.error("Failed to load ML gate model from %s: %s", model_path, exc, exc_info=True)
        return None


def ml_gate_check(
    row: pd.Series,
    model_path: str = DEFAULT_MODEL_PATH,
    threshold: float = 0.65,
) -> Tuple[bool, float]:
    """
    Score a trade signal using the XGBoost model.

    Args:
        row: A pandas Series containing feature columns produced by
             attach_verifiers() (rsi, rsi_5, adx, macd_hist, atr_pct,
             dist_lower, sma50_slope, vol_rel, hour).
        model_path: Path to the saved XGBoost model JSON file.
        threshold: Minimum confidence (probability of class 1) to approve
                   the signal. Signals below this are rejected.

    Returns:
        Tuple of (approved, confidence):
            approved — True if the model approves the signal (or if fail-open).
            confidence — Model's predicted probability (0.0 if fail-open).
    """
    model = _load_model(model_path)
    if model is None:
        # Fail-open: approve everything when model unavailable
        return (True, 0.0)

    try:
        # Extract features, filling missing values with safe defaults
        features: dict = {}
        for col in FEATURE_COLS:
            val = row.get(col, np.nan)
            if col == "hour" and (pd.isna(val) or val is None):
                # Try to extract hour from row index (timestamp)
                if isinstance(row.name, pd.Timestamp):
                    val = row.name.hour
                else:
                    val = 12  # midday default
            features[col] = float(val) if pd.notna(val) else 0.0

        X = pd.DataFrame([features], columns=FEATURE_COLS)

        # Get probability of positive class (class 1 = winner)
        proba = model.predict_proba(X)[0]
        confidence = float(proba[1]) if len(proba) > 1 else float(proba[0])

        approved = confidence >= threshold
        return (approved, confidence)

    except Exception as exc:
        logger.error("ML gate scoring failed: %s", exc, exc_info=True)
        # Fail-open on scoring errors
        return (True, 0.0)
