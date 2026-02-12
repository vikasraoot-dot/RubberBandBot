#!/usr/bin/env python3
"""
Weekly XGBoost model retraining script — Phase 3B of the AI Watchdog system.

Collects training data from performance.jsonl and JSONL trade logs,
trains an XGBoost classifier with the same hyperparameters as train_model.py,
and only replaces the production model if the new accuracy is within 2% of
the old model's accuracy.

Usage:
    python -m RubberBand.scripts.watchdog.retrain_model [--days 90] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

MODEL_PATH = os.path.join(_REPO_ROOT, "rubberband_xgb.json")
PERFORMANCE_PATH = os.path.join(_REPO_ROOT, "results", "watchdog", "performance.jsonl")
TRADE_LOGS_DIR = os.path.join(_REPO_ROOT, "results")
RETRAIN_LOG_PATH = os.path.join(_REPO_ROOT, "results", "watchdog", "retrain_log.jsonl")

# Feature columns — must match train_model.py and ml_gate.py
FEATURE_COLS = [
    "rsi", "rsi_5", "adx", "macd_hist",
    "atr_pct", "dist_lower", "sma50_slope",
    "vol_rel", "hour",
]

# XGBoost hyperparameters — same as train_model.py
XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
}

# Safety margin: only replace model if new_accuracy > old_accuracy - ACCURACY_TOLERANCE
ACCURACY_TOLERANCE = 0.02


def collect_training_data(days: int = 90) -> pd.DataFrame:
    """
    Collect training data from JSONL trade logs.

    Searches for JSONL files in results/ directory that contain per-trade
    records with the required feature columns and a 'pnl' or 'label' field.

    Args:
        days: Number of days of history to include.

    Returns:
        DataFrame with feature columns and 'label' column (1 = winner, 0 = loser).
    """
    records: List[Dict] = []
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)

    # Scan for JSONL trade log files
    log_dir = Path(TRADE_LOGS_DIR)
    jsonl_files = list(log_dir.glob("*.jsonl")) + list(log_dir.glob("**/*.jsonl"))

    for fpath in jsonl_files:
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Check if record has the features we need
                    has_features = all(col in rec for col in FEATURE_COLS)
                    has_outcome = "label" in rec or "pnl" in rec or "realized_pnl" in rec

                    if not (has_features and has_outcome):
                        continue

                    # Date filter
                    ts_str = rec.get("timestamp") or rec.get("date") or rec.get("closed_at") or ""
                    if ts_str:
                        try:
                            ts = pd.Timestamp(ts_str)
                            if ts.tzinfo is None:
                                ts = ts.tz_localize("UTC")
                            if ts < cutoff:
                                continue
                        except Exception as e:
                            logger.debug("Date parse skipped for record: %s", e)

                    # Derive label if not present
                    if "label" not in rec:
                        pnl = float(rec.get("pnl", rec.get("realized_pnl", 0.0)))
                        rec["label"] = 1 if pnl > 0 else 0

                    records.append(rec)
        except Exception as exc:
            logger.warning("Error reading %s: %s", fpath, exc)

    if not records:
        logger.warning("No training records found in %s", TRADE_LOGS_DIR)
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Ensure required columns exist and fill NaN
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df["label"] = df["label"].astype(int)

    logger.info("Collected %d training records from %d files", len(df), len(jsonl_files))
    return df


def evaluate_existing_model(X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Evaluate the existing production model on the test set.

    Args:
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Accuracy score, or 0.0 if model doesn't exist or fails to load.
    """
    if not os.path.isfile(MODEL_PATH):
        logger.info("No existing model at %s — will accept any new model", MODEL_PATH)
        return 0.0

    try:
        import xgboost as xgb
        old_model = xgb.XGBClassifier()
        old_model.load_model(MODEL_PATH)
        from sklearn.metrics import accuracy_score
        y_pred = old_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info("Existing model accuracy: %.4f", acc)
        return float(acc)
    except Exception as exc:
        logger.warning("Failed to evaluate existing model: %s", exc)
        return 0.0


def train_and_evaluate(
    df: pd.DataFrame,
    dry_run: bool = False,
) -> Dict:
    """
    Train a new XGBoost model and compare against the existing one.

    Args:
        df: Training data with feature columns and 'label'.
        dry_run: If True, log results but do not replace the model file.

    Returns:
        Dict with training results (new_accuracy, old_accuracy, replaced, etc.)
    """
    try:
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, classification_report
    except ImportError as exc:
        logger.error("Required packages not installed: %s", exc)
        return {"error": str(exc)}

    X = df[FEATURE_COLS].copy()
    y = df["label"].copy()

    # Time-based split (80/20)
    split_idx = int(len(df) * 0.8)
    if split_idx < 10 or (len(df) - split_idx) < 5:
        logger.error("Insufficient data for train/test split (%d records)", len(df))
        return {"error": f"Insufficient data: {len(df)} records"}

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info("Training on %d samples, testing on %d samples", len(X_train), len(X_test))

    # Train new model
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train)

    # Evaluate new model
    y_pred = model.predict(X_test)
    new_accuracy = float(accuracy_score(y_test, y_pred))
    logger.info("New model accuracy: %.4f", new_accuracy)

    # Evaluate existing model
    old_accuracy = evaluate_existing_model(X_test, y_test)

    # Decision: replace only if new >= old - tolerance
    should_replace = new_accuracy >= (old_accuracy - ACCURACY_TOLERANCE)
    replaced = False

    if should_replace and not dry_run:
        model.save_model(MODEL_PATH)
        replaced = True
        logger.info("Model replaced: new=%.4f >= old=%.4f - %.4f",
                     new_accuracy, old_accuracy, ACCURACY_TOLERANCE)
    elif not should_replace:
        logger.warning(
            "Model NOT replaced: new=%.4f < old=%.4f - %.4f (%.4f)",
            new_accuracy, old_accuracy, ACCURACY_TOLERANCE,
            old_accuracy - ACCURACY_TOLERANCE,
        )
    elif dry_run:
        logger.info("Dry run — model not replaced (would have: %s)", should_replace)

    # Feature importance
    importance = dict(zip(FEATURE_COLS, [float(x) for x in model.feature_importances_]))

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "new_accuracy": round(new_accuracy, 4),
        "old_accuracy": round(old_accuracy, 4),
        "accuracy_threshold": round(old_accuracy - ACCURACY_TOLERANCE, 4),
        "should_replace": should_replace,
        "replaced": replaced,
        "dry_run": dry_run,
        "feature_importance": importance,
    }

    # Log retrain result
    _log_retrain_result(result)

    return result


def _log_retrain_result(result: Dict) -> None:
    """
    Append retrain result to the retrain log JSONL file.

    Args:
        result: Dict of retrain metrics to log.
    """
    try:
        os.makedirs(os.path.dirname(RETRAIN_LOG_PATH), exist_ok=True)
        with open(RETRAIN_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, default=str) + "\n")
    except Exception as exc:
        logger.error("Failed to write retrain log: %s", exc)


def main() -> None:
    """CLI entrypoint for weekly model retraining."""
    parser = argparse.ArgumentParser(description="Retrain XGBoost ML gate model")
    parser.add_argument("--days", type=int, default=90, help="Days of history to use (default: 90)")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate but don't replace model")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print(f"{'='*60}")
    print(f" ML Gate Retraining (last {args.days} days)")
    print(f"{'='*60}")

    # Collect data
    df = collect_training_data(days=args.days)
    if df.empty or len(df) < 20:
        print(f"Insufficient training data ({len(df)} records). Need at least 20.")
        return

    print(f"Collected {len(df)} training records")
    print(f"  Label distribution: {dict(df['label'].value_counts())}")

    # Train and evaluate
    result = train_and_evaluate(df, dry_run=args.dry_run)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n{'='*60}")
    print(f" Results")
    print(f"{'='*60}")
    print(f"  New accuracy : {result['new_accuracy']:.4f}")
    print(f"  Old accuracy : {result['old_accuracy']:.4f}")
    print(f"  Threshold    : {result['accuracy_threshold']:.4f}")
    print(f"  Replaced     : {result['replaced']}")
    print(f"  Dry run      : {result['dry_run']}")
    print(f"\n  Feature Importance:")
    for feat, imp in sorted(result["feature_importance"].items(), key=lambda x: -x[1]):
        print(f"    {feat:15s} : {imp:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
