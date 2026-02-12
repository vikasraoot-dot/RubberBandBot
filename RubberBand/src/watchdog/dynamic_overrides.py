"""
Dynamic Overrides Reader: Load market-condition-based parameter adjustments.

Reads ``results/watchdog/dynamic_overrides.json`` written by the
``MarketConditionClassifier``.  Returns sensible defaults if the file is
missing or corrupt (**fail-open**).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Resolve path relative to this file's location (repo_root/results/watchdog/)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_DEFAULT_OVERRIDES_PATH = os.path.join(_REPO_ROOT, "results", "watchdog", "dynamic_overrides.json")

_NEUTRAL_OVERRIDES: Dict[str, Any] = {
    "market_condition": "RANGE",
    "overrides": {
        "position_size_multiplier": 1.0,
        "tp_r_multiple_adjustment": 0.0,
        "breakeven_trigger_r_adjustment": 0.0,
    },
}


def load_dynamic_overrides(
    path: str = _DEFAULT_OVERRIDES_PATH,
) -> Dict[str, Any]:
    """Load dynamic overrides from disk.

    Args:
        path: Path to ``dynamic_overrides.json``.

    Returns:
        Dict with keys ``market_condition`` (str) and ``overrides`` (dict).
        Returns neutral defaults if the file is missing or corrupt.
    """
    if not os.path.exists(path):
        return dict(_NEUTRAL_OVERRIDES)

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or "overrides" not in data:
            return dict(_NEUTRAL_OVERRIDES)
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Could not read dynamic overrides (%s): %s", path, exc)
        return dict(_NEUTRAL_OVERRIDES)


def read_dynamic_overrides() -> Dict[str, Any]:
    """Read dynamic overrides from results/watchdog/dynamic_overrides.json.

    Canonical reader for dynamic overrides. Fail-open: returns neutral RANGE
    overrides if file is missing or corrupt so that live bots operate with
    default parameters.

    Returns:
        Dict with at least 'market_condition' and 'overrides' keys.
    """
    return load_dynamic_overrides(_DEFAULT_OVERRIDES_PATH)
