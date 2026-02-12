#!/usr/bin/env python3
"""
Apply Approved Changes: Reads approved_changes.yaml and applies them to config.yaml.

Pre-market workflow reads this file, applies approved parameter changes,
and commits the updated config with before/after logging for auditability.

Usage:
    python RubberBand/scripts/watchdog/apply_approved_changes.py
    python RubberBand/scripts/watchdog/apply_approved_changes.py --dry-run
    python RubberBand/scripts/watchdog/apply_approved_changes.py --config path/to/config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

# Ensure repo root is on path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

ET = ZoneInfo("US/Eastern")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("watchdog.apply_changes")


def _get_nested(d: Dict[str, Any], dotted_key: str) -> Any:
    """Get a value from a nested dict using dot notation.

    Args:
        d: The dict to query.
        dotted_key: Key path like ``"filters.rsi_oversold"``.

    Returns:
        The value at that path, or None if not found.
    """
    keys = dotted_key.split(".")
    current = d
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return None
    return current


def _set_nested(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dot notation.

    Creates intermediate dicts if they don't exist.

    Args:
        d: The dict to mutate.
        dotted_key: Key path like ``"filters.rsi_oversold"``.
        value: The value to set.
    """
    keys = dotted_key.split(".")
    current = d
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


def apply_approved_changes(
    approved_path: str = "results/watchdog/approved_changes.yaml",
    config_path: str = "config.yaml",
    alerts_path: str = "results/watchdog/alerts.jsonl",
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Read approved changes and apply them to the trading config.

    Only changes with ``approved: true`` and ``apply_from`` <= today
    are applied.  Each applied change is logged to ``alerts.jsonl``
    with before/after values.

    Args:
        approved_path: Path to the approved_changes.yaml file.
        config_path: Path to the trading config.yaml file.
        alerts_path: Path to append alert logs.
        dry_run: If True, log changes but do not write files.

    Returns:
        List of applied change dicts with before/after values.
    """
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML is required to apply changes")
        return []

    # Load approved changes
    if not os.path.exists(approved_path):
        logger.info("No approved_changes.yaml found at %s", approved_path)
        return []

    try:
        with open(approved_path, "r", encoding="utf-8") as fh:
            approved_data = yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.error("Failed to read approved changes: %s", exc)
        return []

    changes: List[Dict[str, Any]] = approved_data.get("changes", [])
    if not changes:
        logger.info("No changes defined in approved_changes.yaml")
        return []

    # Load config
    if not os.path.exists(config_path):
        logger.error("Config file not found: %s", config_path)
        return []

    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.error("Failed to read config: %s", exc)
        return []

    today = datetime.now(ET).strftime("%Y-%m-%d")
    applied: List[Dict[str, Any]] = []

    for change in changes:
        change_id = change.get("id", "unknown")
        param = change.get("param", "")
        new_value = change.get("value")
        is_approved = change.get("approved", False)
        apply_from = change.get("apply_from", "")
        already_applied = change.get("applied", False)

        # Skip unapproved, not-yet-due, or already-applied changes
        if not is_approved:
            logger.debug("Skipping unapproved change: %s", change_id)
            continue
        if apply_from and apply_from > today:
            logger.debug(
                "Skipping change %s: apply_from %s is in the future",
                change_id, apply_from,
            )
            continue
        if already_applied:
            logger.debug("Skipping already-applied change: %s", change_id)
            continue
        if not param:
            logger.warning("Change %s has no param field, skipping", change_id)
            continue

        old_value = _get_nested(config, param)

        if dry_run:
            logger.info(
                "[DRY RUN] Would apply %s: %s = %s -> %s",
                change_id, param, old_value, new_value,
            )
        else:
            _set_nested(config, param, new_value)
            logger.info(
                "Applied %s: %s = %s -> %s",
                change_id, param, old_value, new_value,
            )

        record = {
            "id": change_id,
            "param": param,
            "old_value": old_value,
            "new_value": new_value,
            "applied_at": datetime.now(ET).isoformat(),
            "dry_run": dry_run,
        }
        applied.append(record)

        # Log to alerts
        alert = {
            "ts": datetime.now(ET).isoformat(),
            "type": "PARAM_CHANGE",
            "change_id": change_id,
            "param": param,
            "old_value": str(old_value),
            "new_value": str(new_value),
            "dry_run": dry_run,
        }
        _append_alert(alerts_path, alert)

    if not dry_run and applied:
        # Write updated config
        try:
            with open(config_path, "w", encoding="utf-8") as fh:
                yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
            logger.info("Config saved to %s", config_path)
        except Exception as exc:
            logger.error("Failed to write config: %s", exc)

        # Mark changes as applied in the approved_changes file
        for change in changes:
            for rec in applied:
                if change.get("id") == rec["id"] and not rec["dry_run"]:
                    change["applied"] = True
                    change["applied_at"] = rec["applied_at"]
        try:
            with open(approved_path, "w", encoding="utf-8") as fh:
                yaml.dump(approved_data, fh, default_flow_style=False, sort_keys=False)
        except Exception as exc:
            logger.error("Failed to update approved_changes: %s", exc)

    return applied


def _append_alert(path: str, alert: Dict[str, Any]) -> None:
    """Append an alert record to the JSONL alert log.

    Args:
        path: Path to alerts.jsonl.
        alert: Alert dict to append.
    """
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        line = json.dumps(alert, separators=(",", ":"), default=str)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        logger.warning("Failed to write alert: %s", exc)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Apply approved parameter changes to config.yaml."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log changes without writing to files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the trading config.yaml.",
    )
    parser.add_argument(
        "--approved",
        type=str,
        default="results/watchdog/approved_changes.yaml",
        help="Path to approved_changes.yaml.",
    )
    args = parser.parse_args()

    applied = apply_approved_changes(
        approved_path=args.approved,
        config_path=args.config,
        dry_run=args.dry_run,
    )

    if applied:
        print(f"\nApplied {len(applied)} change(s):")
        for rec in applied:
            prefix = "[DRY RUN] " if rec["dry_run"] else ""
            print(
                f"  {prefix}{rec['id']}: {rec['param']} = "
                f"{rec['old_value']} -> {rec['new_value']}"
            )
    else:
        print("\nNo changes to apply.")


if __name__ == "__main__":
    main()
