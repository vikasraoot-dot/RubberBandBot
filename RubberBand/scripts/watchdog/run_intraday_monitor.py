#!/usr/bin/env python3
"""
Run Intraday Monitor: GitHub Actions entrypoint for the watchdog health monitor.

Runs one monitoring cycle: fetches live Alpaca data, evaluates per-bot
thresholds, updates pause flags and profit locks, and persists state files.

Usage:
    python RubberBand/scripts/watchdog/run_intraday_monitor.py [--reset]

Flags:
    --reset   Reset daily pause flags and profit locks (run at market open).

Environment:
    APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure repo root is on path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("watchdog.runner")


def main() -> int:
    """Parse arguments and run one monitor cycle.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    parser = argparse.ArgumentParser(description="Run watchdog intraday monitor")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset daily pause flags and profit locks (run at market open)",
    )
    args = parser.parse_args()

    try:
        from RubberBand.src.watchdog.intraday_monitor import IntraDayMonitor

        monitor = IntraDayMonitor()

        if args.reset:
            logger.info("Performing daily reset...")
            monitor.reset_daily()
            return 0

        logger.info("Running intraday monitor cycle...")
        health = monitor.run_cycle()

        if not health:
            logger.error("Monitor cycle returned no data")
            return 1

        logger.info("Cycle complete. Bots: %s", list(health.get("bots", {}).keys()))
        return 0

    except Exception as exc:
        logger.error("Fatal error in watchdog monitor: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
