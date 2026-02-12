#!/usr/bin/env python3
"""
Run Post-Day Analysis: CLI entrypoint for the watchdog post-day analyzer.

Runs all end-of-day analyses and saves the report to
``results/watchdog/daily_analysis/{YYYY-MM-DD}.json``.

Usage:
    python RubberBand/scripts/watchdog/run_post_day_analysis.py
    python RubberBand/scripts/watchdog/run_post_day_analysis.py --date 2026-02-10
    python RubberBand/scripts/watchdog/run_post_day_analysis.py --results-dir results --watchdog-dir results/watchdog
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

# Ensure repo root is on path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

ET = ZoneInfo("US/Eastern")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("watchdog.post_day")


def main() -> int:
    """Parse CLI arguments, run analysis, and report results.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    parser = argparse.ArgumentParser(
        description="Run watchdog post-day analysis for a given date."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now(ET).strftime("%Y-%m-%d"),
        help="Date to analyse (YYYY-MM-DD). Defaults to today (Eastern Time).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Base results directory (default: results).",
    )
    parser.add_argument(
        "--watchdog-dir",
        type=str,
        default="results/watchdog",
        help="Watchdog output directory (default: results/watchdog).",
    )
    args = parser.parse_args()

    logger.info("Post-Day Analyzer starting for date=%s", args.date)

    try:
        from RubberBand.src.watchdog.post_day_analyzer import PostDayAnalyzer

        analyzer = PostDayAnalyzer(
            results_dir=args.results_dir,
            watchdog_dir=args.watchdog_dir,
        )
        report = analyzer.analyze(args.date)
    except Exception:
        logger.exception("Post-day analysis failed")
        return 1

    # Print summary to stdout for GitHub Actions visibility
    bots = report.get("bots_analysed", [])
    filter_summary = report.get("filter_effectiveness", {}).get("_summary", {})
    giveback = report.get("giveback", {})
    auditor = report.get("auditor_vs_bot", {})

    print("\n" + "=" * 60)
    print(f"  Post-Day Analysis Complete: {args.date}")
    print("=" * 60)
    print(f"  Bots analysed        : {bots}")
    print(f"  Total filter blocks  : {filter_summary.get('total_blocks', 0)}")
    print(f"  Net filter value     : ${filter_summary.get('net_filter_value', 0):.2f}")

    for bot_tag, gb in giveback.items():
        print(f"  [{bot_tag}] Peak: ${gb['peak_pnl']:.2f} | "
              f"Close: ${gb['close_pnl']:.2f} | "
              f"Giveback: ${gb['giveback']:.2f}")

    for bot_tag, av in auditor.items():
        print(f"  [{bot_tag}] Real: ${av['real_pnl']:.2f} | "
              f"Shadow: ${av['shadow_pnl']:.2f} | "
              f"Verdict: {av['verdict']}")

    print("=" * 60)

    output_path = os.path.join(
        args.watchdog_dir, "daily_analysis", f"{args.date}.json"
    )
    print(f"  Report saved to: {output_path}")
    logger.info("Post-Day Analyzer finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
