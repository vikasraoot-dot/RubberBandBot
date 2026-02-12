#!/usr/bin/env python3
"""
Generate Weekly Report: Produces a markdown summary of the week's trading performance.

Runs Saturday morning (via GitHub Actions or manually).  Reads performance.jsonl,
daily analysis JSONs, rolling stats, and auditor data to produce a human-readable
markdown report at ``results/watchdog/weekly_reports/{YYYY}-W{##}.md``.

Usage:
    python RubberBand/scripts/watchdog/generate_weekly_report.py
    python RubberBand/scripts/watchdog/generate_weekly_report.py --week 2026-W07
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

# Ensure repo root is on path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.watchdog.utils import to_dec as _dec, dec_to_float as _to_float

ET = ZoneInfo("US/Eastern")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("watchdog.weekly_report")


class WeeklyReportGenerator:
    """Generates a weekly markdown report summarising trading performance.

    Args:
        watchdog_dir: Path to the watchdog results directory.
        results_dir: Path to the base results directory.
    """

    def __init__(
        self,
        watchdog_dir: str = "results/watchdog",
        results_dir: str = "results",
    ) -> None:
        self._watchdog_dir = watchdog_dir
        self._results_dir = results_dir
        self._reports_dir = os.path.join(watchdog_dir, "weekly_reports")
        os.makedirs(self._reports_dir, exist_ok=True)

    def generate(self, week_label: Optional[str] = None) -> str:
        """Generate the weekly report and save it to disk.

        Args:
            week_label: ISO week label like ``"2026-W07"``.
                Defaults to current week.

        Returns:
            Path to the generated markdown file.
        """
        now = datetime.now(ET)
        if not week_label:
            iso = now.isocalendar()
            week_label = f"{iso[0]}-W{iso[1]:02d}"

        week_start, week_end = self._week_date_range(week_label)
        logger.info(
            "Generating weekly report %s (%s to %s)",
            week_label, week_start, week_end,
        )

        # Gather data
        perf_entries = self._load_performance_entries(week_start, week_end)
        daily_analyses = self._load_daily_analyses(week_start, week_end)
        rolling_stats = self._load_rolling_stats()
        prev_week_label = self._previous_week_label(week_label)
        prev_start, prev_end = self._week_date_range(prev_week_label)
        prev_entries = self._load_performance_entries(prev_start, prev_end)

        # Build report
        md = self._build_report(
            week_label=week_label,
            week_start=week_start,
            week_end=week_end,
            perf_entries=perf_entries,
            prev_entries=prev_entries,
            daily_analyses=daily_analyses,
            rolling_stats=rolling_stats,
        )

        # Save
        output_path = os.path.join(self._reports_dir, f"{week_label}.md")
        try:
            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write(md)
            logger.info("Saved weekly report to %s", output_path)
        except OSError as exc:
            logger.error("Failed to save weekly report: %s", exc)

        return output_path

    # ------------------------------------------------------------------
    # Report building
    # ------------------------------------------------------------------

    def _build_report(
        self,
        week_label: str,
        week_start: str,
        week_end: str,
        perf_entries: List[Dict[str, Any]],
        prev_entries: List[Dict[str, Any]],
        daily_analyses: List[Dict[str, Any]],
        rolling_stats: Dict[str, Any],
    ) -> str:
        """Assemble the full markdown report.

        Args:
            week_label: ISO week label.
            week_start: Start date string.
            week_end: End date string.
            perf_entries: Performance DB entries for this week.
            prev_entries: Performance DB entries for previous week.
            daily_analyses: Daily analysis JSONs for this week.
            rolling_stats: Current rolling stats.

        Returns:
            Complete markdown string.
        """
        sections: List[str] = []
        sections.append(f"# Watchdog Weekly Report: {week_label}")
        sections.append(f"\n**Period**: {week_start} to {week_end}")
        sections.append(
            f"**Generated**: {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}\n"
        )

        # 1. Performance summary table
        sections.append(self._section_performance_summary(
            perf_entries, prev_entries
        ))

        # 2. Shadow ledger comparison
        sections.append(self._section_shadow_comparison(perf_entries))

        # 3. Key findings from daily analyses
        sections.append(self._section_key_findings(daily_analyses))

        # 4. Filter effectiveness summary
        sections.append(self._section_filter_summary(daily_analyses))

        # 5. Anomaly flags
        sections.append(self._section_anomaly_flags(rolling_stats))

        # 6. Giveback analysis
        sections.append(self._section_giveback(daily_analyses))

        # 7. Ticker patterns
        sections.append(self._section_ticker_patterns(daily_analyses))

        return "\n".join(sections)

    def _section_performance_summary(
        self,
        entries: List[Dict[str, Any]],
        prev_entries: List[Dict[str, Any]],
    ) -> str:
        """Build the performance summary table.

        Args:
            entries: This week's performance entries.
            prev_entries: Previous week's performance entries.

        Returns:
            Markdown string.
        """
        lines: List[str] = []
        lines.append("## Performance Summary\n")
        lines.append("| Bot | Trades | Win Rate | P&L | vs Last Week | Status |")
        lines.append("|-----|--------|----------|-----|--------------|--------|")

        this_week = self._aggregate_by_bot(entries)
        last_week = self._aggregate_by_bot(prev_entries)

        for bot_tag in sorted(this_week.keys()):
            tw = this_week[bot_tag]
            lw = last_week.get(bot_tag, {})

            trades = tw.get("trades", 0)
            win_rate = tw.get("win_rate", 0.0)
            pnl = tw.get("pnl", 0.0)

            lw_pnl = lw.get("pnl", 0.0)
            delta_pnl = pnl - lw_pnl

            if delta_pnl > 0:
                delta_str = f"BETTER (+${delta_pnl:.0f})"
            elif delta_pnl < 0:
                delta_str = f"WORSE (${delta_pnl:.0f})"
            else:
                delta_str = "SAME"

            status = "OK"
            if win_rate < 35:
                status = "DEGRADED"
            elif pnl < -100:
                status = "LOSING"

            lines.append(
                f"| {bot_tag} | {trades} | {win_rate:.0f}% "
                f"| ${pnl:+.2f} | {delta_str} | {status} |"
            )

        if not this_week:
            lines.append("| *(no data)* | - | - | - | - | - |")

        lines.append("")
        return "\n".join(lines)

    def _section_shadow_comparison(
        self, entries: List[Dict[str, Any]]
    ) -> str:
        """Build the shadow ledger comparison table.

        Args:
            entries: This week's performance entries.

        Returns:
            Markdown string.
        """
        lines: List[str] = []
        lines.append("## Shadow Ledger Comparison (Auditor vs Real)\n")
        lines.append("| Bot | Real P&L | Shadow P&L | Verdict |")
        lines.append("|-----|----------|------------|---------|")

        by_bot = self._aggregate_by_bot(entries)

        for bot_tag in sorted(by_bot.keys()):
            stats = by_bot[bot_tag]
            real = stats.get("pnl", 0.0)
            shadow = stats.get("shadow_pnl", 0.0)
            diff = real - shadow

            if shadow > real and shadow > 0:
                verdict = f"FILTERS_TOO_STRICT (missed ${abs(diff):.0f})"
            elif real > shadow:
                verdict = f"FILTERS_ADDING_VALUE (saved ${abs(diff):.0f})"
            else:
                verdict = "NEUTRAL"

            lines.append(
                f"| {bot_tag} | ${real:+.2f} | ${shadow:+.2f} | {verdict} |"
            )

        if not by_bot:
            lines.append("| *(no data)* | - | - | - |")

        lines.append("")
        return "\n".join(lines)

    def _section_key_findings(
        self, analyses: List[Dict[str, Any]]
    ) -> str:
        """Summarise key findings from daily analyses.

        Args:
            analyses: List of daily analysis dicts.

        Returns:
            Markdown string.
        """
        lines: List[str] = []
        lines.append("## Key Findings\n")

        if not analyses:
            lines.append("- No daily analyses available for this week.\n")
            return "\n".join(lines)

        # Aggregate filter blocks
        total_blocks = 0
        net_filter_value = Decimal("0")
        for a in analyses:
            fe = a.get("filter_effectiveness", {}).get("_summary", {})
            total_blocks += fe.get("total_blocks", 0)
            net_filter_value += _dec(fe.get("net_filter_value", 0))

        lines.append(
            f"- Filters blocked **{total_blocks}** trades this week, "
            f"net value: **${_to_float(net_filter_value):.2f}**"
        )

        # Average giveback
        givebacks: List[Decimal] = []
        for a in analyses:
            for bot_tag, gb in a.get("giveback", {}).items():
                givebacks.append(_dec(gb.get("giveback", 0)))
        if givebacks:
            avg_gb = sum(givebacks, Decimal("0")) / Decimal(str(len(givebacks)))
            lines.append(
                f"- Average daily giveback: **${_to_float(avg_gb):.2f}**"
            )

        # Flagged losers across the week
        flagged: Dict[str, int] = {}
        for a in analyses:
            for sym in a.get("ticker_analysis", {}).get("flagged_losers", []):
                flagged[sym] = flagged.get(sym, 0) + 1
        repeat_losers = [sym for sym, cnt in flagged.items() if cnt >= 2]
        if repeat_losers:
            lines.append(
                f"- Repeat losing tickers: **{', '.join(repeat_losers)}**"
            )

        lines.append("")
        return "\n".join(lines)

    def _section_filter_summary(
        self, analyses: List[Dict[str, Any]]
    ) -> str:
        """Aggregate filter effectiveness across the week.

        Args:
            analyses: List of daily analysis dicts.

        Returns:
            Markdown string.
        """
        lines: List[str] = []
        lines.append("## Filter Effectiveness (Week Total)\n")
        lines.append("| Filter | Blocks | Shadow Winners | Shadow Losers | Net Value |")
        lines.append("|--------|--------|---------------|---------------|-----------|")

        # Aggregate across days
        aggregated: Dict[str, Dict[str, Any]] = {}
        for a in analyses:
            fe = a.get("filter_effectiveness", {})
            for fname, fdata in fe.items():
                if fname.startswith("_") or not isinstance(fdata, dict):
                    continue
                if fname not in aggregated:
                    aggregated[fname] = {
                        "blocks": 0,
                        "shadow_winners": 0,
                        "shadow_losers": 0,
                        "net_value": Decimal("0"),
                    }
                aggregated[fname]["blocks"] += fdata.get("blocked_count", 0)
                aggregated[fname]["shadow_winners"] += fdata.get("shadow_winners", 0)
                aggregated[fname]["shadow_losers"] += fdata.get("shadow_losers", 0)
                aggregated[fname]["net_value"] += _dec(fdata.get("net_value", 0))

        for fname in sorted(aggregated.keys()):
            d = aggregated[fname]
            lines.append(
                f"| {fname} | {d['blocks']} | {d['shadow_winners']} "
                f"| {d['shadow_losers']} | ${_to_float(d['net_value']):.2f} |"
            )

        if not aggregated:
            lines.append("| *(no filter data)* | - | - | - | - |")

        lines.append("")
        return "\n".join(lines)

    def _section_anomaly_flags(
        self, rolling_stats: Dict[str, Any]
    ) -> str:
        """List active anomaly flags from rolling stats.

        Args:
            rolling_stats: Current rolling stats dict.

        Returns:
            Markdown string.
        """
        lines: List[str] = []
        lines.append("## Active Anomaly Flags\n")

        bots = rolling_stats.get("bots", {})
        any_flags = False
        for bot_tag in sorted(bots.keys()):
            flags = bots[bot_tag].get("flags", [])
            for f in flags:
                any_flags = True
                severity = f.get("severity", "INFO")
                flag_name = f.get("flag", "UNKNOWN")
                detail = f.get("detail", "")
                icon = "!!!" if severity == "CRITICAL" else "!"
                lines.append(f"- **[{bot_tag}] {flag_name}** ({severity}) {icon} — {detail}")

        if not any_flags:
            lines.append("- No anomaly flags active.\n")

        lines.append("")
        return "\n".join(lines)

    def _section_giveback(
        self, analyses: List[Dict[str, Any]]
    ) -> str:
        """Summarise giveback patterns across the week.

        Args:
            analyses: List of daily analysis dicts.

        Returns:
            Markdown string.
        """
        lines: List[str] = []
        lines.append("## Giveback Analysis\n")
        lines.append("| Date | Bot | Peak P&L | Close P&L | Giveback | Giveback % |")
        lines.append("|------|-----|----------|-----------|----------|------------|")

        for a in sorted(analyses, key=lambda x: x.get("date", "")):
            date = a.get("date", "?")
            for bot_tag, gb in a.get("giveback", {}).items():
                peak = gb.get("peak_pnl", 0)
                close = gb.get("close_pnl", 0)
                giveback = gb.get("giveback", 0)
                gb_pct = gb.get("giveback_pct", 0)
                if peak != 0 or close != 0:
                    lines.append(
                        f"| {date} | {bot_tag} | ${peak:.2f} "
                        f"| ${close:.2f} | ${giveback:.2f} | {gb_pct:.0f}% |"
                    )

        lines.append("")
        return "\n".join(lines)

    def _section_ticker_patterns(
        self, analyses: List[Dict[str, Any]]
    ) -> str:
        """Identify recurring ticker patterns across the week.

        Args:
            analyses: List of daily analysis dicts.

        Returns:
            Markdown string.
        """
        lines: List[str] = []
        lines.append("## Ticker Patterns\n")

        # Aggregate ticker stats
        ticker_agg: Dict[str, Dict[str, Any]] = {}
        for a in analyses:
            per_ticker = a.get("ticker_analysis", {}).get("per_ticker", {})
            for sym, stats in per_ticker.items():
                if sym not in ticker_agg:
                    ticker_agg[sym] = {
                        "trades": 0,
                        "winners": 0,
                        "losers": 0,
                        "pnl": Decimal("0"),
                    }
                ticker_agg[sym]["trades"] += stats.get("trades", 0)
                ticker_agg[sym]["winners"] += stats.get("winners", 0)
                ticker_agg[sym]["losers"] += stats.get("losers", 0)
                ticker_agg[sym]["pnl"] += _dec(stats.get("total_pnl", 0))

        if not ticker_agg:
            lines.append("- No ticker trade data available.\n")
            return "\n".join(lines)

        # Sort by absolute P&L impact
        sorted_tickers = sorted(
            ticker_agg.items(),
            key=lambda x: abs(x[1]["pnl"]),
            reverse=True,
        )

        lines.append("| Ticker | Trades | Winners | Losers | P&L | Win Rate |")
        lines.append("|--------|--------|---------|--------|-----|----------|")

        for sym, stats in sorted_tickers[:15]:  # Top 15
            count = stats["trades"]
            wr = (
                (stats["winners"] / count * 100) if count > 0 else 0
            )
            lines.append(
                f"| {sym} | {count} | {stats['winners']} | {stats['losers']} "
                f"| ${_to_float(stats['pnl']):.2f} | {wr:.0f}% |"
            )

        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_performance_entries(
        self, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """Load performance DB entries within a date range.

        Args:
            start_date: Start date YYYY-MM-DD (inclusive).
            end_date: End date YYYY-MM-DD (inclusive).

        Returns:
            List of performance dicts.
        """
        db_path = os.path.join(self._watchdog_dir, "performance.jsonl")
        results: List[Dict[str, Any]] = []

        if not os.path.exists(db_path):
            return results

        try:
            with open(db_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    d = entry.get("date", "")
                    if start_date <= d <= end_date:
                        results.append(entry)
        except OSError as exc:
            logger.error("Failed to read performance DB: %s", exc)

        return results

    def _load_daily_analyses(
        self, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """Load daily analysis JSONs within a date range.

        Args:
            start_date: Start date YYYY-MM-DD (inclusive).
            end_date: End date YYYY-MM-DD (inclusive).

        Returns:
            List of analysis dicts.
        """
        analysis_dir = os.path.join(self._watchdog_dir, "daily_analysis")
        results: List[Dict[str, Any]] = []

        if not os.path.isdir(analysis_dir):
            return results

        try:
            for fname in os.listdir(analysis_dir):
                if not fname.endswith(".json"):
                    continue
                date_str = fname.replace(".json", "")
                if start_date <= date_str <= end_date:
                    fpath = os.path.join(analysis_dir, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as fh:
                            results.append(json.load(fh))
                    except (json.JSONDecodeError, OSError) as exc:
                        logger.warning("Failed to read %s: %s", fpath, exc)
        except OSError as exc:
            logger.error("Failed to list analysis dir: %s", exc)

        return results

    def _load_rolling_stats(self) -> Dict[str, Any]:
        """Load rolling_stats.json.

        Returns:
            Rolling stats dict, or empty dict if missing.
        """
        path = os.path.join(self._watchdog_dir, "rolling_stats.json")
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read rolling stats: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _week_date_range(week_label: str) -> Tuple[str, str]:
        """Convert an ISO week label to start/end date strings.

        Args:
            week_label: ISO week like ``"2026-W07"``.

        Returns:
            Tuple of (monday_date, friday_date) as YYYY-MM-DD strings.
        """
        try:
            # Parse ISO week to Monday date
            year, week_num = week_label.split("-W")
            monday = datetime.strptime(f"{year} {week_num} 1", "%G %V %u")
            friday = monday + timedelta(days=4)
            return monday.strftime("%Y-%m-%d"), friday.strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            # Fallback to current week
            now = datetime.now(ET)
            monday = now - timedelta(days=now.weekday())
            friday = monday + timedelta(days=4)
            return monday.strftime("%Y-%m-%d"), friday.strftime("%Y-%m-%d")

    @staticmethod
    def _previous_week_label(week_label: str) -> str:
        """Get the previous week's ISO label.

        Args:
            week_label: Current ISO week like ``"2026-W07"``.

        Returns:
            Previous week label like ``"2026-W06"``.
        """
        try:
            year, week_num = week_label.split("-W")
            prev_monday = datetime.strptime(
                f"{year} {week_num} 1", "%G %V %u"
            ) - timedelta(weeks=1)
            iso = prev_monday.isocalendar()
            return f"{iso[0]}-W{iso[1]:02d}"
        except (ValueError, AttributeError):
            return week_label

    @staticmethod
    def _aggregate_by_bot(
        entries: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate performance entries by bot_tag.

        Uses Decimal for P&L accumulators; converts to float at the end
        so callers see plain floats for display/formatting.

        Args:
            entries: List of performance dicts.

        Returns:
            Dict mapping bot_tag -> aggregated stats.
        """
        accum: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            tag = entry.get("bot_tag", "")
            if not tag:
                continue
            if tag not in accum:
                accum[tag] = {
                    "trades": 0,
                    "winners": 0,
                    "losers": 0,
                    "pnl": Decimal("0"),
                    "shadow_pnl": Decimal("0"),
                }
            accum[tag]["trades"] += entry.get("trades", 0)
            accum[tag]["winners"] += entry.get("winners", 0)
            accum[tag]["losers"] += entry.get("losers", 0)
            accum[tag]["pnl"] += _dec(entry.get("realized_pnl", 0))
            accum[tag]["shadow_pnl"] += _dec(entry.get("auditor_shadow_pnl", 0))

        # Convert Decimals to float and compute win rates
        result: Dict[str, Dict[str, Any]] = {}
        for tag, stats in accum.items():
            total = stats["trades"]
            result[tag] = {
                "trades": stats["trades"],
                "winners": stats["winners"],
                "losers": stats["losers"],
                "pnl": _to_float(stats["pnl"]),
                "shadow_pnl": _to_float(stats["shadow_pnl"]),
                "win_rate": (
                    (stats["winners"] / total * 100) if total > 0 else 0.0
                ),
            }
        return result


def main() -> None:
    """CLI entrypoint for weekly report generation."""
    parser = argparse.ArgumentParser(
        description="Generate watchdog weekly performance report."
    )
    parser.add_argument(
        "--week",
        type=str,
        default=None,
        help="ISO week label (e.g., 2026-W07). Defaults to current week.",
    )
    parser.add_argument(
        "--watchdog-dir",
        type=str,
        default="results/watchdog",
        help="Watchdog output directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Base results directory.",
    )
    args = parser.parse_args()

    generator = WeeklyReportGenerator(
        watchdog_dir=args.watchdog_dir,
        results_dir=args.results_dir,
    )

    output_path = generator.generate(week_label=args.week)
    print(f"\nWeekly report generated: {output_path}")


if __name__ == "__main__":
    main()
