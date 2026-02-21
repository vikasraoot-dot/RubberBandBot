"""
Audit bot P&L from GitHub Actions logs (source of truth).

Pulls each bot's own structured log events (EOD_SUMMARY / BOT_STOP) from
GitHub Actions run logs, avoiding the cross-wiring problem of order-matching.

Usage:
    python RubberBand/scripts/audit_bot_pnl.py                  # all bots, last 10 trading days
    python RubberBand/scripts/audit_bot_pnl.py --days 30        # all bots, last 30 days
    python RubberBand/scripts/audit_bot_pnl.py --bot 15M_STK    # single bot
    python RubberBand/scripts/audit_bot_pnl.py --date 2026-02-19  # single date
    python RubberBand/scripts/audit_bot_pnl.py --no-cache       # force re-fetch from GH

Output: JSON report to stdout with per-bot per-day P&L.
Cache:  results/audit/{date}_{bot_tag}.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

# ── Bot-to-workflow mapping ──────────────────────────────────────────────────
BOT_WORKFLOWS: dict[str, str] = {
    "15M_STK": "[BOT] 15m Stock - Live",
    "15M_OPT": "[BOT] 15m Options - Live",
    "WK_STK":  "[BOT] Weekly Stock - Live",
    "WK_OPT":  "[BOT] Weekly Options - Live",
}

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
AUDIT_DIR = RESULTS_DIR / "audit"


# ── GitHub CLI helpers ───────────────────────────────────────────────────────

def _run_gh(args: list[str], timeout: int = 60) -> str:
    """Run a gh CLI command, return stdout."""
    result = subprocess.run(
        ["gh"] + args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args[:3])}... failed: {result.stderr[:200]}")
    return result.stdout


def get_runs_for_bot(
    bot_tag: str, days: int = 10
) -> dict[str, list[dict[str, Any]]]:
    """
    Fetch GH Actions runs for a bot workflow, grouped by date.

    Returns: {date_str: [{"id": int, "started": str, "conclusion": str}, ...]}
    """
    workflow = BOT_WORKFLOWS[bot_tag]
    # Fetch enough runs to cover the date range (WK_STK has 7/day)
    limit = days * 8
    raw = _run_gh([
        "run", "list",
        f"--workflow={workflow}",
        f"--limit={limit}",
        "--json", "databaseId,startedAt,conclusion",
    ])
    runs = json.loads(raw)

    cutoff = (datetime.now(tz=None) - timedelta(days=days)).strftime("%Y-%m-%d")

    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        date_str = r["startedAt"][:10]
        if date_str < cutoff:
            continue
        by_date[date_str].append({
            "id": r["databaseId"],
            "started": r["startedAt"],
            "conclusion": r["conclusion"],
        })

    # Sort each day's runs by start time (ascending)
    for date_str in by_date:
        by_date[date_str].sort(key=lambda x: x["started"])

    return dict(by_date)


def extract_pnl_from_run(run_id: int) -> Optional[dict[str, Any]]:
    """
    Pull GH Actions log for a run and extract the last EOD_SUMMARY or BOT_STOP.

    Returns the parsed JSON event dict, or None if no P&L event found.
    """
    try:
        raw_log = _run_gh(["run", "view", str(run_id), "--log"], timeout=120)
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        print(f"  WARN: Could not fetch log for run {run_id}: {e}", file=sys.stderr)
        return None

    # Find all EOD_SUMMARY or BOT_STOP JSON events in the log
    pnl_events: list[dict[str, Any]] = []
    has_heartbeat = False

    for line in raw_log.splitlines():
        # Track if the bot ran at all (has HEARTBEAT/FILTER_DIAGNOSTICS events)
        if not has_heartbeat and ('"HEARTBEAT"' in line or '"FILTER_DIAGNOSTICS"' in line):
            has_heartbeat = True

        # Look for EOD_SUMMARY or BOT_STOP JSON events
        start_match = re.search(r'\{"type"\s*:\s*"(?:EOD_SUMMARY|BOT_STOP)"', line)
        if not start_match:
            continue

        # Extract from the start of the JSON to end of line
        json_str = line[start_match.start():]
        # Try to parse directly first
        try:
            evt = json.loads(json_str)
            pnl_events.append(evt)
            continue
        except json.JSONDecodeError:
            pass

        # Fallback: find balanced braces
        depth = 0
        end_idx = 0
        for i, ch in enumerate(json_str):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
        if end_idx > 0:
            try:
                evt = json.loads(json_str[:end_idx])
                pnl_events.append(evt)
            except json.JSONDecodeError:
                pass

    if pnl_events:
        # Return the LAST event (most up-to-date for the day)
        return pnl_events[-1]

    # No P&L events found — if the bot ran (heartbeats exist), it's a zero-trade day
    if has_heartbeat:
        return {
            "type": "EOD_SUMMARY",
            "date": None,
            "total_trades": 0,
            "closed_trades": 0,
            "open_trades": 0,
            "total_pnl": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate_pct": 0.0,
            "trades": [],
            "exit_reasons": {},
            "_synthetic": True,
            "_reason": "bot_ran_but_no_trades",
        }

    return None


# ── Cache management ─────────────────────────────────────────────────────────

def _cache_path(date: str, bot_tag: str) -> Path:
    return AUDIT_DIR / f"{date}_{bot_tag}.json"


def check_cache(date: str, bot_tag: str) -> Optional[dict[str, Any]]:
    """Load cached audit result if it exists."""
    path = _cache_path(date, bot_tag)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def save_cache(date: str, bot_tag: str, data: dict[str, Any]) -> None:
    """Save audit result to cache."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(date, bot_tag)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ── Cross-checking ───────────────────────────────────────────────────────────

def cross_check(date: str, bot_tag: str, bot_pnl: Optional[float]) -> dict[str, Any]:
    """
    Compare bot's own P&L against watchdog daily results.

    Returns: {"watchdog_pnl": float|None, "divergence": float|None, "source": str}
    """
    daily_path = RESULTS_DIR / "daily" / f"{date}.json"
    if not daily_path.exists():
        return {"watchdog_pnl": None, "divergence": None, "source": "no_daily_file"}

    try:
        with open(daily_path, "r", encoding="utf-8") as f:
            daily = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"watchdog_pnl": None, "divergence": None, "source": "parse_error"}

    bots = daily.get("bots", {})
    bot_data = bots.get(bot_tag, {})
    watchdog_pnl = bot_data.get("realized_pnl")

    if watchdog_pnl is None:
        return {"watchdog_pnl": None, "divergence": None, "source": "bot_not_in_daily"}

    divergence = None
    if bot_pnl is not None and watchdog_pnl is not None:
        divergence = round(float(bot_pnl) - float(watchdog_pnl), 2)

    return {
        "watchdog_pnl": float(watchdog_pnl),
        "divergence": divergence,
        "source": "daily_json",
    }


# ── Main audit logic ────────────────────────────────────────────────────────

def normalize_event(evt: dict[str, Any], run_id: int) -> dict[str, Any]:
    """
    Normalize an EOD_SUMMARY or BOT_STOP event into a standard audit row.
    """
    evt_type = evt.get("type", "unknown")

    if evt_type == "EOD_SUMMARY":
        return {
            "source": "gh_actions",
            "event_type": "EOD_SUMMARY",
            "run_id": run_id,
            "total_pnl": evt.get("total_pnl", 0.0),
            "total_trades": evt.get("total_trades", 0),
            "closed_trades": evt.get("closed_trades", 0),
            "open_trades": evt.get("open_trades", 0),
            "win_count": evt.get("win_count", 0),
            "loss_count": evt.get("loss_count", 0),
            "win_rate_pct": evt.get("win_rate_pct", 0.0),
            "total_volume": evt.get("total_volume"),
            "exit_reasons": evt.get("exit_reasons", {}),
            "trades": evt.get("trades", []),
            "timestamp": evt.get("ts") or evt.get("ts_et"),
        }
    elif evt_type == "BOT_STOP":
        return {
            "source": "gh_actions",
            "event_type": "BOT_STOP",
            "run_id": run_id,
            "total_pnl": evt.get("daily_pnl", 0.0),
            "total_trades": None,
            "closed_trades": None,
            "open_trades": len(evt.get("managed_positions", [])),
            "win_count": None,
            "loss_count": None,
            "win_rate_pct": None,
            "total_volume": None,
            "exit_reasons": {},
            "trades": [],
            "managed_at_stop": evt.get("managed_positions", []),
            "stop_reason": evt.get("reason"),
            "timestamp": evt.get("time"),
        }
    else:
        return {
            "source": "gh_actions",
            "event_type": evt_type,
            "run_id": run_id,
            "total_pnl": None,
            "total_trades": None,
            "timestamp": None,
        }


def audit_bot(
    bot_tag: str,
    days: int = 10,
    target_date: Optional[str] = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    Audit a single bot's P&L from its GH Actions logs.

    Returns per-day breakdown + lifetime summary.
    """
    print(f"Auditing {bot_tag}...", file=sys.stderr)

    # Get all runs grouped by date
    runs_by_date = get_runs_for_bot(bot_tag, days)

    if target_date:
        # Filter to specific date
        runs_by_date = {d: r for d, r in runs_by_date.items() if d == target_date}

    day_results: list[dict[str, Any]] = []

    for date_str in sorted(runs_by_date.keys()):
        # Check cache first
        if use_cache:
            cached = check_cache(date_str, bot_tag)
            if cached:
                print(f"  {date_str}: cached", file=sys.stderr)
                day_results.append(cached)
                continue

        runs = runs_by_date[date_str]

        # Try the last run first (most complete data for the day)
        # If it failed, try earlier runs
        result = None
        tried_run_id = None

        for run in reversed(runs):
            tried_run_id = run["id"]
            conclusion = run["conclusion"]

            if conclusion not in ("success", "cancelled"):
                # For failed runs, still try to extract P&L (bot may have logged before crash)
                pass

            print(f"  {date_str}: fetching run {tried_run_id} ({conclusion})...", file=sys.stderr)

            evt = extract_pnl_from_run(tried_run_id)
            if evt is not None:
                result = normalize_event(evt, tried_run_id)
                break

            # Brief delay to be kind to GH API
            time.sleep(0.3)

        if result is None:
            result = {
                "source": "no_data",
                "event_type": None,
                "run_id": tried_run_id,
                "total_pnl": None,
                "total_trades": None,
                "closed_trades": None,
                "open_trades": None,
                "win_count": None,
                "loss_count": None,
                "win_rate_pct": None,
                "trades": [],
                "timestamp": None,
            }

        # Add date and cross-check
        result["date"] = date_str
        result["bot_tag"] = bot_tag
        xcheck = cross_check(date_str, bot_tag, result.get("total_pnl"))
        result["watchdog_pnl"] = xcheck["watchdog_pnl"]
        result["divergence"] = xcheck["divergence"]

        # Cache it
        if use_cache and result.get("source") != "no_data":
            save_cache(date_str, bot_tag, result)

        day_results.append(result)

    # Compute lifetime summary
    pnl_values = [d["total_pnl"] for d in day_results if d.get("total_pnl") is not None]
    total_trades_values = [d["total_trades"] for d in day_results if d.get("total_trades") is not None]

    lifetime_pnl = sum(pnl_values) if pnl_values else 0.0
    total_trades = sum(total_trades_values) if total_trades_values else 0
    trading_days = len(pnl_values)
    avg_daily = round(lifetime_pnl / trading_days, 2) if trading_days > 0 else 0.0

    best_day = max(day_results, key=lambda d: d.get("total_pnl") or float("-inf")) if pnl_values else None
    worst_day = min(day_results, key=lambda d: d.get("total_pnl") or float("inf")) if pnl_values else None

    return {
        "bot_tag": bot_tag,
        "workflow": BOT_WORKFLOWS[bot_tag],
        "days": day_results,
        "summary": {
            "lifetime_pnl": round(lifetime_pnl, 2),
            "total_trades": total_trades,
            "trading_days": trading_days,
            "avg_daily_pnl": avg_daily,
            "best_day": {"date": best_day["date"], "pnl": best_day.get("total_pnl")} if best_day else None,
            "worst_day": {"date": worst_day["date"], "pnl": worst_day.get("total_pnl")} if worst_day else None,
        },
    }


def build_discrepancies(bots: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect all days where bot P&L diverges from watchdog by > $1."""
    discrepancies = []
    for bot_tag, bot_data in bots.items():
        for day in bot_data.get("days", []):
            div = day.get("divergence")
            if div is not None and abs(div) > 1.0:
                discrepancies.append({
                    "date": day["date"],
                    "bot": bot_tag,
                    "bot_pnl": day.get("total_pnl"),
                    "watchdog_pnl": day.get("watchdog_pnl"),
                    "delta": div,
                })
    return sorted(discrepancies, key=lambda d: (d["date"], d["bot"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit bot P&L from GitHub Actions logs"
    )
    parser.add_argument(
        "--days", type=int, default=10,
        help="Number of calendar days to look back (default: 10)"
    )
    parser.add_argument(
        "--bot", type=str, default=None,
        choices=list(BOT_WORKFLOWS.keys()),
        help="Audit a specific bot only"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Audit a specific date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force re-fetch from GitHub Actions (ignore cache)"
    )
    args = parser.parse_args()

    use_cache = not args.no_cache

    # Determine which bots to audit
    bot_tags = [args.bot] if args.bot else list(BOT_WORKFLOWS.keys())

    bots_report: dict[str, Any] = {}
    for tag in bot_tags:
        bots_report[tag] = audit_bot(
            tag,
            days=args.days,
            target_date=args.date,
            use_cache=use_cache,
        )

    discrepancies = build_discrepancies(bots_report)

    report = {
        "audit_timestamp": datetime.now(tz=None).isoformat(),
        "parameters": {
            "days": args.days,
            "bot_filter": args.bot,
            "date_filter": args.date,
            "cache_used": use_cache,
        },
        "bots": bots_report,
        "discrepancies": discrepancies,
        "discrepancy_count": len(discrepancies),
    }

    # Output JSON to stdout
    json.dump(report, sys.stdout, indent=2, default=str)
    print(file=sys.stdout)  # trailing newline

    # Summary to stderr
    print("\n=== AUDIT SUMMARY ===", file=sys.stderr)
    for tag, data in bots_report.items():
        s = data["summary"]
        print(
            f"  {tag}: {s['lifetime_pnl']:+.2f} over {s['trading_days']} days "
            f"({s['total_trades']} trades, avg {s['avg_daily_pnl']:+.2f}/day)",
            file=sys.stderr,
        )
    if discrepancies:
        print(f"\n  WARNING: {len(discrepancies)} discrepancy(ies) vs watchdog!", file=sys.stderr)
        for d in discrepancies[:5]:
            print(
                f"    {d['date']} {d['bot']}: bot={d['bot_pnl']}, "
                f"watchdog={d['watchdog_pnl']}, delta={d['delta']:+.2f}",
                file=sys.stderr,
            )
    else:
        print("\n  No discrepancies found.", file=sys.stderr)


if __name__ == "__main__":
    main()
