"""
Post-Day Analyzer: End-of-day analysis of trading performance and filter effectiveness.

Runs every evening after market close.  Produces a per-date JSON report in
``results/watchdog/daily_analysis/{YYYY-MM-DD}.json`` that contains:

1. Filter effectiveness — cross-referenced with auditor shadow outcomes.
2. Time-of-day P&L breakdown (morning / midday / afternoon).
3. Per-ticker contribution and consistent loser flagging.
4. Regime correlation — P&L by VIXY-based regime.
5. Giveback analysis — peak intraday P&L vs close.
6. Auditor vs Bot comparison — shadow ledger P&L vs real bot P&L.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from RubberBand.src.watchdog.utils import to_dec as _dec, dec_to_float as _to_float

logger = logging.getLogger(__name__)

ET = ZoneInfo("US/Eastern")

# Default bot tags (mirrors position_registry.BOT_TAGS)
_BOT_TAGS = {"15M_STK", "15M_OPT", "WK_STK", "WK_OPT", "SL_ETF"}

# Filter event types that the auditor tracks as skips
_SKIP_EVENT_TYPES = {"SKIP_SLOPE3", "DKF_SKIP", "SPREAD_SKIP"}

# Time-of-day windows (Eastern Time, HH:MM)
_TIME_WINDOWS = {
    "morning":   ("09:45", "11:00"),
    "midday":    ("11:00", "14:00"),
    "afternoon": ("14:00", "15:45"),
}


class PostDayAnalyzer:
    """Runs all end-of-day analyses for a given date.

    Consumes JSONL trade logs, ``auditor_state.json``, and
    ``results/daily/{date}.json`` to produce a comprehensive
    daily analysis JSON report.

    Args:
        results_dir: Base directory for result files (trade logs, daily JSON).
        watchdog_dir: Directory for watchdog output files.
    """

    def __init__(
        self,
        results_dir: str = "results",
        watchdog_dir: str = "results/watchdog",
    ) -> None:
        self._results_dir = results_dir
        self._watchdog_dir = watchdog_dir
        self._output_dir = os.path.join(watchdog_dir, "daily_analysis")
        os.makedirs(self._output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze(self, date: str) -> Dict[str, Any]:
        """Run all analyses for *date* and persist the report.

        Args:
            date: Date string in ``YYYY-MM-DD`` format.

        Returns:
            The complete analysis dict (also saved to disk).
        """
        logger.info("Starting post-day analysis for %s", date)

        bot_logs = self._load_bot_logs(date)
        auditor_data = self._load_auditor_data()
        daily_json = self._load_daily_json(date)

        # Flatten all trade entries across bots for some analyses
        all_trades = self._extract_closed_trades(bot_logs)

        # Build per-bot P&L summary from the daily JSON
        bot_pnl = self._build_bot_pnl_summary(daily_json)

        analysis: Dict[str, Any] = {
            "date": date,
            "generated_at": datetime.now(ET).isoformat(),
            "bots_analysed": list(bot_logs.keys()),
            "filter_effectiveness": self._analyze_filter_effectiveness(
                date, bot_logs, auditor_data
            ),
            "time_of_day": self._analyze_time_of_day(all_trades),
            "ticker_analysis": self._analyze_tickers(all_trades),
            "regime_correlation": self._analyze_regime_correlation(date, all_trades),
            "giveback": self._analyze_giveback(date),
            "auditor_vs_bot": self._compare_auditor_vs_bot(bot_pnl, auditor_data),
        }

        # Signal context analysis (new SCAN_CONTEXT events)
        try:
            analysis["signal_context"] = self._analyze_signal_context(bot_logs)
        except Exception as exc:
            logger.error("Signal context analysis failed: %s", exc, exc_info=True)
            analysis["signal_context"] = {"available": False, "error": str(exc)}

        # Persist
        output_path = os.path.join(self._output_dir, f"{date}.json")
        try:
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(analysis, fh, indent=2, default=str)
            logger.info("Saved daily analysis to %s", output_path)
        except OSError as exc:
            logger.error("Failed to save daily analysis: %s", exc)

        return analysis

    # ------------------------------------------------------------------
    # 1. Filter effectiveness
    # ------------------------------------------------------------------

    def _analyze_filter_effectiveness(
        self,
        date: str,
        bot_logs: Dict[str, List[Dict[str, Any]]],
        auditor_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyse how each filter performed by cross-referencing auditor shadow data.

        For each filter (slope, bearish_bar, DKF, trend / GATE+BLOCK), count how
        many trades it blocked.  Then look at the auditor's closed shadow positions
        to see whether those blocked trades would have been winners or losers.

        Args:
            date: Date string YYYY-MM-DD.
            bot_logs: Trade log entries grouped by bot_tag.
            auditor_data: Parsed auditor_state.json.

        Returns:
            Dict with per-filter counts and shadow cross-reference.
        """
        # Collect all filter-block events across bots
        filter_blocks: Dict[str, List[Dict[str, Any]]] = {}
        for bot_tag, entries in bot_logs.items():
            for entry in entries:
                etype = entry.get("type", "")
                # Skip events that the auditor shadows
                if etype in _SKIP_EVENT_TYPES:
                    key = self._normalise_filter_name(etype)
                    filter_blocks.setdefault(key, []).append(entry)
                elif etype == "GATE" and entry.get("decision") == "BLOCK":
                    # Handle both "reason" (singular string) and
                    # "reasons" (plural list) — callers vary.
                    reason_val = entry.get("reason", "")
                    reasons_val = entry.get("reasons", [])
                    if reason_val:
                        key = self._normalise_filter_name(str(reason_val))
                        filter_blocks.setdefault(key, []).append(entry)
                    elif isinstance(reasons_val, list):
                        for r in reasons_val:
                            key = self._normalise_filter_name(str(r))
                            filter_blocks.setdefault(key, []).append(entry)
                    elif reasons_val:
                        key = self._normalise_filter_name(str(reasons_val))
                        filter_blocks.setdefault(key, []).append(entry)

        # Build index of auditor closed shadow positions by symbol
        closed_shadows = auditor_data.get("closed_positions", [])
        shadow_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        for pos in closed_shadows:
            sym = pos.get("symbol", "")
            if sym:
                shadow_by_symbol.setdefault(sym, []).append(pos)

        # Cross-reference each filter's blocks with shadow outcomes
        result: Dict[str, Any] = {}
        total_value_saved = Decimal("0")
        total_profits_missed = Decimal("0")

        for filter_name, events in filter_blocks.items():
            blocked_count = len(events)
            shadow_winners = 0
            shadow_losers = 0
            shadow_pnl_sum = Decimal("0")

            for event in events:
                sym = event.get("symbol") or event.get("underlying", "")
                if not sym:
                    continue
                # Find matching shadow positions for this symbol
                shadows = shadow_by_symbol.get(sym, [])
                for sp in shadows:
                    pnl = _dec(sp.get("realized_pnl", 0))
                    shadow_pnl_sum += pnl
                    if pnl > 0:
                        shadow_winners += 1
                    elif pnl < 0:
                        shadow_losers += 1

            losses_avoided_raw = Decimal("0")
            profits_missed = Decimal("0")
            for event in events:
                sym = event.get("symbol") or event.get("underlying", "")
                for sp in shadow_by_symbol.get(sym, []):
                    sp_pnl = _dec(sp.get("realized_pnl", 0))
                    if sp_pnl < 0:
                        losses_avoided_raw += sp_pnl
                    elif sp_pnl > 0:
                        profits_missed += sp_pnl
            losses_avoided = abs(losses_avoided_raw)
            net_value = losses_avoided - profits_missed
            total_value_saved += losses_avoided
            total_profits_missed += profits_missed

            result[filter_name] = {
                "blocked_count": blocked_count,
                "shadow_winners": shadow_winners,
                "shadow_losers": shadow_losers,
                "shadow_pnl_sum": _to_float(shadow_pnl_sum),
                "losses_avoided": _to_float(losses_avoided),
                "profits_missed": _to_float(profits_missed),
                "net_value": _to_float(net_value),
            }

        result["_summary"] = {
            "total_blocks": sum(v["blocked_count"] for v in result.values() if isinstance(v, dict) and "blocked_count" in v),
            "total_value_saved": _to_float(total_value_saved),
            "total_profits_missed": _to_float(total_profits_missed),
            "net_filter_value": _to_float(total_value_saved - total_profits_missed),
        }

        return result

    # ------------------------------------------------------------------
    # 2. Time-of-day analysis
    # ------------------------------------------------------------------

    def _analyze_time_of_day(
        self, trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Break down P&L by time-of-day windows (Eastern Time).

        Windows:
        - morning:   09:45 - 11:00
        - midday:    11:00 - 14:00
        - afternoon: 14:00 - 15:45

        Args:
            trades: Flattened list of closed trade entries (EXIT_FILL / SPREAD_EXIT).

        Returns:
            Dict keyed by window name with trade count, total P&L, win rate.
        """
        buckets: Dict[str, List[Decimal]] = {w: [] for w in _TIME_WINDOWS}
        buckets["other"] = []

        for trade in trades:
            pnl = _dec(trade.get("pnl", 0))
            entry_time = self._extract_entry_time_et(trade)
            if not entry_time:
                buckets["other"].append(pnl)
                continue

            placed = False
            for window_name, (start, end) in _TIME_WINDOWS.items():
                if start <= entry_time < end:
                    buckets[window_name].append(pnl)
                    placed = True
                    break
            if not placed:
                buckets["other"].append(pnl)

        result: Dict[str, Any] = {}
        for window_name, pnl_list in buckets.items():
            total = sum(pnl_list, Decimal("0"))
            winners = sum(1 for p in pnl_list if p > 0)
            losers = sum(1 for p in pnl_list if p < 0)
            count = len(pnl_list)
            win_rate = (
                _to_float(Decimal(str(winners)) / Decimal(str(count)) * Decimal("100"))
                if count > 0
                else 0.0
            )
            result[window_name] = {
                "trades": count,
                "winners": winners,
                "losers": losers,
                "total_pnl": _to_float(total),
                "win_rate": win_rate,
            }
        return result

    # ------------------------------------------------------------------
    # 3. Ticker analysis
    # ------------------------------------------------------------------

    def _analyze_tickers(
        self, trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Per-ticker P&L contribution and consistent loser flagging.

        Args:
            trades: Flattened list of closed trade entries.

        Returns:
            Dict with per-ticker stats and a list of flagged losers.
        """
        ticker_stats: Dict[str, Dict[str, Any]] = {}

        for trade in trades:
            sym = trade.get("symbol") or trade.get("underlying", "unknown")
            pnl = _dec(trade.get("pnl", 0))

            if sym not in ticker_stats:
                ticker_stats[sym] = {
                    "trades": 0,
                    "winners": 0,
                    "losers": 0,
                    "total_pnl": Decimal("0"),
                }
            stats = ticker_stats[sym]
            stats["trades"] += 1
            stats["total_pnl"] += pnl
            if pnl > 0:
                stats["winners"] += 1
            elif pnl < 0:
                stats["losers"] += 1

        # Convert Decimals for JSON and flag consistent losers
        per_ticker: Dict[str, Any] = {}
        flagged_losers: List[str] = []

        for sym, stats in ticker_stats.items():
            count = stats["trades"]
            win_rate = (
                _to_float(
                    Decimal(str(stats["winners"])) / Decimal(str(count)) * Decimal("100")
                )
                if count > 0
                else 0.0
            )
            entry = {
                "trades": count,
                "winners": stats["winners"],
                "losers": stats["losers"],
                "total_pnl": _to_float(stats["total_pnl"]),
                "win_rate": win_rate,
            }
            per_ticker[sym] = entry

            # Flag if 2+ trades and 100% loss rate
            if stats["losers"] >= 2 and stats["winners"] == 0:
                flagged_losers.append(sym)

        return {
            "per_ticker": per_ticker,
            "flagged_losers": flagged_losers,
        }

    # ------------------------------------------------------------------
    # 4. Regime correlation
    # ------------------------------------------------------------------

    def _analyze_regime_correlation(
        self,
        date: str,
        trades: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """P&L by regime (PANIC / NORMAL / CALM).

        Reads the daily JSON for regime info if available, otherwise
        groups by regime field in trade entries.

        Args:
            date: Date string YYYY-MM-DD.
            trades: Flattened list of closed trade entries.

        Returns:
            Dict keyed by regime name with trade count, total P&L.
        """
        daily_json = self._load_daily_json(date)

        # Try to get the regime from the daily JSON summary
        day_regime = ""
        if daily_json:
            # Check if there's a regime field in the summary or bots
            day_regime = daily_json.get("summary", {}).get("regime", "")

        # Group trades by the regime they carry (if any)
        regime_pnl: Dict[str, List[Decimal]] = {}
        for trade in trades:
            regime = trade.get("regime", day_regime or "UNKNOWN")
            if not regime:
                regime = "UNKNOWN"
            pnl = _dec(trade.get("pnl", 0))
            regime_pnl.setdefault(regime, []).append(pnl)

        result: Dict[str, Any] = {}
        for regime, pnl_list in regime_pnl.items():
            total = sum(pnl_list, Decimal("0"))
            count = len(pnl_list)
            winners = sum(1 for p in pnl_list if p > 0)
            result[regime] = {
                "trades": count,
                "winners": winners,
                "losers": count - winners,
                "total_pnl": _to_float(total),
            }

        if day_regime:
            result["_day_regime"] = day_regime

        return result

    # ------------------------------------------------------------------
    # 5. Giveback analysis
    # ------------------------------------------------------------------

    def _analyze_giveback(self, date: str) -> Dict[str, Any]:
        """Peak intraday P&L vs close for each bot.

        Uses ``profit_locks.json`` if available (from intraday monitor),
        otherwise falls back to trade log computation.

        Args:
            date: Date string YYYY-MM-DD.

        Returns:
            Dict with per-bot peak P&L, close P&L, and giveback amount.
        """
        result: Dict[str, Any] = {}

        # Try to read profit_locks.json for peak tracking
        profit_locks = self._load_json_file(
            os.path.join(self._watchdog_dir, "profit_locks.json")
        )

        # Try to read intraday_health.json for current day state
        health = self._load_json_file(
            os.path.join(self._watchdog_dir, "intraday_health.json")
        )

        # Get daily JSON for close P&L
        daily_json = self._load_daily_json(date)
        bots_section = daily_json.get("bots", {}) if daily_json else {}

        for bot_tag in _BOT_TAGS:
            close_pnl = Decimal("0")
            peak_pnl = Decimal("0")

            # Close P&L from daily JSON
            bot_data = bots_section.get(bot_tag, {})
            if bot_data:
                close_pnl = _dec(bot_data.get("realized_pnl", 0)) + _dec(
                    bot_data.get("unrealized_pnl", 0)
                )

            # Peak from profit_locks if available
            if profit_locks and bot_tag in profit_locks:
                peak_pnl = _dec(profit_locks[bot_tag].get("peak_pnl_today", 0))

            # Peak from intraday health if available and higher
            # Structure: health["bots"][bot_tag]["profit_lock"]["peak_pnl_today"]
            health_bot = health.get("bots", {}).get(bot_tag, {}) if health else {}
            if health_bot:
                health_peak = _dec(
                    health_bot.get("profit_lock", {}).get("peak_pnl_today", 0)
                )
                if health_peak > peak_pnl:
                    peak_pnl = health_peak

            # If we have no peak data, use close P&L as peak (conservative)
            if peak_pnl == 0 and close_pnl > 0:
                peak_pnl = close_pnl

            giveback = peak_pnl - close_pnl if peak_pnl > close_pnl else Decimal("0")

            if close_pnl != 0 or peak_pnl != 0:
                result[bot_tag] = {
                    "peak_pnl": _to_float(peak_pnl),
                    "close_pnl": _to_float(close_pnl),
                    "giveback": _to_float(giveback),
                    "giveback_pct": (
                        _to_float(giveback / peak_pnl * Decimal("100"))
                        if peak_pnl > 0
                        else 0.0
                    ),
                }

        return result

    # ------------------------------------------------------------------
    # 6. Auditor vs Bot comparison
    # ------------------------------------------------------------------

    def _compare_auditor_vs_bot(
        self,
        bot_pnl: Dict[str, Dict[str, Any]],
        auditor_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare real bot P&L with auditor shadow ledger P&L.

        Args:
            bot_pnl: Per-bot P&L summary from daily JSON.
            auditor_data: Parsed auditor_state.json.

        Returns:
            Dict with per-bot comparison and verdict.
        """
        closed_shadows = auditor_data.get("closed_positions", [])
        open_shadows = auditor_data.get("positions", {})

        # Group shadow P&L by bot_tag
        shadow_by_bot: Dict[str, Dict[str, Any]] = {}
        for pos in closed_shadows:
            tag = pos.get("bot_tag", "")
            if tag not in shadow_by_bot:
                shadow_by_bot[tag] = {
                    "trades": 0,
                    "total_pnl": Decimal("0"),
                    "winners": 0,
                    "losers": 0,
                }
            pnl = _dec(pos.get("realized_pnl", 0))
            shadow_by_bot[tag]["trades"] += 1
            shadow_by_bot[tag]["total_pnl"] += pnl
            if pnl > 0:
                shadow_by_bot[tag]["winners"] += 1
            elif pnl < 0:
                shadow_by_bot[tag]["losers"] += 1

        # Count open shadow positions per bot
        open_count_by_bot: Dict[str, int] = {}
        if isinstance(open_shadows, dict):
            for pos_id, pos in open_shadows.items():
                tag = pos.get("bot_tag", "")
                open_count_by_bot[tag] = open_count_by_bot.get(tag, 0) + 1

        result: Dict[str, Any] = {}
        all_tags = set(bot_pnl.keys()) | set(shadow_by_bot.keys())

        for tag in all_tags:
            real = bot_pnl.get(tag, {})
            real_pnl = _dec(real.get("total_pnl", 0))

            shadow = shadow_by_bot.get(tag, {})
            shadow_pnl = shadow.get("total_pnl", Decimal("0"))
            shadow_trades = shadow.get("trades", 0)
            shadow_winners = shadow.get("winners", 0)

            diff = real_pnl - shadow_pnl

            # Verdict
            if shadow_pnl > real_pnl and shadow_trades > 0:
                verdict = "FILTERS_TOO_STRICT"
                explanation = (
                    f"Shadow outperformed by ${_to_float(abs(diff))} "
                    f"— filters may be blocking profitable trades"
                )
            elif shadow_pnl < real_pnl and shadow_trades > 0:
                verdict = "FILTERS_ADDING_VALUE"
                explanation = (
                    f"Bot outperformed shadow by ${_to_float(abs(diff))} "
                    f"— filters are protecting from losses"
                )
            else:
                verdict = "NEUTRAL"
                explanation = "No meaningful difference or insufficient shadow data"

            result[tag] = {
                "real_pnl": _to_float(real_pnl),
                "shadow_pnl": _to_float(shadow_pnl),
                "shadow_trades": shadow_trades,
                "shadow_winners": shadow_winners,
                "shadow_open_positions": open_count_by_bot.get(tag, 0),
                "difference": _to_float(diff),
                "verdict": verdict,
                "explanation": explanation,
            }

        return result

    # ------------------------------------------------------------------
    # 7. Signal context analysis (SCAN_CONTEXT events)
    # ------------------------------------------------------------------

    def _analyze_signal_context(
        self, bot_logs: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Analyse SCAN_CONTEXT events from trade logs.

        Produces filter effectiveness from the scanner's perspective,
        near-miss detection, indicator distributions, regime breakdowns,
        and per-cycle time distribution.

        Args:
            bot_logs: Trade log entries grouped by bot_tag.

        Returns:
            Dict with analysis sections, or ``{"available": False}``
            if no SCAN_CONTEXT events are found.
        """
        # 1. Extract all SCAN_CONTEXT events
        scan_events: List[Dict[str, Any]] = []
        for _bot_tag, entries in bot_logs.items():
            for entry in entries:
                if entry.get("type") == "SCAN_CONTEXT":
                    scan_events.append(entry)

        if not scan_events:
            return {"available": False}

        # Flatten all symbol records across all cycles
        all_symbols: List[Dict[str, Any]] = []
        for event in scan_events:
            symbols = event.get("symbols", [])
            if isinstance(symbols, list):
                for sym in symbols:
                    # Attach cycle-level context to each symbol record
                    sym_record = dict(sym)
                    sym_record["_regime"] = event.get("regime", "UNKNOWN") or "UNKNOWN"
                    sym_record["_ts_et"] = event.get("ts_et", "")
                    all_symbols.append(sym_record)

        return {
            "available": True,
            "total_cycles": len(scan_events),
            "total_symbol_records": len(all_symbols),
            "filter_effectiveness": self._sc_filter_effectiveness(all_symbols),
            "near_misses": self._sc_near_misses(all_symbols),
            "indicator_summary": self._sc_indicator_summary(all_symbols),
            "regime_breakdown": self._sc_regime_breakdown(all_symbols),
            "time_distribution": self._sc_time_distribution(scan_events),
        }

    # -- signal context sub-analyses -----------------------------------

    def _sc_filter_effectiveness(
        self, all_symbols: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Count how many symbols each filter blocked / passed.

        Uses the ``filters`` dict inside each symbol record.  A filter
        is considered to have *blocked* a symbol when its value is
        ``False``.

        Args:
            all_symbols: Flat list of per-symbol scan records.

        Returns:
            Dict keyed by filter name with blocked/passed/total counts.
        """
        filter_blocked: Dict[str, int] = {}
        filter_passed: Dict[str, int] = {}
        filter_total: Dict[str, int] = {}

        for sym in all_symbols:
            filters = sym.get("filters")
            if not isinstance(filters, dict):
                continue
            for fname, fval in filters.items():
                # Skip the aggregate "signal" key — it is the composite
                if fname == "signal":
                    continue
                filter_total[fname] = filter_total.get(fname, 0) + 1
                if fval is True:
                    filter_passed[fname] = filter_passed.get(fname, 0) + 1
                elif fval is False:
                    filter_blocked[fname] = filter_blocked.get(fname, 0) + 1

        result: Dict[str, Any] = {}
        for fname in sorted(filter_total.keys()):
            result[fname] = {
                "blocked": filter_blocked.get(fname, 0),
                "passed": filter_passed.get(fname, 0),
                "total": filter_total[fname],
            }
        return result

    def _sc_near_misses(
        self, all_symbols: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find symbols that passed all filters except exactly one.

        For each near miss, record the ticker, outcome, and the
        indicator value most likely responsible for the miss.

        Args:
            all_symbols: Flat list of per-symbol scan records.

        Returns:
            List of near-miss dicts sorted by ticker.
        """
        # Map outcome strings to a likely indicator field for diagnostics.
        # Early-exit outcomes (NO_DATA, INSUFFICIENT_BARS, PAUSED_HEALTH,
        # FORMING_BAR_DROPPED) rarely produce near-misses because they fail
        # before multiple filter booleans are computed, but we include them
        # for completeness.
        _outcome_to_indicator: Dict[str, str] = {
            "SKIP_SLOPE": "slope_pct",
            "SKIP_BEARISH_BAR": "close",
            "DKF_SKIP": "close",
            "TREND_BEAR": "is_bull_trend",
            "TREND_NO_DATA": "is_bull_trend",
            "NO_DATA": "ticker",
            "INSUFFICIENT_BARS": "bars_count",
            "PAUSED_HEALTH": "ticker",
            "FORMING_BAR_DROPPED": "ticker",
            "NO_SIGNAL": "rsi",
            "ALREADY_IN_POSITION": "ticker",
            "TRADED_TODAY": "ticker",
            "BAD_TP_SL": "close",
            "QTY_ZERO": "close",
        }

        # Early-exit outcomes that don't produce full filter dicts —
        # skip these since they can't be meaningful near-misses.
        _early_exit_outcomes = {
            "NO_DATA", "INSUFFICIENT_BARS", "PAUSED_HEALTH",
            "FORMING_BAR_DROPPED",
        }

        near_misses: List[Dict[str, Any]] = []
        for sym in all_symbols:
            if sym.get("outcome", "") in _early_exit_outcomes:
                continue

            filters = sym.get("filters")
            if not isinstance(filters, dict):
                continue

            # Exclude the composite "signal" key and "reason" text field
            individual = {
                k: v for k, v in filters.items()
                if k not in ("signal", "reason") and isinstance(v, bool)
            }
            if not individual:
                continue

            failed_filters = [k for k, v in individual.items() if v is False]
            if len(failed_filters) != 1:
                continue

            # This symbol missed by exactly one filter
            blocking_filter = failed_filters[0]
            outcome = sym.get("outcome", "")
            ticker = sym.get("ticker", "unknown")

            # Find the most relevant indicator value for the blocking filter
            indicator_key = _outcome_to_indicator.get(outcome, blocking_filter)
            indicator_value = sym.get(indicator_key)

            entry: Dict[str, Any] = {
                "ticker": ticker,
                "outcome": outcome,
                "blocking_filter": blocking_filter,
                "indicator_key": indicator_key,
            }
            if indicator_value is not None:
                entry["indicator_value"] = indicator_value
            # Include regime and cycle timestamp for context
            entry["regime"] = sym.get("_regime", "")
            entry["ts_et"] = sym.get("_ts_et", "")

            near_misses.append(entry)

        near_misses.sort(key=lambda x: x.get("ticker", ""))
        return near_misses

    def _sc_indicator_summary(
        self, all_symbols: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute summary stats for RSI, slope_pct, ATR.

        Splits symbols into ``entered`` (outcome=SIGNAL) and ``skipped``
        groups.  Only includes symbols that reached indicator computation
        (excludes NO_DATA, INSUFFICIENT_BARS, PAUSED_HEALTH outcomes).

        Args:
            all_symbols: Flat list of per-symbol scan records.

        Returns:
            Dict with ``entered`` and ``skipped`` sub-dicts each
            containing mean / p25 / p75 for each indicator.
        """
        _excluded_outcomes = {"NO_DATA", "INSUFFICIENT_BARS", "PAUSED_HEALTH"}
        _indicator_keys = ("rsi", "slope_pct", "atr")

        entered_vals: Dict[str, List[Decimal]] = {k: [] for k in _indicator_keys}
        skipped_vals: Dict[str, List[Decimal]] = {k: [] for k in _indicator_keys}

        for sym in all_symbols:
            outcome = sym.get("outcome", "")
            if outcome in _excluded_outcomes:
                continue

            target = entered_vals if outcome == "SIGNAL" else skipped_vals
            for key in _indicator_keys:
                raw = sym.get(key)
                if raw is not None:
                    val = _dec(raw)
                    # Only include non-zero / meaningful values
                    target[key].append(val)

        def _stats(values: List[Decimal]) -> Dict[str, Any]:
            """Compute mean, p25, p75 for a list of Decimals."""
            if not values:
                return {"mean": None, "p25": None, "p75": None, "count": 0}
            sorted_v = sorted(values)
            n = len(sorted_v)
            total = sum(sorted_v, Decimal("0"))
            mean = total / Decimal(str(n))
            p25_idx = max(0, int(n * 25 / 100) - 1)
            p75_idx = max(0, min(int(n * 75 / 100) - 1, n - 1))
            return {
                "mean": _to_float(mean),
                "p25": _to_float(sorted_v[p25_idx]),
                "p75": _to_float(sorted_v[p75_idx]),
                "count": n,
            }

        result: Dict[str, Any] = {"entered": {}, "skipped": {}}
        for key in _indicator_keys:
            result["entered"][key] = _stats(entered_vals[key])
            result["skipped"][key] = _stats(skipped_vals[key])

        return result

    def _sc_regime_breakdown(
        self, all_symbols: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Group filter effectiveness counts by regime.

        Args:
            all_symbols: Flat list of per-symbol scan records.

        Returns:
            Dict keyed by regime name, each containing per-filter
            blocked/passed/total counts.
        """
        regime_groups: Dict[str, List[Dict[str, Any]]] = {}
        for sym in all_symbols:
            regime = sym.get("_regime", "UNKNOWN") or "UNKNOWN"
            regime_groups.setdefault(regime, []).append(sym)

        result: Dict[str, Any] = {}
        for regime, symbols in sorted(regime_groups.items()):
            result[regime] = self._sc_filter_effectiveness(symbols)

        return result

    def _sc_time_distribution(
        self, scan_events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Count signals and skips per scan cycle.

        Uses the ``ts_et`` timestamp from each SCAN_CONTEXT event
        and the ``symbols_scanned`` / ``symbols_passed`` fields.

        Args:
            scan_events: List of raw SCAN_CONTEXT event dicts.

        Returns:
            List of per-cycle dicts with ts_et, signals, skips,
            scanned, and regime, sorted chronologically.
        """
        rows: List[Dict[str, Any]] = []
        for event in scan_events:
            ts_et = event.get("ts_et", "")
            scanned = event.get("symbols_scanned", 0) or 0
            passed = event.get("symbols_passed", 0) or 0

            # Also count from the symbols array for accuracy
            symbols = event.get("symbols", [])
            signal_count = 0
            skip_count = 0
            if isinstance(symbols, list):
                for sym in symbols:
                    if sym.get("outcome") == "SIGNAL":
                        signal_count += 1
                    else:
                        skip_count += 1

            rows.append({
                "ts_et": ts_et,
                "regime": event.get("regime", "UNKNOWN") or "UNKNOWN",
                "symbols_scanned": scanned,
                "symbols_passed": passed,
                "signals": signal_count,
                "skips": skip_count,
            })

        # Sort chronologically by timestamp
        rows.sort(key=lambda x: x.get("ts_et", ""))
        return rows

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_bot_logs(
        self, date: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Load JSONL trade logs for the given date, grouped by bot_tag.

        Searches for files matching common naming patterns in the results dir.

        Args:
            date: Date string YYYY-MM-DD.

        Returns:
            Dict mapping bot_tag -> list of JSONL entry dicts.
        """
        date_compact = date.replace("-", "")
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        if not os.path.isdir(self._results_dir):
            logger.warning("Results directory not found: %s", self._results_dir)
            return grouped

        try:
            files = os.listdir(self._results_dir)
        except OSError as exc:
            logger.error("Failed to list results dir %s: %s", self._results_dir, exc)
            return grouped

        candidate_files = [
            f for f in files if f.endswith(".jsonl") and date_compact in f
        ]

        for fname in candidate_files:
            fpath = os.path.join(self._results_dir, fname)
            bot_tag = self._infer_bot_tag_from_filename(fname)

            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        entry_tag = entry.get("bot_tag", bot_tag)
                        if entry_tag:
                            grouped.setdefault(entry_tag, []).append(entry)
            except OSError as exc:
                logger.warning("Failed to read trade log %s: %s", fpath, exc)

        logger.info(
            "Loaded trade logs for %s: %s",
            date,
            {k: len(v) for k, v in grouped.items()},
        )
        return grouped

    def _load_auditor_data(self) -> Dict[str, Any]:
        """Load auditor_state.json from the repo root or relative paths.

        Returns:
            Parsed JSON dict, or empty dict if missing/corrupt.
        """
        candidates = [
            os.path.join(self._results_dir, "..", "auditor_state.json"),
            "auditor_state.json",
            os.path.join(self._results_dir, "auditor_state.json"),
        ]
        for path in candidates:
            norm = os.path.normpath(path)
            if os.path.exists(norm):
                try:
                    with open(norm, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    logger.info("Loaded auditor state from %s", norm)
                    return data
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning(
                        "Failed to read auditor state %s: %s", norm, exc
                    )
                    return {}

        logger.info("auditor_state.json not found — shadow analysis will be empty")
        return {}

    def _load_daily_json(self, date: str) -> Dict[str, Any]:
        """Load results/daily/{date}.json.

        Args:
            date: Date string YYYY-MM-DD.

        Returns:
            Parsed JSON dict, or empty dict if missing/corrupt.
        """
        path = os.path.join(self._results_dir, "daily", f"{date}.json")
        if not os.path.exists(path):
            logger.debug("Daily JSON not found: %s", path)
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to read daily JSON %s: %s", path, exc)
            return {}

    def _load_json_file(self, path: str) -> Dict[str, Any]:
        """Generic JSON file loader with error handling.

        Args:
            path: File path to load.

        Returns:
            Parsed JSON dict, or empty dict if missing/corrupt.
        """
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Failed to read %s: %s", path, exc)
            return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_closed_trades(
        self, bot_logs: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Extract closed trade events (EXIT_FILL, SPREAD_EXIT) from all bot logs.

        Args:
            bot_logs: Trade log entries grouped by bot_tag.

        Returns:
            Flat list of exit event dicts with bot_tag injected.
        """
        trades: List[Dict[str, Any]] = []
        for bot_tag, entries in bot_logs.items():
            for entry in entries:
                etype = entry.get("type", "")
                if etype in ("EXIT_FILL", "SPREAD_EXIT"):
                    entry_copy = dict(entry)
                    entry_copy.setdefault("bot_tag", bot_tag)
                    trades.append(entry_copy)
        return trades

    def _build_bot_pnl_summary(
        self, daily_json: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Build per-bot P&L summary from the daily results JSON.

        Args:
            daily_json: Parsed daily JSON.

        Returns:
            Dict mapping bot_tag -> {total_pnl, realized_pnl, unrealized_pnl}.
        """
        result: Dict[str, Dict[str, Any]] = {}
        bots_section = daily_json.get("bots", {}) if daily_json else {}

        for bot_tag, bot_data in bots_section.items():
            result[bot_tag] = {
                "total_pnl": bot_data.get("total_pnl", 0),
                "realized_pnl": bot_data.get("realized_pnl", 0),
                "unrealized_pnl": bot_data.get("unrealized_pnl", 0),
            }
        return result

    def _extract_entry_time_et(self, trade: Dict[str, Any]) -> str:
        """Extract the Eastern Time HH:MM:SS for a trade's entry time.

        Tries multiple fields that different loggers use.

        Args:
            trade: A single trade/exit event dict.

        Returns:
            Time string in HH:MM:SS format, or empty string if unavailable.
        """
        # Try direct fields first (set by TradeLogger / OptionsTradeLogger)
        entry_time = trade.get("entry_time_et", "")
        if entry_time:
            return entry_time

        # Try ts_et which is "YYYY-MM-DD HH:MM:SS ET"
        ts_et = trade.get("ts_et", "")
        if ts_et and " " in ts_et:
            parts = ts_et.split(" ")
            if len(parts) >= 2:
                return parts[1]

        return ""

    @staticmethod
    def _normalise_filter_name(raw: str) -> str:
        """Normalise a filter event type or reason string to a canonical name.

        Args:
            raw: Raw event type or reason string.

        Returns:
            Normalised filter name.
        """
        upper = raw.upper().strip()
        if "SLOPE" in upper:
            return "slope"
        if "DKF" in upper or "DEAD_KNIFE" in upper or "DEAD KNIFE" in upper:
            return "dead_knife"
        if "BEARISH" in upper:
            return "bearish_bar"
        if "TREND" in upper or "EMA" in upper:
            return "trend"
        if "RSI" in upper:
            return "rsi"
        if "ADX" in upper:
            return "adx"
        if "RVOL" in upper or "VOLUME" in upper:
            return "volume"
        if "PRICE" in upper:
            return "price"
        if "SPREAD" in upper:
            return "spread_skip"
        if "POSITION" in upper or "ALREADY" in upper:
            return "already_in_position"
        return raw.lower().replace(" ", "_")

    @staticmethod
    def _infer_bot_tag_from_filename(fname: str) -> str:
        """Guess bot_tag from a JSONL log filename.

        Args:
            fname: Filename (not full path).

        Returns:
            Bot tag string, or empty string if unrecognised.
        """
        upper = fname.upper()
        if "15M_STK" in upper:
            return "15M_STK"
        if "15M_OPT" in upper:
            return "15M_OPT"
        if "WK_STK" in upper:
            return "WK_STK"
        if "WK_OPT" in upper:
            return "WK_OPT"
        if "SL_ETF" in upper:
            return "SL_ETF"
        # Legacy naming conventions
        if upper.startswith("LIVE_"):
            return "15M_STK"
        if upper.startswith("OPTIONS_TRADES_"):
            return "15M_OPT"
        return ""
