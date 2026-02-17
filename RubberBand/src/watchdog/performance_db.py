"""
Performance Database: Unified append-only JSONL store for daily per-bot metrics.

Ingests data from:
- results/daily/{date}.json  (from persist_daily_results.py)
- results/*.jsonl             (JSONL trade logs from TradeLogger / OptionsTradeLogger)
- auditor_state.json          (from auditor_bot.py shadow ledger)

Provides query and rolling-stats helpers consumed by the watchdog learning engine.
"""
from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from RubberBand.src.watchdog.utils import to_dec as _dec, dec_to_float as _to_float

logger = logging.getLogger(__name__)

ET = ZoneInfo("US/Eastern")

# Default bot tags (mirrors position_registry.BOT_TAGS)
_BOT_TAGS = {"15M_STK", "15M_OPT", "WK_STK", "WK_OPT", "SL_ETF"}


class PerformanceDB:
    """Append-only JSONL database of daily per-bot performance metrics.

    One line per bot per day is appended to ``performance.jsonl``.
    Provides ``query()`` for recent history and ``get_rolling_stats()``
    for windowed aggregation (win rate, avg P&L, Sharpe, max drawdown).

    Args:
        db_path: Path to the performance JSONL file.
    """

    def __init__(self, db_path: str = "results/watchdog/performance.jsonl") -> None:
        self._db_path = db_path
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_daily(
        self,
        date: str,
        results_dir: str = "results",
    ) -> Dict[str, Any]:
        """Ingest a single day's data and append per-bot rows to the DB.

        Reads the daily JSON report, JSONL trade logs, and
        ``auditor_state.json``.  Computes per-bot metrics and appends
        one line per active bot to ``performance.jsonl``.

        Args:
            date: Date string in ``YYYY-MM-DD`` format.
            results_dir: Base directory for result files.

        Returns:
            Dict mapping bot_tag -> ingested metrics row, or an empty
            dict if no data was found for the given date.
        """
        daily_json = self._read_daily_json(date, results_dir)
        trade_entries = self._read_trade_logs(date, results_dir)
        auditor_data = self._read_auditor_state()

        if not daily_json:
            logger.warning(
                "No daily JSON found for %s in %s/daily/", date, results_dir
            )
            return {}

        bots_section: Dict[str, Any] = daily_json.get("bots", {})
        account_section: Dict[str, Any] = daily_json.get("account", {})
        equity_at_close = _dec(account_section.get("equity", 0))

        rows: Dict[str, Dict[str, Any]] = {}

        for bot_tag, bot_data in bots_section.items():
            if bot_tag not in _BOT_TAGS:
                continue

            row = self._build_row(
                date=date,
                bot_tag=bot_tag,
                bot_data=bot_data,
                trade_entries=trade_entries.get(bot_tag, []),
                auditor_data=auditor_data,
                equity_at_close=equity_at_close,
            )
            self._append_row(row)
            rows[bot_tag] = row
            logger.info(
                "Ingested %s for %s: pnl=%s trades=%s",
                date,
                bot_tag,
                row["realized_pnl"],
                row["trades"],
            )

        return rows

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        bot_tag: str = "",
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Return recent performance entries from the database.

        Args:
            bot_tag: If non-empty, filter to this bot only.
            days: Maximum age in days (from today) to include.

        Returns:
            List of performance dicts, newest first.
        """
        cutoff = (datetime.now(ET) - timedelta(days=days)).strftime("%Y-%m-%d")
        results: List[Dict[str, Any]] = []

        if not os.path.exists(self._db_path):
            return results

        try:
            with open(self._db_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    entry_date = entry.get("date", "")
                    if entry_date < cutoff:
                        continue
                    if bot_tag and entry.get("bot_tag") != bot_tag:
                        continue
                    results.append(entry)
        except OSError as exc:
            logger.error("Failed to read performance DB: %s", exc)

        results.sort(key=lambda r: r.get("date", ""), reverse=True)
        return results

    # ------------------------------------------------------------------
    # Rolling stats
    # ------------------------------------------------------------------

    def get_rolling_stats(
        self,
        bot_tag: str,
        window: int = 7,
    ) -> Dict[str, Any]:
        """Compute rolling statistics over the last *window* days.

        Metrics: win_rate, avg_pnl, sharpe, max_drawdown, total_pnl,
        total_trades, total_winners, total_losers.

        Args:
            bot_tag: Bot identifier (e.g. ``"15M_STK"``).
            window: Number of days to include.

        Returns:
            Dict of aggregated statistics.
        """
        entries = self.query(bot_tag=bot_tag, days=window)

        stats: Dict[str, Any] = {
            "bot_tag": bot_tag,
            "window_days": window,
            "entries_found": len(entries),
            "total_trades": 0,
            "total_winners": 0,
            "total_losers": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "total_pnl": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

        if not entries:
            return stats

        total_trades = Decimal("0")
        total_winners = Decimal("0")
        total_losers = Decimal("0")
        total_pnl = Decimal("0")
        pnl_values: List[Decimal] = []
        sum_wins = Decimal("0")
        sum_losses = Decimal("0")

        # For drawdown we walk chronologically (entries are newest-first)
        chronological = sorted(entries, key=lambda e: e.get("date", ""))
        cumulative = Decimal("0")
        peak = Decimal("0")
        max_dd = Decimal("0")

        for entry in chronological:
            trades = _dec(entry.get("trades", 0))
            winners = _dec(entry.get("winners", 0))
            losers = _dec(entry.get("losers", 0))
            rpnl = _dec(entry.get("realized_pnl", 0))

            total_trades += trades
            total_winners += winners
            total_losers += losers
            total_pnl += rpnl
            pnl_values.append(rpnl)
            sum_wins += _dec(entry.get("avg_win", 0)) * winners
            sum_losses += _dec(entry.get("avg_loss", 0)) * losers

            cumulative += rpnl
            if cumulative > peak:
                peak = cumulative
            dd = cumulative - peak
            if dd < max_dd:
                max_dd = dd

        count = Decimal(str(len(entries)))
        avg_pnl = total_pnl / count if count else Decimal("0")

        win_rate = Decimal("0")
        if total_trades > 0:
            win_rate = (total_winners / total_trades) * Decimal("100")

        avg_win = Decimal("0")
        if total_winners > 0:
            avg_win = sum_wins / total_winners

        avg_loss = Decimal("0")
        if total_losers > 0:
            avg_loss = sum_losses / total_losers

        # Sharpe: mean / stdev of daily P&L (annualised is not meaningful here)
        sharpe = Decimal("0")
        if len(pnl_values) >= 2:
            mean = total_pnl / count
            variance = sum((v - mean) ** 2 for v in pnl_values) / (count - 1)
            try:
                std = variance.sqrt()
            except Exception:
                logger.debug(
                    "Decimal.sqrt() failed for bot=%s variance=%s; defaulting std to 0",
                    bot_tag, variance,
                )
                std = Decimal("0")
            if std > 0:
                sharpe = mean / std

        stats.update(
            {
                "total_trades": int(total_trades),
                "total_winners": int(total_winners),
                "total_losers": int(total_losers),
                "win_rate": _to_float(win_rate),
                "avg_pnl": _to_float(avg_pnl),
                "avg_win": _to_float(avg_win),
                "avg_loss": _to_float(avg_loss),
                "total_pnl": _to_float(total_pnl),
                "sharpe": _to_float(sharpe),
                "max_drawdown": _to_float(max_dd),
            }
        )
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_row(
        self,
        date: str,
        bot_tag: str,
        bot_data: Dict[str, Any],
        trade_entries: List[Dict[str, Any]],
        auditor_data: Dict[str, Any],
        equity_at_close: Decimal,
    ) -> Dict[str, Any]:
        """Build one performance row for a bot on a given day.

        Args:
            date: Date string YYYY-MM-DD.
            bot_tag: Bot identifier.
            bot_data: Bot section from daily JSON.
            trade_entries: Parsed JSONL trade log entries for this bot.
            auditor_data: Full auditor_state.json dict.
            equity_at_close: Account equity at close.

        Returns:
            Dict conforming to the performance schema.
        """
        realized_pnl = _dec(bot_data.get("realized_pnl", 0))

        # Compute winners / losers from the trades list in daily JSON
        trades_list: List[Dict[str, Any]] = bot_data.get("trades", [])
        # Group by symbol to compute per-round-trip P&L
        winners, losers = self._count_win_loss_from_daily(bot_data)
        total_trades = winners + losers

        win_rate = Decimal("0")
        if total_trades > 0:
            win_rate = (Decimal(str(winners)) / Decimal(str(total_trades))) * Decimal("100")

        # Compute avg win/loss from JSONL trade logs if available
        avg_win, avg_loss = self._compute_avg_win_loss(trade_entries)

        # Max drawdown intraday and peak from trade log timestamps
        max_dd, peak_pnl = self._compute_intraday_drawdown(trade_entries)

        # Signals and filters breakdown from JSONL logs
        signals_generated, signals_filtered, filters_breakdown = (
            self._count_signals_and_filters(trade_entries)
        )

        # Enrich from SCAN_CONTEXT events when available
        scan_regime, scan_config, scan_filters = self._extract_scan_context(
            trade_entries
        )
        regime = scan_regime or ""
        config_snapshot = scan_config or {}
        if scan_filters:
            filters_breakdown = scan_filters

        # Auditor shadow data
        shadow_pnl, shadow_trades, shadow_win_rate = self._extract_auditor_shadow(
            auditor_data, bot_tag
        )

        return {
            "date": date,
            "bot_tag": bot_tag,
            "trades": total_trades,
            "winners": winners,
            "losers": losers,
            "realized_pnl": _to_float(realized_pnl),
            "win_rate": _to_float(win_rate),
            "avg_win": _to_float(avg_win),
            "avg_loss": _to_float(avg_loss),
            "max_drawdown_intraday": _to_float(max_dd),
            "peak_pnl_before_giveback": _to_float(peak_pnl),
            "regime": regime,
            "signals_generated": signals_generated,
            "signals_filtered": signals_filtered,
            "config_snapshot": config_snapshot,
            "filters_breakdown": filters_breakdown,
            "auditor_shadow_pnl": _to_float(shadow_pnl),
            "auditor_shadow_trades": shadow_trades,
            "auditor_shadow_win_rate": _to_float(shadow_win_rate),
            "equity_at_close": _to_float(equity_at_close),
        }

    def _count_win_loss_from_daily(
        self, bot_data: Dict[str, Any]
    ) -> tuple[int, int]:
        """Derive winner/loser counts from the daily JSON bot section.

        Uses per-symbol buy/sell matching (same approach as
        ``persist_daily_results.calculate_bot_pnl``).

        Args:
            bot_data: A single bot's entry from the daily JSON.

        Returns:
            Tuple of (winners, losers).
        """
        trades: List[Dict[str, Any]] = bot_data.get("trades", [])
        buys: Dict[str, List[Decimal]] = {}
        sells: Dict[str, List[Decimal]] = {}

        for t in trades:
            sym = t.get("symbol", "")
            if not sym:
                continue
            side = t.get("side", "")
            price = _dec(t.get("price", 0))
            qty = _dec(t.get("qty", 0))
            if side == "buy":
                buys.setdefault(sym, []).append(price * qty)
            elif side == "sell":
                sells.setdefault(sym, []).append(price * qty)

        winners = 0
        losers = 0
        for sym in buys:
            if sym in sells:
                buy_total = sum(buys[sym])
                sell_total = sum(sells[sym])
                if sell_total > buy_total:
                    winners += 1
                elif sell_total < buy_total:
                    losers += 1
        return winners, losers

    def _compute_avg_win_loss(
        self, entries: List[Dict[str, Any]]
    ) -> tuple[Decimal, Decimal]:
        """Compute average win and average loss from JSONL trade log entries.

        Args:
            entries: JSONL log entries for a single bot.

        Returns:
            Tuple of (avg_win, avg_loss) as Decimals.
        """
        wins: List[Decimal] = []
        losses: List[Decimal] = []

        for entry in entries:
            etype = entry.get("type", "")
            if etype not in ("EXIT_FILL", "SPREAD_EXIT"):
                continue
            pnl = _dec(entry.get("pnl", 0))
            if pnl > 0:
                wins.append(pnl)
            elif pnl < 0:
                losses.append(pnl)

        avg_win = sum(wins) / Decimal(str(len(wins))) if wins else Decimal("0")
        avg_loss = sum(losses) / Decimal(str(len(losses))) if losses else Decimal("0")
        return avg_win, avg_loss

    def _compute_intraday_drawdown(
        self, entries: List[Dict[str, Any]]
    ) -> tuple[Decimal, Decimal]:
        """Compute intraday max drawdown and peak P&L from trade log events.

        Walks EXIT_FILL / SPREAD_EXIT events in timestamp order and tracks
        cumulative P&L, peak, and max drawdown.

        Args:
            entries: JSONL log entries for a single bot.

        Returns:
            Tuple of (max_drawdown, peak_pnl) as Decimals.
        """
        exit_events = [
            e
            for e in entries
            if e.get("type") in ("EXIT_FILL", "SPREAD_EXIT")
        ]
        exit_events.sort(key=lambda e: e.get("ts", ""))

        cumulative = Decimal("0")
        peak = Decimal("0")
        max_dd = Decimal("0")

        for ev in exit_events:
            cumulative += _dec(ev.get("pnl", 0))
            if cumulative > peak:
                peak = cumulative
            dd = cumulative - peak
            if dd < max_dd:
                max_dd = dd

        return max_dd, peak

    def _count_signals_and_filters(
        self, entries: List[Dict[str, Any]]
    ) -> tuple[int, int, Dict[str, int]]:
        """Count signal generation and filter-block events from JSONL logs.

        Args:
            entries: JSONL log entries for a single bot.

        Returns:
            Tuple of (signals_generated, signals_filtered, filters_breakdown).
        """
        signals_generated = 0
        signals_filtered = 0
        breakdown: Dict[str, int] = {}

        for entry in entries:
            etype = entry.get("type", "")
            if etype in ("SIGNAL", "SPREAD_SIGNAL"):
                signals_generated += 1
            elif etype == "GATE" and entry.get("decision") == "BLOCK":
                signals_filtered += 1
                reason = entry.get("reason", "unknown")
                breakdown[reason] = breakdown.get(reason, 0) + 1
            elif etype in ("SKIP_SLOPE3", "DKF_SKIP", "SPREAD_SKIP"):
                signals_filtered += 1
                breakdown[etype] = breakdown.get(etype, 0) + 1

        return signals_generated, signals_filtered, breakdown

    def _extract_scan_context(
        self, entries: List[Dict[str, Any]]
    ) -> tuple[str, Dict[str, Any], Dict[str, int]]:
        """Extract regime, config snapshot, and filters breakdown from SCAN_CONTEXT events.

        Uses the *last* SCAN_CONTEXT event for config/regime (most representative
        of end-of-day state) and aggregates per-symbol outcome counts across all
        SCAN_CONTEXT cycles for the filters breakdown.

        Args:
            entries: JSONL log entries for a single bot.

        Returns:
            Tuple of (regime, config_snapshot, filters_breakdown).
            All three are empty/falsy when no SCAN_CONTEXT events exist,
            allowing the caller to fall back to existing behaviour.
        """
        scan_events = [e for e in entries if e.get("type") == "SCAN_CONTEXT"]
        if not scan_events:
            return "", {}, {}

        # --- config_snapshot from the last SCAN_CONTEXT event ---
        last_event = scan_events[-1]
        regime = last_event.get("regime", "") or ""

        market_ctx = last_event.get("market_context") or {}
        config_snapshot: Dict[str, Any] = {"regime": regime}
        for _cfg_key in ("market_condition", "tp_r_effective", "size_multiplier"):
            _cfg_val = market_ctx.get(_cfg_key)
            if _cfg_val is not None:
                config_snapshot[_cfg_key] = _cfg_val

        # --- filters_breakdown from outcome counts across all cycles ---
        outcome_counts: Dict[str, int] = {}
        for event in scan_events:
            symbols = event.get("symbols")
            if not isinstance(symbols, list):
                continue
            for sym_record in symbols:
                outcome = sym_record.get("outcome", "")
                if outcome:
                    outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        return regime, config_snapshot, outcome_counts

    def _extract_auditor_shadow(
        self,
        auditor_data: Dict[str, Any],
        bot_tag: str,
    ) -> tuple[Decimal, int, Decimal]:
        """Extract shadow P&L metrics for *bot_tag* from auditor state.

        Args:
            auditor_data: Parsed ``auditor_state.json``.
            bot_tag: Bot identifier.

        Returns:
            Tuple of (shadow_pnl, shadow_trades, shadow_win_rate).
        """
        if not auditor_data:
            return Decimal("0"), 0, Decimal("0")

        closed: List[Dict[str, Any]] = auditor_data.get("closed_positions", [])
        bot_closed = [p for p in closed if p.get("bot_tag", "") == bot_tag]

        if not bot_closed:
            return Decimal("0"), 0, Decimal("0")

        total_pnl = Decimal("0")
        winners = 0
        for pos in bot_closed:
            pnl = _dec(pos.get("realized_pnl", 0))
            total_pnl += pnl
            if pnl > 0:
                winners += 1

        total = len(bot_closed)
        win_rate = (
            (Decimal(str(winners)) / Decimal(str(total))) * Decimal("100")
            if total > 0
            else Decimal("0")
        )
        return total_pnl, total, win_rate

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _read_daily_json(
        self, date: str, results_dir: str
    ) -> Dict[str, Any]:
        """Load ``results/daily/{date}.json``.

        Args:
            date: Date string YYYY-MM-DD.
            results_dir: Base results directory.

        Returns:
            Parsed JSON dict, or empty dict if missing/corrupt.
        """
        path = os.path.join(results_dir, "daily", f"{date}.json")
        if not os.path.exists(path):
            logger.debug("Daily JSON not found: %s", path)
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to read daily JSON %s: %s", path, exc)
            return {}

    def _read_trade_logs(
        self, date: str, results_dir: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Read JSONL trade logs for *date*, grouped by bot_tag.

        Looks for files matching patterns:
        - ``{BOT_TAG}_{YYYYMMDD}.jsonl``
        - ``live_{YYYYMMDD}.jsonl``           (legacy stock logs)
        - ``options_trades_{YYYYMMDD}.jsonl`` (legacy options logs)

        Args:
            date: Date string YYYY-MM-DD.
            results_dir: Base results directory.

        Returns:
            Dict mapping bot_tag -> list of JSONL entry dicts.
        """
        date_compact = date.replace("-", "")
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        if not os.path.isdir(results_dir):
            return grouped

        try:
            files = os.listdir(results_dir)
        except OSError as exc:
            logger.error("Failed to list results dir %s: %s", results_dir, exc)
            return grouped

        candidate_files = [
            f
            for f in files
            if f.endswith(".jsonl") and date_compact in f
        ]

        for fname in candidate_files:
            fpath = os.path.join(results_dir, fname)
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
                        # Use bot_tag from entry if available
                        entry_tag = entry.get("bot_tag", bot_tag)
                        if entry_tag:
                            grouped.setdefault(entry_tag, []).append(entry)
            except OSError as exc:
                logger.warning("Failed to read trade log %s: %s", fpath, exc)

        return grouped

    def _infer_bot_tag_from_filename(self, fname: str) -> str:
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

    def _read_auditor_state(self) -> Dict[str, Any]:
        """Load ``auditor_state.json`` from the repo root.

        Returns:
            Parsed JSON dict, or empty dict if missing/corrupt.
        """
        # auditor_state.json lives at repo root (same level as results/)
        candidates = [
            "auditor_state.json",
            os.path.join("..", "auditor_state.json"),
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        return json.load(fh)
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Failed to read auditor state %s: %s", path, exc)
                    return {}
        logger.debug("auditor_state.json not found (expected at repo root)")
        return {}

    def _append_row(self, row: Dict[str, Any]) -> None:
        """Append a single performance row to the JSONL database.

        Args:
            row: Performance metrics dict.
        """
        try:
            line = json.dumps(row, separators=(",", ":"), ensure_ascii=False, default=str)
            with open(self._db_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            logger.error("Failed to append to performance DB: %s", exc)
