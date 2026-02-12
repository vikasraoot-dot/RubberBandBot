"""
Rolling Performance Tracker: Computes 7/30/90-day rolling statistics and anomaly flags.

Reads from ``performance.jsonl`` (populated by ``PerformanceDB.ingest_daily()``)
and produces ``results/watchdog/rolling_stats.json`` with per-bot rolling metrics
and anomaly detection flags.

Anomaly flags:
- DEGRADED:          7-day win rate below configurable threshold (default 35%).
- LOSING:            7-day cumulative P&L below configurable threshold (default -$300).
- ASYMMETRIC_RISK:   Average loss growing while average win stays flat.
- FILTERS_TOO_STRICT:  Auditor shadow consistently outperforms bot over 7 days.
- FILTERS_ADDING_VALUE:  Bot consistently outperforms auditor shadow over 7 days.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from RubberBand.src.watchdog.utils import to_dec as _dec, dec_to_float as _to_float

logger = logging.getLogger(__name__)

ET = ZoneInfo("US/Eastern")

# Default bot tags
_BOT_TAGS = {"15M_STK", "15M_OPT", "WK_STK", "WK_OPT", "SL_ETF"}


class RollingTracker:
    """Computes rolling statistics and anomaly flags from the performance DB.

    Reads ``performance.jsonl`` and the watchdog config for thresholds,
    computes per-bot rolling stats across multiple windows, and saves
    the result to ``rolling_stats.json``.

    Args:
        db_path: Path to the performance JSONL file.
        config_path: Path to the watchdog config YAML.
        output_path: Path to write rolling_stats.json.
    """

    def __init__(
        self,
        db_path: str = "results/watchdog/performance.jsonl",
        config_path: str = "results/watchdog/watchdog_config.yaml",
        output_path: str = "results/watchdog/rolling_stats.json",
    ) -> None:
        self._db_path = db_path
        self._config_path = config_path
        self._output_path = output_path
        self._config = self._load_config()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def compute(self) -> Dict[str, Any]:
        """Compute rolling stats for all bots and save to disk.

        Returns:
            The full rolling stats dict (also persisted to disk).
        """
        logger.info("Computing rolling statistics")
        all_entries = self._read_performance_db()

        if not all_entries:
            logger.warning("No entries in performance DB; generating empty stats")

        # Group by bot_tag
        by_bot: Dict[str, List[Dict[str, Any]]] = {}
        for entry in all_entries:
            tag = entry.get("bot_tag", "")
            if tag:
                by_bot.setdefault(tag, []).append(entry)

        result: Dict[str, Any] = {
            "generated_at": datetime.now(ET).isoformat(),
            "bots": {},
        }

        for bot_tag in sorted(by_bot.keys()):
            entries = sorted(by_bot[bot_tag], key=lambda e: e.get("date", ""))
            bot_stats: Dict[str, Any] = {}
            for window in (7, 30, 90):
                window_entries = self._filter_window(entries, window)
                bot_stats[f"{window}d"] = self._compute_window_stats(
                    bot_tag, window_entries, window
                )
            bot_stats["flags"] = self._compute_anomaly_flags(bot_tag, entries)
            result["bots"][bot_tag] = bot_stats

        # Persist
        try:
            os.makedirs(os.path.dirname(self._output_path) or ".", exist_ok=True)
            with open(self._output_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, default=str)
            logger.info("Saved rolling stats to %s", self._output_path)
        except OSError as exc:
            logger.error("Failed to save rolling stats: %s", exc)

        return result

    # ------------------------------------------------------------------
    # Rolling window stats
    # ------------------------------------------------------------------

    def _compute_window_stats(
        self,
        bot_tag: str,
        entries: List[Dict[str, Any]],
        window: int,
    ) -> Dict[str, Any]:
        """Compute statistics for a single window of entries.

        Args:
            bot_tag: Bot identifier.
            entries: Chronologically sorted performance entries within window.
            window: Window size in days.

        Returns:
            Dict with aggregated statistics.
        """
        stats: Dict[str, Any] = {
            "window_days": window,
            "entries_found": len(entries),
            "total_trades": 0,
            "total_winners": 0,
            "total_losers": 0,
            "win_rate": 0.0,
            "avg_pnl_per_day": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "total_pnl": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "trend": "FLAT",
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

        # Drawdown tracking
        cumulative = Decimal("0")
        peak = Decimal("0")
        max_dd = Decimal("0")

        for entry in entries:
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

        # Sharpe: mean / stdev of daily P&L
        sharpe = Decimal("0")
        if len(pnl_values) >= 2:
            mean = total_pnl / count
            variance = sum(
                (v - mean) ** 2 for v in pnl_values
            ) / (count - 1)
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

        # Trend detection: compare first half avg vs second half avg
        trend = "FLAT"
        if len(pnl_values) >= 4:
            mid = len(pnl_values) // 2
            first_half = sum(pnl_values[:mid], Decimal("0")) / Decimal(str(mid))
            second_half = sum(pnl_values[mid:], Decimal("0")) / Decimal(
                str(len(pnl_values) - mid)
            )
            delta = second_half - first_half
            if delta > Decimal("5"):
                trend = "IMPROVING"
            elif delta < Decimal("-5"):
                trend = "DEGRADING"

        stats.update(
            {
                "total_trades": int(total_trades),
                "total_winners": int(total_winners),
                "total_losers": int(total_losers),
                "win_rate": _to_float(win_rate),
                "avg_pnl_per_day": _to_float(avg_pnl),
                "avg_win": _to_float(avg_win),
                "avg_loss": _to_float(avg_loss),
                "total_pnl": _to_float(total_pnl),
                "sharpe": _to_float(sharpe),
                "max_drawdown": _to_float(max_dd),
                "trend": trend,
            }
        )
        return stats

    # ------------------------------------------------------------------
    # Anomaly flags
    # ------------------------------------------------------------------

    def _compute_anomaly_flags(
        self,
        bot_tag: str,
        entries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Detect anomaly conditions from the performance history.

        Args:
            bot_tag: Bot identifier.
            entries: Full chronologically sorted performance entries.

        Returns:
            List of anomaly flag dicts with name, severity, detail.
        """
        flags: List[Dict[str, Any]] = []

        # Use 7-day window for flag calculations
        recent_7d = self._filter_window(entries, 7)
        if not recent_7d:
            return flags

        # Load thresholds from config
        degradation = self._config.get("degradation_flags", {})
        min_win_rate_7d = _dec(degradation.get("min_win_rate_7d", 35))
        max_cum_loss_7d = _dec(degradation.get("max_cumulative_loss_7d", -300))

        # Aggregate 7-day metrics
        total_trades_7d = sum(_dec(e.get("trades", 0)) for e in recent_7d)
        total_winners_7d = sum(_dec(e.get("winners", 0)) for e in recent_7d)
        total_pnl_7d = sum(
            _dec(e.get("realized_pnl", 0)) for e in recent_7d
        )

        win_rate_7d = Decimal("0")
        if total_trades_7d > 0:
            win_rate_7d = (total_winners_7d / total_trades_7d) * Decimal("100")

        # Flag: DEGRADED
        if total_trades_7d > 0 and win_rate_7d < min_win_rate_7d:
            flags.append({
                "flag": "DEGRADED",
                "severity": "WARNING",
                "detail": (
                    f"7-day win rate {_to_float(win_rate_7d)}% "
                    f"< {_to_float(min_win_rate_7d)}% threshold"
                ),
            })

        # Flag: LOSING
        if total_pnl_7d < max_cum_loss_7d:
            flags.append({
                "flag": "LOSING",
                "severity": "CRITICAL",
                "detail": (
                    f"7-day cumulative P&L ${_to_float(total_pnl_7d)} "
                    f"< ${_to_float(max_cum_loss_7d)} threshold"
                ),
            })

        # Flag: ASYMMETRIC_RISK
        # Check if avg loss is growing while avg win is flat/declining
        if len(recent_7d) >= 4:
            mid = len(recent_7d) // 2
            first_half = recent_7d[:mid]
            second_half = recent_7d[mid:]

            fh_avg_loss = self._avg_field(first_half, "avg_loss")
            sh_avg_loss = self._avg_field(second_half, "avg_loss")
            fh_avg_win = self._avg_field(first_half, "avg_win")
            sh_avg_win = self._avg_field(second_half, "avg_win")

            # avg_loss is negative, so "growing" means more negative
            if sh_avg_loss < fh_avg_loss - Decimal("2") and sh_avg_win <= fh_avg_win + Decimal("2"):
                flags.append({
                    "flag": "ASYMMETRIC_RISK",
                    "severity": "WARNING",
                    "detail": (
                        f"Avg loss worsening ({_to_float(fh_avg_loss)} -> "
                        f"{_to_float(sh_avg_loss)}) while avg win flat "
                        f"({_to_float(fh_avg_win)} -> {_to_float(sh_avg_win)})"
                    ),
                })

        # Flag: FILTERS_TOO_STRICT / FILTERS_ADDING_VALUE
        # Compare auditor shadow P&L vs bot P&L over 7 days
        shadow_pnl_7d = sum(
            _dec(e.get("auditor_shadow_pnl", 0)) for e in recent_7d
        )
        if total_pnl_7d != 0 or shadow_pnl_7d != 0:
            if shadow_pnl_7d > total_pnl_7d and shadow_pnl_7d > 0:
                flags.append({
                    "flag": "FILTERS_TOO_STRICT",
                    "severity": "INFO",
                    "detail": (
                        f"Shadow P&L ${_to_float(shadow_pnl_7d)} > "
                        f"Bot P&L ${_to_float(total_pnl_7d)} over 7 days"
                    ),
                })
            elif total_pnl_7d > shadow_pnl_7d:
                flags.append({
                    "flag": "FILTERS_ADDING_VALUE",
                    "severity": "INFO",
                    "detail": (
                        f"Bot P&L ${_to_float(total_pnl_7d)} > "
                        f"Shadow P&L ${_to_float(shadow_pnl_7d)} over 7 days"
                    ),
                })

        return flags

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _filter_window(
        self,
        entries: List[Dict[str, Any]],
        days: int,
    ) -> List[Dict[str, Any]]:
        """Return entries from the last *days* calendar days.

        Args:
            entries: Chronologically sorted performance entries.
            days: Number of days to include.

        Returns:
            Filtered list (newest last).
        """
        if not entries:
            return []
        # Entries are sorted chronologically; take the tail
        # Use date comparison to handle gaps
        today = datetime.now(ET).strftime("%Y-%m-%d")
        from datetime import timedelta
        cutoff = (datetime.now(ET) - timedelta(days=days)).strftime("%Y-%m-%d")
        return [e for e in entries if e.get("date", "") >= cutoff]

    @staticmethod
    def _avg_field(
        entries: List[Dict[str, Any]], field: str
    ) -> Decimal:
        """Compute the average of a numeric field across entries.

        Args:
            entries: List of performance dicts.
            field: Key name to average.

        Returns:
            Average as Decimal, or Decimal("0") if empty.
        """
        if not entries:
            return Decimal("0")
        total = sum((_dec(e.get(field, 0)) for e in entries), Decimal("0"))
        return total / Decimal(str(len(entries)))

    def _read_performance_db(self) -> List[Dict[str, Any]]:
        """Read all entries from the performance JSONL database.

        Returns:
            List of performance dicts, chronologically sorted.
        """
        results: List[Dict[str, Any]] = []
        if not os.path.exists(self._db_path):
            logger.debug("Performance DB not found: %s", self._db_path)
            return results

        try:
            with open(self._db_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        results.append(entry)
                    except json.JSONDecodeError:
                        continue
        except OSError as exc:
            logger.error("Failed to read performance DB: %s", exc)

        results.sort(key=lambda r: r.get("date", ""))
        return results

    def _load_config(self) -> Dict[str, Any]:
        """Load watchdog config YAML.

        Returns:
            Config dict, or empty dict with defaults if missing.
        """
        if not os.path.exists(self._config_path):
            logger.debug("Watchdog config not found: %s", self._config_path)
            return {}
        try:
            import yaml
            with open(self._config_path, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except ImportError:
            logger.warning("PyYAML not installed; using default config")
            return {}
        except Exception as exc:
            logger.error("Failed to read watchdog config: %s", exc)
            return {}
