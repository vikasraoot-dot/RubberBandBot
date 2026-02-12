"""
Intraday Health Monitor: Real-time per-bot P&L tracking with graduated responses.

Runs every ~10 minutes during market hours (via GitHub Actions).  Reads live
account/position data from Alpaca, attributes P&L to bots via client_order_id,
and enforces configurable thresholds:

    -$50  -> LOG warning
    -$100 -> PAUSE new entries
    -$200 -> PAUSE + tighten SL on losing positions only
    Account < -3% -> PAUSE all bots

Also implements the profit-lock mechanism: once a bot's daily P&L exceeds a
threshold (+$20 default), 50% of the peak is locked as a floor.  If P&L drops
below the floor, the bot is paused.

State files written:
    results/watchdog/intraday_health.json  - current day snapshot
    results/watchdog/bot_pause_flags.json  - pause/resume flags per bot
    results/watchdog/profit_locks.json     - per-bot profit lock state
    results/watchdog/alerts.jsonl          - append-only alert log
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import yaml

from RubberBand.src.watchdog.utils import to_dec as _dec, dec_to_float as _to_float
from RubberBand.src.position_registry import BOT_TAGS

logger = logging.getLogger(__name__)

ET = ZoneInfo("US/Eastern")

# Default paths (relative to repo root / CWD)
_CONFIG_PATH = "results/watchdog/watchdog_config.yaml"
_HEALTH_PATH = "results/watchdog/intraday_health.json"
_PAUSE_FLAGS_PATH = "results/watchdog/bot_pause_flags.json"
_PROFIT_LOCKS_PATH = "results/watchdog/profit_locks.json"
_ALERTS_PATH = "results/watchdog/alerts.jsonl"


def _now_et() -> datetime:
    """Return current time in US/Eastern.

    Returns:
        Timezone-aware datetime in ET.
    """
    return datetime.now(ET)


def _load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file, returning empty dict if missing or corrupt.

    Args:
        path: File path to load.

    Returns:
        Parsed JSON dict.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return {}


def _save_json(path: str, data: Any) -> None:
    """Save data as pretty-printed JSON.

    Args:
        path: Destination file path.
        data: JSON-serialisable object.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
    except OSError as exc:
        logger.error("Failed to write %s: %s", path, exc)


def _append_alert(alert: Dict[str, Any], path: str = _ALERTS_PATH) -> None:
    """Append a single alert line to the alerts JSONL file.

    Args:
        alert: Alert dict.
        path: Path to the alerts JSONL file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    alert.setdefault("ts", _now_et().isoformat())
    try:
        line = json.dumps(alert, separators=(",", ":"), ensure_ascii=False, default=str)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        logger.error("Failed to append alert: %s", exc)


class IntraDayMonitor:
    """Real-time per-bot P&L monitor with graduated response thresholds.

    Designed to be called once per cycle (every ~10 min).  Each call:
    1. Fetches account + positions + daily fills from Alpaca.
    2. Attributes P&L to each bot via ``parse_client_order_id()``.
    3. Evaluates thresholds (warn, pause, emergency) and profit locks.
    4. Writes state files (pause flags, profit locks, intraday health).

    Args:
        config_path: Path to ``watchdog_config.yaml``.
        health_path: Path for ``intraday_health.json`` output.
        pause_flags_path: Path for ``bot_pause_flags.json`` output.
        profit_locks_path: Path for ``profit_locks.json`` output.
        alerts_path: Path for ``alerts.jsonl`` output.
    """

    def __init__(
        self,
        config_path: str = _CONFIG_PATH,
        health_path: str = _HEALTH_PATH,
        pause_flags_path: str = _PAUSE_FLAGS_PATH,
        profit_locks_path: str = _PROFIT_LOCKS_PATH,
        alerts_path: str = _ALERTS_PATH,
    ) -> None:
        self._config_path = config_path
        self._health_path = health_path
        self._pause_flags_path = pause_flags_path
        self._profit_locks_path = profit_locks_path
        self._alerts_path = alerts_path

        self._config = self._load_config()
        self._pause_flags = _load_json(self._pause_flags_path)
        self._profit_locks = _load_json(self._profit_locks_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_cycle(self) -> Dict[str, Any]:
        """Execute one monitoring cycle.

        Fetches live data from Alpaca, computes per-bot P&L, evaluates
        thresholds, updates pause flags and profit locks, and persists
        all state to disk.

        Returns:
            Intraday health snapshot dict.
        """
        from RubberBand.src.data import get_account, get_positions, get_daily_fills
        from RubberBand.src.position_registry import parse_client_order_id

        now = _now_et()
        today = now.strftime("%Y-%m-%d")

        # Fetch live data
        account = get_account()
        positions = get_positions()
        fills = get_daily_fills()

        if not account:
            logger.error("Failed to fetch account data — skipping cycle")
            return {}

        equity = _dec(account.get("equity", 0))
        last_equity = _dec(account.get("last_equity", 0))

        # Account-level daily P&L %
        account_pnl_pct = Decimal("0")
        if last_equity > 0:
            account_pnl_pct = ((equity - last_equity) / last_equity) * Decimal("100")

        # Attribute P&L per bot from fills
        bot_pnl = self._compute_bot_pnl(fills, positions)

        # Evaluate thresholds for each bot
        thresholds = self._config.get("thresholds", {})
        warn_loss = _dec(thresholds.get("warn_loss", -50))
        pause_loss = _dec(thresholds.get("pause_loss", -100))
        emergency_loss = _dec(thresholds.get("emergency_loss", -200))
        account_pause_pct = _dec(thresholds.get("account_pause_pct", -3.0))
        profit_lock_activation = _dec(thresholds.get("profit_lock_activation", 20))
        profit_lock_pct = _dec(thresholds.get("profit_lock_pct", Decimal("0.50")))

        # Check account-level pause
        account_paused = account_pnl_pct < account_pause_pct
        if account_paused:
            logger.critical(
                "ACCOUNT-LEVEL PAUSE: daily P&L %.2f%% < %.2f%% threshold",
                float(account_pnl_pct),
                float(account_pause_pct),
            )
            _append_alert(
                {
                    "level": "CRITICAL",
                    "event": "ACCOUNT_PAUSE",
                    "account_pnl_pct": _to_float(account_pnl_pct),
                    "threshold": _to_float(account_pause_pct),
                },
                self._alerts_path,
            )

        health: Dict[str, Any] = {
            "date": today,
            "updated_at": now.isoformat(),
            "account": {
                "equity": _to_float(equity),
                "last_equity": _to_float(last_equity),
                "daily_pnl_pct": _to_float(account_pnl_pct),
                "account_paused": account_paused,
            },
            "bots": {},
        }

        for bot_tag in sorted(BOT_TAGS):
            pnl = bot_pnl.get(bot_tag, Decimal("0"))
            bot_health = self._evaluate_bot(
                bot_tag=bot_tag,
                daily_pnl=pnl,
                warn_loss=warn_loss,
                pause_loss=pause_loss,
                emergency_loss=emergency_loss,
                account_paused=account_paused,
                profit_lock_activation=profit_lock_activation,
                profit_lock_pct=profit_lock_pct,
                now=now,
            )
            health["bots"][bot_tag] = bot_health

        # Persist state
        _save_json(self._health_path, health)
        _save_json(self._pause_flags_path, self._pause_flags)
        _save_json(self._profit_locks_path, self._profit_locks)

        logger.info(
            "Cycle complete: equity=$%s, account_pnl=%.2f%%, bots=%s",
            _to_float(equity),
            float(account_pnl_pct),
            {k: _to_float(v) for k, v in bot_pnl.items()},
        )
        return health

    # ------------------------------------------------------------------
    # Per-bot evaluation
    # ------------------------------------------------------------------

    def _evaluate_bot(
        self,
        bot_tag: str,
        daily_pnl: Decimal,
        warn_loss: Decimal,
        pause_loss: Decimal,
        emergency_loss: Decimal,
        account_paused: bool,
        profit_lock_activation: Decimal,
        profit_lock_pct: Decimal,
        now: datetime,
    ) -> Dict[str, Any]:
        """Evaluate thresholds and profit lock for a single bot.

        Args:
            bot_tag: Bot identifier.
            daily_pnl: Realised daily P&L for this bot.
            warn_loss: Warning threshold (negative).
            pause_loss: Pause threshold (negative).
            emergency_loss: Emergency threshold (negative).
            account_paused: Whether the whole account is paused.
            profit_lock_activation: P&L level to activate profit lock.
            profit_lock_pct: Fraction of peak to lock as floor.
            now: Current timestamp.

        Returns:
            Bot health snapshot dict.
        """
        status = "OK"
        pause_reason = ""
        paused = False

        # --- Loss thresholds (graduated) ---
        if daily_pnl <= emergency_loss:
            status = "EMERGENCY"
            pause_reason = (
                f"Daily loss ${_to_float(daily_pnl)} exceeded "
                f"${_to_float(emergency_loss)} threshold"
            )
            paused = True
            self._emit_alert(bot_tag, "EMERGENCY", pause_reason)
        elif daily_pnl <= pause_loss:
            status = "PAUSED"
            pause_reason = (
                f"Daily loss ${_to_float(daily_pnl)} exceeded "
                f"${_to_float(pause_loss)} threshold"
            )
            paused = True
            self._emit_alert(bot_tag, "PAUSE", pause_reason)
        elif daily_pnl <= warn_loss:
            status = "WARNING"
            self._emit_alert(
                bot_tag,
                "WARNING",
                f"Daily P&L ${_to_float(daily_pnl)} below "
                f"${_to_float(warn_loss)} warn level",
            )

        # --- Account-level override ---
        if account_paused and not paused:
            status = "ACCOUNT_PAUSED"
            pause_reason = "Account-level daily loss threshold exceeded"
            paused = True

        # --- Profit lock ---
        lock_state = self._update_profit_lock(
            bot_tag, daily_pnl, profit_lock_activation, profit_lock_pct, now
        )
        if lock_state.get("lock_triggered") and not paused:
            status = "PROFIT_LOCKED"
            pause_reason = (
                f"P&L ${_to_float(daily_pnl)} dropped below lock floor "
                f"${lock_state.get('lock_floor', 0)}"
            )
            paused = True
            self._emit_alert(bot_tag, "PROFIT_LOCK", pause_reason)

        # --- Update pause flags ---
        self._set_pause_flag(bot_tag, paused, pause_reason, now)

        return {
            "bot_tag": bot_tag,
            "daily_pnl": _to_float(daily_pnl),
            "status": status,
            "paused": paused,
            "pause_reason": pause_reason,
            "profit_lock": lock_state,
        }

    # ------------------------------------------------------------------
    # Profit lock
    # ------------------------------------------------------------------

    def _update_profit_lock(
        self,
        bot_tag: str,
        daily_pnl: Decimal,
        activation: Decimal,
        lock_pct: Decimal,
        now: datetime,
    ) -> Dict[str, Any]:
        """Update profit lock state for a bot.

        Tracks the daily high-water mark.  Once P&L exceeds *activation*,
        a lock floor is set at ``peak * lock_pct``.  If P&L later drops
        below the floor the lock triggers (bot should be paused).  If P&L
        recovers above the floor the lock un-triggers (dynamic, not one-way).

        Args:
            bot_tag: Bot identifier.
            daily_pnl: Current daily P&L.
            activation: Minimum P&L to activate the lock.
            lock_pct: Fraction of peak to set as floor.
            now: Current timestamp.

        Returns:
            Lock state dict for this bot.
        """
        lock = self._profit_locks.get(bot_tag, {
            "peak_pnl_today": 0.0,
            "lock_floor": 0.0,
            "lock_active": False,
            "lock_triggered": False,
            "lock_triggered_at": None,
        })

        pnl_f = _to_float(daily_pnl)
        peak = _dec(lock.get("peak_pnl_today", 0))

        # Update peak
        if daily_pnl > peak:
            peak = daily_pnl
            lock["peak_pnl_today"] = _to_float(peak)

        # Activate lock once peak exceeds threshold
        if peak >= activation:
            lock["lock_active"] = True
            lock["lock_floor"] = _to_float(peak * lock_pct)

        # Evaluate trigger (dynamic: can un-trigger if P&L recovers)
        if lock.get("lock_active"):
            floor = _dec(lock.get("lock_floor", 0))
            if daily_pnl < floor:
                if not lock.get("lock_triggered"):
                    lock["lock_triggered"] = True
                    lock["lock_triggered_at"] = now.isoformat()
            else:
                # P&L recovered above floor — un-trigger
                lock["lock_triggered"] = False
                lock["lock_triggered_at"] = None

        self._profit_locks[bot_tag] = lock
        return lock

    # ------------------------------------------------------------------
    # Pause flags
    # ------------------------------------------------------------------

    def _set_pause_flag(
        self,
        bot_tag: str,
        paused: bool,
        reason: str,
        now: datetime,
    ) -> None:
        """Write the pause flag for a bot.

        Pauses auto-reset at market open next day (``resume_tomorrow: true``).

        Args:
            bot_tag: Bot identifier.
            paused: Whether the bot should be paused.
            reason: Human-readable pause reason.
            now: Current timestamp.
        """
        current = self._pause_flags.get(bot_tag, {
            "paused": False,
            "reason": "",
            "paused_at": None,
            "resume_tomorrow": True,
        })

        if paused:
            current["paused"] = True
            current["reason"] = reason
            if not current.get("paused_at"):
                current["paused_at"] = now.isoformat()
            current["resume_tomorrow"] = True
        else:
            current["paused"] = False
            current["reason"] = ""
            current["paused_at"] = None
            current["resume_tomorrow"] = True

        self._pause_flags[bot_tag] = current

    # ------------------------------------------------------------------
    # P&L attribution
    # ------------------------------------------------------------------

    def _compute_bot_pnl(
        self,
        fills: List[Dict[str, Any]],
        positions: List[Dict[str, Any]],
    ) -> Dict[str, Decimal]:
        """Attribute daily realised + unrealised P&L to each bot from fills.

        Uses ``parse_client_order_id()`` to tag fills.  Untagged sells are
        matched to tagged buys for the same symbol (bracket order pattern).

        Args:
            fills: List of filled orders from ``get_daily_fills()``.
            positions: Current open positions from ``get_positions()``.

        Returns:
            Dict mapping bot_tag -> Decimal daily P&L.
        """
        from RubberBand.src.position_registry import parse_client_order_id

        # Step 1: tag every fill
        tagged_buys: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}  # bot -> sym -> [fill]
        untagged_sells: Dict[str, List[Dict[str, Any]]] = {}  # sym -> [fill]
        tagged_sells: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}  # bot -> sym -> [fill]

        for fill in fills:
            coid = fill.get("client_order_id", "")
            parsed = parse_client_order_id(coid)
            bot = parsed.get("bot_tag", "")
            side = fill.get("side", "")
            sym = fill.get("symbol", "")

            if bot:
                bucket = tagged_buys if side == "buy" else tagged_sells
                bucket.setdefault(bot, {}).setdefault(sym, []).append(fill)
            elif side == "sell":
                untagged_sells.setdefault(sym, []).append(fill)

        # Step 2: match untagged sells to tagged buys by symbol
        for bot, sym_map in tagged_buys.items():
            for sym in sym_map:
                if sym in untagged_sells:
                    tagged_sells.setdefault(bot, {}).setdefault(sym, []).extend(
                        untagged_sells.pop(sym, [])
                    )

        # Step 3: compute realised P&L per bot
        bot_pnl: Dict[str, Decimal] = {}
        for bot in sorted(BOT_TAGS):
            pnl = Decimal("0")
            buy_syms = tagged_buys.get(bot, {})
            sell_syms = tagged_sells.get(bot, {})

            all_syms = set(buy_syms.keys()) | set(sell_syms.keys())
            for sym in all_syms:
                buy_fills = buy_syms.get(sym, [])
                sell_fills = sell_syms.get(sym, [])

                buy_value = sum(
                    _dec(f.get("filled_qty", 0)) * _dec(f.get("filled_avg_price", 0))
                    for f in buy_fills
                )
                buy_qty = sum(_dec(f.get("filled_qty", 0)) for f in buy_fills)
                sell_value = sum(
                    _dec(f.get("filled_qty", 0)) * _dec(f.get("filled_avg_price", 0))
                    for f in sell_fills
                )
                sell_qty = sum(_dec(f.get("filled_qty", 0)) for f in sell_fills)

                # Realised P&L on matched qty
                matched = min(buy_qty, sell_qty)
                if matched > 0 and buy_qty > 0 and sell_qty > 0:
                    avg_buy = buy_value / buy_qty
                    avg_sell = sell_value / sell_qty
                    pnl += matched * (avg_sell - avg_buy)

                # Unrealised on open portion
                open_qty = buy_qty - sell_qty
                if open_qty > 0 and buy_qty > 0:
                    avg_buy = buy_value / buy_qty
                    for pos in positions:
                        if pos.get("symbol") == sym:
                            current = _dec(pos.get("current_price", 0))
                            pnl += open_qty * (current - avg_buy)
                            break

            if pnl != 0:
                bot_pnl[bot] = pnl

        return bot_pnl

    # ------------------------------------------------------------------
    # Config & alerts
    # ------------------------------------------------------------------

    def _load_config(self) -> Dict[str, Any]:
        """Load watchdog_config.yaml.

        Returns:
            Parsed YAML config dict.  Empty dict if missing/corrupt.
        """
        if not os.path.exists(self._config_path):
            logger.warning("Watchdog config not found: %s — using defaults", self._config_path)
            return {}
        try:
            with open(self._config_path, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except Exception as exc:
            logger.error("Failed to parse watchdog config: %s", exc)
            return {}

    def _emit_alert(self, bot_tag: str, level: str, message: str) -> None:
        """Log and persist an alert.

        Args:
            bot_tag: Bot identifier.
            level: Severity (WARNING, PAUSE, EMERGENCY, PROFIT_LOCK, etc).
            message: Human-readable description.
        """
        logger.warning("[%s] %s: %s", bot_tag, level, message)
        _append_alert(
            {
                "bot_tag": bot_tag,
                "level": level,
                "message": message,
            },
            self._alerts_path,
        )

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def reset_daily(self) -> None:
        """Reset pause flags and profit locks for a new trading day.

        Called at market open.  Only resets flags marked with
        ``resume_tomorrow: true``.
        """
        for bot_tag, flag in self._pause_flags.items():
            if flag.get("resume_tomorrow", True):
                flag["paused"] = False
                flag["reason"] = ""
                flag["paused_at"] = None

        self._profit_locks = {}

        _save_json(self._pause_flags_path, self._pause_flags)
        _save_json(self._profit_locks_path, self._profit_locks)
        logger.info("Daily reset complete — all pause flags and profit locks cleared")
