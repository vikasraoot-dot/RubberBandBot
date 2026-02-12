"""
Pause Flag Reader: Check whether the watchdog has paused a specific bot.

Follows the same pattern as ``ticker_health.py`` ``is_paused()`` (lines 39-72).
Reads ``bot_pause_flags.json`` written by ``IntraDayMonitor``.

**Fail-open**: if the file is missing or corrupt, returns ``(False, "")``
so the bot continues normally.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Tuple

logger = logging.getLogger(__name__)

_DEFAULT_FLAGS_PATH = "results/watchdog/bot_pause_flags.json"


def check_bot_paused(
    bot_tag: str,
    flags_path: str = _DEFAULT_FLAGS_PATH,
) -> Tuple[bool, str]:
    """Read watchdog pause flag for this bot.

    Args:
        bot_tag: Bot identifier (e.g. ``"15M_STK"``).
        flags_path: Path to ``bot_pause_flags.json``.

    Returns:
        Tuple of ``(is_paused, reason)``.  Returns ``(False, "")`` if the
        file is missing, corrupt, or the bot has no entry (fail-open).
    """
    if not os.path.exists(flags_path):
        return False, ""

    try:
        with open(flags_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Could not read pause flags (%s): %s", flags_path, exc)
        return False, ""

    entry = data.get(bot_tag)
    if not entry or not isinstance(entry, dict):
        return False, ""

    if entry.get("paused", False):
        reason = entry.get("reason", "Watchdog paused (no reason given)")
        return True, reason

    return False, ""
