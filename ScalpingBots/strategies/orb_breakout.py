"""
Strategy 2: Opening Range Breakout (ORB)
==========================================
CONCEPT: The first 15 minutes of trading establish a range.
A breakout from this range with volume tends to continue in that direction.

WHY HIGH WIN RATE:
- Institutional order flow creates the opening range
- Breakout direction often aligns with the day's trend
- We filter for high-quality setups (gap direction, volume, narrow range)

ENTRY RULES:
- Wait for first 15 minutes to define Opening Range (OR)
- LONG: Price breaks above OR High with:
  - Volume > 1.5x average on breakout bar
  - Gap direction agrees (gap up = bullish bias)
  - OR range is not too wide (< 1.5% of price - avoids volatile mornings)
  - RSI not overbought (< 70) to avoid chasing
- SHORT: Price breaks below OR Low (mirror conditions)

EXIT RULES:
- Take Profit: 1.5x the opening range size (risk:reward ~1.5:1)
- Stop Loss: Opposite end of opening range (natural support/resistance)
- Trailing stop: After 1R profit, trail at 50% of range
- Time stop: Close by 12:00 ET (ORB momentum fades afternoon)
- EOD flatten

RISK MANAGEMENT:
- Risk = OR range (distance from entry to opposite range boundary)
- Position size: 1.5% equity at risk per trade
- Max 2 ORB trades per day (one per direction)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np


@dataclass
class ORBConfig:
    # Opening range
    or_minutes: int = 15  # First N minutes define the range
    max_or_range_pct: float = 1.5  # Skip if OR > 1.5% of price
    min_or_range_pct: float = 0.1  # Skip if OR too tight (< 0.1%)

    # Entry filters
    min_breakout_rvol: float = 1.2  # Volume on breakout bar vs avg
    max_rsi_long: float = 70.0      # Don't chase overbought
    min_rsi_short: float = 30.0     # Don't chase oversold
    gap_alignment: bool = True       # Require gap direction agrees with breakout
    require_close_outside: bool = True  # Bar must CLOSE outside range, not just wick

    # Exit - TUNED: More realistic TP, later time stop
    tp_range_mult: float = 1.0     # Reduced from 1.5x - take profits sooner
    sl_at_opposite_range: bool = True
    sl_buffer_pct: float = 0.15    # Slightly wider buffer
    trailing_trigger_r: float = 0.5  # Start trailing at 0.5R (earlier)
    trailing_pct: float = 0.4      # Tighter trail
    time_stop_hour: int = 14       # Extended from 12 to 14:00 ET - let winners run
    time_stop_minute: int = 0
    flatten_eod: bool = True
    max_hold_bars: int = 60        # Extended from 40

    # Risk
    risk_per_trade_pct: float = 1.5
    max_trades_per_day: int = 2
    max_daily_loss: float = 200.0
    max_notional: float = 3000.0


@dataclass
class ORBTrade:
    symbol: str
    entry_time: object
    entry_price: float
    qty: int
    side: str  # "LONG" or "SHORT"
    or_high: float = 0.0
    or_low: float = 0.0
    or_range: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    exit_time: object = None
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl: float = 0.0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    bars_held: int = 0
    gap_pct: float = 0.0


def backtest(df: pd.DataFrame, cfg: ORBConfig = None,
             start_cash: float = 10000.0, symbol: str = "?",
             verbose: bool = False) -> Dict:
    """
    Run ORB backtest on a single symbol.
    Expects df to have: open, high, low, close, volume, rsi, rvol, atr, gap_pct,
                        or_high, or_low, or_range columns.
    """
    cfg = cfg or ORBConfig()

    if df is None or df.empty or len(df) < 30:
        return {"trades": [], "summary": _empty_summary(symbol)}

    trades: List[ORBTrade] = []
    in_pos = False
    current_trade: Optional[ORBTrade] = None
    daily_pnl = 0.0
    current_date = None
    daily_trade_count = 0
    equity = start_cash
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')

    for i in range(1, len(df)):
        cur = df.iloc[i]
        prev = df.iloc[i - 1]
        et = cur.name.tz_convert("US/Eastern")
        bar_date = et.date()
        bar_minutes = et.hour * 60 + et.minute
        minutes_from_open = bar_minutes - (9 * 60 + 30)

        # Reset daily counters
        if bar_date != current_date:
            daily_pnl = 0.0
            daily_trade_count = 0
            current_date = bar_date

        if not in_pos:
            # Only look for entries AFTER opening range is established
            if minutes_from_open < cfg.or_minutes:
                continue

            # Daily limits
            if daily_pnl <= -cfg.max_daily_loss:
                continue
            if daily_trade_count >= cfg.max_trades_per_day:
                continue

            # Get opening range values
            or_high = float(cur.get("or_high", 0))
            or_low = float(cur.get("or_low", 0))
            or_range = or_high - or_low

            if or_high <= 0 or or_low <= 0 or or_range <= 0:
                continue

            # Filter: OR range not too wide or too narrow
            or_range_pct = (or_range / or_low) * 100
            if or_range_pct > cfg.max_or_range_pct or or_range_pct < cfg.min_or_range_pct:
                continue

            price = float(cur["close"]) if cfg.require_close_outside else float(cur["high"])
            price_low = float(cur["close"]) if cfg.require_close_outside else float(cur["low"])

            rvol = float(cur.get("rvol", 0))
            rsi = float(prev.get("rsi", 50))
            gap_pct = float(cur.get("gap_pct", 0))
            atr_val = float(prev.get("atr", 0))

            # LONG Breakout
            if price > or_high and rvol >= cfg.min_breakout_rvol and rsi < cfg.max_rsi_long:
                if not cfg.gap_alignment or gap_pct >= 0:
                    entry_px = float(cur["close"])
                    sl_px = or_low - (or_range * cfg.sl_buffer_pct)
                    tp_px = entry_px + (or_range * cfg.tp_range_mult)
                    risk = entry_px - sl_px

                    if risk <= 0:
                        continue

                    max_risk = equity * (cfg.risk_per_trade_pct / 100.0)
                    qty_risk = int(max_risk / risk)
                    qty_notional = int(cfg.max_notional / entry_px) if entry_px > 0 else 0
                    qty = max(1, min(qty_risk, qty_notional))

                    current_trade = ORBTrade(
                        symbol=symbol, entry_time=cur.name, entry_price=entry_px,
                        qty=qty, side="LONG", or_high=or_high, or_low=or_low,
                        or_range=or_range, sl_price=sl_px, tp_price=tp_px,
                        gap_pct=gap_pct,
                    )
                    in_pos = True
                    daily_trade_count += 1
                    highest_since_entry = entry_px
                    lowest_since_entry = entry_px
                    continue

            # SHORT Breakout
            if price_low < or_low and rvol >= cfg.min_breakout_rvol and rsi > cfg.min_rsi_short:
                if not cfg.gap_alignment or gap_pct <= 0:
                    entry_px = float(cur["close"])
                    sl_px = or_high + (or_range * cfg.sl_buffer_pct)
                    tp_px = entry_px - (or_range * cfg.tp_range_mult)
                    risk = sl_px - entry_px

                    if risk <= 0:
                        continue

                    max_risk = equity * (cfg.risk_per_trade_pct / 100.0)
                    qty_risk = int(max_risk / risk)
                    qty_notional = int(cfg.max_notional / entry_px) if entry_px > 0 else 0
                    qty = max(1, min(qty_risk, qty_notional))

                    current_trade = ORBTrade(
                        symbol=symbol, entry_time=cur.name, entry_price=entry_px,
                        qty=qty, side="SHORT", or_high=or_high, or_low=or_low,
                        or_range=or_range, sl_price=sl_px, tp_price=tp_px,
                        gap_pct=gap_pct,
                    )
                    in_pos = True
                    daily_trade_count += 1
                    highest_since_entry = entry_px
                    lowest_since_entry = entry_px
                    continue

        else:
            # Manage position
            current_trade.bars_held += 1
            highest_since_entry = max(highest_since_entry, float(cur["high"]))
            lowest_since_entry = min(lowest_since_entry, float(cur["low"]))

            exit_px = None
            reason = ""
            is_long = current_trade.side == "LONG"

            # MFE / MAE tracking
            if is_long:
                current_trade.max_favorable = max(
                    current_trade.max_favorable,
                    (cur["high"] - current_trade.entry_price) * current_trade.qty)
                current_trade.max_adverse = min(
                    current_trade.max_adverse,
                    (cur["low"] - current_trade.entry_price) * current_trade.qty)
            else:
                current_trade.max_favorable = max(
                    current_trade.max_favorable,
                    (current_trade.entry_price - cur["low"]) * current_trade.qty)
                current_trade.max_adverse = min(
                    current_trade.max_adverse,
                    (current_trade.entry_price - cur["high"]) * -current_trade.qty)

            # Priority 1: Stop Loss
            if is_long and cur["low"] <= current_trade.sl_price:
                exit_px = current_trade.sl_price
                reason = "SL"
            elif not is_long and cur["high"] >= current_trade.sl_price:
                exit_px = current_trade.sl_price
                reason = "SL"

            # Priority 2: Take Profit
            if exit_px is None:
                if is_long and cur["high"] >= current_trade.tp_price:
                    exit_px = current_trade.tp_price
                    reason = "TP"
                elif not is_long and cur["low"] <= current_trade.tp_price:
                    exit_px = current_trade.tp_price
                    reason = "TP"

            # Priority 3: Trailing stop
            if exit_px is None and cfg.trailing_trigger_r > 0:
                risk = abs(current_trade.entry_price - current_trade.sl_price)
                if is_long:
                    profit = highest_since_entry - current_trade.entry_price
                    if risk > 0 and profit >= risk * cfg.trailing_trigger_r:
                        trail_level = highest_since_entry - (current_trade.or_range * cfg.trailing_pct)
                        if cur["low"] <= trail_level and trail_level > current_trade.entry_price:
                            exit_px = trail_level
                            reason = "TRAIL"
                else:
                    profit = current_trade.entry_price - lowest_since_entry
                    if risk > 0 and profit >= risk * cfg.trailing_trigger_r:
                        trail_level = lowest_since_entry + (current_trade.or_range * cfg.trailing_pct)
                        if cur["high"] >= trail_level and trail_level < current_trade.entry_price:
                            exit_px = trail_level
                            reason = "TRAIL"

            # Priority 4: Time stop (ORB fades afternoon)
            if exit_px is None:
                if et.hour * 60 + et.minute >= cfg.time_stop_hour * 60 + cfg.time_stop_minute:
                    exit_px = float(cur["close"])
                    reason = "TIME"

            # Priority 5: Max hold
            if exit_px is None and current_trade.bars_held >= cfg.max_hold_bars:
                exit_px = float(cur["close"])
                reason = "MAX_HOLD"

            # Priority 6: EOD
            if exit_px is None and cfg.flatten_eod and et.hour * 60 + et.minute >= 15 * 60 + 50:
                exit_px = float(cur["close"])
                reason = "EOD"

            if exit_px is not None:
                if is_long:
                    pnl = (exit_px - current_trade.entry_price) * current_trade.qty
                else:
                    pnl = (current_trade.entry_price - exit_px) * current_trade.qty

                current_trade.exit_time = cur.name
                current_trade.exit_price = exit_px
                current_trade.exit_reason = reason
                current_trade.pnl = pnl
                trades.append(current_trade)

                daily_pnl += pnl
                equity += pnl
                in_pos = False
                current_trade = None

                if verbose:
                    sign = "+" if pnl >= 0 else ""
                    print(f"  [{symbol}] ORB {reason}: {sign}${pnl:.2f}")

    # Force close
    if in_pos and current_trade:
        last = df.iloc[-1]
        if current_trade.side == "LONG":
            pnl = (float(last["close"]) - current_trade.entry_price) * current_trade.qty
        else:
            pnl = (current_trade.entry_price - float(last["close"])) * current_trade.qty
        current_trade.exit_time = last.name
        current_trade.exit_price = float(last["close"])
        current_trade.exit_reason = "FORCE_CLOSE"
        current_trade.pnl = pnl
        trades.append(current_trade)
        equity += pnl

    summary = _compute_summary(trades, symbol, start_cash, equity)
    return {"trades": trades, "summary": summary}


def _compute_summary(trades: List[ORBTrade], symbol: str, start_cash: float, equity: float) -> Dict:
    if not trades:
        return _empty_summary(symbol)

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in trades)
    total_trades = len(trades)

    reason_counts = {}
    reason_pnl = {}
    for t in trades:
        reason_counts[t.exit_reason] = reason_counts.get(t.exit_reason, 0) + 1
        reason_pnl[t.exit_reason] = reason_pnl.get(t.exit_reason, 0) + t.pnl

    daily_pnl = {}
    for t in trades:
        d = t.exit_time.tz_convert("US/Eastern").date() if t.exit_time else None
        if d:
            daily_pnl[d] = daily_pnl.get(d, 0) + t.pnl

    positive_days = sum(1 for v in daily_pnl.values() if v > 0)
    negative_days = sum(1 for v in daily_pnl.values() if v < 0)

    running_equity = start_cash
    peak = start_cash
    max_dd = 0
    for t in trades:
        running_equity += t.pnl
        peak = max(peak, running_equity)
        dd = (running_equity - peak) / peak * 100 if peak > 0 else 0
        max_dd = min(max_dd, dd)

    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
    loss_total = sum(t.pnl for t in losses)
    profit_factor = abs(sum(t.pnl for t in wins) / loss_total) if loss_total != 0 else float('inf')

    # Long vs Short breakdown
    long_trades = [t for t in trades if t.side == "LONG"]
    short_trades = [t for t in trades if t.side == "SHORT"]

    return {
        "symbol": symbol,
        "total_trades": total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / total_trades * 100, 1),
        "net_pnl": round(total_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "final_equity": round(equity, 2),
        "return_pct": round((equity / start_cash - 1) * 100, 2),
        "positive_days": positive_days,
        "negative_days": negative_days,
        "total_days": len(daily_pnl),
        "daily_win_rate": round(positive_days / len(daily_pnl) * 100, 1) if daily_pnl else 0,
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "long_pnl": round(sum(t.pnl for t in long_trades), 2),
        "short_pnl": round(sum(t.pnl for t in short_trades), 2),
        "reason_counts": reason_counts,
        "reason_pnl": {k: round(v, 2) for k, v in reason_pnl.items()},
        "avg_bars_held": round(np.mean([t.bars_held for t in trades]), 1),
    }


def _empty_summary(symbol: str) -> Dict:
    return {
        "symbol": symbol,
        "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
        "net_pnl": 0, "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
        "max_drawdown_pct": 0, "final_equity": 0, "return_pct": 0,
        "positive_days": 0, "negative_days": 0, "total_days": 0,
        "daily_win_rate": 0, "long_trades": 0, "short_trades": 0,
        "long_pnl": 0, "short_pnl": 0, "reason_counts": {}, "reason_pnl": {},
        "avg_bars_held": 0,
    }
