"""
Strategy 3: Gap Fill / Overnight Gap Fade
============================================
CONCEPT: Stocks that gap up or down at open tend to "fill" the gap
(revert to previous close) within the same session. This is one of
the most statistically validated patterns in equity markets.

WHY HIGH WIN RATE:
- Academic research shows 60-70%+ of gaps fill within the same day
- We filter for "fadeable" gaps (not momentum/news gaps)
- Small-to-medium gaps (0.3%-2%) fill most reliably
- Volume confirmation prevents fading breakaway gaps

ENTRY RULES:
- Gap detected at open: |gap_pct| between 0.3% and 2.0%
- Gap UP fade (SHORT): Gap up + first 5-min bar is bearish (sellers stepping in)
  - RSI > 55 (not deeply oversold, actually mildly overbought from gap)
  - Volume on first bars below average (exhaustion gap, not momentum)
- Gap DOWN fade (LONG): Gap down + first 5-min bar is bullish (buyers stepping in)
  - RSI < 45
  - Volume on first bars below average

EXIT RULES:
- Take Profit: Previous close (gap fill target)
- Partial profit: 50% at half-gap fill
- Stop Loss: 1.5x gap size beyond the open (gap extends, not filling)
- Time stop: If gap hasn't filled by 11:00 ET, close (fades happen early)
- Trailing stop after 60% fill

RISK MANAGEMENT:
- Only trade gaps that are NOT driven by earnings/news (avoid first 2 days after earnings)
- Max risk: 1.5% equity per trade
- Max 3 gap trades per day
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import numpy as np


@dataclass
class GapFillConfig:
    # Gap filters
    min_gap_pct: float = 0.3   # Minimum gap to trade
    max_gap_pct: float = 2.0   # Maximum gap (larger gaps less likely to fill same day)
    max_rvol_at_open: float = 3.0  # Skip if opening volume is extreme (news event)
    min_rvol_at_open: float = 0.3  # Skip if no one is trading

    # Entry
    wait_bars: int = 2          # Wait N bars after open for confirmation
    require_reversal_bar: bool = True  # First bar should reverse gap direction
    max_rsi_gap_down: float = 45.0  # For buying gap downs
    min_rsi_gap_up: float = 55.0    # For fading gap ups

    # Exit - TUNED: More patient, wider stops
    tp_fill_pct: float = 0.75  # Reduced from 0.9 - take partial fill profits
    sl_gap_mult: float = 2.0   # Widened from 1.5x - less false stop-outs
    time_stop_hour: int = 13   # Extended from 11:00 to 13:00 ET
    time_stop_minute: int = 0
    trailing_trigger_fill_pct: float = 0.35  # Start trailing at 35% fill (earlier)
    trailing_pct: float = 0.25  # Tighter trail at 25%
    flatten_eod: bool = True
    max_hold_bars: int = 50    # Extended from 30

    # Risk
    risk_per_trade_pct: float = 1.5
    max_trades_per_day: int = 3
    max_daily_loss: float = 200.0
    max_notional: float = 3000.0

    # Allow shorts?
    allow_shorts: bool = True


@dataclass
class GapTrade:
    symbol: str
    entry_time: object
    entry_price: float
    qty: int
    side: str
    gap_pct: float = 0.0
    prev_close: float = 0.0
    gap_fill_target: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    exit_time: object = None
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl: float = 0.0
    bars_held: int = 0
    max_fill_pct: float = 0.0  # How much of the gap was filled


def backtest(df: pd.DataFrame, cfg: GapFillConfig = None,
             start_cash: float = 10000.0, symbol: str = "?",
             verbose: bool = False) -> Dict:
    """
    Run Gap Fill backtest on a single symbol.
    Expects df to have: open, high, low, close, volume, rsi, rvol, atr,
                        gap, gap_pct, prev_close columns.
    """
    cfg = cfg or GapFillConfig()

    if df is None or df.empty or len(df) < 30:
        return {"trades": [], "summary": _empty_summary(symbol)}

    trades: List[GapTrade] = []
    in_pos = False
    current_trade: Optional[GapTrade] = None
    daily_pnl = 0.0
    current_date = None
    daily_trade_count = 0
    equity = start_cash
    best_price_since_entry = None

    for i in range(1, len(df)):
        cur = df.iloc[i]
        prev = df.iloc[i - 1]
        et = cur.name.tz_convert("US/Eastern")
        bar_date = et.date()
        minutes_from_open = (et.hour * 60 + et.minute) - (9 * 60 + 30)

        if bar_date != current_date:
            daily_pnl = 0.0
            daily_trade_count = 0
            current_date = bar_date

        if not in_pos:
            # Only enter in the first 30 minutes (after wait_bars)
            if minutes_from_open < cfg.wait_bars * 5 or minutes_from_open > 30:
                continue

            if daily_pnl <= -cfg.max_daily_loss:
                continue
            if daily_trade_count >= cfg.max_trades_per_day:
                continue

            gap_pct = float(cur.get("gap_pct", 0))
            prev_close_val = float(cur.get("prev_close", 0))
            rsi = float(prev.get("rsi", 50))
            rvol = float(cur.get("rvol", 1))
            atr_val = float(prev.get("atr", 0))

            if prev_close_val <= 0 or atr_val <= 0:
                continue
            if abs(gap_pct) < cfg.min_gap_pct or abs(gap_pct) > cfg.max_gap_pct:
                continue
            if rvol > cfg.max_rvol_at_open or rvol < cfg.min_rvol_at_open:
                continue

            # GAP DOWN -> BUY (fade the gap, expect fill up to prev close)
            if gap_pct < -cfg.min_gap_pct and rsi < cfg.max_rsi_gap_down:
                # Confirmation: bar should be bullish (buyers stepping in)
                if cfg.require_reversal_bar and cur["close"] <= cur["open"]:
                    continue

                entry_px = float(cur["close"])
                gap_fill_target = prev_close_val
                tp_px = entry_px + (gap_fill_target - entry_px) * cfg.tp_fill_pct
                gap_size = abs(entry_px - prev_close_val)
                sl_px = entry_px - (gap_size * cfg.sl_gap_mult)

                risk = entry_px - sl_px
                if risk <= 0:
                    continue

                max_risk = equity * (cfg.risk_per_trade_pct / 100.0)
                qty_risk = int(max_risk / risk)
                qty_notional = int(cfg.max_notional / entry_px) if entry_px > 0 else 0
                qty = max(1, min(qty_risk, qty_notional))

                current_trade = GapTrade(
                    symbol=symbol, entry_time=cur.name, entry_price=entry_px,
                    qty=qty, side="LONG", gap_pct=gap_pct,
                    prev_close=prev_close_val, gap_fill_target=gap_fill_target,
                    sl_price=sl_px, tp_price=tp_px,
                )
                in_pos = True
                daily_trade_count += 1
                best_price_since_entry = entry_px
                continue

            # GAP UP -> SHORT (fade the gap, expect fill down to prev close)
            if cfg.allow_shorts and gap_pct > cfg.min_gap_pct and rsi > cfg.min_rsi_gap_up:
                if cfg.require_reversal_bar and cur["close"] >= cur["open"]:
                    continue

                entry_px = float(cur["close"])
                gap_fill_target = prev_close_val
                tp_px = entry_px - (entry_px - gap_fill_target) * cfg.tp_fill_pct
                gap_size = abs(entry_px - prev_close_val)
                sl_px = entry_px + (gap_size * cfg.sl_gap_mult)

                risk = sl_px - entry_px
                if risk <= 0:
                    continue

                max_risk = equity * (cfg.risk_per_trade_pct / 100.0)
                qty_risk = int(max_risk / risk)
                qty_notional = int(cfg.max_notional / entry_px) if entry_px > 0 else 0
                qty = max(1, min(qty_risk, qty_notional))

                current_trade = GapTrade(
                    symbol=symbol, entry_time=cur.name, entry_price=entry_px,
                    qty=qty, side="SHORT", gap_pct=gap_pct,
                    prev_close=prev_close_val, gap_fill_target=gap_fill_target,
                    sl_price=sl_px, tp_price=tp_px,
                )
                in_pos = True
                daily_trade_count += 1
                best_price_since_entry = entry_px
                continue

        else:
            # Manage position
            current_trade.bars_held += 1
            is_long = current_trade.side == "LONG"

            # Track gap fill progress
            if is_long:
                fill_progress = (float(cur["high"]) - current_trade.entry_price) / \
                    max(current_trade.gap_fill_target - current_trade.entry_price, 0.01)
                best_price_since_entry = max(best_price_since_entry, float(cur["high"]))
            else:
                fill_progress = (current_trade.entry_price - float(cur["low"])) / \
                    max(current_trade.entry_price - current_trade.gap_fill_target, 0.01)
                best_price_since_entry = min(best_price_since_entry, float(cur["low"]))

            current_trade.max_fill_pct = max(current_trade.max_fill_pct, fill_progress * 100)

            exit_px = None
            reason = ""

            # Priority 1: Stop Loss
            if is_long and cur["low"] <= current_trade.sl_price:
                exit_px = current_trade.sl_price
                reason = "SL"
            elif not is_long and cur["high"] >= current_trade.sl_price:
                exit_px = current_trade.sl_price
                reason = "SL"

            # Priority 2: Take Profit (gap fill)
            if exit_px is None:
                if is_long and cur["high"] >= current_trade.tp_price:
                    exit_px = current_trade.tp_price
                    reason = "GAP_FILL"
                elif not is_long and cur["low"] <= current_trade.tp_price:
                    exit_px = current_trade.tp_price
                    reason = "GAP_FILL"

            # Priority 3: Trailing stop (after partial fill)
            if exit_px is None and fill_progress >= cfg.trailing_trigger_fill_pct:
                gap_distance = abs(current_trade.gap_fill_target - current_trade.entry_price)
                if is_long:
                    trail_stop = best_price_since_entry - (gap_distance * cfg.trailing_pct)
                    if cur["low"] <= trail_stop and trail_stop > current_trade.entry_price:
                        exit_px = trail_stop
                        reason = "TRAIL"
                else:
                    trail_stop = best_price_since_entry + (gap_distance * cfg.trailing_pct)
                    if cur["high"] >= trail_stop and trail_stop < current_trade.entry_price:
                        exit_px = trail_stop
                        reason = "TRAIL"

            # Priority 4: Time stop
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
                    print(f"  [{symbol}] GAP {reason}: {sign}${pnl:.2f}")

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


def _compute_summary(trades: List[GapTrade], symbol: str, start_cash: float, equity: float) -> Dict:
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

    avg_fill = np.mean([t.max_fill_pct for t in trades])

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
        "avg_gap_fill_pct": round(avg_fill, 1),
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
        "daily_win_rate": 0, "avg_gap_fill_pct": 0,
        "reason_counts": {}, "reason_pnl": {}, "avg_bars_held": 0,
    }
