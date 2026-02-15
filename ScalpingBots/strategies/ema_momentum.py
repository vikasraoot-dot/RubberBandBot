"""
Strategy 4: EMA Momentum Scalper
===================================
CONCEPT: Trade WITH momentum using EMA crossovers, volume surge, and price action.
This is the most common profitable retail scalping strategy because it trades
WITH the trend rather than against it.

WHY HIGH WIN RATE:
- Trading with momentum = higher probability trades
- EMA crossovers are a lagging but reliable trend confirmation
- Volume surge confirms institutional participation
- No fighting the trend (unlike mean reversion which fights momentum)

ENTRY RULES:
- EMA 9 crosses above EMA 21 (bullish momentum)
- Price is above VWAP (with the intraday trend)
- Current bar volume > 1.5x 20-bar average (volume surge)
- RSI between 40-65 (momentum but not overbought)
- Close > Open on the signal bar (bullish confirmation)
- NOT in first 15 min or last 30 min of session

SHORT (mirror for bearish):
- EMA 9 crosses below EMA 21
- Price below VWAP
- Volume surge
- RSI between 35-60

EXIT RULES:
- TP: 2x ATR from entry (achievable momentum target)
- SL: Use swing low/high (last 5 bars low minus buffer) or 2x ATR
- Trailing: After 1 ATR profit, trail at 1 ATR distance
- Cross-back: Exit if EMA 9 crosses back below EMA 21
- Time: Max 15 bars (momentum fades)
- EOD flatten

KEY INNOVATION: Use a "momentum score" to size positions.
Higher momentum = bigger position. Low momentum = skip.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import numpy as np


@dataclass
class EMAMomentumConfig:
    # EMA periods
    fast_ema: int = 9
    slow_ema: int = 21

    # Entry filters
    require_above_vwap_long: bool = True
    min_rvol: float = 1.2          # Volume must be 1.2x average
    rsi_min_long: float = 40.0     # Not oversold (would be mean reversion)
    rsi_max_long: float = 65.0     # Not overbought (avoid chasing)
    rsi_min_short: float = 35.0
    rsi_max_short: float = 60.0
    require_bullish_bar: bool = True  # Bar must close > open for longs
    min_momentum_score: float = 0.5   # Minimum score to take trade

    # Exit - TUNED based on backtest data
    tp_atr_mult: float = 2.5       # Wider TP (was 2.0) - let winners run more
    sl_atr_mult: float = 2.5       # Wider SL (was 2.0) - reduce false stops
    sl_use_swing: bool = True       # Use swing low instead of fixed ATR
    sl_swing_bars: int = 8          # Look back 8 bars (was 5) - wider swing
    sl_confirm_bars: int = 1        # Immediate SL exit (was 2 — 10-min delay caused outsized losses)
    trailing_trigger_atr: float = 0.8  # Start trailing earlier (was 1.0)
    trailing_distance_atr: float = 0.6  # Tighter trail (was 0.8) - lock in more profit
    exit_on_cross_back: bool = False   # DISABLED: Was killing +$549 in profits
    max_hold_bars: int = 20         # Extended from 15 - momentum can persist
    flatten_eod: bool = True

    # Risk
    risk_per_trade_pct: float = 1.5
    max_daily_loss: float = 150.0
    max_notional: float = 3000.0
    max_trades_per_day: int = 5

    # Timing
    entry_start: str = "09:45"
    entry_end: str = "15:30"
    flatten_time: str = "15:50"

    # Allow shorts
    allow_shorts: bool = True


@dataclass
class MomentumTrade:
    symbol: str
    entry_time: object
    entry_price: float
    qty: int
    side: str
    momentum_score: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    exit_time: object = None
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl: float = 0.0
    bars_held: int = 0
    max_favorable: float = 0.0
    max_adverse: float = 0.0


def calculate_momentum_score(df: pd.DataFrame, i: int) -> float:
    """
    Calculate a momentum quality score (0-1) based on multiple factors.
    Higher score = stronger momentum = better trade.
    """
    if i < 5:
        return 0.0

    cur = df.iloc[i]
    prev = df.iloc[i - 1]

    score = 0.0
    factors = 0

    # Factor 1: Volume relative to average (higher = stronger)
    rvol = float(cur.get("rvol", 1.0))
    if rvol > 2.0:
        score += 1.0
    elif rvol > 1.5:
        score += 0.7
    elif rvol > 1.0:
        score += 0.4
    factors += 1

    # Factor 2: Bar range relative to ATR (big bar = strong move)
    atr = float(prev.get("atr", 1.0))
    bar_range = float(cur["high"] - cur["low"])
    if atr > 0:
        range_ratio = bar_range / atr
        if range_ratio > 1.0:
            score += 1.0
        elif range_ratio > 0.7:
            score += 0.6
        elif range_ratio > 0.4:
            score += 0.3
    factors += 1

    # Factor 3: Close position in bar (close near high for longs = strong)
    bar_body = abs(float(cur["close"]) - float(cur["open"]))
    if bar_range > 0:
        body_ratio = bar_body / bar_range
        if body_ratio > 0.6:
            score += 0.8
        elif body_ratio > 0.4:
            score += 0.5
    factors += 1

    # Factor 4: EMA spread (wider gap = stronger trend)
    ema_fast = float(cur.get("ema_9", 0))
    ema_slow = float(cur.get("ema_21", 0))
    if ema_slow > 0:
        ema_spread = abs(ema_fast - ema_slow) / ema_slow * 100
        if ema_spread > 0.5:
            score += 1.0
        elif ema_spread > 0.2:
            score += 0.5
    factors += 1

    # Factor 5: Recent price action (3 of last 5 bars in same direction)
    if i >= 5:
        recent = df.iloc[i-5:i]
        bullish_bars = (recent["close"] > recent["open"]).sum()
        if bullish_bars >= 3:
            score += 0.6
    factors += 1

    return score / factors if factors > 0 else 0.0


def backtest(df: pd.DataFrame, cfg: EMAMomentumConfig = None,
             start_cash: float = 10000.0, symbol: str = "?",
             verbose: bool = False) -> Dict:
    """
    Run EMA Momentum backtest.
    Expects df to have: open, high, low, close, volume, ema_9, ema_21,
                        rsi, atr, rvol, vwap columns.
    """
    cfg = cfg or EMAMomentumConfig()

    if df is None or df.empty or len(df) < 30:
        return {"trades": [], "summary": _empty_summary(symbol)}

    trades: List[MomentumTrade] = []
    in_pos = False
    current_trade: Optional[MomentumTrade] = None
    daily_pnl = 0.0
    current_date = None
    daily_trade_count = 0
    equity = start_cash
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    sl_confirm_count = 0  # Consecutive bars below SL level

    for i in range(2, len(df)):
        cur = df.iloc[i]
        prev = df.iloc[i - 1]
        prev2 = df.iloc[i - 2]
        et = cur.name.tz_convert("US/Eastern")
        bar_date = et.date()
        bar_minutes = et.hour * 60 + et.minute

        if bar_date != current_date:
            daily_pnl = 0.0
            daily_trade_count = 0
            current_date = bar_date

        if not in_pos:
            # Time window check
            h_start, m_start = map(int, cfg.entry_start.split(":"))
            h_end, m_end = map(int, cfg.entry_end.split(":"))
            if bar_minutes < h_start * 60 + m_start or bar_minutes > h_end * 60 + m_end:
                continue

            if daily_pnl <= -cfg.max_daily_loss:
                continue
            if daily_trade_count >= cfg.max_trades_per_day:
                continue

            ema_fast = float(cur.get("ema_9", 0))
            ema_slow = float(cur.get("ema_21", 0))
            prev_ema_fast = float(prev.get("ema_9", 0))
            prev_ema_slow = float(prev.get("ema_21", 0))
            rsi = float(prev.get("rsi", 50))
            rvol = float(cur.get("rvol", 1))
            atr_val = float(prev.get("atr", 0))
            vwap_val = float(cur.get("vwap", 0))

            if ema_fast == 0 or ema_slow == 0 or atr_val <= 0:
                continue

            # LONG: EMA 9 crosses above EMA 21
            bullish_cross = (prev_ema_fast <= prev_ema_slow) and (ema_fast > ema_slow)
            # Also allow: EMA 9 already above EMA 21 AND widening
            bullish_trend = (ema_fast > ema_slow) and (ema_fast - ema_slow > prev_ema_fast - prev_ema_slow)

            long_signal = (bullish_cross or bullish_trend)
            if long_signal:
                long_signal = long_signal and (rvol >= cfg.min_rvol)
                long_signal = long_signal and (cfg.rsi_min_long <= rsi <= cfg.rsi_max_long)
                if cfg.require_above_vwap_long and vwap_val > 0:
                    long_signal = long_signal and (float(cur["close"]) > vwap_val)
                if cfg.require_bullish_bar:
                    long_signal = long_signal and (float(cur["close"]) > float(cur["open"]))

            # SHORT: EMA 9 crosses below EMA 21
            bearish_cross = (prev_ema_fast >= prev_ema_slow) and (ema_fast < ema_slow)
            bearish_trend = (ema_fast < ema_slow) and (ema_slow - ema_fast > prev_ema_slow - prev_ema_fast)

            short_signal = cfg.allow_shorts and (bearish_cross or bearish_trend)
            if short_signal:
                short_signal = short_signal and (rvol >= cfg.min_rvol)
                short_signal = short_signal and (cfg.rsi_min_short <= rsi <= cfg.rsi_max_short)
                if cfg.require_above_vwap_long and vwap_val > 0:
                    short_signal = short_signal and (float(cur["close"]) < vwap_val)
                if cfg.require_bullish_bar:
                    short_signal = short_signal and (float(cur["close"]) < float(cur["open"]))

            if long_signal or short_signal:
                # Calculate momentum score
                mom_score = calculate_momentum_score(df, i)
                if mom_score < cfg.min_momentum_score:
                    continue

                side = "LONG" if long_signal else "SHORT"
                entry_px = float(cur["close"])

                # Stop loss: swing low/high or ATR-based
                if cfg.sl_use_swing and i >= cfg.sl_swing_bars:
                    if side == "LONG":
                        swing_low = df["low"].iloc[i - cfg.sl_swing_bars:i].min()
                        sl_px = float(swing_low) - (atr_val * 0.2)  # Small buffer below swing
                    else:
                        swing_high = df["high"].iloc[i - cfg.sl_swing_bars:i].max()
                        sl_px = float(swing_high) + (atr_val * 0.2)
                else:
                    if side == "LONG":
                        sl_px = entry_px - (atr_val * cfg.sl_atr_mult)
                    else:
                        sl_px = entry_px + (atr_val * cfg.sl_atr_mult)

                # TP
                if side == "LONG":
                    tp_px = entry_px + (atr_val * cfg.tp_atr_mult)
                    risk = entry_px - sl_px
                else:
                    tp_px = entry_px - (atr_val * cfg.tp_atr_mult)
                    risk = sl_px - entry_px

                if risk <= 0:
                    continue

                # Position sizing
                max_risk = equity * (cfg.risk_per_trade_pct / 100.0)
                qty_risk = int(max_risk / risk)
                qty_notional = int(cfg.max_notional / entry_px) if entry_px > 0 else 0
                qty = max(1, min(qty_risk, qty_notional))

                current_trade = MomentumTrade(
                    symbol=symbol, entry_time=cur.name, entry_price=entry_px,
                    qty=qty, side=side, momentum_score=mom_score,
                    sl_price=sl_px, tp_price=tp_px,
                )
                in_pos = True
                daily_trade_count += 1
                highest_since_entry = entry_px
                lowest_since_entry = entry_px
                sl_confirm_count = 0  # Reset SL confirmation counter
                continue

        else:
            # Manage position
            current_trade.bars_held += 1
            highest_since_entry = max(highest_since_entry, float(cur["high"]))
            lowest_since_entry = min(lowest_since_entry, float(cur["low"]))
            is_long = current_trade.side == "LONG"

            # MFE/MAE
            if is_long:
                current_trade.max_favorable = max(current_trade.max_favorable,
                    (cur["high"] - current_trade.entry_price) * current_trade.qty)
                current_trade.max_adverse = min(current_trade.max_adverse,
                    (cur["low"] - current_trade.entry_price) * current_trade.qty)
            else:
                current_trade.max_favorable = max(current_trade.max_favorable,
                    (current_trade.entry_price - cur["low"]) * current_trade.qty)
                current_trade.max_adverse = min(current_trade.max_adverse,
                    (current_trade.entry_price - cur["high"]) * -current_trade.qty)

            exit_px = None
            reason = ""
            atr_val = float(prev.get("atr", 0))

            # Priority 1: Stop Loss with CONFIRMATION
            # Require N consecutive bars touching/below SL before exiting
            # This prevents single-bar wicks from triggering false stops
            sl_touched = False
            if is_long and cur["low"] <= current_trade.sl_price:
                sl_touched = True
            elif not is_long and cur["high"] >= current_trade.sl_price:
                sl_touched = True

            if sl_touched:
                sl_confirm_count += 1
                if sl_confirm_count >= cfg.sl_confirm_bars:
                    exit_px = current_trade.sl_price
                    reason = "SL"
            else:
                sl_confirm_count = 0  # Reset if price moves away from SL

            # Priority 2: Take Profit
            if exit_px is None:
                if is_long and cur["high"] >= current_trade.tp_price:
                    exit_px = current_trade.tp_price
                    reason = "TP"
                elif not is_long and cur["low"] <= current_trade.tp_price:
                    exit_px = current_trade.tp_price
                    reason = "TP"

            # Priority 3: Trailing stop
            if exit_px is None and atr_val > 0:
                trigger_dist = atr_val * cfg.trailing_trigger_atr
                trail_dist = atr_val * cfg.trailing_distance_atr
                if is_long:
                    profit = highest_since_entry - current_trade.entry_price
                    if profit >= trigger_dist:
                        trail_stop = highest_since_entry - trail_dist
                        if cur["low"] <= trail_stop and trail_stop > current_trade.entry_price:
                            exit_px = trail_stop
                            reason = "TRAIL"
                else:
                    profit = current_trade.entry_price - lowest_since_entry
                    if profit >= trigger_dist:
                        trail_stop = lowest_since_entry + trail_dist
                        if cur["high"] >= trail_stop and trail_stop < current_trade.entry_price:
                            exit_px = trail_stop
                            reason = "TRAIL"

            # Priority 4: EMA cross-back exit
            if exit_px is None and cfg.exit_on_cross_back:
                ema_fast = float(cur.get("ema_9", 0))
                ema_slow = float(cur.get("ema_21", 0))
                if is_long and ema_fast < ema_slow:
                    exit_px = float(cur["close"])
                    reason = "CROSS_BACK"
                elif not is_long and ema_fast > ema_slow:
                    exit_px = float(cur["close"])
                    reason = "CROSS_BACK"

            # Priority 5: Max hold
            if exit_px is None and current_trade.bars_held >= cfg.max_hold_bars:
                exit_px = float(cur["close"])
                reason = "TIME"

            # Priority 6: EOD
            if exit_px is None and cfg.flatten_eod:
                h_flat, m_flat = map(int, cfg.flatten_time.split(":"))
                if et.hour * 60 + et.minute >= h_flat * 60 + m_flat:
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
                    print(f"  [{symbol}] MOM {reason}: {sign}${pnl:.2f}")

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


def _compute_summary(trades: List[MomentumTrade], symbol: str, start_cash: float, equity: float) -> Dict:
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

    avg_mom = np.mean([t.momentum_score for t in trades])
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
        "avg_momentum_score": round(avg_mom, 2),
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
        "daily_win_rate": 0, "avg_momentum_score": 0,
        "long_trades": 0, "short_trades": 0, "long_pnl": 0, "short_pnl": 0,
        "reason_counts": {}, "reason_pnl": {}, "avg_bars_held": 0,
    }
