"""
Strategy 1: VWAP Mean Reversion Scalper
=========================================
CONCEPT: Price tends to revert to VWAP. When price pulls back to VWAP
(or VWAP bands) with volume confirmation, enter for a bounce.

WHY HIGH WIN RATE:
- VWAP is the institutional benchmark price
- Mean reversion around VWAP is one of the most statistically robust patterns
- We ONLY trade when conditions stack (trend + volume + price location)

ENTRY RULES:
- Price touches or crosses below VWAP lower band (1 std)
- RSI < 35 (oversold on timeframe)
- Relative volume > 1.0 (above average activity)
- ADX > 20 (trending, not dead sideways)
- Previous bar was bearish (confirms pullback)
- Current bar shows buying (close > open OR close > VWAP)

EXIT RULES:
- Take Profit: Price reaches VWAP (mean reversion target)
- Extended TP: Price reaches VWAP upper band (momentum continuation)
- Stop Loss: Price falls below VWAP lower band 2 (2 std) OR 1.5x ATR
- Time stop: 20 bars max hold (avoids overnight risk)
- EOD flatten: Close all before market close

RISK MANAGEMENT:
- Max 2% of capital per trade
- Max 3 concurrent positions
- Daily loss limit: Stop after -$200 (configurable)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class VWAPConfig:
    # Entry - TUNED: Require deeper dip for higher quality entries
    rsi_oversold: float = 30.0       # Tightened from 35 - only deep oversold
    min_rvol: float = 0.8            # Lowered - allow normal volume too
    min_adx: float = 15.0            # Lowered - work in less trendy markets too
    vwap_band_entry: int = 1         # 1 = touch lower1, 2 = touch lower2
    require_bearish_prev: bool = False  # Removed - was filtering out good setups

    # Exit - TUNED: Wider SL, target VWAP upper for better R:R
    tp_target: str = "vwap_upper1"   # Changed: target upper band for bigger wins
    sl_atr_mult: float = 2.5         # Widened from 1.5 - stop getting stopped out
    max_hold_bars: int = 30          # Extended from 20 - give trades more time
    flatten_eod: bool = True
    trailing_stop_trigger_r: float = 0.5   # Start trailing earlier (at 0.5R)
    trailing_stop_pct: float = 0.4         # Tighter trail (40%)

    # Risk
    risk_per_trade_pct: float = 1.5  # Reduced from 2% - preserve capital
    max_concurrent: int = 3
    max_daily_loss: float = 150.0    # Tighter daily loss limit
    max_notional: float = 2000.0

    # Timing
    entry_start: str = "09:45"  # Skip first 15 min
    entry_end: str = "15:30"    # No entries in last 30 min
    flatten_time: str = "15:50"


@dataclass
class Trade:
    symbol: str
    entry_time: object
    entry_price: float
    qty: int
    side: str = "LONG"
    sl_price: float = 0.0
    tp_price: float = 0.0
    exit_time: object = None
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl: float = 0.0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    entry_rsi: float = 0.0
    entry_rvol: float = 0.0
    entry_adx: float = 0.0
    bars_held: int = 0


def generate_signals(df: pd.DataFrame, cfg: VWAPConfig) -> pd.DataFrame:
    """
    Attach entry/exit signals to DataFrame.
    Expects df to already have: vwap, vwap_lower1, vwap_upper1, rsi, atr, rvol, adx columns.
    """
    df = df.copy()

    # Entry conditions (all must be True)
    c_price_below_vwap = df["close"] <= df["vwap_lower1"]
    c_rsi_oversold = df["rsi"] < cfg.rsi_oversold
    c_volume = df["rvol"] >= cfg.min_rvol
    c_adx = df["adx"] >= cfg.min_adx

    # Previous bar was bearish (pullback confirmation)
    prev_bearish = df["close"].shift(1) < df["open"].shift(1)

    # Current bar shows buying pressure
    buying_pressure = (df["close"] > df["open"]) | (df["close"] > df["vwap"])

    if cfg.require_bearish_prev:
        df["long_signal"] = (c_price_below_vwap & c_rsi_oversold & c_volume &
                             c_adx & prev_bearish)
    else:
        df["long_signal"] = c_price_below_vwap & c_rsi_oversold & c_volume & c_adx

    # Time window filter
    eastern = df.index.tz_convert("US/Eastern")
    h_start, m_start = map(int, cfg.entry_start.split(":"))
    h_end, m_end = map(int, cfg.entry_end.split(":"))
    time_ok = (eastern.hour * 60 + eastern.minute >= h_start * 60 + m_start) & \
              (eastern.hour * 60 + eastern.minute <= h_end * 60 + m_end)
    df["long_signal"] = df["long_signal"] & time_ok

    return df


def backtest(df: pd.DataFrame, cfg: VWAPConfig = None,
             start_cash: float = 10000.0, symbol: str = "?",
             verbose: bool = False) -> Dict:
    """
    Run backtest for VWAP Bounce strategy on a single symbol.

    Returns dict with trades, win_rate, net_pnl, etc.
    """
    cfg = cfg or VWAPConfig()

    if df is None or df.empty or len(df) < 30:
        return {"trades": [], "summary": _empty_summary(symbol)}

    df = generate_signals(df, cfg)

    trades: List[Trade] = []
    in_pos = False
    current_trade: Optional[Trade] = None
    daily_pnl = 0.0
    current_date = None
    equity = start_cash

    for i in range(1, len(df)):
        cur = df.iloc[i]
        prev = df.iloc[i - 1]
        bar_date = cur.name.tz_convert("US/Eastern").date()

        # Reset daily P&L
        if bar_date != current_date:
            daily_pnl = 0.0
            current_date = bar_date

        if not in_pos:
            # Check daily loss limit
            if daily_pnl <= -cfg.max_daily_loss:
                continue

            if prev.get("long_signal", False):
                entry_px = float(cur["open"])
                if entry_px <= 0:
                    continue

                # Position sizing
                atr_val = float(prev.get("atr", 0))
                if atr_val <= 0:
                    continue

                sl_px = entry_px - (atr_val * cfg.sl_atr_mult)
                risk_per_share = entry_px - sl_px
                if risk_per_share <= 0:
                    continue

                # Risk-based sizing: risk X% of equity per trade
                max_risk = equity * (cfg.risk_per_trade_pct / 100.0)
                qty_risk = int(max_risk / risk_per_share)

                # Notional cap
                qty_notional = int(cfg.max_notional / entry_px) if entry_px > 0 else 0
                qty = max(1, min(qty_risk, qty_notional))

                # TP target
                if cfg.tp_target == "vwap":
                    tp_px = float(prev.get("vwap", entry_px * 1.01))
                elif cfg.tp_target == "vwap_upper1":
                    tp_px = float(prev.get("vwap_upper1", entry_px * 1.02))
                else:
                    tp_px = entry_px + (atr_val * 2.0)

                # Ensure TP is above entry
                if tp_px <= entry_px:
                    tp_px = entry_px + atr_val * 0.5

                current_trade = Trade(
                    symbol=symbol,
                    entry_time=cur.name,
                    entry_price=entry_px,
                    qty=qty,
                    sl_price=sl_px,
                    tp_price=tp_px,
                    entry_rsi=float(prev.get("rsi", 0)),
                    entry_rvol=float(prev.get("rvol", 0)),
                    entry_adx=float(prev.get("adx", 0)),
                )
                in_pos = True
                continue

        else:
            # Manage position
            current_trade.bars_held += 1

            # Track MFE/MAE
            current_trade.max_favorable = max(
                current_trade.max_favorable,
                (cur["high"] - current_trade.entry_price) * current_trade.qty
            )
            current_trade.max_adverse = min(
                current_trade.max_adverse,
                (cur["low"] - current_trade.entry_price) * current_trade.qty
            )

            exit_px = None
            reason = ""

            # Priority 1: Stop Loss (check low first - worst case)
            if cur["low"] <= current_trade.sl_price:
                exit_px = current_trade.sl_price
                reason = "SL"

            # Priority 2: Take Profit
            elif cur["high"] >= current_trade.tp_price:
                exit_px = current_trade.tp_price
                reason = "TP"

            # Priority 3: Trailing stop (if in profit)
            elif cfg.trailing_stop_trigger_r > 0:
                risk = current_trade.entry_price - current_trade.sl_price
                profit = cur["high"] - current_trade.entry_price
                if risk > 0 and profit >= risk * cfg.trailing_stop_trigger_r:
                    trail_stop = cur["high"] - (profit * cfg.trailing_stop_pct)
                    if cur["low"] <= trail_stop:
                        exit_px = max(trail_stop, current_trade.entry_price)  # At least breakeven
                        reason = "TRAIL"

            # Priority 4: Max hold time
            if exit_px is None and current_trade.bars_held >= cfg.max_hold_bars:
                exit_px = float(cur["close"])
                reason = "TIME"

            # Priority 5: EOD flatten
            if exit_px is None and cfg.flatten_eod:
                et = cur.name.tz_convert("US/Eastern")
                h_flat, m_flat = map(int, cfg.flatten_time.split(":"))
                if et.hour * 60 + et.minute >= h_flat * 60 + m_flat:
                    exit_px = float(cur["close"])
                    reason = "EOD"

            if exit_px is not None:
                pnl = (exit_px - current_trade.entry_price) * current_trade.qty
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
                    print(f"  [{symbol}] {reason}: {sign}${pnl:.2f} (equity=${equity:.2f})")

    # Force close if still in position
    if in_pos and current_trade:
        last = df.iloc[-1]
        pnl = (float(last["close"]) - current_trade.entry_price) * current_trade.qty
        current_trade.exit_time = last.name
        current_trade.exit_price = float(last["close"])
        current_trade.exit_reason = "FORCE_CLOSE"
        current_trade.pnl = pnl
        trades.append(current_trade)
        equity += pnl

    summary = _compute_summary(trades, symbol, start_cash, equity)
    return {"trades": trades, "summary": summary}


def _compute_summary(trades: List[Trade], symbol: str, start_cash: float, equity: float) -> Dict:
    if not trades:
        return _empty_summary(symbol)

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in trades)
    total_trades = len(trades)

    # Exit reason breakdown
    reason_counts = {}
    reason_pnl = {}
    for t in trades:
        reason_counts[t.exit_reason] = reason_counts.get(t.exit_reason, 0) + 1
        reason_pnl[t.exit_reason] = reason_pnl.get(t.exit_reason, 0) + t.pnl

    # Daily P&L for consistency check
    daily_pnl = {}
    for t in trades:
        d = t.exit_time.tz_convert("US/Eastern").date() if t.exit_time else None
        if d:
            daily_pnl[d] = daily_pnl.get(d, 0) + t.pnl

    positive_days = sum(1 for v in daily_pnl.values() if v > 0)
    negative_days = sum(1 for v in daily_pnl.values() if v < 0)
    total_days = len(daily_pnl)

    # Max drawdown
    running_equity = start_cash
    peak = start_cash
    max_dd = 0
    for t in trades:
        running_equity += t.pnl
        peak = max(peak, running_equity)
        dd = (running_equity - peak) / peak * 100
        max_dd = min(max_dd, dd)

    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
    profit_factor = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else float('inf')

    return {
        "symbol": symbol,
        "total_trades": total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / total_trades * 100, 1) if total_trades else 0,
        "net_pnl": round(total_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "final_equity": round(equity, 2),
        "return_pct": round((equity / start_cash - 1) * 100, 2),
        "positive_days": positive_days,
        "negative_days": negative_days,
        "total_days": total_days,
        "daily_win_rate": round(positive_days / total_days * 100, 1) if total_days else 0,
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
        "daily_win_rate": 0, "reason_counts": {}, "reason_pnl": {},
        "avg_bars_held": 0,
    }
