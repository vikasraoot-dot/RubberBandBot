"""
Trade simulation engine (daily bars, long only).

Execution model
---------------
* Signal evaluated at the close of session ``s``; entry at the **open of s+1**.
* Initial stop is fixed at entry.  If the entry open is already at/below the
  stop (gap through the stop) the trade is skipped -- this is knowable at the
  open and is how a live stop-entry order would behave.
* Intraday stop: if ``low < stop`` the position exits at the stop, or at the
  open if the session gapped below the stop.
* Close-based exit signals (e.g. close < EMA20) are evaluated at the close and
  filled at the **next open**.
* A trade still open on the last session of the evaluation window is closed at
  that session's close (``period_end``) so development trades never use
  out-of-sample prices.
* Costs: ``cost_bps`` per side, applied to both fills.

Trade-quality metrics
---------------------
* MFE / MAE: best high / worst low while the position was held (intraday),
  relative to the entry price.  On a stop day only the open and the stop price
  are counted because the intraday order of high and low is unknown.
* Peak unrealized return: best *close* while held (what a close-based trader
  could actually have locked in).  ``giveback_pp`` = peak - realized (gross),
  measured on trades whose peak close was above entry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config as C
from .features import EXIT_COLUMNS
from .setups import EntrySpec, signals


@dataclass(frozen=True)
class ExitSpec:
    """Exit rule + initial stop."""

    rule: str = "B_close<EMA20"
    stop: str = "atr1.5"                 # none|atr1.0|atr1.5|atr2.0|signal_low|swing_low
    horizon: Optional[int] = None        # fixed-horizon exit (sessions) for T-rules

    def label(self) -> str:
        """Compact label."""
        h = f"|h={self.horizon}" if self.horizon else ""
        return f"{self.rule}|stop={self.stop}{h}"


class StockData:
    """Numpy views of one stock's features plus a per-spec signal cache."""

    def __init__(self, ticker: str, feat: pd.DataFrame):
        self.ticker = ticker
        self.feat = feat
        self.dates = feat.index
        self.o = feat["open"].to_numpy(float)
        self.h = feat["high"].to_numpy(float)
        self.l = feat["low"].to_numpy(float)
        self.c = feat["close"].to_numpy(float)
        self.atr = feat["atr14"].to_numpy(float)
        self.exit_arrays = {k: feat[v].to_numpy(bool) for k, v in EXIT_COLUMNS.items()}
        self._sig_cache: Dict[EntrySpec, np.ndarray] = {}

    def signal_mask(self, spec: EntrySpec) -> np.ndarray:
        """Boolean numpy mask of entry signals for ``spec`` (cached)."""
        if spec not in self._sig_cache:
            self._sig_cache[spec] = signals(self.feat, spec).to_numpy(bool)
        return self._sig_cache[spec]

    def index_range(self, start: str, end: str) -> Tuple[int, int]:
        """First/last positional index of sessions inside [start, end]."""
        s = int(self.dates.searchsorted(pd.Timestamp(start), side="left"))
        e = int(self.dates.searchsorted(pd.Timestamp(end), side="right")) - 1
        return s, e


def _initial_stop(sd: StockData, s: int, entry: float, stop: str) -> float:
    if stop == "none":
        return -np.inf
    if stop.startswith("atr"):
        return entry - float(stop[3:]) * sd.atr[s]
    if stop == "signal_low":
        return sd.l[s]
    if stop == "swing_low":
        return float(np.min(sd.l[max(0, s - C.PULLBACK_WINDOW + 1):s + 1]))
    raise KeyError(stop)


def simulate_one(sd: StockData, s: int, xs: ExitSpec, end_i: int, cost: float) -> Optional[Dict[str, object]]:
    """
    Simulate one trade for a signal at index ``s``.

    Args:
        sd: stock arrays.
        s: signal index (close of session s).
        xs: exit specification.
        end_i: last index of the evaluation window (forced close at its close).
        cost: one-way cost as a fraction (5 bps = 0.0005).

    Returns:
        Trade dict, or None if there is no next session in the window or the
        open gapped through the stop.
    """
    e = s + 1
    if e > end_i:
        return None
    entry = sd.o[e]
    stop = _initial_stop(sd, s, entry, xs.stop)
    if not np.isfinite(entry) or entry <= stop:
        return None
    init_stop = stop
    risk_px = (entry - stop) if np.isfinite(stop) else C.BASELINE_ATR_STOP * sd.atr[s]
    exit_arr = sd.exit_arrays[xs.rule]
    trailing = xs.rule == "F_chandelier"
    horizon = xs.horizon
    hi_since = entry
    mfe_hi, mae_lo, peak_c = entry, entry, -np.inf
    j = e
    while True:
        if sd.l[j] < stop:
            gap = j > e and sd.o[j] < stop
            px = sd.o[j] if gap else stop
            mfe_hi = max(mfe_hi, sd.o[j])
            mae_lo = min(mae_lo, px)
            exit_i, exit_type = j, ("open" if gap else "intraday")
            reason = "trail_stop" if (trailing and stop > init_stop) else "stop"
            break
        mfe_hi = max(mfe_hi, sd.h[j])
        mae_lo = min(mae_lo, sd.l[j])
        peak_c = max(peak_c, sd.c[j])
        if j >= end_i:
            px, exit_i, exit_type, reason = sd.c[j], j, "close", "period_end"
            break
        if (horizon is not None and j - e + 1 >= horizon) or exit_arr[j]:
            px, exit_i, exit_type = sd.o[j + 1], j + 1, "open"
            reason = "time" if (horizon is not None and j - e + 1 >= horizon) else "signal"
            mfe_hi = max(mfe_hi, px)
            mae_lo = min(mae_lo, px)
            break
        if trailing:
            hi_since = max(hi_since, sd.h[j])
            stop = max(stop, hi_since - C.CHANDELIER_ATR * sd.atr[j])
        j += 1

    gross = px / entry - 1.0
    net = (px * (1 - cost)) / (entry * (1 + cost)) - 1.0
    peak_ret = (peak_c / entry - 1.0) if np.isfinite(peak_c) else np.nan
    risk_pct = risk_px / entry
    return {
        "ticker": sd.ticker,
        "signal_i": s,
        "entry_i": e,
        "exit_i": exit_i,
        "signal_date": sd.dates[s],
        "entry_date": sd.dates[e],
        "exit_date": sd.dates[exit_i],
        "entry_px": entry,
        "stop_px": init_stop if np.isfinite(init_stop) else np.nan,
        "exit_px": px,
        "exit_type": exit_type,
        "exit_reason": reason,
        "ret": net,
        "ret_gross": gross,
        "risk_pct": risk_pct,
        "R": net / risk_pct if risk_pct > 0 else np.nan,
        "hold_days": exit_i - e,
        "mfe": mfe_hi / entry - 1.0,
        "mae": mae_lo / entry - 1.0,
        "peak_close_ret": peak_ret,
        "giveback_pp": (peak_ret - gross) if (np.isfinite(peak_ret) and peak_ret > 0) else np.nan,
    }


def gate_mask(sd: StockData, gate: Optional[pd.Series]) -> np.ndarray:
    """Align a market-gate Series (on the SPY calendar) to this stock's sessions."""
    if gate is None:
        return np.ones(len(sd.dates), dtype=bool)
    return gate.reindex(sd.dates).fillna(False).to_numpy(bool)


def run_trades(stocks: Dict[str, StockData], spec: EntrySpec, xs: ExitSpec, start: str, end: str,
               gate: Optional[pd.Series] = None, cost_bps: float = C.COST_BPS_PER_SIDE,
               sequential: bool = True, extra_mask: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
    """
    Run one strategy across the universe.

    Args:
        stocks: ticker -> StockData.
        spec: entry rule.
        xs: exit rule.
        start/end: signal-date window (inclusive); trades are force-closed at ``end``.
        gate: market gate (True = new longs allowed) indexed by session.
        cost_bps: per-side cost.
        sequential: True = one position per stock at a time (trade-level study);
            False = simulate every signal independently (input to the portfolio sim).
        extra_mask: optional ticker -> boolean mask ANDed with the signals.

    Returns:
        DataFrame of trades.
    """
    cost = cost_bps / 1e4
    rows: List[Dict[str, object]] = []
    for tkr, sd in stocks.items():
        s0, e0 = sd.index_range(start, end)
        if e0 <= s0:
            continue
        mask = sd.signal_mask(spec) & gate_mask(sd, gate)
        if extra_mask is not None:
            mask = mask & extra_mask[tkr]
        idx = np.flatnonzero(mask[s0:e0 + 1]) + s0
        last_exit_i, last_exit_type = -1, "open"
        for s in idx:
            e = s + 1
            if sequential and (e < last_exit_i or (e == last_exit_i and last_exit_type != "open")):
                continue
            tr = simulate_one(sd, int(s), xs, e0, cost)
            if tr is None:
                continue
            rows.append(tr)
            if sequential:
                last_exit_i, last_exit_type = int(tr["exit_i"]), str(tr["exit_type"])
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["entry_date", "ticker"]).reset_index(drop=True)
    return df


def forward_returns(stocks: Dict[str, StockData], spec: Optional[EntrySpec], start: str, end: str,
                    horizons: Iterable[int] = C.FIXED_HORIZONS, gate: Optional[pd.Series] = None,
                    cost_bps: float = C.COST_BPS_PER_SIDE) -> pd.DataFrame:
    """
    Exit-agnostic forward returns (open[s+1] -> open[s+1+H]) for every signal.

    ``spec=None`` returns the unconditional baseline over all warmed-up stock-days.
    Forward windows are truncated at ``end`` (no out-of-window prices).
    """
    cost = cost_bps / 1e4
    out = []
    for tkr, sd in stocks.items():
        s0, e0 = sd.index_range(start, end)
        if spec is None:
            mask = sd.feat["sma200"].notna().to_numpy()
        else:
            mask = sd.signal_mask(spec)
        mask = mask & gate_mask(sd, gate)
        idx = np.flatnonzero(mask[s0:e0 + 1]) + s0
        if len(idx) == 0:
            continue
        rec = {"ticker": tkr, "signal_i": idx}
        for H in horizons:
            ent = idx + 1
            ext = idx + 1 + H
            ok = ext <= e0
            r = np.full(len(idx), np.nan)
            r[ok] = (sd.o[ext[ok]] * (1 - cost)) / (sd.o[ent[ok]] * (1 + cost)) - 1.0
            rec[f"fwd{H}"] = r
        out.append(pd.DataFrame(rec))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def per_stock_daily_returns(sd: StockData, trades: pd.DataFrame, start_i: int, end_i: int,
                            cost_bps: float) -> np.ndarray:
    """
    Daily return path of a fully-invested, one-position-at-a-time $X strategy on one stock.

    Flat days return 0.  Used for the per-ticker equity/Sharpe/drawdown comparison.
    """
    cost = cost_bps / 1e4
    g = np.ones(end_i - start_i + 1)
    for tr in trades.itertuples():
        e, x = int(tr.entry_i), int(tr.exit_i)
        prev = tr.entry_px * (1 + cost)
        last_full = x - 1 if tr.exit_type in ("open", "intraday") else x
        for j in range(e, last_full + 1):
            if start_i <= j <= end_i:
                g[j - start_i] *= sd.c[j] / prev
            prev = sd.c[j]
        if start_i <= x <= end_i:
            if tr.exit_type in ("open", "intraday"):
                g[x - start_i] *= tr.exit_px * (1 - cost) / prev
            else:  # period_end at the close: charge the exit cost on the last day
                g[x - start_i] *= (1 - cost)
    return g - 1.0
