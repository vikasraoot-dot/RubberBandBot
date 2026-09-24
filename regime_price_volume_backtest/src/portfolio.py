"""
Capital-constrained portfolio simulation ($50K, long only, no leverage).

Inputs are *every* signal's pre-simulated trade path (``run_trades(...,
sequential=False)``): exits depend only on the stock's own prices, so the
portfolio only decides which signals it can afford and how many shares.

Daily order of events for session d
-----------------------------------
1. Exits that fill at the open (close-based exit signals from d-1, gap stops).
2. Entries at the open (signals from d-1's close), sized with the equity
   marked at d-1's close:
       shares = min(risk$ / (entry - stop),
                    max_position_pct x equity / entry,
                    (exposure_cap x equity - gross_exposure) / entry,
                    cash / entry)
   risk$ = equity x risk_per_trade x state multiplier (ON 1.0, REDUCED 0.5).
   When more signals arrive than capital allows, they are taken in a seeded
   pseudo-random order (no hidden ranking edge); results are checked across
   seeds.
3. Intraday stop exits and period-end closes.
4. Mark to market at the close.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config as C
from .engine import StockData
from .metrics import equity_stats


@dataclass(frozen=True)
class PortfolioParams:
    """Portfolio construction parameters."""

    capital: float = C.PORTFOLIO_CAPITAL
    risk_per_trade: float = C.RISK_PER_TRADE
    max_position_pct: float = C.MAX_POSITION_PCT
    min_position_pct: float = C.MIN_POSITION_PCT
    cap_on: float = C.EXPOSURE_CAP_ON
    cap_reduced: float = C.EXPOSURE_CAP_REDUCED
    reduced_risk_mult: float = C.REDUCED_RISK_MULT
    cost_bps: float = C.COST_BPS_PER_SIDE
    seed: int = 0


def simulate_portfolio(trades: pd.DataFrame, stocks: Dict[str, StockData], calendar: pd.DatetimeIndex,
                       state: Optional[pd.Series], params: PortfolioParams) -> Dict[str, object]:
    """
    Run the portfolio over ``calendar``.

    Args:
        trades: all candidate trades (already filtered by the entry gate).
        stocks: ticker -> StockData (for daily closes).
        calendar: sessions to simulate (SPY calendar within the window).
        state: Series of 'ON'/'REDUCED'/'OFF' by signal date (None = always ON).
        params: PortfolioParams.

    Returns:
        dict with ``equity`` (Series), ``daily`` (DataFrame of exposure/positions),
        ``taken`` (DataFrame of executed trades with shares and $ P&L), ``stats``.
    """
    cost = params.cost_bps / 1e4
    rng = np.random.default_rng(params.seed)
    tr = trades.copy()
    if len(tr):
        tr["prio"] = rng.random(len(tr))
        if state is not None:
            tr["state"] = state.reindex(tr["signal_date"]).fillna("OFF").to_numpy()
        else:
            tr["state"] = "ON"
        tr = tr[tr["state"] != "OFF"]
    by_entry = {d: g.sort_values("prio") for d, g in tr.groupby("entry_date")} if len(tr) else {}
    date_pos = {tkr: {d: i for i, d in enumerate(sd.dates)} for tkr, sd in stocks.items()}

    cash = params.capital
    equity_prev = params.capital
    held: Dict[str, Dict[str, object]] = {}
    last_close: Dict[str, float] = {}
    eq_rows: List[float] = []
    daily_rows: List[Dict[str, float]] = []
    taken: List[Dict[str, object]] = []

    def _close_pos(tkr: str, px: float, d: pd.Timestamp) -> None:
        nonlocal cash
        p = held.pop(tkr)
        proceeds = p["shares"] * px * (1 - cost)
        cash += proceeds
        rec = dict(p["trade"])
        rec.update({"shares": p["shares"], "cost_basis": p["basis"],
                    "pnl_usd": proceeds - p["basis"], "size_state": p["state"]})
        taken.append(rec)

    for d in calendar:
        # 1) exits at the open
        for tkr in [t for t, p in held.items() if p["exit_date"] == d and p["exit_type"] == "open"]:
            _close_pos(tkr, held[tkr]["exit_px"], d)
        # 2) entries at the open
        cands = by_entry.get(d)
        if cands is not None:
            gross = sum(p["shares"] * last_close.get(t, p["entry_px"]) for t, p in held.items())
            for row in cands.itertuples(index=False):
                tkr = row.ticker
                if tkr in held:
                    continue
                st = row.state
                cap = params.cap_on if st == "ON" else params.cap_reduced
                mult = 1.0 if st == "ON" else params.reduced_risk_mult
                entry = float(row.entry_px)
                risk_ps = float(row.risk_pct) * entry
                if risk_ps <= 0 or entry <= 0:
                    continue
                risk_usd = equity_prev * params.risk_per_trade * mult
                n_risk = risk_usd / risk_ps
                n_cap = params.max_position_pct * equity_prev / entry
                n_room = (cap * equity_prev - gross) / entry
                n_cash = cash / (entry * (1 + cost))
                shares = int(np.floor(min(n_risk, n_cap, n_room, n_cash)))
                if shares < 1 or shares * entry < params.min_position_pct * equity_prev:
                    continue
                basis = shares * entry * (1 + cost)
                cash -= basis
                gross += shares * entry
                held[tkr] = {"shares": shares, "entry_px": entry, "basis": basis, "state": st,
                             "exit_date": row.exit_date, "exit_type": row.exit_type,
                             "exit_px": float(row.exit_px), "trade": row._asdict()}
        # 3) intraday stops / period-end closes
        for tkr in [t for t, p in held.items() if p["exit_date"] == d and p["exit_type"] in ("intraday", "close")]:
            _close_pos(tkr, held[tkr]["exit_px"], d)
        # 4) mark to market
        mv = 0.0
        for tkr, p in held.items():
            i = date_pos[tkr].get(d)
            if i is not None:
                last_close[tkr] = stocks[tkr].c[i]
            mv += p["shares"] * last_close.get(tkr, p["entry_px"])
        equity = cash + mv
        eq_rows.append(equity)
        daily_rows.append({"gross": mv, "exposure": mv / equity if equity > 0 else 0.0,
                           "positions": len(held), "cash": cash})
        equity_prev = equity

    eq = pd.Series(eq_rows, index=calendar)
    start_pt = pd.Series([params.capital], index=[calendar[0] - pd.Timedelta(days=1)])
    eq_full = pd.concat([start_pt, eq])
    daily = pd.DataFrame(daily_rows, index=calendar)
    taken_df = pd.DataFrame(taken)
    stats = equity_stats(eq_full)
    stats.update({
        "avg_exposure": float(daily["exposure"].mean()),
        "avg_capital_deployed": float(daily["gross"].mean()),
        "peak_capital_deployed": float(daily["gross"].max()),
        "avg_positions": float(daily["positions"].mean()),
        "max_positions": int(daily["positions"].max()),
        "portfolio_trades": int(len(taken_df)),
        "candidate_signals": int(len(tr)),
    })
    return {"equity": eq_full, "daily": daily, "taken": taken_df, "stats": stats}
