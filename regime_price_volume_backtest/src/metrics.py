"""
Performance statistics for trade lists and equity curves.

Conventions
-----------
* Returns are simple returns; annualisation uses 252 sessions.
* Sharpe/Sortino use a 0% risk-free rate (stated in the report).
* Downside volatility = sqrt(mean(min(r, 0)^2)) * sqrt(252) (semi-deviation vs 0).
* All ratios guard against division by zero and return NaN when undefined.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

ANN = 252


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b not in (0, 0.0) and np.isfinite(b) else float("nan")


def trade_stats(tr: pd.DataFrame, ret_col: str = "ret") -> Dict[str, float]:
    """
    Summary statistics for a trade list.

    Returns a dict with count, win rate, average winner/loser, median, profit
    factor, expectancy (mean return per trade), t-stat of the mean, average
    R-multiple, holding period, MFE/MAE and profit giveback.
    """
    if tr is None or len(tr) == 0:
        return {"trades": 0}
    r = tr[ret_col].astype(float)
    wins, losses = r[r > 0], r[r <= 0]
    std = r.std(ddof=1) if len(r) > 1 else np.nan
    gb = tr["giveback_pp"].dropna() if "giveback_pp" in tr else pd.Series(dtype=float)
    peak = tr.loc[tr["peak_close_ret"] > 0, "peak_close_ret"] if "peak_close_ret" in tr else pd.Series(dtype=float)
    return {
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "median_ret": float(r.median()),
        "expectancy": float(r.mean()),
        "t_stat": _safe_div(r.mean(), std / np.sqrt(len(r))) if len(r) > 1 else np.nan,
        "profit_factor": _safe_div(wins.sum(), -losses.sum()) if len(losses) else np.inf,
        "sum_ret": float(r.sum()),
        "avg_R": float(tr["R"].mean()) if "R" in tr else np.nan,
        "avg_hold": float(tr["hold_days"].mean()) if "hold_days" in tr else np.nan,
        "avg_mfe": float(tr["mfe"].mean()) if "mfe" in tr else np.nan,
        "avg_mae": float(tr["mae"].mean()) if "mae" in tr else np.nan,
        "avg_giveback_pp": float(gb.mean()) if len(gb) else np.nan,
        "giveback_pct_of_peak": _safe_div(gb.sum(), peak.sum()) if len(peak) else np.nan,
    }


def equity_stats(equity: pd.Series) -> Dict[str, float]:
    """
    Statistics for a daily equity curve (index = sessions).

    Returns total return, CAGR, max drawdown, annualised/downside volatility,
    Sharpe, Sortino, Calmar and ending equity.
    """
    eq = equity.dropna()
    if len(eq) < 2 or eq.iloc[0] <= 0:
        return {}
    r = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = eq.iloc[-1] / eq.iloc[0] - 1.0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else -1.0
    dd = eq / eq.cummax() - 1.0
    vol = r.std(ddof=1) * np.sqrt(ANN)
    dvol = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) * np.sqrt(ANN)
    mean_ann = r.mean() * ANN
    return {
        "start_equity": float(eq.iloc[0]),
        "end_equity": float(eq.iloc[-1]),
        "total_return": float(total),
        "cagr": float(cagr),
        "max_dd": float(dd.min()),
        "ann_vol": float(vol),
        "downside_vol": float(dvol),
        "sharpe": _safe_div(mean_ann, vol),
        "sortino": _safe_div(mean_ann, dvol),
        "calmar": _safe_div(cagr, abs(dd.min())),
    }


def returns_to_equity(daily_ret: np.ndarray, index: pd.DatetimeIndex, start_value: float) -> pd.Series:
    """Compound a daily return array into an equity curve that starts at ``start_value``."""
    return pd.Series(start_value * np.cumprod(1.0 + np.asarray(daily_ret)), index=index)


def buy_and_hold_equity(o: pd.Series, c: pd.Series, start_value: float, cost_bps: float = 0.0) -> pd.Series:
    """Buy at the first open, hold to the end; equity marked at each close."""
    cost = cost_bps / 1e4
    shares = start_value / (o.iloc[0] * (1 + cost))
    eq = shares * c
    eq.iloc[-1] = eq.iloc[-1] * (1 - cost)
    first = pd.Series([start_value], index=[c.index[0] - pd.Timedelta(days=1)])
    return pd.concat([first, eq])
