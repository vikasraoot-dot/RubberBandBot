"""
Shared research helpers used by both ``run_dev.py`` and ``run_oos.py``.

A *Model* = entry rule + exit rule + market gate (binary) or ON/REDUCED/OFF
state rule.  Models are plain dataclasses so the development-period choices
can be frozen to JSON and read back unchanged for the out-of-sample run.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from .engine import ExitSpec, StockData, forward_returns, per_stock_daily_returns, run_trades
from .features import compute_features
from .metrics import buy_and_hold_equity, equity_stats, returns_to_equity, trade_stats
from .portfolio import PortfolioParams, simulate_portfolio
from .regime import gate_series
from .setups import EntrySpec

log = logging.getLogger(__name__)

PERIODS = {
    "dev": (C.BACKTEST_START, C.DEV_END),
    "oos": (C.OOS_START, None),   # None -> latest available session
    "full": (C.BACKTEST_START, None),
}

SIGNAL_FEATURES = ["atr_pct", "ext", "ext_atr", "pb_depth", "pb_depth_atr", "pb_vol_ratio", "range_atr",
                   "clv", "body_frac", "vol_ratio", "low_to_ema20", "ret_20d", "base_range_atr"]
REGIME_FEATURES = ["market_state", "vol_regime", "breadth_state", "vix", "vix_pct", "vix_roc5", "vix_gt_sma",
                   "pct_above_50", "pct_above_200", "spy_above_200", "spy_50_gt_200", "spy_ema20_rising",
                   "qqq_above_200"]


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────
def load_universe() -> pd.DataFrame:
    """Frozen universe table."""
    return pd.read_csv(C.OUT_DIR / "universe_100.csv")


def load_stocks(tickers: Iterable[str]) -> Dict[str, StockData]:
    """Load cached bars for ``tickers`` and compute features."""
    out: Dict[str, StockData] = {}
    for t in tickers:
        df = D.load_cached(t, C.RAW_DIR)
        out[t] = StockData(t, compute_features(df))
    return out


def load_regime() -> pd.DataFrame:
    """Cached daily regime frame (built by ``build_regime.py``)."""
    reg = pd.read_csv(C.DERIVED_DIR / "regime.csv.gz", index_col="date", parse_dates=["date"])
    for col in ("spy_above_200", "spy_50_gt_200", "spy_ema20_rising", "qqq_above_200", "qqq_50_gt_200",
                "qqq_ema20_rising", "vix_gt_sma"):
        if col in reg:
            reg[col] = reg[col].astype(bool)
    return reg


def data_end() -> str:
    """Last session with complete data for SPY/QQQ/VIX and all universe stocks."""
    return json.loads((C.META_DIR / "data_end.json").read_text())["data_end"]


def period_bounds(name: str, reg: pd.DataFrame) -> Tuple[str, str]:
    """Resolve a period name to concrete [start, end] session dates."""
    start, end = PERIODS[name]
    end = end or data_end()
    sessions = reg.loc[start:end].index
    return sessions[0].strftime("%Y-%m-%d"), sessions[-1].strftime("%Y-%m-%d")


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Model:
    """Entry + exit + regime layer."""

    name: str
    entry: EntrySpec
    exit: ExitSpec = field(default_factory=ExitSpec)
    gate: str = "none"                 # binary gate expression (see regime.gate_series)
    state_rule: Optional[str] = None   # name of an ON/REDUCED/OFF rule (overrides gate for sizing)

    def to_json(self) -> Dict[str, object]:
        """Serialisable dict."""
        return {"name": self.name, "entry": asdict(self.entry), "exit": asdict(self.exit),
                "gate": self.gate, "state_rule": self.state_rule}

    @staticmethod
    def from_json(d: Dict[str, object]) -> "Model":
        """Inverse of ``to_json``."""
        return Model(name=d["name"], entry=EntrySpec(**d["entry"]), exit=ExitSpec(**d["exit"]),
                     gate=d.get("gate", "none"), state_rule=d.get("state_rule"))


STATE_RULES = {
    # a-priori three-state framework; components chosen on the dev period are written to frozen_models.json
    "onoff_v1": {
        "off": "not spy200 OR vol SHOCK",
        "on": "spy200 AND spy50_200 AND breadth50>=50 AND vol in (LOW, NORMAL)",
    },
    "onoff_v2": {
        "off": "not spy200 OR vol SHOCK OR breadth50<40",
        "on": "spy200 AND spy50_200 AND vol in (LOW, NORMAL)",
    },
}


def state_series(reg: pd.DataFrame, rule: str) -> pd.Series:
    """Map a named ON/REDUCED/OFF rule onto the regime frame."""
    spy200 = reg["spy_above_200"].astype(bool)
    s50 = reg["spy_50_gt_200"].astype(bool)
    shock = reg["vol_regime"] == "SHOCK"
    calm = reg["vol_regime"].isin(["LOW", "NORMAL"])
    b50 = reg["pct_above_50"]
    if rule == "onoff_v1":
        off = ~spy200 | shock
        on = spy200 & s50 & (b50 >= 50) & calm
    elif rule == "onoff_v2":
        off = ~spy200 | shock | (b50 < 40)
        on = spy200 & s50 & calm
    else:
        raise KeyError(rule)
    return pd.Series(np.where(off, "OFF", np.where(on, "ON", "REDUCED")), index=reg.index)


def model_gate(model: Model, reg: pd.DataFrame) -> pd.Series:
    """Boolean 'new longs allowed' series for a model."""
    if model.state_rule:
        return state_series(reg, model.state_rule) != "OFF"
    return gate_series(reg, model.gate)


# ──────────────────────────────────────────────────────────────────────────────
# Running
# ──────────────────────────────────────────────────────────────────────────────
def run_model_trades(model: Model, stocks: Dict[str, StockData], reg: pd.DataFrame, period: str,
                     cost_bps: float = C.COST_BPS_PER_SIDE, sequential: bool = True) -> pd.DataFrame:
    """Trade list for a model over a named period."""
    start, end = period_bounds(period, reg)
    return run_trades(stocks, model.entry, model.exit, start, end, gate=model_gate(model, reg),
                      cost_bps=cost_bps, sequential=sequential)


def enrich(trades: pd.DataFrame, stocks: Dict[str, StockData], reg: pd.DataFrame,
           universe: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Attach signal-bar stock features, regime-at-signal and universe metadata to trades."""
    if trades.empty:
        return trades
    tr = trades.copy()
    feats = []
    for row in tr[["ticker", "signal_i"]].itertuples(index=False):
        f = stocks[row.ticker].feat
        feats.append(f.iloc[int(row.signal_i)][SIGNAL_FEATURES].to_numpy())
    tr[[f"sig_{c}" for c in SIGNAL_FEATURES]] = np.array(feats, dtype=float)
    rg = reg[REGIME_FEATURES].reindex(pd.DatetimeIndex(tr["signal_date"]))
    for c in REGIME_FEATURES:
        tr[f"reg_{c}"] = rg[c].to_numpy()
    if universe is not None:
        meta = universe.set_index("ticker")[["sector", "bucket", "vol_bucket", "atr_pct_2019"]]
        tr = tr.join(meta, on="ticker")
    tr["year"] = pd.DatetimeIndex(tr["signal_date"]).year
    return tr


def summarize(trades: pd.DataFrame, label: Dict[str, object], fwd: Optional[pd.DataFrame] = None,
              fwd_base: Optional[float] = None) -> Dict[str, object]:
    """One summary row: label fields + trade stats + optional forward-return edge."""
    row = dict(label)
    row.update(trade_stats(trades))
    if fwd is not None and len(fwd):
        row["fwd10_mean"] = float(np.nanmean(fwd["fwd10"]))
        row["fwd10_n"] = int(np.isfinite(fwd["fwd10"]).sum())
        if fwd_base is not None:
            row["fwd10_excess_vs_all_days"] = row["fwd10_mean"] - fwd_base
    return row


def run_portfolio(model: Model, stocks: Dict[str, StockData], reg: pd.DataFrame, period: str,
                  params: PortfolioParams = PortfolioParams()) -> Dict[str, object]:
    """Capital-constrained $50K simulation of a model over a named period."""
    start, end = period_bounds(period, reg)
    cand = run_trades(stocks, model.entry, model.exit, start, end, gate=model_gate(model, reg),
                      cost_bps=params.cost_bps, sequential=False)
    cal = reg.loc[start:end].index
    state = state_series(reg, model.state_rule) if model.state_rule else None
    return simulate_portfolio(cand, stocks, cal, state, params)


def benchmark_stats(reg: pd.DataFrame, period: str, capital: float = C.PORTFOLIO_CAPITAL) -> Dict[str, Dict]:
    """Buy-and-hold SPY and QQQ over a period (adjusted, total return)."""
    start, end = period_bounds(period, reg)
    out = {}
    for sym in ("SPY", "QQQ"):
        o = reg.loc[start:end, f"{sym.lower()}_open"]
        c = reg.loc[start:end, f"{sym.lower()}_close"]
        eq = buy_and_hold_equity(o, c, capital)
        st = equity_stats(eq)
        st.update({"avg_exposure": 1.0})
        out[sym] = {"stats": st, "equity": eq}
    return out


def per_ticker_table(model: Model, trades: pd.DataFrame, stocks: Dict[str, StockData], reg: pd.DataFrame,
                     period: str, universe: pd.DataFrame, cost_bps: float = C.COST_BPS_PER_SIDE) -> pd.DataFrame:
    """
    Per-stock $10K fully-invested strategy vs buy-and-hold of the same stock.

    Returns one row per ticker with strategy and B&H total return / CAGR / Sharpe / max DD
    plus trade stats.
    """
    start, end = period_bounds(period, reg)
    rows = []
    for tkr, sd in stocks.items():
        s0, e0 = sd.index_range(start, end)
        if e0 <= s0:
            continue
        tt = trades[trades["ticker"] == tkr] if len(trades) else trades
        r = per_stock_daily_returns(sd, tt, s0, e0, cost_bps)
        idx = sd.dates[s0:e0 + 1]
        eq = pd.concat([pd.Series([C.PER_STOCK_CAPITAL], index=[idx[0] - pd.Timedelta(days=1)]),
                        returns_to_equity(r, idx, C.PER_STOCK_CAPITAL)])
        st = equity_stats(eq)
        bh = equity_stats(buy_and_hold_equity(pd.Series(sd.o[s0:e0 + 1], index=idx),
                                              pd.Series(sd.c[s0:e0 + 1], index=idx), C.PER_STOCK_CAPITAL,
                                              cost_bps))
        ts = trade_stats(tt)
        in_mkt = float(np.mean(r != 0)) if len(r) else 0.0
        rows.append({
            "model": model.name, "period": period, "ticker": tkr,
            "trades": ts.get("trades", 0), "win_rate": ts.get("win_rate"), "expectancy": ts.get("expectancy"),
            "profit_factor": ts.get("profit_factor"), "sum_trade_ret": ts.get("sum_ret", 0.0),
            "strat_total_return": st.get("total_return"), "strat_cagr": st.get("cagr"),
            "strat_sharpe": st.get("sharpe"), "strat_max_dd": st.get("max_dd"),
            "bh_total_return": bh.get("total_return"), "bh_cagr": bh.get("cagr"),
            "bh_sharpe": bh.get("sharpe"), "bh_max_dd": bh.get("max_dd"), "time_in_market": in_mkt,
        })
    out = pd.DataFrame(rows)
    return out.merge(universe[["ticker", "sector", "bucket", "vol_bucket", "atr_pct_2019"]], on="ticker", how="left")


def breadth_of_results(pt: pd.DataFrame) -> Dict[str, float]:
    """Section-54 statistics over a per-ticker table."""
    traded = pt[pt["trades"] > 0]
    q = pt["strat_total_return"].quantile([0.25, 0.5, 0.75])
    return {
        "tickers": int(len(pt)),
        "tickers_traded": int(len(traded)),
        "pct_profitable": float((pt["strat_total_return"] > 0).mean()),
        "pct_positive_expectancy": float((traded["expectancy"] > 0).mean()) if len(traded) else np.nan,
        "pct_beat_bh_return": float((pt["strat_total_return"] > pt["bh_total_return"]).mean()),
        "pct_better_sharpe": float((pt["strat_sharpe"] > pt["bh_sharpe"]).mean()),
        "pct_smaller_dd": float((pt["strat_max_dd"] > pt["bh_max_dd"]).mean()),
        "median_ticker_return": float(q.loc[0.5]),
        "p25_ticker_return": float(q.loc[0.25]),
        "p75_ticker_return": float(q.loc[0.75]),
        "median_bh_return": float(pt["bh_total_return"].median()),
    }


def save_json(obj: object, path: Path) -> None:
    """Pretty JSON writer."""
    path.write_text(json.dumps(obj, indent=2, default=str))


def portfolio_seeds(model: Model, stocks: Dict[str, StockData], reg: pd.DataFrame, period: str,
                    params: PortfolioParams = PortfolioParams(), seeds: Iterable[int] = range(5)) -> Dict[str, object]:
    """
    Run the portfolio for several priority seeds and average the statistics.

    Candidate trades are simulated once; only the capital-allocation order changes
    with the seed.  Returns ``{"mean": {...}, "min_sharpe": x, "max_sharpe": y, "runs": [...]}``.
    """
    start, end = period_bounds(period, reg)
    cand = run_trades(stocks, model.entry, model.exit, start, end, gate=model_gate(model, reg),
                      cost_bps=params.cost_bps, sequential=False)
    cal = reg.loc[start:end].index
    state = state_series(reg, model.state_rule) if model.state_rule else None
    runs = []
    for sd in seeds:
        p = PortfolioParams(**{**asdict(params), "seed": sd})
        runs.append(simulate_portfolio(cand, stocks, cal, state, p))
    stats = pd.DataFrame([r["stats"] for r in runs])
    return {"mean": stats.mean(numeric_only=True).to_dict(),
            "min_sharpe": float(stats["sharpe"].min()), "max_sharpe": float(stats["sharpe"].max()),
            "runs": runs}


def equal_weight_universe_equity(stocks: Dict[str, StockData], reg: pd.DataFrame, period: str,
                                 capital: float = C.PORTFOLIO_CAPITAL) -> pd.Series:
    """Equal-dollar buy-and-hold of every universe stock from the period's first open (no rebalancing)."""
    start, end = period_bounds(period, reg)
    cal = reg.loc[start:end].index
    per = capital / len(stocks)
    total = pd.Series(0.0, index=cal)
    for sd in stocks.values():
        c = sd.feat["close"].reindex(cal).ffill()
        o0 = sd.feat["open"].reindex(cal).dropna().iloc[0]
        total += per * c / o0
    return pd.concat([pd.Series([capital], index=[cal[0] - pd.Timedelta(days=1)]), total])
