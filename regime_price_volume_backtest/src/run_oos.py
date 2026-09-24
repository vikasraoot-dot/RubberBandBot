"""
Step 5: evaluate the FROZEN models on the development period and, unchanged,
on the out-of-sample period (2023-01-01 .. latest complete session).

Nothing in this script selects or tunes anything; every model, ladder rung,
ablation and sensitivity variant is read from ``frozen_models.json`` (written
by ``freeze.py`` and committed before this script was first run).

Outputs (package root): portfolio_results.csv, setup_comparison.csv,
ablation_results.csv, parameter_sensitivity.csv, regime_results.csv,
per_ticker_results.csv, winners_losers_analysis.csv, trades.csv,
volatility_results.csv, sector_results.csv, concentration_results.csv,
plus equity curves in data/derived/ and case studies in qc/.

Run:  python -m src.run_oos
"""
from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config as C
from .engine import forward_returns
from .metrics import equity_stats, trade_stats
from .portfolio import PortfolioParams
from .research import (Model, benchmark_stats, breadth_of_results, enrich, equal_weight_universe_equity,
                       load_regime, load_stocks, load_universe, model_gate, per_ticker_table, period_bounds,
                       portfolio_seeds, run_model_trades, summarize)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_oos")

PERIODS = ("dev", "oos")
SEEDS = range(10)
OUT = C.OUT_DIR


class Ctx:
    """Shared state: data, frozen spec, forward-return baselines."""

    def __init__(self) -> None:
        self.frozen = json.loads((OUT / "frozen_models.json").read_text())
        self.models = {k: Model.from_json(v) for k, v in self.frozen["models"].items()}
        self.uni = load_universe()
        self.stocks = load_stocks(self.uni["ticker"])
        self.reg = load_regime()
        self.base_fwd10: Dict[str, float] = {}
        self.base_fwd: Dict[str, pd.DataFrame] = {}
        for p in PERIODS:
            s, e = period_bounds(p, self.reg)
            fw = forward_returns(self.stocks, None, s, e)
            self.base_fwd[p] = fw
            self.base_fwd10[p] = float(np.nanmean(fw["fwd10"]))
        self._trade_cache: Dict[tuple, pd.DataFrame] = {}

    def trades(self, m: Model, period: str) -> pd.DataFrame:
        """Sequential trade list (cached)."""
        key = (json.dumps(m.to_json(), sort_keys=True, default=str), period)
        if key not in self._trade_cache:
            self._trade_cache[key] = run_model_trades(m, self.stocks, self.reg, period)
        return self._trade_cache[key]

    def row(self, m: Model, period: str, label: Dict[str, object], portfolio_seeds_n: int = 0) -> Dict[str, object]:
        """Trade stats + fwd10 edge (+ seed-averaged portfolio) for one model/period."""
        s, e = period_bounds(period, self.reg)
        tr = self.trades(m, period)
        fwd = forward_returns(self.stocks, m.entry, s, e, gate=model_gate(m, self.reg))
        r = summarize(tr, {"period": period, "model": m.name, **label}, fwd, self.base_fwd10[period])
        if portfolio_seeds_n:
            ps = portfolio_seeds(m, self.stocks, self.reg, period, seeds=range(portfolio_seeds_n))
            r.update({f"pf_{k}": v for k, v in ps["mean"].items()})
        return r


def _pooled_trade_stats(runs: List[Dict[str, object]]) -> Dict[str, float]:
    taken = pd.concat([r["taken"] for r in runs if len(r["taken"])], ignore_index=True)
    st = trade_stats(taken)
    st["trades"] = st.get("trades", 0) / max(len(runs), 1)
    return st


def portfolio_section(X: Ctx) -> pd.DataFrame:
    """Top-level table, risk/cost grids and benchmarks (Sections 36-41, 53)."""
    rows = []
    derived = C.DERIVED_DIR
    for p in PERIODS:
        for name in X.frozen["top_table"] + X.frozen.get("portfolio_extra", []):
            m = X.models[name]
            for risk in C.RISK_GRID:
                for cost in (0.0, C.COST_BPS_PER_SIDE, 15.0):
                    if (risk != C.RISK_PER_TRADE or cost != C.COST_BPS_PER_SIDE) and name not in (X.frozen["final"], X.frozen["compare"]):
                        continue
                    if risk != C.RISK_PER_TRADE and cost != C.COST_BPS_PER_SIDE:
                        continue
                    params = PortfolioParams(risk_per_trade=risk, cost_bps=cost)
                    ps = portfolio_seeds(m, X.stocks, X.reg, p, params, SEEDS)
                    ts = _pooled_trade_stats(ps["runs"])
                    row = {"period": p, "model": name, "risk_per_trade": risk, "cost_bps": cost,
                           "gate": m.state_rule or m.gate, "exit": m.exit.label(),
                           **{f"pf_{k}": v for k, v in ps["mean"].items()},
                           "pf_min_sharpe": ps["min_sharpe"], "pf_max_sharpe": ps["max_sharpe"],
                           **{f"tr_{k}": v for k, v in ts.items()}}
                    rows.append(row)
                    if risk == C.RISK_PER_TRADE and cost == C.COST_BPS_PER_SIDE:
                        eqs = pd.concat([r["equity"] for r in ps["runs"]], axis=1).mean(axis=1)
                        expo = pd.concat([r["daily"]["exposure"] for r in ps["runs"]], axis=1).mean(axis=1)
                        pd.DataFrame({"equity": eqs}).to_csv(derived / f"equity_{p}_{name}.csv")
                        pd.DataFrame({"exposure": expo}).to_csv(derived / f"exposure_{p}_{name}.csv")
                        if name == X.frozen["final"]:
                            ps["runs"][0]["taken"].to_csv(derived / f"portfolio_taken_{p}_{name}_seed0.csv",
                                                          index=False)
        # exposure-cap variants for the ON/REDUCED/OFF model
        onoff = X.frozen.get("onoff_model")
        if onoff:
            for cap_red in (0.25, 0.5, 0.75):
                params = PortfolioParams(cap_reduced=cap_red)
                ps = portfolio_seeds(X.models[onoff], X.stocks, X.reg, p, params, SEEDS)
                rows.append({"period": p, "model": f"{onoff}|cap_reduced={cap_red}", "risk_per_trade": C.RISK_PER_TRADE,
                             "cost_bps": C.COST_BPS_PER_SIDE, "gate": X.models[onoff].state_rule,
                             **{f"pf_{k}": v for k, v in ps["mean"].items()},
                             **{f"tr_{k}": v for k, v in _pooled_trade_stats(ps["runs"]).items()}})
        comb = X.frozen.get("combined")
        if comb:
            from .portfolio import simulate_portfolio
            from .engine import run_trades
            s0, e0 = period_bounds(p, X.reg)
            parts = []
            for nm in comb["members"]:
                mm = X.models[nm]
                parts.append(run_trades(X.stocks, mm.entry, mm.exit, s0, e0, gate=model_gate(mm, X.reg),
                                        sequential=False).assign(member=nm))
            cand = pd.concat(parts, ignore_index=True)
            # same stock + same entry day from both setups: keep the first-listed member's trade
            cand = cand.drop_duplicates(["ticker", "entry_date"], keep="first")
            runs = [simulate_portfolio(cand, X.stocks, X.reg.loc[s0:e0].index, None, PortfolioParams(seed=sd))
                    for sd in SEEDS]
            st = pd.DataFrame([r["stats"] for r in runs]).mean(numeric_only=True).to_dict()
            rows.append({"period": p, "model": comb["name"], "risk_per_trade": C.RISK_PER_TRADE,
                         "cost_bps": C.COST_BPS_PER_SIDE, "gate": "per-member",
                         **{f"pf_{k}": v for k, v in st.items()},
                         **{f"tr_{k}": v for k, v in _pooled_trade_stats(runs).items()}})
            eqs = pd.concat([r["equity"] for r in runs], axis=1).mean(axis=1)
            pd.DataFrame({"equity": eqs}).to_csv(derived / f"equity_{p}_{comb['name']}.csv")
        bm = benchmark_stats(X.reg, p)
        for sym, v in bm.items():
            rows.append({"period": p, "model": f"BENCH_{sym}_buy_hold", **{f"pf_{k}": x for k, x in v["stats"].items()}})
            v["equity"].to_frame("equity").to_csv(derived / f"equity_{p}_BENCH_{sym}.csv")
        ew = equal_weight_universe_equity(X.stocks, X.reg, p)
        ew.to_frame("equity").to_csv(derived / f"equity_{p}_BENCH_EW100.csv")
        rows.append({"period": p, "model": "BENCH_EW_universe_buy_hold",
                     **{f"pf_{k}": x for k, x in equity_stats(ew).items()}, "pf_avg_exposure": 1.0})
        log.info("portfolio %s done", p)
    return pd.DataFrame(rows)


def setup_section(X: Ctx) -> pd.DataFrame:
    """Setup-family comparison, overall and by regime slice (Sections 28, 44)."""
    rows = []
    for p in PERIODS:
        for name in X.frozen["families"]:
            m = X.models[name]
            for gate_label, mm in (("ungated", Model(m.name, m.entry, m.exit, "none")),
                                   (X.frozen["setup_gate"], Model(m.name, m.entry, m.exit, X.frozen["setup_gate"]))):
                r = X.row(mm, p, {"family": name, "gating": gate_label, "slice": "ALL", "slice_value": "ALL"},
                          portfolio_seeds_n=5)
                rows.append(r)
            tr = enrich(X.trades(Model(m.name, m.entry, m.exit, "none"), p), X.stocks, X.reg, X.uni)
            for col in ("reg_market_state", "reg_vol_regime", "reg_breadth_state"):
                for val, g in tr.groupby(col):
                    rows.append({"period": p, "model": name, "family": name, "gating": "ungated", "slice": col,
                                 "slice_value": val, **trade_stats(g)})
    return pd.DataFrame(rows)


def ablation_section(X: Ctx) -> pd.DataFrame:
    """Strategy 0-5 ladder and one-at-a-time component removal (Sections 50, 51)."""
    rows = []
    for p in PERIODS:
        for item in X.frozen["ladder"]:
            m = Model.from_json(item["model"])
            rows.append(X.row(m, p, {"experiment": item.get("experiment", "ladder"), "step": item["name"]},
                              portfolio_seeds_n=5))
        for base, items in X.frozen["ablation"].items():
            for item in items:
                m = Model.from_json(item["model"])
                rows.append(X.row(m, p, {"experiment": f"ablation_of_{base}", "step": item["removed"]},
                                  portfolio_seeds_n=5))
        log.info("ablation %s done", p)
    return pd.DataFrame(rows)


def sensitivity_section(X: Ctx) -> pd.DataFrame:
    """One-at-a-time parameter sensitivity around the final model (Section 49)."""
    rows = []
    for p in PERIODS:
        for item in X.frozen["sensitivity"]:
            m = Model.from_json(item["model"])
            rows.append(X.row(m, p, {"param": item["param"], "value": str(item["value"]),
                                     "is_frozen_value": bool(item.get("is_frozen_value", False))},
                              portfolio_seeds_n=3))
    return pd.DataFrame(rows)


def _bucket(s: pd.Series, edges: List[float], labels: List[str]) -> pd.Series:
    return pd.cut(s, bins=[-np.inf] + edges + [np.inf], labels=labels)


def regime_section(X: Ctx) -> pd.DataFrame:
    """Conditional expectancy by regime dimension and combination (Sections 42, 43)."""
    rows = []
    reg = X.reg.copy()
    reg["vix_level_bucket"] = _bucket(reg["vix"], [15, 20, 25, 30], ["<15", "15-20", "20-25", "25-30", ">=30"])
    reg["vix_pct_bucket"] = _bucket(reg["vix_pct"], [25, 50, 75, 90], ["<25", "25-50", "50-75", "75-90", ">=90"])
    reg["vix_roc_bucket"] = _bucket(reg["vix_roc5"], [-0.10, 0.0, 0.10, 0.25],
                                    ["<-10%", "-10..0%", "0..+10%", "+10..25%", ">=+25%"])
    reg["vix_vs_sma"] = np.where(reg["vix_gt_sma"], "VIX>SMA20(rising)", "VIX<=SMA20")
    reg["spy_trend"] = np.where(reg["spy_above_200"], "SPY>SMA200", "SPY<SMA200")
    reg["vix_state2"] = np.where(reg["vol_regime"].isin(["LOW", "NORMAL"]), "VIX low/normal", "VIX high/shock")
    reg["breadth200_bucket"] = _bucket(reg["pct_above_200"], [40, 60], ["<40", "40-60", ">=60"])
    reg["mkt_x_vix"] = reg["spy_trend"] + " & " + reg["vix_state2"]
    reg["mkt_x_vix_x_breadth"] = reg["mkt_x_vix"] + " & breadth " + reg["breadth_state"].astype(str)
    dims = ["market_state", "spy_trend", "vol_regime", "vix_state2", "vix_level_bucket", "vix_pct_bucket",
            "vix_roc_bucket", "vix_vs_sma", "breadth_state", "breadth200_bucket", "mkt_x_vix",
            "mkt_x_vix_x_breadth"]
    for p in PERIODS:
        # baseline: every stock-day forward 10-session return in the same regime
        fw = X.base_fwd[p]
        fw_sorted = pd.concat([pd.DataFrame({"date": X.stocks[t].dates[g["signal_i"].to_numpy()],
                                             "fwd10": g["fwd10"].to_numpy()}) for t, g in fw.groupby("ticker")])
        for name in X.frozen["regime_models"]:
            m = X.models[name]
            tr = X.trades(Model(m.name, m.entry, m.exit, "none"), p)
            if tr.empty:
                continue
            s, e = period_bounds(p, X.reg)
            fwd = forward_returns(X.stocks, m.entry, s, e)
            fwd["date"] = [X.stocks[t].dates[i] for t, i in zip(fwd["ticker"], fwd["signal_i"])]
            for dim in dims:
                tr_dim = reg[dim].reindex(pd.DatetimeIndex(tr["signal_date"])).astype(str).to_numpy()
                fw_dim = reg[dim].reindex(pd.DatetimeIndex(fwd["date"])).astype(str).to_numpy()
                base_dim = reg[dim].reindex(pd.DatetimeIndex(fw_sorted["date"])).astype(str).to_numpy()
                for val in sorted(set(tr_dim)):
                    g = tr[tr_dim == val]
                    st = trade_stats(g)
                    rows.append({"period": p, "model": name, "dimension": dim, "value": val,
                                 "share_of_days": float((reg.loc[s:e, dim].astype(str) == val).mean()),
                                 **st,
                                 "sig_fwd10": float(np.nanmean(fwd.loc[fw_dim == val, "fwd10"])) if (fw_dim == val).any() else np.nan,
                                 "all_days_fwd10": float(np.nanmean(fw_sorted.loc[base_dim == val, "fwd10"]))})
        log.info("regime %s done", p)
    return pd.DataFrame(rows)


def per_ticker_section(X: Ctx) -> pd.DataFrame:
    """Per-stock $10K results for the final model and its ungated control (Section 54)."""
    rows = []
    for p in PERIODS:
        for name in (X.frozen["final"], X.frozen["compare"]):
            m = X.models[name]
            rows.append(per_ticker_table(m, X.trades(m, p), X.stocks, X.reg, p, X.uni))
    return pd.concat(rows, ignore_index=True)


def group_section(X: Ctx, pt: pd.DataFrame, key: str) -> pd.DataFrame:
    """Results by volatility bucket or sector (Sections 45, 46)."""
    rows = []
    for p in PERIODS:
        for name in (X.frozen["final"], X.frozen["compare"]):
            m = X.models[name]
            tr = enrich(X.trades(m, p), X.stocks, X.reg, X.uni)
            if key == "vol_bucket" and len(tr):
                q = pd.qcut(tr["sig_atr_pct"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
                for val, g in tr.groupby(q, observed=True):
                    rows.append({"period": p, "model": name, "grouping": "signal-time ATR% quartile",
                                 "group": str(val), "atr_pct_range": f"{g['sig_atr_pct'].min():.3f}-{g['sig_atr_pct'].max():.3f}",
                                 **trade_stats(g)})
            for val, g in tr.groupby(key):
                ptg = pt[(pt["period"] == p) & (pt["model"] == name) & (pt[key] == val)]
                rows.append({"period": p, "model": name, "grouping": key, "group": val, **trade_stats(g),
                             "tickers": int(len(ptg)),
                             "median_ticker_cagr": float(ptg["strat_cagr"].median()),
                             "median_ticker_max_dd": float(ptg["strat_max_dd"].median()),
                             "median_ticker_sharpe": float(ptg["strat_sharpe"].median()),
                             "median_bh_cagr": float(ptg["bh_cagr"].median()),
                             "pct_tickers_profitable": float((ptg["strat_total_return"] > 0).mean())})
    return pd.DataFrame(rows)


def concentration_section(X: Ctx, pt: pd.DataFrame) -> pd.DataFrame:
    """Share of total profit from the top 5/10/20 stocks (Section 47)."""
    rows = []
    for p in PERIODS:
        for name in (X.frozen["final"], X.frozen["compare"]):
            sub = pt[(pt["period"] == p) & (pt["model"] == name)]
            pnl = (sub.set_index("ticker")["strat_total_return"] * C.PER_STOCK_CAPITAL).sort_values(ascending=False)
            total = pnl.sum()
            gross_profit = pnl[pnl > 0].sum()
            for k in (5, 10, 20):
                rows.append({"period": p, "model": name, "basis": "per-stock $10K strategy P&L", "top_k": k,
                             "share_of_net_profit": float(pnl.head(k).sum() / total) if total > 0 else np.nan,
                             "share_of_gross_profit": float(pnl.head(k).sum() / gross_profit) if gross_profit > 0 else np.nan,
                             "top_tickers": " ".join(pnl.head(k).index),
                             "net_profit_usd": float(total),
                             "net_profit_ex_top_k_usd": float(total - pnl.head(k).sum())})
    return pd.DataFrame(rows)


def winners_losers_section(X: Ctx) -> pd.DataFrame:
    """Feature differences between winning and losing trades (Section 48)."""
    feats = ["sig_atr_pct", "sig_ext", "sig_ext_atr", "sig_pb_depth", "sig_pb_depth_atr", "sig_pb_vol_ratio",
             "sig_range_atr", "sig_clv", "sig_body_frac", "sig_vol_ratio", "sig_ret_20d", "reg_vix", "reg_vix_pct",
             "reg_vix_roc5", "reg_pct_above_50", "reg_pct_above_200"]
    rows = []
    for name in X.frozen["winner_loser_models"]:
        m = X.models[name]
        mm = Model(m.name, m.entry, m.exit, "none")
        dev = enrich(X.trades(mm, "dev"), X.stocks, X.reg, X.uni)
        edges = {f: dev[f].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy() for f in feats}
        for p in PERIODS:
            tr = dev if p == "dev" else enrich(X.trades(mm, p), X.stocks, X.reg, X.uni)
            win = tr["ret"] > 0
            for f in feats:
                x = tr[f].astype(float)
                qb = np.searchsorted(edges[f], x.to_numpy(), side="right")
                q_exp = [float(tr["ret"][qb == k].mean()) if (qb == k).any() else np.nan for k in range(5)]
                q_n = [int((qb == k).sum()) for k in range(5)]
                rows.append({"model": name, "period": p, "feature": f,
                             "mean_winners": float(x[win].mean()), "mean_losers": float(x[~win].mean()),
                             "median_winners": float(x[win].median()), "median_losers": float(x[~win].median()),
                             "corr_with_return": float(np.corrcoef(x.fillna(x.median()), tr["ret"])[0, 1]),
                             **{f"q{k+1}_expectancy": q_exp[k] for k in range(5)},
                             **{f"q{k+1}_n": q_n[k] for k in range(5)},
                             "q5_minus_q1": q_exp[4] - q_exp[0],
                             "dev_quintile_edges": " / ".join(f"{v:.4g}" for v in edges[f])})
            for col in ("reg_market_state", "reg_vol_regime", "reg_breadth_state"):
                share_w = tr.loc[win, col].value_counts(normalize=True)
                share_l = tr.loc[~win, col].value_counts(normalize=True)
                for val in sorted(set(share_w.index) | set(share_l.index)):
                    rows.append({"model": name, "period": p, "feature": f"{col}={val}",
                                 "mean_winners": float(share_w.get(val, 0.0)), "mean_losers": float(share_l.get(val, 0.0))})
    return pd.DataFrame(rows)


def trades_section(X: Ctx) -> pd.DataFrame:
    """All trade-level trades for the models listed in ``trade_export_models``."""
    out = []
    for p in PERIODS:
        for name in X.frozen["trade_export_models"]:
            tr = enrich(X.trades(X.models[name], p), X.stocks, X.reg, X.uni)
            tr.insert(0, "period", p)
            tr.insert(0, "model", name)
            out.append(tr)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    """Run the full frozen-model evaluation."""
    t0 = time.time()
    X = Ctx()
    log.info("frozen models: %s (fwd10 base dev %.4f / oos %.4f)", list(X.models), X.base_fwd10["dev"],
             X.base_fwd10["oos"])
    trades_section(X).to_csv(OUT / "trades.csv", index=False, float_format="%.6g")
    pt = per_ticker_section(X)
    pt.to_csv(OUT / "per_ticker_results.csv", index=False)
    summ = []
    for (p, m), g in pt.groupby(["period", "model"]):
        summ.append({"period": p, "model": m, **breadth_of_results(g)})
    pd.DataFrame(summ).to_csv(OUT / "per_ticker_summary.csv", index=False)
    group_section(X, pt, "vol_bucket").to_csv(OUT / "volatility_results.csv", index=False)
    group_section(X, pt, "bucket").to_csv(OUT / "sector_results.csv", index=False)
    concentration_section(X, pt).to_csv(OUT / "concentration_results.csv", index=False)
    log.info("per-ticker/groups done %.0fs", time.time() - t0)
    winners_losers_section(X).to_csv(OUT / "winners_losers_analysis.csv", index=False)
    regime_section(X).to_csv(OUT / "regime_results.csv", index=False)
    setup_section(X).to_csv(OUT / "setup_comparison.csv", index=False)
    log.info("setup done %.0fs", time.time() - t0)
    ablation_section(X).to_csv(OUT / "ablation_results.csv", index=False)
    sensitivity_section(X).to_csv(OUT / "parameter_sensitivity.csv", index=False)
    log.info("ablation/sensitivity done %.0fs", time.time() - t0)
    portfolio_section(X).to_csv(OUT / "portfolio_results.csv", index=False)
    log.info("all done %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
