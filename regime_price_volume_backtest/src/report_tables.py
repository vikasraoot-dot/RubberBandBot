"""
Render every REPORT.md table from the result CSVs (so report numbers are traceable).

Writes qc/report_tables.md; REPORT.md embeds these tables verbatim.

Run:  python -m src.report_tables
"""
from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import config as C

R = C.OUT_DIR


def pct(v: float, d: int = 1) -> str:
    """Format a fraction as a percentage."""
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v * 100:.{d}f}%"


def num(v: float, d: int = 2) -> str:
    """Format a number."""
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def md(df: pd.DataFrame, fmts: Dict[str, Callable[[float], str]], cols: Sequence[str],
       headers: Optional[Sequence[str]] = None, align_left: int = 1) -> str:
    """Markdown table with per-column formatters."""
    headers = list(headers or cols)
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join([":--"] * align_left + ["--:"] * (len(cols) - align_left)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            f = fmts.get(c)
            cells.append(f(v) if f and v is not None and not (isinstance(v, str)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


LABELS = {
    "FINAL": "**FINAL — pullback to EMA20, no regime gate** (= no-regime-filter row)",
    "SPY_only": "FINAL + SPY>SMA200 gate",
    "SPY_VIX": "FINAL + SPY>SMA200 + VIX no-shock",
    "SPY_VIX_BREADTH": "FINAL + SPY + VIX + breadth≥40%",
    "ONOFF": "FINAL with ON / REDUCED / OFF sizing",
    "A_full_hypothesis": "Full hypothesis (trend+pullback+candle+volume+SPY+VIX+breadth)",
    "C_breakout": "Setup C contraction breakout + SPY gate",
    "B_failed_breakdown": "Setup B failed breakdown (swing low)",
    "REF_random_uptrend_day": "Control: random uptrend-day entry, same exit",
    "REF_random_every_day": "Control: random entry any day, same exit",
    "COMBINED_C_breakout+FINAL": "FINAL + Setup C combined",
    "BENCH_SPY_buy_hold": "SPY buy & hold",
    "BENCH_QQQ_buy_hold": "QQQ buy & hold",
    "BENCH_EW_universe_buy_hold": "Equal-weight 100-stock buy & hold",
}


def top_table(pr: pd.DataFrame, period: str) -> str:
    """Section-53 headline table."""
    base = pr[(pr["period"] == period) & ((pr["risk_per_trade"].isna()) | ((pr["risk_per_trade"] == C.RISK_PER_TRADE)
                                                                            & (pr["cost_bps"] == C.COST_BPS_PER_SIDE)))]
    order = list(LABELS)
    base = base[base["model"].isin(order)].copy()
    base["order"] = base["model"].map({k: i for i, k in enumerate(order)})
    base = base.sort_values("order")
    base["label"] = base["model"].map(LABELS)
    cols = ["label", "pf_cagr", "pf_max_dd", "pf_sharpe", "tr_profit_factor", "tr_win_rate", "tr_trades",
            "tr_giveback_pct_of_peak", "pf_avg_exposure"]
    f = {"pf_cagr": pct, "pf_max_dd": pct, "pf_sharpe": num, "tr_profit_factor": num, "tr_win_rate": lambda v: pct(v, 0),
         "tr_trades": lambda v: "–" if not np.isfinite(v) else f"{v:,.0f}", "tr_giveback_pct_of_peak": lambda v: pct(v, 0),
         "pf_avg_exposure": lambda v: pct(v, 0)}
    return md(base, f, cols, ["Strategy", "CAGR", "Max DD", "Sharpe", "Profit factor", "Win rate", "Trades",
                              "Giveback (% of peak)", "Avg exposure"])


def main() -> None:
    """Write qc/report_tables.md."""
    parts: List[str] = []
    pr = pd.read_csv(R / "portfolio_results.csv")
    for p, title in (("oos", "OUT-OF-SAMPLE 2023-01-03 → 2026-09-21"), ("dev", "DEVELOPMENT 2020-01-02 → 2022-12-30")):
        parts += [f"## TOP TABLE — {title}", top_table(pr, p), ""]
    # seed ranges for key models
    k = pr[(pr["model"].isin(["FINAL", "SPY_only", "REF_random_uptrend_day", "A_full_hypothesis", "ONOFF"]))
           & (pr["risk_per_trade"] == C.RISK_PER_TRADE) & (pr["cost_bps"] == C.COST_BPS_PER_SIDE)]
    parts += ["## Sharpe range across 10 priority seeds",
              md(k, {"pf_sharpe": num, "pf_min_sharpe": num, "pf_max_sharpe": num}, ["period", "model", "pf_sharpe", "pf_min_sharpe", "pf_max_sharpe"],
                 ["Period", "Model", "Mean Sharpe", "Min", "Max"], align_left=2), ""]
    # risk / cost grid
    g = pr[pr["model"].isin(["FINAL", "SPY_only"]) & pr["risk_per_trade"].notna()].copy()
    parts += ["## Risk-per-trade and cost grid",
              md(g, {"risk_per_trade": lambda v: pct(v, 1), "cost_bps": lambda v: f"{v:.0f}", "pf_cagr": pct, "pf_max_dd": pct,
                     "pf_sharpe": num, "pf_sortino": num, "pf_calmar": num, "pf_avg_exposure": lambda v: pct(v, 0),
                     "pf_avg_positions": lambda v: num(v, 1), "pf_max_positions": lambda v: f"{v:.0f}",
                     "pf_avg_capital_deployed": lambda v: f"${v:,.0f}", "pf_peak_capital_deployed": lambda v: f"${v:,.0f}"},
                 ["period", "model", "risk_per_trade", "cost_bps", "pf_cagr", "pf_max_dd", "pf_sharpe", "pf_sortino",
                  "pf_calmar", "pf_avg_exposure", "pf_avg_positions", "pf_max_positions", "pf_avg_capital_deployed",
                  "pf_peak_capital_deployed"],
                 ["Period", "Model", "Risk/trade", "Cost bps", "CAGR", "Max DD", "Sharpe", "Sortino", "Calmar",
                  "Exposure", "Avg pos", "Max pos", "Avg deployed", "Peak deployed"], align_left=2), ""]
    cap = pr[pr["model"].str.contains("cap_reduced", na=False)]
    if len(cap):
        parts += ["## ON/REDUCED/OFF exposure cap in REDUCED state",
                  md(cap, {"pf_cagr": pct, "pf_max_dd": pct, "pf_sharpe": num, "pf_avg_exposure": lambda v: pct(v, 0)},
                     ["period", "model", "pf_cagr", "pf_max_dd", "pf_sharpe", "pf_avg_exposure"], align_left=2), ""]
    # full benchmark stats
    b = pr[pr["model"].str.startswith("BENCH")]
    parts += ["## Benchmarks",
              md(b, {"pf_total_return": pct, "pf_cagr": pct, "pf_max_dd": pct, "pf_ann_vol": pct, "pf_downside_vol": pct,
                     "pf_sharpe": num, "pf_sortino": num, "pf_calmar": num},
                 ["period", "model", "pf_total_return", "pf_cagr", "pf_max_dd", "pf_ann_vol", "pf_downside_vol",
                  "pf_sharpe", "pf_sortino", "pf_calmar"], align_left=2), ""]
    # full metrics for top models
    t = pr[pr["model"].isin(["FINAL", "SPY_only", "A_full_hypothesis", "REF_random_uptrend_day", "COMBINED_C_breakout+FINAL"])
           & ((pr["risk_per_trade"] == C.RISK_PER_TRADE) & (pr["cost_bps"] == C.COST_BPS_PER_SIDE) | pr["risk_per_trade"].isna())]
    parts += ["## Full portfolio metrics (1% risk, 5 bps)",
              md(t, {"pf_total_return": pct, "pf_cagr": pct, "pf_end_equity": lambda v: f"${v:,.0f}", "pf_max_dd": pct,
                     "pf_ann_vol": pct, "pf_downside_vol": pct, "pf_sharpe": num, "pf_sortino": num, "pf_calmar": num,
                     "tr_avg_win": pct, "tr_avg_loss": pct, "tr_median_ret": lambda v: pct(v, 2),
                     "tr_expectancy": lambda v: pct(v, 2), "tr_avg_hold": lambda v: num(v, 1)},
                 ["period", "model", "pf_total_return", "pf_cagr", "pf_end_equity", "pf_max_dd", "pf_ann_vol",
                  "pf_downside_vol", "pf_sharpe", "pf_sortino", "pf_calmar", "tr_avg_win", "tr_avg_loss", "tr_median_ret",
                  "tr_expectancy", "tr_avg_hold"],
                 ["Period", "Model", "Total", "CAGR", "End $", "MaxDD", "Vol", "DownVol", "Sharpe", "Sortino", "Calmar",
                  "Avg win", "Avg loss", "Median", "Expectancy", "Hold"], align_left=2), ""]

    # setup families
    sc = pd.read_csv(R / "setup_comparison.csv")
    s = sc[sc["slice"] == "ALL"].copy()
    s["family"] = s["family"].str.replace("FAM_", "", regex=False)
    s["exp_per_day"] = s["expectancy"] / s["avg_hold"].replace(0, np.nan)
    tf = {"trades": lambda v: f"{v:,.0f}", "win_rate": lambda v: pct(v, 0), "expectancy": lambda v: pct(v, 2),
          "t_stat": lambda v: num(v, 1), "profit_factor": num, "avg_hold": lambda v: num(v, 1),
          "exp_per_day": lambda v: pct(v, 3), "fwd10_excess_vs_all_days": lambda v: pct(v, 2),
          "giveback_pct_of_peak": lambda v: pct(v, 0), "pf_cagr": pct, "pf_max_dd": pct, "pf_sharpe": num,
          "avg_mfe": pct, "avg_mae": pct}
    parts += ["## Setup families (same exit: close<EMA20 → next open, 1.5 ATR stop)",
              md(s, tf, ["period", "family", "gating", "trades", "win_rate", "expectancy", "t_stat", "profit_factor",
                         "avg_hold", "exp_per_day", "fwd10_excess_vs_all_days", "pf_cagr", "pf_max_dd", "pf_sharpe"],
                 ["Period", "Family", "Gate", "Trades", "Win", "Expect.", "t", "PF", "Hold", "Exp/day",
                  "10d fwd excess", "PF CAGR", "PF MaxDD", "PF Sharpe"], align_left=3), ""]
    sl = sc[sc["slice"] != "ALL"].copy()
    sl["family"] = sl["family"].str.replace("FAM_", "", regex=False)
    piv = sl.pivot_table(index=["family", "slice", "slice_value"], columns="period", values=["trades", "expectancy"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    piv["slice"] = piv["slice"].str.replace("reg_", "", regex=False)
    parts += ["## Setup family × regime (ungated trades)",
              md(piv, {"expectancy_dev": lambda v: pct(v, 2), "expectancy_oos": lambda v: pct(v, 2),
                       "trades_dev": lambda v: "–" if not np.isfinite(v) else f"{v:,.0f}",
                       "trades_oos": lambda v: "–" if not np.isfinite(v) else f"{v:,.0f}"},
                 ["family", "slice", "slice_value", "trades_dev", "expectancy_dev", "trades_oos", "expectancy_oos"],
                 ["Family", "Regime", "State", "Dev n", "Dev exp.", "OOS n", "OOS exp."], align_left=3), ""]

    # ladder / ablation
    ab = pd.read_csv(R / "ablation_results.csv")
    ab["exp_per_day"] = ab["expectancy"] / ab["avg_hold"].replace(0, np.nan)
    for exp, title in (("ladder", "Strategy ladder (Section 51)"), ("ablation_of_A_full_hypothesis", "Ablation of the full hypothesis model"),
                       ("ablation_of_FINAL", "Ablation / add-back around FINAL")):
        d = ab[ab["experiment"].str.startswith(exp)].copy()
        piv = d.pivot_table(index="step", columns="period", values=["trades", "expectancy", "t_stat", "exp_per_day",
                                                                    "fwd10_excess_vs_all_days", "pf_sharpe", "pf_max_dd"],
                            sort=False)
        piv.columns = [f"{a}_{b}" for a, b in piv.columns]
        piv = piv.reset_index()
        cols = ["step"] + [f"{m}_{p}" for p in ("dev", "oos") for m in ("trades", "expectancy", "t_stat", "exp_per_day",
                                                                          "fwd10_excess_vs_all_days", "pf_sharpe", "pf_max_dd")]
        f2 = {c: (lambda v: f"{v:,.0f}") if c.startswith("trades") else
              (lambda v: pct(v, 2)) if c.startswith("expectancy") or c.startswith("fwd10") else
              (lambda v: pct(v, 3)) if c.startswith("exp_per_day") else
              (lambda v: pct(v, 1)) if c.startswith("pf_max_dd") else (lambda v: num(v, 2)) for c in cols}
        parts += [f"## {title}",
                  md(piv, f2, cols, ["Step"] + [f"{p} {m}" for p in ("Dev", "OOS") for m in
                                                ("n", "exp.", "t", "exp/day", "10d excess", "PF Sharpe", "PF MaxDD")]), ""]

    # sensitivity
    ps = pd.read_csv(R / "parameter_sensitivity.csv")
    ps["exp_per_day"] = ps["expectancy"] / ps["avg_hold"].replace(0, np.nan)
    piv = ps.pivot_table(index=["param", "value"], columns="period",
                         values=["trades", "expectancy", "t_stat", "exp_per_day", "giveback_pct_of_peak", "avg_mae", "pf_sharpe"], sort=False)
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    fz = ps.groupby(["param", "value"], sort=False)["is_frozen_value"].first().reset_index()
    piv = piv.merge(fz, on=["param", "value"])
    piv["value"] = np.where(piv["is_frozen_value"], "★ " + piv["value"].astype(str), piv["value"].astype(str))
    cols = ["param", "value"] + [f"{m}_{p}" for p in ("dev", "oos") for m in
                                 ("trades", "expectancy", "t_stat", "exp_per_day", "giveback_pct_of_peak", "avg_mae", "pf_sharpe")]
    f3 = {c: (lambda v: f"{v:,.0f}") if c.startswith("trades") else (lambda v: pct(v, 2)) if c.startswith("expectancy") else
          (lambda v: pct(v, 3)) if c.startswith("exp_per_day") else (lambda v: pct(v, 0)) if c.startswith("giveback") else
          (lambda v: pct(v, 1)) if c.startswith("avg_mae") else (lambda v: num(v, 2)) for c in cols}
    parts += ["## Parameter sensitivity (★ = frozen value)",
              md(piv, f3, cols, ["Param", "Value"] + [f"{p} {m}" for p in ("Dev", "OOS") for m in
                                                      ("n", "exp.", "t", "exp/day", "giveback", "MAE", "PF Sharpe")], align_left=2), ""]

    # regime
    rr = pd.read_csv(R / "regime_results.csv")
    for model in ("FINAL", "A_full_hypothesis", "C_breakout"):
        d = rr[rr["model"] == model]
        piv = d.pivot_table(index=["dimension", "value"], columns="period",
                            values=["trades", "expectancy", "profit_factor", "all_days_fwd10", "share_of_days"], sort=False)
        piv.columns = [f"{a}_{b}" for a, b in piv.columns]
        piv = piv.reset_index()
        cols = ["dimension", "value"] + [f"{m}_{p}" for p in ("dev", "oos") for m in
                                         ("share_of_days", "trades", "expectancy", "profit_factor", "all_days_fwd10")]
        f4 = {c: (lambda v: "–" if not np.isfinite(v) else f"{v:,.0f}") if c.startswith("trades") else
              (lambda v: pct(v, 0)) if c.startswith("share") else
              (lambda v: pct(v, 2)) if c.startswith("expectancy") or c.startswith("all_days") else (lambda v: num(v, 2))
              for c in cols}
        parts += [f"## Regime-conditional results — {model} (ungated trades)",
                  md(piv, f4, cols, ["Dimension", "State"] + [f"{p} {m}" for p in ("Dev", "OOS") for m in
                                                              ("% days", "n", "exp.", "PF", "any-day 10d fwd")], align_left=2), ""]

    # per ticker summary
    pts = pd.read_csv(R / "per_ticker_summary.csv")
    parts += ["## Breadth of results across the 100 stocks ($10K per stock, fully invested per signal)",
              md(pts, {c: (lambda v: pct(v, 0)) for c in pts.columns if c.startswith("pct_") or c.endswith("_return")},
                 ["period", "model", "tickers_traded", "pct_profitable", "pct_positive_expectancy", "pct_beat_bh_return",
                  "pct_better_sharpe", "pct_smaller_dd", "p25_ticker_return", "median_ticker_return", "p75_ticker_return",
                  "median_bh_return"],
                 ["Period", "Model", "Traded", "% profitable", "% +expectancy", "% beat B&H", "% better Sharpe",
                  "% smaller DD", "P25 return", "Median", "P75", "Median B&H"], align_left=2), ""]
    # volatility / sector
    for fn, title in (("volatility_results.csv", "By stock volatility"), ("sector_results.csv", "By sector bucket")):
        d = pd.read_csv(R / fn)
        d = d[d["model"] == "FINAL"]
        piv = d.pivot_table(index=["grouping", "group"], columns="period",
                            values=["trades", "expectancy", "profit_factor", "median_ticker_cagr", "median_ticker_max_dd",
                                    "median_bh_cagr"], sort=False)
        piv.columns = [f"{a}_{b}" for a, b in piv.columns]
        piv = piv.reset_index()
        cols = ["grouping", "group"] + [f"{m}_{p}" for p in ("dev", "oos") for m in
                                        ("trades", "expectancy", "profit_factor", "median_ticker_cagr", "median_ticker_max_dd",
                                         "median_bh_cagr") if f"{m}_{p}" in piv]
        f5 = {c: (lambda v: "–" if not np.isfinite(v) else f"{v:,.0f}") if c.startswith("trades") else
              (lambda v: pct(v, 2)) if c.startswith("expectancy") else
              (lambda v: num(v, 2)) if c.startswith("profit") else (lambda v: pct(v, 1)) for c in cols}
        parts += [f"## {title} — FINAL", md(piv, f5, cols, [c.replace("_", " ") for c in cols], align_left=2), ""]
    conc = pd.read_csv(R / "concentration_results.csv")
    parts += ["## Profit concentration (per-stock $10K strategy P&L)",
              md(conc, {"share_of_net_profit": lambda v: pct(v, 0), "share_of_gross_profit": lambda v: pct(v, 0),
                        "net_profit_usd": lambda v: f"${v:,.0f}", "net_profit_ex_top_k_usd": lambda v: f"${v:,.0f}"},
                 ["period", "model", "top_k", "share_of_net_profit", "share_of_gross_profit", "net_profit_usd",
                  "net_profit_ex_top_k_usd", "top_tickers"], align_left=2), ""]
    wl = pd.read_csv(R / "winners_losers_analysis.csv")
    d = wl[(wl["model"] == "FINAL") & wl["q1_expectancy"].notna()]
    piv = d.pivot_table(index="feature", columns="period",
                        values=["mean_winners", "mean_losers", "q1_expectancy", "q5_expectancy", "q5_minus_q1"], sort=False)
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    cols = ["feature"] + [f"{m}_{p}" for p in ("dev", "oos") for m in ("mean_winners", "mean_losers", "q1_expectancy",
                                                                         "q5_expectancy", "q5_minus_q1")]
    f6 = {c: (lambda v: pct(v, 2)) if ("expectancy" in c or "q5_minus" in c) else (lambda v: f"{v:.4g}") for c in cols}
    parts += ["## Winners vs losers — FINAL (quintile edges fixed on dev trades)",
              md(d.assign() if False else piv, f6, cols, ["Feature"] + [f"{p} {m}" for p in ("Dev", "OOS") for m in
                                                                     ("mean W", "mean L", "Q1 exp", "Q5 exp", "Q5−Q1")]), ""]
    cs = json.loads((C.ROOT / "qc" / "case_studies.json").read_text()) if (C.ROOT / "qc" / "case_studies.json").exists() else None
    if cs:
        parts += ["## Case-study picks", "```json", json.dumps(cs, indent=1, default=str)[:6000], "```"]
    (C.ROOT / "qc" / "report_tables.md").write_text("\n".join(parts))
    print("wrote qc/report_tables.md")


if __name__ == "__main__":
    main()
