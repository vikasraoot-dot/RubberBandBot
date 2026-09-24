"""
Step 4: development-period research (signals 2020-01-01 .. 2022-12-31).

Everything here is computed on the development window only; trades still open
on the last dev session are closed at that close, so no 2023+ price is used.
Outputs go to ``dev_results/``.  The researcher reviews them, writes
``frozen_models.json`` (see ``freeze.py``) and commits it *before* running
``run_oos.py``.

Run:  python -m src.run_dev
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from . import config as C
from .engine import ExitSpec, forward_returns
from .portfolio import PortfolioParams
from .research import (Model, benchmark_stats, enrich, load_regime, load_stocks, load_universe,
                       period_bounds, run_model_trades, run_portfolio, summarize)
from .setups import BASE_A, BASE_CANDLE, FAMILY_BASELINES, EntrySpec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_dev")

OUT = C.ROOT / "dev_results"
PERIOD = "dev"
BASE_EXIT = ExitSpec("B_close<EMA20", "atr1.5")
T10 = ExitSpec("T10_time", "none", horizon=10)

GATES_MARKET = ["none", "spy200", "spy200+spy50_200", "spy200+spyema20", "spy50_200", "qqq200"]
GATES_VIX = ["vix_lt20", "vix_lt25", "vix_lt30", "vix_pct_lt80", "vix_pct_lt90", "vix_le_sma",
             "vix_roc_lt20", "vix_roc_lt10", "vix_noshock", "vix_calm"]
GATES_BREADTH = ["br50_40", "br50_50", "br50_60", "br200_50"]


class Runner:
    """Caches stocks/regime and collects summary rows."""

    def __init__(self) -> None:
        self.uni = load_universe()
        self.stocks = load_stocks(self.uni["ticker"])
        self.reg = load_regime()
        self.start, self.end = period_bounds(PERIOD, self.reg)
        fwd_all = forward_returns(self.stocks, None, self.start, self.end)
        self.base_fwd10 = float(np.nanmean(fwd_all["fwd10"]))
        log.info("dev window %s..%s  all-stock-day fwd10 baseline %.4f", self.start, self.end, self.base_fwd10)

    def trades(self, m: Model) -> pd.DataFrame:
        """Sequential trade list for a model on the dev window."""
        return run_model_trades(m, self.stocks, self.reg, PERIOD)

    def row(self, m: Model, experiment: str, extra: Dict[str, object] | None = None,
            portfolio: bool = False) -> Dict[str, object]:
        """Summary row for a model (trade stats + fwd-return edge [+ portfolio stats])."""
        from .research import model_gate
        tr = self.trades(m)
        fwd = forward_returns(self.stocks, m.entry, self.start, self.end, gate=model_gate(m, self.reg))
        label = {"experiment": experiment, "model": m.name, "gate": m.state_rule or m.gate,
                 "exit": m.exit.label(), "entry": m.entry.label()}
        if extra:
            label.update(extra)
        r = summarize(tr, label, fwd, self.base_fwd10)
        if portfolio:
            pf = run_portfolio(m, self.stocks, self.reg, PERIOD)
            r.update({f"pf_{k}": v for k, v in pf["stats"].items()})
        return r


def main() -> None:
    """Run every development experiment and write CSVs."""
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    R = Runner()
    rows: Dict[str, List[Dict[str, object]]] = {k: [] for k in
                                                ("family", "ladder", "candle", "gates", "ablation",
                                                 "exits", "extension", "sensitivity")}

    # 1) Setup-family comparison ─────────────────────────────────────────────
    for fam, spec in FAMILY_BASELINES.items():
        for xs in (BASE_EXIT, T10):
            for g in ("none", "spy200"):
                m = Model(f"{fam}", spec, xs, g)
                rows["family"].append(R.row(m, "family", {"family": fam}, portfolio=(xs == BASE_EXIT)))
    # reference: trend-only every day and every day (no setup at all)
    every_day = EntrySpec(family="CANDLE", candle="none", trend="none", trig_vol=None, pb_vol=False, ext_max=None)
    trend_day = every_day.with_(trend="full")
    for nm, sp in (("REF_every_day", every_day), ("REF_trend_every_day", trend_day)):
        for g in ("none", "spy200"):
            rows["family"].append(R.row(Model(nm, sp, BASE_EXIT, g), "family", {"family": nm}))
    log.info("family done %.0fs", time.time() - t0)

    # 2) Strategy 0-5 ladder (location-free candle) and the same ladder with the pullback location
    ladder = [
        ("S0_candle_only", BASE_CANDLE, "none"),
        ("S1_candle+trend", BASE_CANDLE.with_(trend="full"), "none"),
        ("S2_candle+trend+volume", BASE_CANDLE.with_(trend="full", trig_vol=1.0), "none"),
        ("S3_+SPY_regime", BASE_CANDLE.with_(trend="full", trig_vol=1.0), "spy200"),
        ("S4_+VIX", BASE_CANDLE.with_(trend="full", trig_vol=1.0), "spy200+vix_noshock"),
        ("S5_+breadth", BASE_CANDLE.with_(trend="full", trig_vol=1.0), "spy200+vix_noshock+br50_40"),
    ]
    for nm, sp, g in ladder:
        for xs in (BASE_EXIT, T10):
            rows["ladder"].append(R.row(Model(nm, sp, xs, g), "ladder_no_location", portfolio=(xs == BASE_EXIT)))
            spA = sp.with_(family="A", pb_vol=True, ext_max=C.EXTENSION_MAX)
            rows["ladder"].append(R.row(Model(nm.replace("S", "A", 1), spA, xs, g), "ladder_with_pullback",
                                        portfolio=(xs == BASE_EXIT)))
    log.info("ladder done %.0fs", time.time() - t0)

    # 3) Candle-type comparison inside Setup A (no gate and SPY gate) ─────────
    for cdl in ("any", "engulf", "hammer", "wrb", "strong", "upclose", "none"):
        for g in ("none", "spy200"):
            for xs in (BASE_EXIT, T10):
                rows["candle"].append(R.row(Model(f"A_candle={cdl}", BASE_A.with_(candle=cdl), xs, g),
                                            "candle_type", {"candle": cdl}))
    log.info("candle done %.0fs", time.time() - t0)

    # 4) Gate research on Setup A baseline ───────────────────────────────────
    gate_list = list(GATES_MARKET)
    gate_list += [f"spy200+{v}" for v in GATES_VIX] + [f"spy200+spy50_200+{v}" for v in ("vix_noshock", "vix_calm", "vix_lt25")]
    gate_list += [f"spy200+vix_noshock+{b}" for b in GATES_BREADTH] + [f"spy200+{b}" for b in GATES_BREADTH]
    gate_list += [f"{v}" for v in ("vix_noshock", "vix_calm", "vix_lt25", "vix_roc_lt20")]
    gate_list += [f"{b}" for b in GATES_BREADTH]
    for g in gate_list:
        rows["gates"].append(R.row(Model(f"A|{g}", BASE_A, BASE_EXIT, g), "gate", portfolio=True))
    for rule in ("onoff_v1", "onoff_v2"):
        rows["gates"].append(R.row(Model(f"A|{rule}", BASE_A, BASE_EXIT, "none", state_rule=rule), "gate",
                                   portfolio=True))
    log.info("gates done %.0fs", time.time() - t0)

    # conditional expectancy tables (ungated Setup A trades, regime at signal)
    trA = enrich(R.trades(Model("A_nogate", BASE_A, BASE_EXIT, "none")), R.stocks, R.reg, R.uni)
    trA.to_csv(OUT / "dev_trades_A_nogate.csv", index=False)

    # 5) Ablation of Setup A baseline (one component removed at a time) ───────
    full_gate = "spy200+vix_noshock+br50_40"
    abl = [
        ("full", BASE_A, full_gate),
        ("no_breadth", BASE_A, "spy200+vix_noshock"),
        ("no_vix", BASE_A, "spy200+br50_40"),
        ("no_market_gate", BASE_A, "vix_noshock+br50_40"),
        ("no_regime_at_all", BASE_A, "none"),
        ("no_stock_trend", BASE_A.with_(trend="none"), full_gate),
        ("trend_above200_only", BASE_A.with_(trend="above200"), full_gate),
        ("no_trigger_volume", BASE_A.with_(trig_vol=None), full_gate),
        ("no_pullback_volume", BASE_A.with_(pb_vol=False), full_gate),
        ("no_candle", BASE_A.with_(candle="none"), full_gate),
        ("candle->upclose", BASE_A.with_(candle="upclose"), full_gate),
        ("no_pullback_location", BASE_A.with_(family="CANDLE"), full_gate),
        ("no_extension_filter", BASE_A.with_(ext_max=None), full_gate),
        ("bare_pullback_trend_only", BASE_A.with_(candle="none", trig_vol=None, pb_vol=False, ext_max=None), "none"),
    ]
    for nm, sp, g in abl:
        for xs in (BASE_EXIT, T10):
            rows["ablation"].append(R.row(Model(f"abl_{nm}", sp, xs, g), "ablation", {"removed": nm},
                                          portfolio=(xs == BASE_EXIT)))
    log.info("ablation done %.0fs", time.time() - t0)

    # 6) Exit x stop grid on Setup A (SPY gate) ───────────────────────────────
    for rule in ("A_close<EMA9", "A2_2closes<EMA9", "W_EMA9warn+confirm", "B_close<EMA20", "C_EMA9xEMA20",
                 "D_swinglow", "E_pricevol", "F_chandelier"):
        for stop in ("none", "signal_low", "swing_low", "atr1.0", "atr1.5", "atr2.0"):
            if rule == "F_chandelier" and stop == "none":
                continue
            xs = ExitSpec(rule, stop)
            rows["exits"].append(R.row(Model(f"exit_{rule}_{stop}", BASE_A, xs, "spy200"), "exit",
                                       {"exit_rule": rule, "stop": stop}, portfolio=(stop in ("atr1.5", "none", "signal_low"))))
    for h in C.FIXED_HORIZONS:
        rows["exits"].append(R.row(Model(f"exit_T{h}", BASE_A, ExitSpec("T10_time", "none", horizon=h), "spy200"),
                                   "exit", {"exit_rule": f"T{h}_time", "stop": "none"}))
    log.info("exits done %.0fs", time.time() - t0)

    # 7) Extension filter research ────────────────────────────────────────────
    for ext in (None, 0.03, 0.05, 0.07):
        rows["extension"].append(R.row(Model(f"ext_pct={ext}", BASE_A.with_(ext_max=ext), BASE_EXIT, "spy200"),
                                       "extension", {"ext_max": ext, "ext_atr_max": None}, portfolio=True))
    for ea in (1.0, 1.5, 2.0):
        rows["extension"].append(R.row(Model(f"ext_atr={ea}", BASE_A.with_(ext_max=None, ext_atr_max=ea), BASE_EXIT,
                                             "spy200"), "extension", {"ext_max": None, "ext_atr_max": ea}, portfolio=True))
    # extension on the location-free candle strategy (where chasing is possible)
    for ext in (None, 0.03, 0.05, 0.07):
        sp = BASE_CANDLE.with_(trend="full", trig_vol=1.0, ext_max=ext)
        rows["extension"].append(R.row(Model(f"candle_ext_pct={ext}", sp, BASE_EXIT, "spy200"), "extension_candle",
                                       {"ext_max": ext}))
    log.info("extension done %.0fs", time.time() - t0)

    # 8) Parameter sensitivity (one at a time around Setup A baseline) ─────────
    sens = []
    for d in (0.01, 0.02, 0.03):
        sens.append(("pb_dist", d, BASE_A.with_(pb_dist=d), BASE_EXIT))
    sens.append(("pb_mode", "touch", BASE_A.with_(pb_mode="touch"), BASE_EXIT))
    for v in (1.0, 1.2, 1.5):
        sens.append(("trig_vol", v, BASE_A.with_(trig_vol=v), BASE_EXIT))
    for clv in (0.70, 0.75, 0.80):
        sens.append(("strong_close_clv", clv, BASE_A.with_(candle="strong", clv_min=clv), BASE_EXIT))
    for pbv in (0.8, 0.9, 1.0):
        sens.append(("pb_vol_max", pbv, BASE_A.with_(pb_vol_max=pbv), BASE_EXIT))
    for k in ("atr1.0", "atr1.5", "atr2.0"):
        sens.append(("atr_stop", k, BASE_A, ExitSpec("B_close<EMA20", k)))
    for name, val, sp, xs in sens:
        rows["sensitivity"].append(R.row(Model(f"sens_{name}={val}", sp, xs, "spy200"), "sensitivity",
                                         {"param": name, "value": val}))
    log.info("sensitivity done %.0fs", time.time() - t0)

    for k, v in rows.items():
        pd.DataFrame(v).to_csv(OUT / f"dev_{k}.csv", index=False)
    bm = benchmark_stats(R.reg, PERIOD)
    pd.DataFrame({k: v["stats"] for k, v in bm.items()}).T.to_csv(OUT / "dev_benchmarks.csv")
    log.info("all done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
