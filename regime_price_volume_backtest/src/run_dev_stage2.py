"""
Step 4b: development-period stage 2 (still 2020-2022 only).

Stage 1 (``run_dev.py``) showed that none of the candle / volume / extension /
regime components improved the a-priori Setup A, and that slower trend exits
beat fast ones.  Stage 2 therefore:

1. Compares exits on the *simplified* pullback entry ("A_simple": stock trend +
   pullback to EMA20, nothing else), on Setup C, and on the a-priori full
   Setup A, with portfolio statistics averaged over 5 priority seeds.
2. Adds controls: random entry with the same exits/sizing ("every day" and
   "every uptrend day") and an equal-weight buy-and-hold of the universe.
3. Re-runs the regime-gate menu on the chosen entry+exit with seed-averaged
   portfolio statistics, and tabulates conditional expectancy by regime.

Pre-declared selection rules (applied by the researcher when writing
``frozen_models.json``):
* Exit: highest seed-averaged dev portfolio Sharpe among exits whose trade
  profit factor > 1.15; ties (< 0.10 Sharpe) go to the simpler rule.
* A regime layer is kept only if, versus the model without it, it raises the
  seed-averaged dev Sharpe by >= 0.10 AND does not worsen max drawdown AND
  does not lower per-trade expectancy.  Otherwise the layer is dropped.

Run:  python -m src.run_dev_stage2
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from . import config as C
from .engine import ExitSpec, forward_returns
from .metrics import equity_stats
from .research import (Model, benchmark_stats, enrich, equal_weight_universe_equity, model_gate,
                       portfolio_seeds, summarize)
from .run_dev import OUT, PERIOD, Runner
from .setups import BASE_A, BASE_C, EntrySpec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_dev_stage2")

A_SIMPLE = EntrySpec(family="A", candle="none", trend="full", trig_vol=None, pb_vol=False, ext_max=None)
REF_EVERY = EntrySpec(family="CANDLE", candle="none", trend="none", trig_vol=None, pb_vol=False, ext_max=None)
REF_TREND = REF_EVERY.with_(trend="full")
ENTRIES = {"A_simple": A_SIMPLE, "C_breakout": BASE_C, "A_full_hypothesis": BASE_A,
           "REF_random_every_day": REF_EVERY, "REF_random_uptrend_day": REF_TREND}
EXITS = [ExitSpec("B_close<EMA20", "atr1.5"), ExitSpec("A_close<EMA9", "atr1.5"),
         ExitSpec("W_EMA9warn+confirm", "atr1.5"), ExitSpec("E_pricevol", "atr1.5"),
         ExitSpec("C_EMA9xEMA20", "none"), ExitSpec("C_EMA9xEMA20", "atr1.5"), ExitSpec("C_EMA9xEMA20", "atr2.0"),
         ExitSpec("D_swinglow", "none"), ExitSpec("D_swinglow", "atr1.5"), ExitSpec("D_swinglow", "atr2.0"),
         ExitSpec("D_swinglow", "swing_low"), ExitSpec("F_chandelier", "atr2.0"),
         ExitSpec("T10_time", "none", horizon=10), ExitSpec("T10_time", "none", horizon=20)]
SEEDS = range(5)


def prow(R: Runner, m: Model, experiment: str, extra: Dict[str, object] | None = None,
         seeds=SEEDS) -> Dict[str, object]:
    """Trade stats + fwd edge + seed-averaged portfolio stats."""
    r = R.row(m, experiment, extra)
    ps = portfolio_seeds(m, R.stocks, R.reg, PERIOD, seeds=seeds)
    r.update({f"pf_{k}": v for k, v in ps["mean"].items()})
    r["pf_min_sharpe"], r["pf_max_sharpe"] = ps["min_sharpe"], ps["max_sharpe"]
    return r


def main() -> None:
    """Run stage-2 dev experiments."""
    t0 = time.time()
    R = Runner()
    rows: Dict[str, List[Dict[str, object]]] = {"s2_exits": [], "s2_gates": [], "s2_gates_C": []}

    # 1) exits x entries (seed-averaged portfolio) ─────────────────────────────
    for en, spec in ENTRIES.items():
        for xs in EXITS:
            rows["s2_exits"].append(prow(R, Model(f"{en}", spec, xs, "none"), "s2_exit",
                                         {"entry_name": en, "exit_rule": xs.rule, "stop": xs.stop,
                                          "horizon": xs.horizon}))
        log.info("exits %s done %.0fs", en, time.time() - t0)
    ex = pd.DataFrame(rows["s2_exits"])
    ex.to_csv(OUT / "dev_s2_exits.csv", index=False)

    # benchmarks incl. equal-weight universe
    bm = benchmark_stats(R.reg, PERIOD)
    ew = equity_stats(equal_weight_universe_equity(R.stocks, R.reg, PERIOD))
    bmt = pd.DataFrame({k: v["stats"] for k, v in bm.items()}).T
    bmt.loc["EW_universe_100"] = pd.Series(ew)
    bmt.to_csv(OUT / "dev_s2_benchmarks.csv")

    # 2) choose the exit per entry by the pre-declared rule (logged; researcher confirms)
    chosen = {}
    for en in ("A_simple", "C_breakout"):
        sub = ex[(ex["entry_name"] == en) & (ex["profit_factor"] > 1.15)].sort_values("pf_sharpe", ascending=False)
        chosen[en] = sub.iloc[0] if len(sub) else ex[ex["entry_name"] == en].sort_values("pf_sharpe").iloc[-1]
        log.info("rule-based exit for %s: %s/%s (Sharpe %.2f, PF %.2f)", en, chosen[en]["exit_rule"],
                 chosen[en]["stop"], chosen[en]["pf_sharpe"], chosen[en]["profit_factor"])

    def _xs(row: pd.Series) -> ExitSpec:
        h = row["horizon"]
        return ExitSpec(row["exit_rule"], row["stop"], None if pd.isna(h) else int(h))

    # 3) gate menu on the chosen A_simple and C models ─────────────────────────
    gates = ["none", "spy200", "spy50_200", "spy200+spy50_200", "spy200+spyema20", "qqq200",
             "spy200+vix_lt25", "spy200+vix_lt30", "spy200+vix_pct_lt80", "spy200+vix_le_sma",
             "spy200+vix_roc_lt20", "spy200+vix_noshock", "spy200+vix_calm",
             "spy200+spy50_200+vix_noshock", "spy200+spy50_200+vix_lt30",
             "spy200+vix_noshock+br50_40", "spy200+vix_noshock+br50_50", "spy200+vix_noshock+br50_60",
             "spy200+vix_noshock+br200_50", "spy200+br50_50", "vix_noshock", "vix_lt30", "vix_roc_lt20",
             "br50_40", "br50_50"]
    for en, key in (("A_simple", "s2_gates"), ("C_breakout", "s2_gates_C")):
        spec, xs = ENTRIES[en], _xs(chosen[en])
        for g in gates:
            rows[key].append(prow(R, Model(f"{en}|{g}", spec, xs, g), "s2_gate", {"entry_name": en}))
        for rule in ("onoff_v1", "onoff_v2"):
            rows[key].append(prow(R, Model(f"{en}|{rule}", spec, xs, "none", state_rule=rule), "s2_gate",
                                  {"entry_name": en}))
        pd.DataFrame(rows[key]).to_csv(OUT / f"dev_{key}.csv", index=False)
        log.info("gates %s done %.0fs", en, time.time() - t0)

    # 4) conditional expectancy by regime for the ungated chosen models ─────────
    for en in ("A_simple", "C_breakout"):
        m = Model(en, ENTRIES[en], _xs(chosen[en]), "none")
        tr = enrich(R.trades(m), R.stocks, R.reg, R.uni)
        tr.to_csv(OUT / f"dev_s2_trades_{en}_nogate.csv", index=False)
    log.info("stage 2 done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
