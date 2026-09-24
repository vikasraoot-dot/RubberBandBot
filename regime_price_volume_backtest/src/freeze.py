"""
Step 4c: FREEZE the development-period decisions into ``frozen_models.json``.

This file is edited once, after reviewing ``dev_results/`` (2020-2022 only) and
BEFORE ``run_oos.py`` is run for the first time.  It is committed to git
together with the dev results so the freeze is auditable.

Dev-period decisions and the evidence behind them (see dev_results/*.csv):

1. Entry.  Stage-1 ablation removed every component of the a-priori Setup A
   one at a time; none of candle shape, trigger volume, pullback-volume
   contraction, extension cap, VIX, breadth or SPY gate improved dev
   expectancy.  The bare "uptrend + pullback to EMA20" rule had the best
   expectancy (+0.36%/trade, t=2.8, 2,542 trades) -> FINAL entry = A_simple.
2. Exit.  Pre-declared rule (max seed-averaged dev portfolio Sharpe with trade
   PF > 1.15) picked close<EMA20 with a 1.5 ATR initial stop (Sharpe 0.99),
   which is also the a-priori baseline exit -> no exit tuning.
3. Regime gate for FINAL.  Pre-declared keep rule (dSharpe >= +0.10, max DD
   not worse, expectancy not lower).  Every market / VIX / breadth / ON-OFF
   gate lowered dev Sharpe and expectancy -> FINAL has NO regime gate.
4. Setup C (contraction breakout) is frozen as a secondary candidate: its
   rule-chosen exit is close < prior 10-day low with a 1.5 ATR stop; SPY>SMA200
   passed the keep rule (Sharpe 0.81 -> 1.00, DD -15.0% -> -11.1%,
   expectancy 1.72% -> 1.74%); SMA50>SMA200 (+0.09), VIX (+0.02) and breadth
   (+0.04) layers did not.
5. The layered SPY / +VIX / +breadth / ON-REDUCED-OFF rows keep their
   a-priori definitions (spy200, vix_noshock, br50_40, onoff_v1) because no
   dev alternative beat them by >= 0.10 Sharpe.
6. The literal research hypothesis (Setup A with every filter + SPY + VIX +
   breadth) is frozen UNCHANGED so the out-of-sample test covers it.

Run:  python -m src.freeze
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from typing import Dict, List

from . import config as C
from .engine import ExitSpec
from .research import Model
from .setups import BASE_A, BASE_B, BASE_B_EMA, BASE_C, BASE_CANDLE, EntrySpec

# ─── Dev-period choices ──────────────────────────────────────────────────────
A_SIMPLE = EntrySpec(family="A", candle="none", trend="full", trig_vol=None, pb_vol=False, ext_max=None)
FINAL_EXIT = ExitSpec("B_close<EMA20", "atr1.5")
FINAL_GATE = "none"
C_EXIT = ExitSpec("D_swinglow", "atr1.5")
C_GATE = "spy200"
SPY_GATE = "spy200"
VIX_LAYER = "vix_noshock"
BREADTH_LAYER = "br50_40"
ONOFF_RULE = "onoff_v1"
HYPOTHESIS_GATE = f"{SPY_GATE}+{VIX_LAYER}+{BREADTH_LAYER}"
REF_EVERY = EntrySpec(family="CANDLE", candle="none", trend="none", trig_vol=None, pb_vol=False, ext_max=None)
REF_TREND = REF_EVERY.with_(trend="full")


def _m(name: str, entry: EntrySpec, xs: ExitSpec = FINAL_EXIT, gate: str = "none",
       state_rule: str | None = None) -> Dict[str, object]:
    return Model(name, entry, xs, gate, state_rule).to_json()


def build() -> Dict[str, object]:
    """Assemble the frozen specification."""
    models = {
        "FINAL": _m("FINAL", A_SIMPLE, FINAL_EXIT, FINAL_GATE),
        "SPY_only": _m("SPY_only", A_SIMPLE, FINAL_EXIT, SPY_GATE),
        "SPY_VIX": _m("SPY_VIX", A_SIMPLE, FINAL_EXIT, f"{SPY_GATE}+{VIX_LAYER}"),
        "SPY_VIX_BREADTH": _m("SPY_VIX_BREADTH", A_SIMPLE, FINAL_EXIT, HYPOTHESIS_GATE),
        "ONOFF": _m("ONOFF", A_SIMPLE, FINAL_EXIT, "none", ONOFF_RULE),
        "A_full_hypothesis": _m("A_full_hypothesis", BASE_A, FINAL_EXIT, HYPOTHESIS_GATE),
        "C_breakout": _m("C_breakout", BASE_C, C_EXIT, C_GATE),
        "B_failed_breakdown": _m("B_failed_breakdown", BASE_B, FINAL_EXIT, "none"),
        "REF_random_uptrend_day": _m("REF_random_uptrend_day", REF_TREND, FINAL_EXIT, "none"),
        "REF_random_every_day": _m("REF_random_every_day", REF_EVERY, FINAL_EXIT, "none"),
        # like-for-like family comparison: every family with the same exit
        "FAM_A_simple": _m("FAM_A_simple", A_SIMPLE),
        "FAM_A_full_candle_volume": _m("FAM_A_full_candle_volume", BASE_A),
        "FAM_B_failed_breakdown_swing": _m("FAM_B_failed_breakdown_swing", BASE_B),
        "FAM_B_reclaim_ema20": _m("FAM_B_reclaim_ema20", BASE_B_EMA),
        "FAM_C_contraction_breakout": _m("FAM_C_contraction_breakout", BASE_C),
        "FAM_REF_random_uptrend_day": _m("FAM_REF_random_uptrend_day", REF_TREND),
    }

    ladder: List[Dict[str, object]] = []
    steps = [("S0_candle_only", BASE_CANDLE, "none"),
             ("S1_+stock_trend", BASE_CANDLE.with_(trend="full"), "none"),
             ("S2_+volume", BASE_CANDLE.with_(trend="full", trig_vol=1.0), "none"),
             ("S3_+SPY_regime", BASE_CANDLE.with_(trend="full", trig_vol=1.0), SPY_GATE),
             ("S4_+VIX", BASE_CANDLE.with_(trend="full", trig_vol=1.0), f"{SPY_GATE}+{VIX_LAYER}"),
             ("S5_+breadth", BASE_CANDLE.with_(trend="full", trig_vol=1.0), HYPOTHESIS_GATE)]
    for nm, sp, g in steps:
        ladder.append({"name": nm, "experiment": "ladder_no_location", "model": _m(nm, sp, FINAL_EXIT, g)})
        spa = sp.with_(family="A", pb_vol=True, ext_max=C.EXTENSION_MAX)
        nma = nm.replace("S", "A", 1)
        ladder.append({"name": nma, "experiment": "ladder_with_pullback", "model": _m(nma, spa, FINAL_EXIT, g)})
    ladder.append({"name": "REF_uptrend_every_day", "experiment": "ladder_reference", "model": models["REF_random_uptrend_day"]})
    ladder.append({"name": "REF_every_day", "experiment": "ladder_reference", "model": models["REF_random_every_day"]})

    abl_h = [("full", BASE_A, HYPOTHESIS_GATE), ("no_breadth", BASE_A, f"{SPY_GATE}+{VIX_LAYER}"),
             ("no_vix", BASE_A, f"{SPY_GATE}+{BREADTH_LAYER}"), ("no_market_gate", BASE_A, f"{VIX_LAYER}+{BREADTH_LAYER}"),
             ("no_regime_at_all", BASE_A, "none"), ("no_stock_trend", BASE_A.with_(trend="none"), HYPOTHESIS_GATE),
             ("no_trigger_volume", BASE_A.with_(trig_vol=None), HYPOTHESIS_GATE),
             ("no_pullback_volume", BASE_A.with_(pb_vol=False), HYPOTHESIS_GATE),
             ("no_candle", BASE_A.with_(candle="none"), HYPOTHESIS_GATE),
             ("candle->plain_up_close", BASE_A.with_(candle="upclose"), HYPOTHESIS_GATE),
             ("no_pullback_location", BASE_A.with_(family="CANDLE"), HYPOTHESIS_GATE),
             ("no_extension_filter", BASE_A.with_(ext_max=None), HYPOTHESIS_GATE)]
    abl_f = [("FINAL", A_SIMPLE, "none"),
             ("remove_stock_trend", A_SIMPLE.with_(trend="none"), "none"),
             ("trend_only_close>SMA200", A_SIMPLE.with_(trend="above200"), "none"),
             ("remove_pullback_location(=random uptrend day)", REF_TREND, "none"),
             ("add_any_bullish_candle", A_SIMPLE.with_(candle="any"), "none"),
             ("add_strong_close_candle", A_SIMPLE.with_(candle="strong"), "none"),
             ("add_engulfing_candle", A_SIMPLE.with_(candle="engulf"), "none"),
             ("add_plain_up_close", A_SIMPLE.with_(candle="upclose"), "none"),
             ("add_trigger_volume>=1.0", A_SIMPLE.with_(trig_vol=1.0), "none"),
             ("add_trigger_volume>=1.5", A_SIMPLE.with_(trig_vol=1.5), "none"),
             ("add_pullback_volume_contraction", A_SIMPLE.with_(pb_vol=True), "none"),
             ("add_extension<=5%", A_SIMPLE.with_(ext_max=0.05), "none"),
             ("add_SPY>SMA200", A_SIMPLE, SPY_GATE),
             ("add_SPY_SMA50>SMA200", A_SIMPLE, "spy200+spy50_200"),
             ("add_VIX_noshock", A_SIMPLE, VIX_LAYER),
             ("add_breadth50>=40", A_SIMPLE, BREADTH_LAYER)]
    ablation = {
        "A_full_hypothesis": [{"removed": nm, "model": _m(f"ablH_{nm}", sp, FINAL_EXIT, g)} for nm, sp, g in abl_h],
        "FINAL": [{"removed": nm, "model": _m(f"ablF_{nm}", sp, FINAL_EXIT, g)} for nm, sp, g in abl_f],
    }

    sens: List[Dict[str, object]] = []

    def add(param: str, value: object, entry: EntrySpec, xs: ExitSpec = FINAL_EXIT, frozen: bool = False) -> None:
        sens.append({"param": param, "value": value, "is_frozen_value": frozen,
                     "model": _m(f"sens_{param}={value}", entry, xs, FINAL_GATE)})

    for d in (0.01, 0.02, 0.03):
        add("pullback_dist", d, A_SIMPLE.with_(pb_dist=d), frozen=(d == 0.02))
    add("pullback_mode", "touch(low<=EMA20<close)", A_SIMPLE.with_(pb_mode="touch"))
    for tmode in ("above200", "above200_50_200", "full"):
        add("stock_trend", tmode, A_SIMPLE.with_(trend=tmode), frozen=(tmode == "full"))
    for stop in ("none", "atr1.0", "atr1.5", "atr2.0", "signal_low", "swing_low"):
        add("initial_stop", stop, A_SIMPLE, ExitSpec("B_close<EMA20", stop), frozen=(stop == "atr1.5"))
    for rule in ("A_close<EMA9", "A2_2closes<EMA9", "W_EMA9warn+confirm", "B_close<EMA20", "C_EMA9xEMA20",
                 "D_swinglow", "E_pricevol", "F_chandelier"):
        add("exit_rule", rule, A_SIMPLE, ExitSpec(rule, "atr1.5" if rule != "F_chandelier" else "atr2.0"),
            frozen=(rule == "B_close<EMA20"))
    for h in (10, 20):
        add("exit_rule", f"T{h}_time", A_SIMPLE, ExitSpec("T10_time", "none", horizon=h))
    for v in (None, 1.0, 1.2, 1.5):
        add("trigger_volume", v, A_SIMPLE.with_(trig_vol=v), frozen=(v is None))
    for clv in (0.70, 0.75, 0.80):
        add("strong_close_clv(added)", clv, A_SIMPLE.with_(candle="strong", clv_min=clv))
    for ext in (None, 0.03, 0.05, 0.07):
        add("extension_cap", ext, A_SIMPLE.with_(ext_max=ext), frozen=(ext is None))
    for ea in (1.0, 1.5, 2.0):
        add("extension_cap_atr", ea, A_SIMPLE.with_(ext_atr_max=ea))

    return {
        "protocol": {
            "dev_period": [C.BACKTEST_START, C.DEV_END],
            "oos_period": [C.OOS_START, "latest complete session (data/meta/data_end.json)"],
            "selection_rules": {
                "exit": "max seed-averaged dev portfolio Sharpe among exits with trade PF > 1.15; ties (<0.10) to simpler",
                "regime_layer_keep": "dSharpe >= +0.10 AND max DD not worse AND expectancy not lower vs model without it",
            },
            "note": "Nothing below may be changed after the first OOS run.",
        },
        "models": models,
        "final": "FINAL",
        "compare": "SPY_only",
        "setup_gate": SPY_GATE,
        "top_table": ["FINAL", "SPY_only", "SPY_VIX", "SPY_VIX_BREADTH", "ONOFF", "A_full_hypothesis",
                      "C_breakout", "B_failed_breakdown", "REF_random_uptrend_day", "REF_random_every_day"],
        "onoff_model": "ONOFF",
        "combined": {"name": "COMBINED_C_breakout+FINAL", "members": ["C_breakout", "FINAL"]},
        "families": ["FAM_A_simple", "FAM_A_full_candle_volume", "FAM_B_failed_breakdown_swing",
                     "FAM_B_reclaim_ema20", "FAM_C_contraction_breakout", "FAM_REF_random_uptrend_day"],
        "ladder": ladder,
        "ablation": ablation,
        "sensitivity": sens,
        "regime_models": ["FINAL", "A_full_hypothesis", "C_breakout"],
        "winner_loser_models": ["FINAL", "A_full_hypothesis"],
        "trade_export_models": ["FINAL", "SPY_only", "A_full_hypothesis", "C_breakout"],
    }


def main() -> None:
    """Write frozen_models.json with provenance (git HEAD, dev-result hashes, timestamp)."""
    spec = build()
    hashes = {}
    for p in sorted((C.ROOT / "dev_results").glob("*.csv")):
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=C.ROOT).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        head = "unknown"
    spec["provenance"] = {"frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                          "git_head_before_freeze": head, "dev_result_sha256_16": hashes}
    (C.OUT_DIR / "frozen_models.json").write_text(json.dumps(spec, indent=2, default=str))
    print(f"frozen {len(spec['models'])} models, {len(spec['ladder'])} ladder rungs, "
          f"{sum(len(v) for v in spec['ablation'].values())} ablations, {len(spec['sensitivity'])} sensitivity variants")


if __name__ == "__main__":
    main()
