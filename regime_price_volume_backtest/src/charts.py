"""
Step 6: charts for REPORT.md (static PNG, light theme).

Palette = the validated reference categorical order (blue, orange, aqua,
yellow, magenta, green, violet, red) used in fixed slot order; text in ink
tokens; hairline recessive grids; 2px lines; legends whenever >= 2 series;
no dual axes (volume and drawdown get their own panels).

Also selects and draws the Section-55 case studies (qc/case_studies.json).

Run:  python -m src.charts
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402
from . import data as D  # noqa: E402
from .features import compute_features  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b3"
POS, NEG = "#2a78d6", "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "font.family": "DejaVu Sans",
})
OUT = C.CHART_DIR


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _pct(ax: plt.Axes, axis: str = "y", decimals: int = 1) -> None:
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def equity_charts(period: str, series: List[tuple]) -> None:
    """Equity (indexed to $50K) and drawdown panels for the given (label, file) series."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for k, (label, fname) in enumerate(series):
        p = C.DERIVED_DIR / fname
        if not p.exists():
            continue
        eq = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
        col = SERIES[k]
        a1.plot(eq.index, eq.values, color=col, label=label, lw=2 if k == 0 else 1.6)
        if k in (0, 3, 4):  # strategy lines converge at the end; label one of them plus the benchmarks
            a1.annotate(f"${eq.iloc[-1]/1000:,.0f}K", (eq.index[-1], eq.iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK2)
        dd = eq / eq.cummax() - 1
        a2.plot(dd.index, dd.values, color=col, lw=1.4 if k == 0 else 1.1)
    a1.set_title(f"Portfolio equity, $50K start — {'2023 → 2026-09 out-of-sample' if period == 'oos' else '2020–2022 development'}")
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v/1000:,.0f}K"))
    a1.legend(loc="upper left", ncol=2, fontsize=9)
    a2.set_title("Drawdown", fontsize=10)
    _pct(a2, decimals=0)
    a2.xaxis.set_major_locator(mdates.YearLocator())
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save(fig, f"equity_{period}.png")


def regime_timeline() -> None:
    """SPY vs SMA200, VIX and breadth as three aligned panels with SPY<SMA200 shading."""
    reg = pd.read_csv(C.DERIVED_DIR / "regime.csv.gz", index_col="date", parse_dates=["date"])
    end = json.loads((C.META_DIR / "data_end.json").read_text())["data_end"]
    reg = reg.loc["2019-10-01":end]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [1.4, 1, 1]})
    bear = ~reg["spy_above_200"].astype(bool)
    for ax in axes:
        ax.fill_between(reg.index, 0, 1, where=bear, transform=ax.get_xaxis_transform(), color="#f0efec", lw=0)
        ax.axvline(pd.Timestamp(C.OOS_START), color=INK2, lw=1)
    axes[0].plot(reg.index, reg["spy_close"], color=SERIES[0], label="SPY (adjusted)")
    axes[0].plot(reg.index, reg["spy_sma200"], color=SERIES[1], lw=1.4, label="SMA200")
    axes[0].set_title("Market regime layers (grey bands: SPY below its 200-day SMA; vertical line: start of out-of-sample)")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[1].plot(reg.index, reg["vix"], color=SERIES[0], lw=1.3)
    axes[1].axhline(25, color=INK2, lw=0.8)
    axes[1].set_ylabel("VIX")
    shock = reg["vol_regime"] == "SHOCK"
    axes[1].scatter(reg.index[shock], reg["vix"][shock], s=6, color=SERIES[7], zorder=3, label="SHOCK regime day")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[2].plot(reg.index, reg["pct_above_50"], color=SERIES[0], lw=1.2)
    for y in (40, 60):
        axes[2].axhline(y, color=INK2, lw=0.8)
    axes[2].set_ylabel("% S&P > SMA50")
    axes[2].set_ylim(0, 100)
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save(fig, "regime_timeline.png")


def paired_bars(df: pd.DataFrame, label_col: str, value_col: str, title: str, fname: str,
                xlabel: str = "Expectancy per trade (net of 5 bps/side)", ref: Optional[float] = None) -> None:
    """Horizontal bars, dev vs OOS, one row per label."""
    labels = list(dict.fromkeys(df[label_col]))
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(labels) + 1.4))
    y = np.arange(len(labels))
    h = 0.36
    for k, p in enumerate(("dev", "oos")):
        vals = [df[(df[label_col] == lab) & (df["period"] == p)][value_col].mean() for lab in labels]
        ax.barh(y + (k - 0.5) * h * 1.08, vals, height=h, color=SERIES[k],
                label="2020–22 development" if p == "dev" else "2023–26 out-of-sample")
    ax.axvline(0, color=INK2, lw=1)
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=1, ls=(0, (1, 2)))
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    _pct(ax, "x", 2)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", visible=False)
    _save(fig, fname)


def per_ticker_scatter(pt: pd.DataFrame) -> None:
    """Per-stock OOS strategy total return vs buy-and-hold of the same stock."""
    d = pt[(pt["period"] == "oos") & (pt["model"] == "FINAL")]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(d["bh_total_return"], d["strat_total_return"], s=36, color=SERIES[0], edgecolor=SURFACE, lw=1.5)
    lim = [min(d["bh_total_return"].min(), d["strat_total_return"].min()) - 0.1,
           max(d["bh_total_return"].max(), d["strat_total_return"].max()) + 0.1]
    ax.plot(lim, lim, color=INK2, lw=1)
    for _, r in d.nlargest(4, "strat_total_return").iterrows():
        ax.annotate(r["ticker"], (r["bh_total_return"], r["strat_total_return"]), xytext=(4, 2),
                    textcoords="offset points", fontsize=8, color=INK2)
    for _, r in d.nlargest(3, "bh_total_return").iterrows():
        ax.annotate(r["ticker"], (r["bh_total_return"], r["strat_total_return"]), xytext=(4, -8),
                    textcoords="offset points", fontsize=8, color=INK2)
    _pct(ax, "x", 0)
    _pct(ax, "y", 0)
    ax.set_xlabel("Buy-and-hold total return (2023 → 2026-09)")
    ax.set_ylabel("FINAL strategy total return, $10K fully invested per signal")
    ax.set_title("Per-stock: strategy vs buy-and-hold (line = equal)")
    _save(fig, "per_ticker_vs_buyhold_oos.png")


def mfe_mae(tr: pd.DataFrame) -> None:
    """MAE vs realised return and MFE vs realised return for FINAL OOS trades."""
    d = tr[(tr["model"] == "FINAL") & (tr["period"] == "oos")]
    win = d["ret"] > 0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, col, title in ((axes[0], "mae", "MAE vs realised return"), (axes[1], "mfe", "MFE vs realised return")):
        ax.scatter(d.loc[~win, col], d.loc[~win, "ret"], s=9, color=SERIES[7], alpha=0.5, lw=0, label="losers")
        ax.scatter(d.loc[win, col], d.loc[win, "ret"], s=9, color=SERIES[0], alpha=0.5, lw=0, label="winners")
        ax.axhline(0, color=INK2, lw=1)
        _pct(ax, "x", 0)
        _pct(ax, "y", 0)
        ax.set_xlabel(col.upper() + " (intraday, vs entry)")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("Realised return")
    _save(fig, "mfe_mae_oos.png")


def regime_conditional(rr: pd.DataFrame) -> None:
    """FINAL expectancy by SPY-trend x VIX state vs the all-stock-day forward return in that state."""
    d = rr[(rr["model"] == "FINAL") & (rr["dimension"] == "mkt_x_vix")]
    labels = sorted(set(d["value"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, p in zip(axes, ("dev", "oos")):
        sub = d[d["period"] == p].set_index("value").reindex(labels)
        y = np.arange(len(labels))
        ax.barh(y - 0.19, sub["expectancy"], height=0.36, color=SERIES[0], label="FINAL trade expectancy")
        ax.barh(y + 0.19, sub["all_days_fwd10"], height=0.36, color=SERIES[1], label="any stock-day, next 10 sessions")
        for yi, n in zip(y, sub["trades"]):
            if pd.notna(n):
                ax.annotate(f"n={int(n)}", (0, yi - 0.19), xytext=(-4, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=7, color=INK2)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(y, labels)
        _pct(ax, "x", 1)
        ax.set_title("2020–22 development" if p == "dev" else "2023–26 out-of-sample", fontsize=11)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Conditional expectancy by market trend × volatility state", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    _save(fig, "regime_conditional.png")


def sensitivity_chart(ps: pd.DataFrame) -> None:
    """Small multiples: expectancy per parameter value, dev vs OOS, frozen value marked."""
    params = list(dict.fromkeys(ps["param"]))
    n = len(params)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.1 * rows))
    for ax, prm in zip(axes.flat, params):
        sub = ps[ps["param"] == prm]
        vals = list(dict.fromkeys(sub["value"]))
        x = np.arange(len(vals))
        for k, p in enumerate(("dev", "oos")):
            v = [sub[(sub["value"] == val) & (sub["period"] == p)]["expectancy"].mean() for val in vals]
            ax.bar(x + (k - 0.5) * 0.38, v, width=0.36, color=SERIES[k],
                   label="dev" if p == "dev" else "OOS")
        frozen = sub[sub["is_frozen_value"]]["value"].unique()
        ax.set_xticks(x, [("★ " if val in frozen else "") + str(val)[:16] for val in vals], rotation=30,
                      ha="right", fontsize=7.5)
        ax.axhline(0, color=INK2, lw=1)
        _pct(ax, "y", 1)
        ax.set_title(prm, fontsize=10)
        ax.grid(axis="x", visible=False)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Parameter sensitivity around FINAL (★ = frozen value; expectancy per trade)", x=0.01,
                 ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "parameter_sensitivity.png")


def _candles(ax: plt.Axes, f: pd.DataFrame) -> None:
    x = mdates.date2num(f.index.to_pydatetime())
    w = 0.6
    up = f["close"] >= f["open"]
    for xi, (o, h, l, c), u in zip(x, f[["open", "high", "low", "close"]].to_numpy(), up):
        col = SERIES[0] if u else SERIES[7]
        ax.vlines(xi, l, h, color=col, lw=0.8)
        ax.bar(xi, max(abs(c - o), 1e-9), bottom=min(o, c), width=w, color=col if not u else SURFACE,
               edgecolor=col, lw=0.8)


def case_chart(tkr: str, row: pd.Series, title: str, fname: str) -> None:
    """Candlestick chart for one trade with EMA9/EMA20/SMA50, signal/entry/exit/stop and a volume panel."""
    df = D.load_cached(tkr, C.RAW_DIR)
    f = compute_features(df)
    s, x = pd.Timestamp(row["signal_date"]), pd.Timestamp(row["exit_date"])
    i0 = max(0, f.index.get_loc(s) - 45)
    i1 = min(len(f) - 1, f.index.get_loc(x) + 12)
    w = f.iloc[i0:i1 + 1]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.5, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    _candles(a1, w)
    a1.plot(w.index, w["ema9"], color=SERIES[2], lw=1.3, label="EMA9")
    a1.plot(w.index, w["ema20"], color=SERIES[1], lw=1.6, label="EMA20")
    a1.plot(w.index, w["sma50"], color=SERIES[6], lw=1.3, label="SMA50")
    a1.hlines(row["stop_px"], pd.Timestamp(row["entry_date"]), x, color=SERIES[7], lw=1, ls=(0, (3, 2)),
              label="initial stop")
    a1.scatter([s], [f.loc[s, "low"] * 0.985], marker="^", s=70, color=INK, zorder=4, label="signal (close)")
    a1.scatter([pd.Timestamp(row["entry_date"])], [row["entry_px"]], marker="o", s=55, color=SERIES[0],
               edgecolor=SURFACE, lw=2, zorder=5, label="entry (next open)")
    a1.scatter([x], [row["exit_px"]], marker="X", s=70, color=SERIES[7], edgecolor=SURFACE, lw=1.5, zorder=5,
               label=f"exit ({row['exit_reason']})")
    a1.set_title(title, fontsize=11)
    a1.legend(loc="upper left", fontsize=8, ncol=4)
    a2.bar(w.index, w["volume"], color=[SERIES[0] if u else SERIES[7] for u in (w["close"] >= w["open"])], width=0.7)
    a2.plot(w.index, w["vol_avg20_prev"], color=INK2, lw=1)
    a2.set_ylabel("volume")
    a2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%Y"))
    _save(fig, fname)


def case_studies(tr: pd.DataFrame) -> List[Dict[str, object]]:
    """Pick representative trades by rule (not by eye) and chart them."""
    fin = tr[tr["model"] == "FINAL"]
    oos = fin[fin["period"] == "oos"].reset_index(drop=True)
    picks: List[Dict[str, object]] = []

    def nearest(d: pd.DataFrame, q: float) -> pd.Series:
        target = d["ret"].quantile(q)
        return d.iloc[(d["ret"] - target).abs().argmin()]

    wins, losses = oos[oos["ret"] > 0], oos[oos["ret"] <= 0]
    picks.append(("strong_winner", "Strong winner (OOS trade nearest the 97.5th return percentile)", nearest(oos, 0.975)))
    picks.append(("normal_winner", "Normal winner (median OOS winning trade)", nearest(wins, 0.5)))
    picks.append(("false_signal_loser", "False signal (median OOS losing trade)", nearest(losses, 0.5)))
    # regime filter cases: FINAL trades the SPY>SMA200 gate would have blocked
    blocked = fin[~fin["reg_spy_above_200"].astype(bool)]
    b_src = blocked[blocked["period"] == "oos"] if (blocked["period"] == "oos").sum() >= 10 else blocked
    picks.append(("regime_prevented_loss", "SPY<SMA200 gate would have blocked this loser (median blocked loser)",
                  nearest(b_src[b_src["ret"] <= 0], 0.5)))
    picks.append(("regime_missed_winner", "SPY<SMA200 gate would have blocked this winner (median blocked winner)",
                  nearest(b_src[b_src["ret"] > 0], 0.5)))
    out = []
    for key, title, row in picks:
        t = (f"{title}\n{row['ticker']}: signal {str(row['signal_date'])[:10]}, "
             f"entry {row['entry_px']:.2f} → exit {row['exit_px']:.2f} ({row['ret']:+.1%}), "
             f"MFE {row['mfe']:+.1%}, MAE {row['mae']:+.1%}, held {int(row['hold_days'])} sessions")
        case_chart(row["ticker"], row, t, f"case_{key}.png")
        out.append({"case": key, "title": title, "ticker": row["ticker"], "period": row["period"],
                    "signal_date": str(row["signal_date"])[:10], "entry_date": str(row["entry_date"])[:10],
                    "exit_date": str(row["exit_date"])[:10], "entry_px": row["entry_px"], "stop_px": row["stop_px"],
                    "exit_px": row["exit_px"], "exit_reason": row["exit_reason"], "ret": row["ret"],
                    "mfe": row["mfe"], "mae": row["mae"], "peak_close_ret": row["peak_close_ret"],
                    "hold_days": int(row["hold_days"]), "sig_ext": row["sig_ext"], "sig_atr_pct": row["sig_atr_pct"],
                    "sig_vol_ratio": row["sig_vol_ratio"], "sig_clv": row["sig_clv"],
                    "sig_low_to_ema20": row["sig_low_to_ema20"], "reg_market_state": row["reg_market_state"],
                    "reg_vix": row["reg_vix"], "reg_vol_regime": row["reg_vol_regime"],
                    "reg_pct_above_50": row["reg_pct_above_50"], "blocked_pool": "oos" if b_src is not blocked else "all periods"})
    agg = {}
    for p in ("dev", "oos"):
        bp = blocked[blocked["period"] == p]
        agg[p] = {"blocked_trades": int(len(bp)), "blocked_win_rate": float((bp["ret"] > 0).mean()) if len(bp) else None,
                  "blocked_expectancy": float(bp["ret"].mean()) if len(bp) else None,
                  "blocked_sum_of_returns": float(bp["ret"].sum()) if len(bp) else 0.0}
    (C.ROOT / "qc" / "case_studies.json").write_text(json.dumps({"cases": out, "blocked_by_spy_gate": agg},
                                                                 indent=2, default=str))
    return out


def main() -> None:
    """Draw every report chart."""
    regime_timeline()
    for p in ("dev", "oos"):
        equity_charts(p, [("FINAL (no regime gate)", f"equity_{p}_FINAL.csv"),
                          ("FINAL + SPY>SMA200 gate", f"equity_{p}_SPY_only.csv"),
                          ("Random uptrend-day entry, same exits", f"equity_{p}_REF_random_uptrend_day.csv"),
                          ("SPY buy & hold", f"equity_{p}_BENCH_SPY.csv"),
                          ("Equal-weight 100-stock buy & hold", f"equity_{p}_BENCH_EW100.csv")])
    sc = pd.read_csv(C.OUT_DIR / "setup_comparison.csv")
    sc_all = sc[(sc["slice"] == "ALL") & (sc["gating"] == "ungated")].copy()
    sc_all["label"] = sc_all["family"].str.replace("FAM_", "", regex=False)
    paired_bars(sc_all, "label", "expectancy", "Setup families, same exit (close<EMA20, 1.5 ATR stop), no regime gate",
                "setup_comparison.png")
    sc_all["fwd10_excess_vs_all_days"] = sc_all["fwd10_excess_vs_all_days"]
    paired_bars(sc_all, "label", "fwd10_excess_vs_all_days",
                "Entry edge only: 10-session forward return minus the any-stock-day average",
                "setup_fwd10_excess.png", xlabel="Excess 10-session forward return per signal")
    ab = pd.read_csv(C.OUT_DIR / "ablation_results.csv")
    lad = ab[ab["experiment"].str.startswith("ladder")].copy()
    paired_bars(lad, "step", "expectancy", "Strategy ladder S0–S5 (no location) and A0–A5 (pullback location)",
                "ladder.png")
    af = ab[ab["experiment"] == "ablation_of_FINAL"].copy()
    ref = af[(af["step"] == "FINAL") & (af["period"] == "oos")]["expectancy"].mean()
    paired_bars(af, "step", "expectancy", "FINAL: remove / add one component (dotted line = FINAL OOS)",
                "ablation_final.png", ref=ref)
    ah = ab[ab["experiment"] == "ablation_of_A_full_hypothesis"].copy()
    paired_bars(ah, "step", "expectancy", "Full hypothesis stack: remove one component at a time",
                "ablation_hypothesis.png")
    rr = pd.read_csv(C.OUT_DIR / "regime_results.csv")
    regime_conditional(rr)
    pt = pd.read_csv(C.OUT_DIR / "per_ticker_results.csv")
    per_ticker_scatter(pt)
    tr = pd.read_csv(C.OUT_DIR / "trades.csv", parse_dates=["signal_date", "entry_date", "exit_date"])
    mfe_mae(tr)
    ps = pd.read_csv(C.OUT_DIR / "parameter_sensitivity.csv")
    sensitivity_chart(ps)
    vr = pd.read_csv(C.OUT_DIR / "volatility_results.csv")
    v = vr[(vr["model"] == "FINAL") & (vr["grouping"] == "vol_bucket")].copy()
    v["group"] = pd.Categorical(v["group"], ["Low", "Medium", "High", "Very High"], ordered=True)
    paired_bars(v.sort_values("group"), "group", "expectancy", "FINAL by stock volatility bucket (2019 ATR% quartile)",
                "by_volatility.png")
    s = pd.read_csv(C.OUT_DIR / "sector_results.csv")
    s = s[(s["model"] == "FINAL")]
    order = s[s["period"] == "oos"].sort_values("expectancy", ascending=False)["group"].tolist()
    s["group"] = pd.Categorical(s["group"], order, ordered=True)
    paired_bars(s.sort_values("group"), "group", "expectancy", "FINAL by sector bucket", "by_sector.png")
    case_studies(tr)


if __name__ == "__main__":
    main()
