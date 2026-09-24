# Regime-gated price/volume + candlestick study (100 stocks, daily, 2020 → 2026-09)

Research package, fully separate from the live bots. Nothing in here places orders or imports
live-trading logic (the only repo import is the optional Alpaca data client).

**Start with [`REPORT.md`](REPORT.md).** This file explains how to reproduce it.

## Pipeline

| Step | Command | Output |
|---|---|---|
| 1. Download & cache | `python -m src.download` | `data/raw/` (SPY, QQQ, ^VIX), `data/breadth_raw/` (~590 point-in-time S&P members, gitignored) |
| 2. Universe, validation, regime | `python -m src.build_data` | `universe_100.csv`, `data/raw/<100 stocks>`, `data/derived/regime.csv.gz`, `data/meta/*` |
| 3. Look-ahead test | `python -m src.lookahead_check` | `qc/lookahead_check.csv` |
| 4. Development research (2020–22) | `python -m src.run_dev` then `python -m src.run_dev_stage2` | `dev_results/*.csv` |
| 5. **Freeze** | `python -m src.freeze` → commit | `frozen_models.json` (commit `eccc073`, made before any OOS run) |
| 6. Frozen evaluation, dev + OOS | `python -m src.run_oos` | `portfolio_results.csv`, `setup_comparison.csv`, `ablation_results.csv`, `parameter_sensitivity.csv`, `regime_results.csv`, `per_ticker_results.csv`, `per_ticker_summary.csv`, `winners_losers_analysis.csv`, `volatility_results.csv`, `sector_results.csv`, `concentration_results.csv`, `trades.csv`, `data/derived/equity_*.csv` |
| 7. Manual audit | `python -m src.audit` | `qc/audit_report.md` |
| 8. Charts + case studies | `python -m src.charts` | `charts/*.png`, `qc/case_studies.json` |
| 9. Report tables | `python -m src.report_tables` | `qc/report_tables.md` (every REPORT.md table, generated from the CSVs) |

Steps 2–8 run offline from the committed cache. Only step 1 (and the breadth part of step 2)
needs the network. Dependencies: `pandas numpy matplotlib requests lxml pyarrow`. Total runtime is
about 45 minutes on one core, most of it the random-entry controls in steps 4 and 6.

## Data provenance

* **Prices:** adjusted daily OHLCV. The research brief preferred Alpaca, but this cloud session had
  no `APCA_API_KEY_ID`/`APCA_API_SECRET_KEY`, so bars came from Yahoo's chart API. `src/data.py`
  switches to Alpaca (`adjustment=all`, SIP feed, via the repo's `AlpacaHttpClient`) automatically
  when those env vars are set. Yahoo's split-adjusted OHLC is scaled by `adjclose/close`, so every
  bar is split- **and** dividend-adjusted, the same convention as Alpaca `adjustment=all`. The
  as-traded close is rebuilt from split events and used only for the 2019 "price > $10" filter.
* **VIX:** Yahoo `^VIX` close. A FRED `VIXCLS` cross-check is attempted but FRED timed out in this session.
* **Membership:** point-in-time S&P 500 constituents from
  [fja05680/sp500](https://github.com/fja05680/sp500) (`data/meta/sp500_hist.csv.gz`). GICS
  classifications come from the Wikipedia constituents page (snapshot in `data/meta/`), plus a
  manual table for 48 names that have since left the index.
* **End date:** 2026-09-21, the last session where SPY, QQQ, VIX and all 100 stocks have a bar.
  Yahoo returned null bars for most stocks on 2026-09-22 (`data/meta/data_end.json`).

## Key conventions

* Signal at the close of day T, fill at the **open of T+1**. Exits on a close-based rule fill at the
  next open. Stops fill intraday at the stop, or at the open on a gap through it. If the entry open
  is already below the stop, the trade is skipped.
* Dev trades are force-closed on 2022-12-30, so no 2023 price ever touches a development result.
* Costs: 5 bps per side. `portfolio_results.csv` also has a frictionless run and a 15 bps/side run, which stands in for stop-fill slippage.
* Portfolio: $50K, 1% risk per trade (0.5% / 1.5% variants), 25% single-name cap, 2% minimum
  position, 100% gross cap (50% in REDUCED state). When capital is short, signals are taken in a
  seeded random order, and every portfolio figure is a **mean over 10 seeds** (5 in the dev screens).
* Sharpe/Sortino use a 0% risk-free rate. Money math is float64. This is research code that never
  places orders; the repo's `Decimal` rule applies to live P&L code.

## QC fixes made after the freeze (model definitions unchanged)

* **Breadth guard for spin-offs.** An independent code review found that the reused-ticker guard
  (`membership.data_usable_for`) dropped spin-offs that joined the index on their first trading day,
  such as CARR, OTIS, GEHC and CEG. The fix changes breadth by at most 0.78 pp and flips the
  breadth≥40% gate on 4 of 1,688 sessions. It does not change the universe (same sha256) or FINAL,
  which has no breadth gate. `dev_results/` were produced before the fix and are kept as the
  historical evidence for the freeze. `run_oos.py` recomputes dev and OOS with the corrected breadth.
* The look-ahead test now also covers QQQ, VIX ROC and breadth.

## Layout

```
src/            config, data, membership, universe, features, setups, regime, engine,
                portfolio, metrics, research, run_dev(_stage2), freeze, run_oos, audit, charts
data/raw/       cached adjusted bars for the 100 stocks + SPY/QQQ/^VIX (committed)
data/derived/   regime frame, equity curves
data/meta/      membership, GICS snapshot, validation, download status, data_end
dev_results/    every development-period table (the evidence behind the freeze)
qc/             look-ahead test, manual audit, case-study picks
charts/         report figures
```
