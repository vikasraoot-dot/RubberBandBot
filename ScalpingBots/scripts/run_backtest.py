#!/usr/bin/env python3
"""
Unified backtester for all scalping strategies.
Fetches data with caching, runs all strategies, and produces consolidated reports.

Usage:
  python ScalpingBots/scripts/run_backtest.py --strategies vwap,orb,gap --days 30 --timeframe 5m
  python ScalpingBots/scripts/run_backtest.py --strategies all --days 60 --symbols AAPL,TSLA,NVDA
  python ScalpingBots/scripts/run_backtest.py --strategies vwap --days 30 --top 20
"""
import os
import sys
import json
import argparse
import datetime as dt
from typing import List, Dict

import pandas as pd
import numpy as np

# Setup paths
_THIS = os.path.abspath(os.path.dirname(__file__))
_PROJECT = os.path.abspath(os.path.join(_THIS, ".."))
_REPO = os.path.abspath(os.path.join(_PROJECT, ".."))
for p in [_PROJECT, _REPO]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ScalpingBots.src.data_cache import get_bars, bulk_fetch, get_cache_stats
from ScalpingBots.src.indicators import (
    add_vwap, add_ema, add_rsi, add_atr, add_bollinger, add_keltner,
    add_macd, add_volume_profile, add_rvol, add_adx, add_squeeze,
    add_opening_range, add_gap, add_stochastic,
)
from ScalpingBots.strategies.vwap_bounce import (
    backtest as vwap_backtest, VWAPConfig,
)
from ScalpingBots.strategies.orb_breakout import (
    backtest as orb_backtest, ORBConfig,
)
from ScalpingBots.strategies.gap_fill import (
    backtest as gap_backtest, GapFillConfig,
)
from ScalpingBots.strategies.ema_momentum import (
    backtest as ema_backtest, EMAMomentumConfig,
)


# Default top-30 high-liquidity tickers
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "AMZN", "GOOGL",
    "NFLX", "AVGO", "CRM", "ORCL", "PLTR", "COIN", "HOOD", "SOFI",
    "BA", "JPM", "GS", "BAC", "XOM", "CVX", "DIS", "NKE",
    "UBER", "PYPL", "SQ", "SHOP", "CRWD", "SNOW",
]


def load_tickers(tickers_file: str = None) -> List[str]:
    """Load tickers from file or return defaults."""
    if tickers_file and os.path.exists(tickers_file):
        tickers = []
        with open(tickers_file) as f:
            for line in f:
                t = line.strip().upper()
                if t and not t.startswith("#"):
                    tickers.append(t)
        return tickers
    return DEFAULT_TICKERS


def prepare_data(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Add all required indicators for a strategy."""
    if df is None or df.empty or len(df) < 20:
        return pd.DataFrame()

    # Common indicators
    df = add_rsi(df, 14)
    df = add_atr(df, 14)
    df = add_rvol(df, 20)
    df = add_adx(df, 14)
    df = add_ema(df, 9, name="ema_9")
    df = add_ema(df, 21, name="ema_21")

    if strategy in ("vwap", "all"):
        df = add_vwap(df)
        df = add_bollinger(df)
        df = add_keltner(df)

    if strategy in ("orb", "all"):
        df = add_opening_range(df, minutes=15)
        df = add_gap(df)

    if strategy in ("gap", "all"):
        df = add_gap(df)

    if strategy in ("ema", "all"):
        if "vwap" not in df.columns:
            df = add_vwap(df)

    # Drop NaN rows at the beginning (indicator warmup)
    df = df.dropna(subset=["rsi", "atr"]).copy()
    return df


def run_strategy(strategy: str, df: pd.DataFrame, symbol: str,
                 start_cash: float, verbose: bool) -> Dict:
    """Run a single strategy and return results."""
    if strategy == "vwap":
        cfg = VWAPConfig()
        return vwap_backtest(df, cfg, start_cash, symbol, verbose)
    elif strategy == "orb":
        cfg = ORBConfig()
        return orb_backtest(df, cfg, start_cash, symbol, verbose)
    elif strategy == "gap":
        cfg = GapFillConfig()
        return gap_backtest(df, cfg, start_cash, symbol, verbose)
    elif strategy == "ema":
        cfg = EMAMomentumConfig()
        return ema_backtest(df, cfg, start_cash, symbol, verbose)
    else:
        return {"trades": [], "summary": {"symbol": symbol, "total_trades": 0}}


def print_summary_table(all_results: List[Dict], strategy: str):
    """Print formatted summary table."""
    if not all_results:
        print(f"\n  No results for {strategy}")
        return

    summaries = [r["summary"] for r in all_results if r["summary"]["total_trades"] > 0]
    if not summaries:
        print(f"\n  No trades generated for {strategy}")
        return

    df = pd.DataFrame(summaries)

    # Sort by net P&L
    df = df.sort_values("net_pnl", ascending=False)

    print(f"\n{'='*80}")
    print(f"  STRATEGY: {strategy.upper()}")
    print(f"{'='*80}")

    # Per-symbol results
    cols = ["symbol", "total_trades", "win_rate", "net_pnl", "profit_factor",
            "max_drawdown_pct", "daily_win_rate", "avg_bars_held"]
    available_cols = [c for c in cols if c in df.columns]
    print(df[available_cols].to_string(index=False))

    # Aggregate stats
    total_trades = df["total_trades"].sum()
    total_pnl = df["net_pnl"].sum()
    total_wins = df["wins"].sum()
    total_losses = df["losses"].sum()
    agg_wr = round(total_wins / max(total_trades, 1) * 100, 1)
    pos_days = df["positive_days"].sum()
    neg_days = df["negative_days"].sum()
    tot_days = df["total_days"].max()  # All symbols share same date range
    daily_wr = round(pos_days / max(pos_days + neg_days, 1) * 100, 1)

    print(f"\n  AGGREGATE: {total_trades} trades | WR: {agg_wr}% | Net P&L: ${total_pnl:,.2f}")
    print(f"  Wins: {total_wins} | Losses: {total_losses}")
    if pos_days + neg_days > 0:
        print(f"  Daily: {pos_days} green / {neg_days} red = {daily_wr}% daily WR")

    # Exit reason breakdown
    all_reasons = {}
    all_reason_pnl = {}
    for s in summaries:
        for k, v in s.get("reason_counts", {}).items():
            all_reasons[k] = all_reasons.get(k, 0) + v
        for k, v in s.get("reason_pnl", {}).items():
            all_reason_pnl[k] = all_reason_pnl.get(k, 0) + v

    if all_reasons:
        print(f"\n  Exit Reasons:")
        for reason in sorted(all_reasons.keys()):
            count = all_reasons[reason]
            pnl = all_reason_pnl.get(reason, 0)
            print(f"    {reason:12s}: {count:4d} trades, P&L: ${pnl:>8,.2f}")


def print_portfolio_summary(portfolio_results: Dict[str, List[Dict]], start_cash: float):
    """Print combined portfolio summary across all strategies."""
    print(f"\n{'='*80}")
    print(f"  PORTFOLIO SUMMARY (Combined Strategies)")
    print(f"{'='*80}")

    total_trades = 0
    total_pnl = 0.0
    total_wins = 0
    total_losses = 0
    all_daily_pnl = {}

    for strategy, results in portfolio_results.items():
        strat_trades = sum(r["summary"]["total_trades"] for r in results)
        strat_pnl = sum(r["summary"]["net_pnl"] for r in results)
        strat_wins = sum(r["summary"]["wins"] for r in results)
        strat_losses = sum(r["summary"]["losses"] for r in results)
        strat_wr = round(strat_wins / max(strat_trades, 1) * 100, 1)

        print(f"\n  {strategy.upper():8s}: {strat_trades:4d} trades | WR: {strat_wr}% | P&L: ${strat_pnl:>10,.2f}")

        total_trades += strat_trades
        total_pnl += strat_pnl
        total_wins += strat_wins
        total_losses += strat_losses

        # Collect daily PnL for consistency analysis
        for r in results:
            for t in r.get("trades", []):
                if hasattr(t, 'exit_time') and t.exit_time:
                    d = t.exit_time.tz_convert("US/Eastern").date()
                    all_daily_pnl[d] = all_daily_pnl.get(d, 0) + t.pnl

    agg_wr = round(total_wins / max(total_trades, 1) * 100, 1)
    print(f"\n  {'TOTAL':8s}: {total_trades:4d} trades | WR: {agg_wr}% | P&L: ${total_pnl:>10,.2f}")
    print(f"  Starting Capital: ${start_cash:,.2f} | Return: {(total_pnl/start_cash)*100:.1f}%")

    # Daily consistency
    if all_daily_pnl:
        daily_values = sorted(all_daily_pnl.items())
        pos_days = sum(1 for _, v in daily_values if v > 0)
        neg_days = sum(1 for _, v in daily_values if v < 0)
        flat_days = sum(1 for _, v in daily_values if v == 0)
        total_days = len(daily_values)
        daily_wr = round(pos_days / max(total_days, 1) * 100, 1)

        daily_returns = [v for _, v in daily_values]
        avg_daily = np.mean(daily_returns)
        std_daily = np.std(daily_returns)
        sharpe_approx = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0

        best_day = max(daily_values, key=lambda x: x[1])
        worst_day = min(daily_values, key=lambda x: x[1])

        # Max drawdown
        running = start_cash
        peak = start_cash
        max_dd_pct = 0
        for _, pnl in daily_values:
            running += pnl
            peak = max(peak, running)
            dd = (running - peak) / peak * 100
            max_dd_pct = min(max_dd_pct, dd)

        # Consecutive losing days
        max_consec_loss = 0
        curr_consec = 0
        for _, pnl in daily_values:
            if pnl < 0:
                curr_consec += 1
                max_consec_loss = max(max_consec_loss, curr_consec)
            else:
                curr_consec = 0

        print(f"\n  DAILY CONSISTENCY:")
        print(f"    Trading Days: {total_days}")
        print(f"    Green Days: {pos_days} ({daily_wr}%)")
        print(f"    Red Days: {neg_days}")
        print(f"    Avg Daily P&L: ${avg_daily:,.2f}")
        print(f"    Best Day: {best_day[0]} (+${best_day[1]:,.2f})")
        print(f"    Worst Day: {worst_day[0]} (${worst_day[1]:,.2f})")
        print(f"    Max Drawdown: {max_dd_pct:.2f}%")
        print(f"    Max Consecutive Losses: {max_consec_loss} days")
        print(f"    Approx Sharpe Ratio: {sharpe_approx:.2f}")

        # Monthly breakdown
        monthly = {}
        for d, pnl in daily_values:
            month_key = d.strftime("%Y-%m")
            monthly[month_key] = monthly.get(month_key, 0) + pnl

        if monthly:
            print(f"\n  MONTHLY P&L:")
            for m in sorted(monthly.keys()):
                sign = "+" if monthly[m] >= 0 else ""
                print(f"    {m}: {sign}${monthly[m]:,.2f}")

            profitable_months = sum(1 for v in monthly.values() if v > 0)
            print(f"    Profitable Months: {profitable_months}/{len(monthly)}")


def save_results(portfolio_results: Dict, output_dir: str):
    """Save results to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)

    for strategy, results in portfolio_results.items():
        # Save summaries
        summaries = [r["summary"] for r in results if r["summary"]["total_trades"] > 0]
        if summaries:
            df_sum = pd.DataFrame(summaries)
            df_sum.to_csv(os.path.join(output_dir, f"{strategy}_summary.csv"), index=False)

        # Save all trades
        all_trades = []
        for r in results:
            for t in r.get("trades", []):
                trade_dict = {
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_time": str(t.entry_time),
                    "exit_time": str(t.exit_time),
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "qty": t.qty,
                    "pnl": round(t.pnl, 2),
                    "exit_reason": t.exit_reason,
                    "bars_held": t.bars_held,
                }
                all_trades.append(trade_dict)

        if all_trades:
            df_trades = pd.DataFrame(all_trades)
            df_trades.to_csv(os.path.join(output_dir, f"{strategy}_trades.csv"), index=False)

    print(f"\n  Results saved to {output_dir}/")


def main():
    ap = argparse.ArgumentParser(description="Scalping Bot Backtester")
    ap.add_argument("--strategies", default="all",
                    help="Comma-separated: vwap,orb,gap or 'all'")
    ap.add_argument("--days", type=int, default=30,
                    help="Days of history to backtest")
    ap.add_argument("--timeframe", default="5m",
                    help="Bar timeframe (1m, 5m, 15m)")
    ap.add_argument("--symbols", default="",
                    help="Comma-separated symbols (overrides tickers file)")
    ap.add_argument("--tickers", default="",
                    help="Path to tickers file")
    ap.add_argument("--top", type=int, default=0,
                    help="Use top N tickers from default list")
    ap.add_argument("--cash", type=float, default=10000.0,
                    help="Starting capital per strategy")
    ap.add_argument("--feed", default="iex",
                    help="Alpaca data feed (iex or sip)")
    ap.add_argument("--end-date", default=None,
                    help="End date YYYY-MM-DD (default: today)")
    ap.add_argument("--force-refresh", action="store_true",
                    help="Force re-fetch data from API (ignore cache)")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--output", default="ScalpingBots/results",
                    help="Output directory for results")
    args = ap.parse_args()

    # Determine symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.tickers:
        symbols = load_tickers(args.tickers)
    elif args.top > 0:
        symbols = DEFAULT_TICKERS[:args.top]
    else:
        symbols = DEFAULT_TICKERS

    # Determine strategies
    if args.strategies == "all":
        strategies = ["vwap", "orb", "gap", "ema"]
    else:
        strategies = [s.strip().lower() for s in args.strategies.split(",")]

    print(f"  Scalping Bot Backtester")
    print(f"  Strategies: {', '.join(strategies)}")
    print(f"  Symbols: {len(symbols)} ({', '.join(symbols[:5])}{'...' if len(symbols)>5 else ''})")
    print(f"  Timeframe: {args.timeframe} | Days: {args.days} | Cash: ${args.cash:,.0f}")
    print(f"  Feed: {args.feed}")

    # Show cache stats
    stats = get_cache_stats()
    print(f"  Cache: {stats['unique_symbols']} symbols, {stats['total_bars']:,} bars cached")

    # Phase 1: Fetch all data
    print(f"\n  Phase 1: Fetching data...")
    bars_map = bulk_fetch(
        symbols, args.timeframe, args.days, args.feed,
        args.end_date, rth_only=True, force_refresh=args.force_refresh,
        delay=0.15
    )

    if not bars_map:
        print("  ERROR: No data fetched. Check API keys and network.")
        return

    # Phase 2: Run strategies
    print(f"\n  Phase 2: Running backtests...")
    portfolio_results = {}

    for strategy in strategies:
        print(f"\n  --- Running {strategy.upper()} strategy ---")
        results = []

        for sym in symbols:
            raw_df = bars_map.get(sym)
            if raw_df is None or raw_df.empty:
                continue

            df = prepare_data(raw_df.copy(), strategy)
            if df.empty or len(df) < 30:
                continue

            result = run_strategy(strategy, df, sym, args.cash, args.verbose)
            results.append(result)

            if result["summary"]["total_trades"] > 0:
                s = result["summary"]
                if args.verbose:
                    print(f"    {sym}: {s['total_trades']} trades, WR={s['win_rate']}%, P&L=${s['net_pnl']:,.2f}")

        portfolio_results[strategy] = results
        print_summary_table(results, strategy)

    # Phase 3: Portfolio summary
    print_portfolio_summary(portfolio_results, args.cash)

    # Phase 4: Save results
    save_results(portfolio_results, args.output)

    # Updated cache stats
    stats = get_cache_stats()
    print(f"\n  Cache now has {stats['unique_symbols']} symbols, {stats['total_bars']:,} bars")


if __name__ == "__main__":
    main()
