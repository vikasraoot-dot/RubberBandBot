#!/usr/bin/env python3
"""
Unified Ticker Scanner for RubberBandBot

Scans the ticker universe for optimal candidates for each bot type:
- 15M_STK: 15-minute stock bot
- 15M_OPT: 15-minute options spread bot
- WK_STK: Weekly stock bot
- WK_OPT: Weekly options bot

Usage:
    python RubberBand/scripts/scan_for_bot.py --bot-type 15M_OPT --tickers tickers_full_list.txt
    python RubberBand/scripts/scan_for_bot.py --bot-type ALL --output scan_results.csv
"""
from __future__ import annotations

import os
import sys
import argparse
import pandas as pd
from typing import Dict, List, Any, Optional
import requests

# Ensure repo root is on path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.data import fetch_latest_bars, load_symbols_from_file


# ──────────────────────────────────────────────────────────────────────────────
# Bot Profiles (Default Criteria)
# ──────────────────────────────────────────────────────────────────────────────
BOT_PROFILES = {
    "15M_STK": {
        "description": "15-Minute Stock Bot - Quick mean reversion on liquid stocks",
        "price_min": 20,
        "price_max": 1000,
        "atr_pct_min": 1.5,
        "atr_pct_max": 5.0,
        "dollar_vol_min": 20_000_000,  # $20M daily (Raised from $5M to improve quality)
        "sma_period": 0,  # 0 = Disabled (Mean Reversion catches falling knives)
        "require_options": False,
    },
    "15M_OPT": {
        "description": "15-Minute Options Spread Bot - Bull call spreads on oversold bounces",
        "price_min": 50,
        "price_max": 500,
        "atr_pct_min": 2.0,
        "atr_pct_max": 6.0,
        "dollar_vol_min": 20_000_000,  # $20M daily
        "sma_period": 100, # 100-day SMA for trend following (required for options)
        "require_options": True,
    },
    "WK_STK": {
        "description": "Weekly Stock Bot - Swing trades on weekly timeframe",
        "price_min": 40,
        "price_max": 1000,
        "atr_pct_min": 4.0,
        "atr_pct_max": 15.0,
        "dollar_vol_min": 50_000_000,  # $50M daily
        "sma_period": 0,  # 0 = Disabled (Allow falling knives for mean reversion)
        "require_options": False,
    },
    "WK_OPT": {
        "description": "Weekly Options Bot - Deep ITM calls on stable uptrends",
        "price_min": 50,
        "price_max": 400,
        "atr_pct_min": 2.5,
        "atr_pct_max": 5.0,
        "dollar_vol_min": 30_000_000,  # $30M daily
        "sma_period": 0,  # 0 = Disabled (Allow falling knives for mean reversion)
        "require_options": True,
    },
}


def get_profile(bot_type: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get bot profile with optional overrides."""
    if bot_type not in BOT_PROFILES:
        raise ValueError(f"Unknown bot type: {bot_type}. Valid: {list(BOT_PROFILES.keys())}")
    
    profile = BOT_PROFILES[bot_type].copy()
    if overrides:
        for key, value in overrides.items():
            if value is not None and key in profile:
                profile[key] = value
    return profile


# ──────────────────────────────────────────────────────────────────────────────
# Indicator Calculation
# ──────────────────────────────────────────────────────────────────────────────
def calculate_indicators(df: pd.DataFrame, sma_period: int = 20) -> Dict[str, Any]:
    """
    Calculate indicators for screening.
    
    Returns dict with: price, atr_14, atr_pct, dollar_vol, sma, in_uptrend
    """
    if len(df) < max(sma_period if sma_period > 0 else 20, 20):
        return {}
    
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    
    current_price = float(close.iloc[-1])
    if current_price <= 0:
        return {}
    
    # ATR-14
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = float(tr.rolling(window=14).mean().iloc[-1])
    
    # ATR% = ATR / Price * 100
    atr_pct = (atr_14 / current_price) * 100.0
    
    # Dollar Volume (20-day average daily)
    dollar_vol = float((close * volume).rolling(window=20).mean().iloc[-1])
    
    # SMA for trend filter
    if sma_period > 0:
        sma = float(close.rolling(window=sma_period).mean().iloc[-1])
        in_uptrend = current_price > sma
    else:
        sma = 0.0
        in_uptrend = True # Bypass filter
    
    return {
        "price": round(current_price, 2),
        "atr_14": round(atr_14, 2),
        "atr_pct": round(atr_pct, 2),
        "dollar_vol": dollar_vol,
        "dollar_vol_m": round(dollar_vol / 1_000_000, 2),
        "sma": round(sma, 2),
        "sma_period": sma_period,
        "in_uptrend": in_uptrend,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Options Availability Check
# ──────────────────────────────────────────────────────────────────────────────
def check_options_available(symbol: str) -> bool:
    """
    Check if a symbol has options available via Alpaca API.
    
    Returns True if weekly options are available.
    """
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    key = os.getenv("APCA_API_KEY_ID", "") or os.getenv("ALPACA_KEY_ID", "")
    secret = os.getenv("APCA_API_SECRET_KEY", "") or os.getenv("ALPACA_SECRET_KEY", "")
    
    if not key or not secret:
        # Can't check, assume available
        return True
    
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }
    
    try:
        # Query options contracts for this symbol
        url = f"{base_url}/v2/options/contracts?underlying_symbols={symbol}&limit=1"
        resp = requests.get(url, headers=headers, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            # If we get any contracts back, options are available
            contracts = data.get("option_contracts", [])
            return len(contracts) > 0
        else:
            # API error, assume available
            return True
    except Exception as e:
        # Fail open - assume available if API error
        print(f"  [OPTIONS CHECK] Error checking {symbol}: {e}")
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Main Scanner
# ──────────────────────────────────────────────────────────────────────────────
def scan_for_bot(
    symbols: List[str],
    bot_type: str,
    profile: Dict[str, Any],
    check_options: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Scan symbols for a specific bot type.
    
    Returns list of qualifying symbol dicts with indicators.
    """
    candidates = []
    stats = {
        "total": len(symbols),
        "with_data": 0,
        "passed_price": 0,
        "passed_atr": 0,
        "passed_volume": 0,
        "passed_trend": 0,
        "passed_options": 0,
    }
    
    sma_period = profile.get("sma_period", 20)
    # Handle sma=0
    sma_period_calc = sma_period if sma_period > 0 else 20
    
    require_options = profile.get("require_options", False)
    
    # Fetch data in batches
    BATCH_SIZE = 50
    history_days = max(sma_period_calc * 3, 100)  # Enough history for SMA
    
    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i+BATCH_SIZE]
        if verbose:
            print(f"  [{bot_type}] Processing batch {i//BATCH_SIZE + 1}/{(len(symbols) + BATCH_SIZE - 1)//BATCH_SIZE}...")
        
        try:
            bars_map, _ = fetch_latest_bars(
                symbols=batch,
                timeframe="1Day",
                history_days=history_days,
                feed="sip",
                rth_only=False,
                verbose=False,
            )
        except Exception as e:
            print(f"  Error fetching batch: {e}")
            continue
        
        for sym in batch:
            try:
                df = bars_map.get(sym)
                if df is None or df.empty:
                    continue
                
                stats["with_data"] += 1
                
                # Calculate indicators
                inds = calculate_indicators(df, sma_period)
                if not inds:
                    continue
                
                price = inds["price"]
                atr_pct = inds["atr_pct"]
                dollar_vol = inds["dollar_vol"]
                in_uptrend = inds["in_uptrend"]
                
                # Apply filters
                if not (profile["price_min"] <= price <= profile["price_max"]):
                    continue
                stats["passed_price"] += 1
                
                if not (profile["atr_pct_min"] <= atr_pct <= profile["atr_pct_max"]):
                    continue
                stats["passed_atr"] += 1
                
                if dollar_vol < profile["dollar_vol_min"]:
                    continue
                stats["passed_volume"] += 1
                
                if not in_uptrend:
                    # Filter only if sma_period > 0 (handled inside calculate_indicators via in_uptrend=True)
                    continue
                stats["passed_trend"] += 1
                
                # Options check (only for options bots)
                has_options = True
                if require_options and check_options:
                    has_options = check_options_available(sym)
                    if not has_options:
                        continue
                stats["passed_options"] += 1
                
                # Passed all filters!
                candidates.append({
                    "symbol": sym,
                    "bot_type": bot_type,
                    "price": price,
                    "atr_pct": atr_pct,
                    "dollar_vol_m": inds["dollar_vol_m"],
                    "sma": inds["sma"],
                    "in_uptrend": in_uptrend,
                    "has_options": has_options,
                })
                
            except Exception as e:
                print(f"    Warning: Error processing {sym}: {e}")
                continue
    
    if verbose:
        print(f"  [{bot_type}] Stats: {stats['with_data']} data → {stats['passed_price']} price → {stats['passed_atr']} ATR → {stats['passed_volume']} vol → {stats['passed_trend']} trend → {stats['passed_options']} final")
    
    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Unified Ticker Scanner for RubberBandBot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Bot Types:
  15M_STK  - 15-Minute Stock Bot
  15M_OPT  - 15-Minute Options Spread Bot
  WK_STK   - Weekly Stock Bot
  WK_OPT   - Weekly Options Bot
  ALL      - Scan for all bot types

Examples:
  python scan_for_bot.py --bot-type 15M_OPT --tickers tickers_full_list.txt
  python scan_for_bot.py --bot-type ALL --price-min 50 --atr-min 2.0
        """
    )
    
    # Required
    parser.add_argument("--bot-type", required=True, 
                        choices=["15M_STK", "15M_OPT", "WK_STK", "WK_OPT", "ALL"],
                        help="Bot type to scan for")
    parser.add_argument("--tickers", default="tickers_full_list.txt",
                        help="Path to ticker list file")
    
    # Output
    parser.add_argument("--output", default="scan_results.csv",
                        help="Output CSV file path")
    
    # Override defaults
    parser.add_argument("--price-min", type=float, default=None,
                        help="Override minimum price filter")
    parser.add_argument("--price-max", type=float, default=None,
                        help="Override maximum price filter")
    parser.add_argument("--atr-min", type=float, default=None,
                        help="Override minimum ATR%% filter")
    parser.add_argument("--atr-max", type=float, default=None,
                        help="Override maximum ATR%% filter")
    parser.add_argument("--volume-min", type=float, default=None,
                        help="Override minimum dollar volume (in millions)")
    parser.add_argument("--sma-period", type=int, default=None,
                        help="Override SMA period for trend filter")
    
    # Options
    parser.add_argument("--skip-options-check", action="store_true",
                        help="Skip options availability check (faster)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of tickers to scan (for testing)")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output")
    
    args = parser.parse_args()
    
    # Load tickers
    ticker_path = args.tickers
    if not os.path.isabs(ticker_path):
        ticker_path = os.path.join(_REPO_ROOT, ticker_path)
    
    if not os.path.exists(ticker_path):
        print(f"Error: Ticker file not found: {ticker_path}")
        return 1
    
    symbols = load_symbols_from_file(ticker_path)
    print(f"Loaded {len(symbols)} tickers from {ticker_path}")
    
    if args.limit > 0:
        symbols = symbols[:args.limit]
        print(f"Limited to first {args.limit} tickers")
    
    # Build overrides dict
    overrides = {}
    if args.price_min is not None:
        overrides["price_min"] = args.price_min
    if args.price_max is not None:
        overrides["price_max"] = args.price_max
    if args.atr_min is not None:
        overrides["atr_pct_min"] = args.atr_min
    if args.atr_max is not None:
        overrides["atr_pct_max"] = args.atr_max
    if args.volume_min is not None:
        overrides["dollar_vol_min"] = args.volume_min * 1_000_000  # Convert to raw
    if args.sma_period is not None:
        overrides["sma_period"] = args.sma_period
    
    # Determine bot types to scan
    if args.bot_type == "ALL":
        bot_types = list(BOT_PROFILES.keys())
    else:
        bot_types = [args.bot_type]
    
    print(f"\n{'='*80}")
    print("UNIFIED TICKER SCANNER FOR RUBBERBANDBOT")
    print(f"{'='*80}")
    print(f"Bot Type(s): {', '.join(bot_types)}")
    if overrides:
        print(f"Overrides: {overrides}")
    print(f"Options Check: {'SKIP' if args.skip_options_check else 'ENABLED'}")
    print(f"{'='*80}\n")
    
    # Run scans
    all_candidates = []
    
    for bot_type in bot_types:
        profile = get_profile(bot_type, overrides)
        print(f"\n[{bot_type}] {profile['description']}")
        print(f"  Criteria: ${profile['price_min']}-${profile['price_max']}, ATR {profile['atr_pct_min']}-{profile['atr_pct_max']}%, Vol>${profile['dollar_vol_min']/1e6:.0f}M, SMA-{profile['sma_period']}")
        
        candidates = scan_for_bot(
            symbols=symbols,
            bot_type=bot_type,
            profile=profile,
            check_options=not args.skip_options_check,
            verbose=not args.quiet,
        )
        
        print(f"  ✓ Found {len(candidates)} candidates for {bot_type}")
        all_candidates.extend(candidates)
    
    # Output results
    if not all_candidates:
        print("\nNo candidates found matching criteria.")
        return 0
    
    df = pd.DataFrame(all_candidates)
    df = df.sort_values(["bot_type", "atr_pct"], ascending=[True, False])
    
    # Save to CSV
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(_REPO_ROOT, output_path)
    
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"SCAN COMPLETE - {len(df)} total candidates")
    print(f"{'='*80}")
    
    # Summary by bot type
    for bt in df["bot_type"].unique():
        count = len(df[df["bot_type"] == bt])
        print(f"  {bt}: {count} candidates")
    
    print(f"\nTop 10 by ATR%:")
    print(df.head(10).to_string(index=False))
    
    print(f"\nSaved to {output_path}")
    
    # Also save per-bot ticker files in same directory as output CSV
    output_dir = os.path.dirname(output_path) or "."
    for bt in df["bot_type"].unique():
        bot_df = df[df["bot_type"] == bt]
        bot_file = os.path.join(output_dir, f"tickers_{bt.lower()}_scan.txt")
        with open(bot_file, "w") as f:
            for sym in bot_df["symbol"]:
                f.write(f"{sym}\n")
        print(f"  → {bt}: {len(bot_df)} tickers saved to {bot_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
