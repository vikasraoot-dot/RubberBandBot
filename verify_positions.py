#!/usr/bin/env python3
"""Verify option positions and their spread/single classification."""

import os
import requests
import re
from collections import defaultdict

def _alpaca_creds():
    base = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    key = os.environ.get("APCA_API_KEY_ID", "")
    secret = os.environ.get("APCA_API_SECRET_KEY", "")
    if not key:
        key = os.environ.get("ALPACA_KEY_ID", "")
        secret = os.environ.get("ALPACA_SECRET_KEY", "")
    return base, key, secret

def parse_option_symbol(symbol):
    """Parse OCC option symbol."""
    match = re.match(r'^([A-Z]+)(\d{6})([CP])(\d{8})$', symbol)
    if not match:
        return None
    return {
        "underlying": match.group(1),
        "expiration": f"20{match.group(2)[:2]}-{match.group(2)[2:4]}-{match.group(2)[4:6]}",
        "type": "Call" if match.group(3) == "C" else "Put",
        "strike": int(match.group(4)) / 1000,
    }

def main():
    base, key, secret = _alpaca_creds()
    if not key:
        print("[ERROR] No Alpaca credentials found")
        return
    
    # Fetch positions
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    resp = requests.get(f"{base}/v2/positions", headers=headers, timeout=30)
    positions = resp.json()
    
    # Filter options only
    options = [p for p in positions if p.get("asset_class") == "us_option"]
    stocks = [p for p in positions if p.get("asset_class") == "us_equity"]
    
    print(f"\n{'='*70}")
    print(f"  POSITION VERIFICATION")
    print(f"{'='*70}")
    print(f"\nTotal positions: {len(positions)}")
    print(f"  Stocks: {len(stocks)}")
    print(f"  Options: {len(options)}")
    
    # Group options by underlying + expiration
    groups = defaultdict(list)
    for pos in options:
        parsed = parse_option_symbol(pos["symbol"])
        if parsed:
            key = (parsed["underlying"], parsed["expiration"])
            groups[key].append({
                "symbol": pos["symbol"],
                "qty": int(pos["qty"]),
                "strike": parsed["strike"],
                "type": parsed["type"],
                "unrealized_pl": float(pos.get("unrealized_pl", 0)),
            })
    
    print(f"\n\nOption Groups by Underlying/Expiration:")
    print("-" * 70)
    
    spreads = []
    singles = []
    
    for (underlying, exp), legs in sorted(groups.items()):
        print(f"\n{underlying} ({exp}):")
        
        # Check if this is a spread (2 legs, opposite qty signs)
        if len(legs) == 2:
            leg1, leg2 = legs
            if leg1["qty"] * leg2["qty"] < 0:  # Opposite signs
                spreads.append((underlying, exp, legs))
                long_leg = leg1 if leg1["qty"] > 0 else leg2
                short_leg = leg2 if leg1["qty"] > 0 else leg1
                combined_pnl = leg1["unrealized_pl"] + leg2["unrealized_pl"]
                print(f"  [SPREAD] {long_leg['strike']}/{short_leg['strike']}C")
                print(f"    Long:  {long_leg['symbol']} qty={long_leg['qty']} PnL=${long_leg['unrealized_pl']:.2f}")
                print(f"    Short: {short_leg['symbol']} qty={short_leg['qty']} PnL=${short_leg['unrealized_pl']:.2f}")
                print(f"    Combined PnL: ${combined_pnl:.2f}")
                continue
        
        # Single legs or unmatched
        for leg in legs:
            singles.append((underlying, exp, leg))
            print(f"  [SINGLE] {leg['symbol']} qty={leg['qty']} strike={leg['strike']} PnL=${leg['unrealized_pl']:.2f}")
    
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Spreads detected: {len(spreads)}")
    print(f"  Singles detected: {len(singles)}")
    
    # Check if singles might actually be spreads with different criteria
    print(f"\n\nANALYSIS:")
    print("-" * 70)
    
    # Group singles by underlying only (ignoring expiration)
    underlying_groups = defaultdict(list)
    for underlying, exp, leg in singles:
        underlying_groups[underlying].append((exp, leg))
    
    for underlying, positions in underlying_groups.items():
        if len(positions) > 1:
            print(f"\n{underlying}: Multiple single positions detected!")
            for exp, leg in positions:
                print(f"  - {leg['symbol']} (exp={exp}) qty={leg['qty']}")
            print("  ⚠️  These might be separate trades, not spread legs (different expirations or same direction)")

if __name__ == "__main__":
    main()
