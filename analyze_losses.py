"""
Analyze today's options trades to understand losses.
"""
import os
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import re

PAPER_BASE = "https://paper-api.alpaca.markets"
DATA_BASE = "https://data.alpaca.markets"
key = os.environ.get("APCA_API_KEY_ID", "")
secret = os.environ.get("APCA_API_SECRET_KEY", "")
headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

print("=" * 100)
print("OPTIONS BOT LOSS ANALYSIS - Today (Jan 8, 2026)")
print("=" * 100)

# Get today's date
today = "2026-01-08"

# Get all orders from today
url = f"{PAPER_BASE}/v2/orders"
params = {
    "status": "all",
    "after": f"{today}T00:00:00Z",
    "limit": 500,
    "direction": "asc"
}

resp = requests.get(url, headers=headers, params=params)
orders = resp.json() if resp.status_code == 200 else []

print(f"\nTotal orders today: {len(orders)}")

# Filter to options orders
option_orders = [o for o in orders if len(o.get("symbol", "")) > 10 and any(c.isdigit() for c in o.get("symbol", ""))]
print(f"Options orders: {len(option_orders)}")

# Group by underlying
def parse_underlying(occ):
    match = re.match(r'^([A-Z]+)\d', occ)
    return match.group(1) if match else None

by_underlying = defaultdict(list)
for o in option_orders:
    underlying = parse_underlying(o.get("symbol", ""))
    if underlying:
        by_underlying[underlying].append(o)

print(f"\n{'Underlying':<10} {'Symbol':<30} {'Side':<6} {'Qty':<5} {'Price':>8} {'Status':<10} {'Time':<8}")
print("-" * 90)

for underlying in sorted(by_underlying.keys()):
    for o in by_underlying[underlying]:
        sym = o.get("symbol", "")[:28]
        side = o.get("side", "")
        qty = o.get("filled_qty") or o.get("qty", "")
        price = o.get("filled_avg_price", "")
        price_str = f"${float(price):.2f}" if price else "N/A"
        status = o.get("status", "")
        filled = (o.get("filled_at") or "")
        time_str = filled[11:16] if filled else "N/A"
        
        print(f"{underlying:<10} {sym:<30} {side:<6} {qty:<5} {price_str:>8} {status:<10} {time_str:<8}")

# Get positions with P&L
print("\n" + "=" * 100)
print("CURRENT OPTIONS POSITIONS WITH P&L")
print("=" * 100)

pos_resp = requests.get(f"{PAPER_BASE}/v2/positions", headers=headers)
positions = pos_resp.json() if pos_resp.status_code == 200 else []

options_pos = [p for p in positions if p.get("asset_class") == "us_option"]

total_unrealized = 0
print(f"\n{'Symbol':<30} {'Qty':<6} {'Entry':>10} {'Current':>10} {'P&L':>12} {'P&L%':>8}")
print("-" * 80)

for p in sorted(options_pos, key=lambda x: float(x.get("unrealized_pl", 0))):
    sym = p.get("symbol", "")[:28]
    qty = p.get("qty", "")
    entry = float(p.get("avg_entry_price", 0))
    current = float(p.get("current_price", 0))
    pnl = float(p.get("unrealized_pl", 0))
    pnl_pct = float(p.get("unrealized_plpc", 0)) * 100
    total_unrealized += pnl
    
    print(f"{sym:<30} {qty:<6} ${entry:>8.2f} ${current:>8.2f} ${pnl:>+10.2f} {pnl_pct:>+7.1f}%")

print("-" * 80)
print(f"{'TOTAL UNREALIZED P&L':<60} ${total_unrealized:>+10.2f}")

# Get activities/trades for realized P&L
print("\n" + "=" * 100)
print("TODAY'S CLOSED TRADES (REALIZED P&L)")
print("=" * 100)

# Use activities API for trade history
activities_url = f"{PAPER_BASE}/v2/account/activities/FILL"
params = {"date": today, "direction": "desc"}
activities_resp = requests.get(activities_url, headers=headers, params=params)

if activities_resp.status_code == 200:
    activities = activities_resp.json()
    
    # Group by symbol to find round trips
    fills_by_sym = defaultdict(list)
    for a in activities:
        sym = a.get("symbol", "")
        if len(sym) > 10:  # Options
            fills_by_sym[sym].append(a)
    
    print(f"\nFilled options trades today:")
    realized_pnl = 0
    
    for sym, fills in fills_by_sym.items():
        buys = [f for f in fills if f.get("side") == "buy"]
        sells = [f for f in fills if f.get("side") == "sell"]
        
        underlying = parse_underlying(sym)
        
        for sell in sells:
            sell_price = float(sell.get("price", 0))
            sell_qty = float(sell.get("qty", 0))
            
            # Find matching buy
            for buy in buys:
                buy_price = float(buy.get("price", 0))
                buy_qty = float(buy.get("qty", 0))
                
                if buy_qty > 0:
                    matched_qty = min(sell_qty, buy_qty)
                    trade_pnl = (sell_price - buy_price) * matched_qty * 100
                    realized_pnl += trade_pnl
                    
                    print(f"   {underlying}: Buy ${buy_price:.2f} -> Sell ${sell_price:.2f} x {matched_qty:.0f} = ${trade_pnl:+.2f}")
                    break
    
    print(f"\n   TOTAL REALIZED P&L TODAY: ${realized_pnl:+.2f}")

# Summary
print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"\n   Unrealized P&L (open positions): ${total_unrealized:+,.2f}")
print(f"   This includes positions opened before today")

# Check for big losers
print("\n   TOP LOSERS:")
for p in sorted(options_pos, key=lambda x: float(x.get("unrealized_pl", 0)))[:5]:
    sym = p.get("symbol", "")
    pnl = float(p.get("unrealized_pl", 0))
    entry = float(p.get("avg_entry_price", 0))
    current = float(p.get("current_price", 0))
    if pnl < 0:
        print(f"      {sym}: ${pnl:+,.2f} (Entry: ${entry:.2f}, Now: ${current:.2f})")
