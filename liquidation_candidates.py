"""
Identify Most Profitable Positions to Liquidate
Goal: Reduce invested capital below $100K
"""
import os
import requests

def _alpaca_creds():
    base = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    key = os.environ.get("APCA_API_KEY_ID", "")
    secret = os.environ.get("APCA_API_SECRET_KEY", "")
    return base, key, secret

base, key, secret = _alpaca_creds()
headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

# Get positions
positions = requests.get(f"{base}/v2/positions", headers=headers).json()

# Calculate total invested
total_cost = sum(float(p.get("cost_basis", 0)) for p in positions)
target = 100000
excess = total_cost - target

print("=" * 70)
print("LIQUIDATION CANDIDATES - Top Profitable Positions")
print("=" * 70)
print(f"\nCurrent Total Invested: ${total_cost:,.2f}")
print(f"Target Max Capital:     ${target:,.2f}")
print(f"Need to Liquidate:      ${excess:,.2f}")
print()

# Sort by unrealized P&L (most profitable first)
ranked = []
for p in positions:
    symbol = p.get("symbol", "")
    asset_class = p.get("asset_class", "")
    cost = float(p.get("cost_basis", 0))
    market_val = float(p.get("market_value", 0))
    unrealized_pnl = float(p.get("unrealized_pl", 0))
    pnl_pct = float(p.get("unrealized_plpc", 0)) * 100
    qty = int(p.get("qty", 0))
    
    ranked.append({
        "symbol": symbol,
        "type": "OPTION" if asset_class == "us_option" else "STOCK",
        "qty": qty,
        "cost": cost,
        "value": market_val,
        "pnl": unrealized_pnl,
        "pnl_pct": pnl_pct,
    })

# Sort by P&L descending (most profitable first)
ranked.sort(key=lambda x: x["pnl"], reverse=True)

print(f"{'Symbol':<30} {'Type':<7} {'Qty':>5} {'Cost':>12} {'P&L':>10} {'P&L%':>8}")
print("-" * 70)

running_cost = 0
recommended = []

for pos in ranked:
    if pos["pnl"] > 0:  # Only profitable positions
        print(f"{pos['symbol']:<30} {pos['type']:<7} {pos['qty']:>5} ${pos['cost']:>10,.0f} ${pos['pnl']:>9,.0f} {pos['pnl_pct']:>7.1f}%")
        
        if running_cost < excess:
            running_cost += pos["cost"]
            recommended.append(pos)
            
print()
print("=" * 70)
print("RECOMMENDED LIQUIDATIONS (to get under $100K)")
print("=" * 70)

if recommended:
    total_freed = sum(p["cost"] for p in recommended)
    total_profit = sum(p["pnl"] for p in recommended)
    
    for pos in recommended:
        print(f"✅ SELL: {pos['symbol']} - Free ${pos['cost']:,.0f} (Profit: ${pos['pnl']:,.0f})")
    
    print()
    print(f"Total Capital Freed: ${total_freed:,.2f}")
    print(f"Total Realized Profit: ${total_profit:,.2f}")
    print(f"New Invested Capital: ${total_cost - total_freed:,.2f}")
else:
    print("No profitable positions to liquidate!")

# Show negative positions too
print()
print("=" * 70)
print("LOSING POSITIONS (avoid selling these)")
print("=" * 70)
losers = [p for p in ranked if p["pnl"] < 0]
losers.sort(key=lambda x: x["pnl"])  # Most losing first

for pos in losers[:5]:
    print(f"❌ {pos['symbol']:<30} ${pos['cost']:>10,.0f} P&L: ${pos['pnl']:>9,.0f}")
