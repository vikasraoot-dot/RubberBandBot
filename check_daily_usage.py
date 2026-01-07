"""Check typical trade sizes per day."""
import json
from pathlib import Path

ORDERS_CACHE = Path("data/cache/orders")

print("=" * 70)
print("TYPICAL DAILY CAPITAL USAGE")
print("=" * 70)

for cache_file in sorted(ORDERS_CACHE.glob("*.json"), reverse=True)[:5]:
    date = cache_file.stem
    orders = json.load(open(cache_file))
    
    daily_buys = 0
    trade_count = 0
    
    for order in orders:
        if order.get("status") != "filled":
            continue
        if order.get("side") != "buy":
            continue
        
        # Calculate trade value
        qty = float(order.get("filled_qty", 0) or 0)
        price = float(order.get("filled_avg_price", 0) or 0)
        
        # For options, multiply by 100
        if "us_option" in str(order.get("asset_class", "")):
            value = qty * price * 100
        else:
            value = qty * price
        
        if value > 0:
            daily_buys += value
            trade_count += 1
    
    print(f"{date}: {trade_count:>3} trades, total buys = ${daily_buys:>10,.0f}")

print()
print("=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print("""
Based on recent activity:
- 15m Stock Bot: ~$500-2000 per trade, 10-35 trades/day = $5K-$20K capital needed
- Weekly Bots: ~$1000-3000 per trade, 1-5 trades/day = $1K-$15K capital needed

$3K buffer is TIGHT - especially on high signal days.
Recommend freeing $10-15K to have room for 1-2 trading days.
""")
