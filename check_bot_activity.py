import json
from pathlib import Path
from collections import defaultdict

ORDERS_CACHE = Path("data/cache/orders")

# Check all cached dates
dates_data = defaultdict(lambda: {"15m_stock": 0, "15m_options": 0, "weekly_stock": 0, "weekly_options": 0})

for cache_file in sorted(ORDERS_CACHE.glob("*.json"), reverse=True):
    date = cache_file.stem
    orders = json.load(open(cache_file))
    
    for order in orders:
        if order.get("status") not in ("filled", "partially_filled"):
            continue
        
        cid = (order.get("client_order_id") or "").upper()
        if cid.startswith("15M_STK"):
            dates_data[date]["15m_stock"] += 1
        elif cid.startswith("15M_OPT"):
            dates_data[date]["15m_options"] += 1
        elif cid.startswith("WK_STK"):
            dates_data[date]["weekly_stock"] += 1
        elif cid.startswith("WK_OPT"):
            dates_data[date]["weekly_options"] += 1

print("=" * 80)
print("Bot Activity by Date (from cached orders)")
print("=" * 80)
print(f"{'Date':<12} {'15m Stock':>12} {'15m Options':>12} {'Weekly Stock':>14} {'Weekly Options':>16}")
print("-" * 80)

for date in sorted(dates_data.keys(), reverse=True):
    data = dates_data[date]
    if any(v > 0 for v in data.values()):
        print(f"{date:<12} {data['15m_stock']:>12} {data['15m_options']:>12} {data['weekly_stock']:>14} {data['weekly_options']:>16}")
