import requests

# Test the API
resp = requests.get("http://localhost:5000/api/pnl?date=2026-01-06&bot=all")
data = resp.json()

print("=== PnL API Response ===")
print(f"Date: {data.get('date')}")
print(f"Bot Filter: {data.get('bot_filter')}")
print()
print("Bots data:")
for bot, info in data['bots'].items():
    print(f"  {bot}:")
    print(f"    entries={info.get('entries', 0)}")
    print(f"    trades={info.get('trades', 0)}")
    print(f"    winners={info.get('winners', 0)}")
    print(f"    losers={info.get('losers', 0)}")
    print(f"    realized_pnl=${info.get('realized_pnl', 0):.2f}")
    if 'open_count' in info:
        print(f"    open_count={info.get('open_count', 0)}")
        print(f"    unrealized_pnl=${info.get('unrealized_pnl', 0):.2f}")
    print()
