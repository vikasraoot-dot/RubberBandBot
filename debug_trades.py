import json

orders = json.load(open('data/cache/orders/2026-01-06.json'))
filled = [o for o in orders if o.get('status') == 'filled'][:10]

print("Sample filled orders:")
for o in filled:
    symbol = o.get('symbol', '')
    side = o.get('side', '')
    cid = o.get('client_order_id', '')[:30]
    legs = o.get('legs', [])
    print(f"symbol={symbol!r}, side={side!r}, cid={cid!r}, has_legs={len(legs) if legs else 0}")
    if legs:
        for leg in legs[:2]:
            print(f"  leg: symbol={leg.get('symbol')!r}, side={leg.get('side')!r}")
