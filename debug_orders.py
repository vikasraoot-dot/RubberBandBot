import json

orders = json.load(open('data/cache/orders/2026-01-06.json'))
filled = [o for o in orders if o.get('status') == 'filled']

for o in filled:
    sym = o.get('symbol', '')
    if sym in ['UBS', 'LVS', 'JCI']:
        cid = o.get('client_order_id', '')[:30]
        print(f"{sym}: {o.get('side')} qty={o.get('filled_qty')} @ {o.get('filled_avg_price')} cid={cid}")
