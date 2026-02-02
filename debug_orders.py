import json
orders = json.load(open('data/cache/orders/2026-01-07.json'))

print("BUY orders with 15M_STK prefix:")
buys = [o for o in orders if o.get('status')=='filled' and o.get('side')=='buy']
for o in buys:
    cid = o.get('client_order_id', '')
    if '15M_STK' in cid or cid == '':
        print(f"  {o.get('symbol')}: {cid}")

print("\nSELL orders:")
sells = [o for o in orders if o.get('status')=='filled' and o.get('side')=='sell']
for o in sells:
    cid = o.get('client_order_id', '')
    print(f"  {o.get('symbol')}: {cid}")
