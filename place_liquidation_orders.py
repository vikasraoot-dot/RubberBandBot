"""
Place Sell Orders for HOOD and ORCL to execute at market open tomorrow.
Orders will be queued and execute when market opens.
"""
import os
import requests
import json

def _alpaca_creds():
    base = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    key = os.environ.get("APCA_API_KEY_ID", "")
    secret = os.environ.get("APCA_API_SECRET_KEY", "")
    return base, key, secret

base, key, secret = _alpaca_creds()
headers = {
    "APCA-API-KEY-ID": key,
    "APCA-API-SECRET-KEY": secret,
    "Content-Type": "application/json",
}

# Get current positions to confirm quantities
positions = requests.get(f"{base}/v2/positions", headers=headers).json()

hood_pos = next((p for p in positions if p["symbol"] == "HOOD" and p["asset_class"] == "us_equity"), None)
orcl_pos = next((p for p in positions if p["symbol"] == "ORCL" and p["asset_class"] == "us_equity"), None)

print("=" * 60)
print("SELL ORDERS - Capital Reduction")
print("=" * 60)

if not hood_pos or not orcl_pos:
    print("ERROR: Could not find HOOD or ORCL positions!")
    print(f"HOOD: {hood_pos}")
    print(f"ORCL: {orcl_pos}")
    exit(1)

print(f"\n📊 Current Positions:")
print(f"  HOOD: {hood_pos['qty']} shares @ ${float(hood_pos['avg_entry_price']):.2f}")
print(f"  ORCL: {orcl_pos['qty']} shares @ ${float(orcl_pos['avg_entry_price']):.2f}")

# Place sell orders
orders_to_place = [
    {"symbol": "HOOD", "qty": abs(int(hood_pos["qty"])), "side": "sell"},
    {"symbol": "ORCL", "qty": abs(int(orcl_pos["qty"])), "side": "sell"},
]

print(f"\n📋 Orders to Submit:")
for order in orders_to_place:
    print(f"  SELL {order['qty']} {order['symbol']} at MARKET (on open)")

print("\n" + "=" * 60)
confirm = input("Type 'CONFIRM' to place these orders: ")

if confirm.strip().upper() != "CONFIRM":
    print("❌ Aborted - no orders placed")
    exit(0)

print("\n🚀 Placing orders...")

for order_spec in orders_to_place:
    order_data = {
        "symbol": order_spec["symbol"],
        "qty": str(order_spec["qty"]),
        "side": "sell",
        "type": "market",
        "time_in_force": "opg",  # "On Open" - executes at market open
        "client_order_id": f"LIQUIDATE_{order_spec['symbol']}_20260107",
    }
    
    resp = requests.post(
        f"{base}/v2/orders",
        headers=headers,
        json=order_data,
    )
    
    if resp.status_code in (200, 201):
        result = resp.json()
        print(f"✅ {order_spec['symbol']}: Order queued - ID: {result.get('id', 'N/A')[:8]}...")
        print(f"   Status: {result.get('status')}, Type: {result.get('type')}, TIF: {result.get('time_in_force')}")
    else:
        print(f"❌ {order_spec['symbol']}: FAILED - {resp.status_code}")
        print(f"   {resp.text}")

print("\n" + "=" * 60)
print("Orders will execute at market open tomorrow (9:30 AM ET)")
print("=" * 60)
