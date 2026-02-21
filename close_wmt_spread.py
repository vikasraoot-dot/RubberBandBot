"""
Close WMT $128/$129 Feb27 spread at market open.

Run this during market hours (9:30 AM - 4:00 PM ET):
    python close_wmt_spread.py

What it does:
  1. Buys back WMT260227C00129000 (short leg) at market
  2. Sells WMT260227C00128000 (long leg) at market
  3. Verifies both positions are closed

Delete this file after use.
"""
import sys
import time
import requests

# Resolve credentials from environment (same as the bots do)
import os
key = (os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_KEY_ID") or "").strip()
secret = (os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY") or "").strip()
base = (os.getenv("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")

if not key or not secret:
    print("ERROR: Alpaca credentials not found in environment.")
    print("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY, then re-run.")
    sys.exit(1)

headers = {
    "APCA-API-KEY-ID": key,
    "APCA-API-SECRET-KEY": secret,
    "Content-Type": "application/json",
}

# Check market is open
clock = requests.get(f"{base}/v2/clock", headers=headers, timeout=10).json()
if not clock.get("is_open"):
    print(f"WARNING: Market is currently CLOSED (next open: {clock.get('next_open', '?')})")
    print("Options market orders are only allowed during market hours.")
    resp = input("Try anyway? (y/n): ").strip().lower()
    if resp != "y":
        print("Aborted.")
        sys.exit(0)

# Step 1: Check positions exist
print("Checking positions...")
positions = requests.get(f"{base}/v2/positions", headers=headers, timeout=10).json()
wmt_long = None
wmt_short = None
for p in positions:
    if p["symbol"] == "WMT260227C00128000":
        wmt_long = p
    elif p["symbol"] == "WMT260227C00129000":
        wmt_short = p

if not wmt_long and not wmt_short:
    print("Both WMT positions already closed. Nothing to do.")
    sys.exit(0)

if wmt_long:
    print(f"  LONG:  WMT260227C00128000  qty={wmt_long['qty']}  now=${wmt_long['current_price']}  uPnL=${wmt_long['unrealized_pl']}")
if wmt_short:
    print(f"  SHORT: WMT260227C00129000  qty={wmt_short['qty']}  now=${wmt_short['current_price']}  uPnL=${wmt_short['unrealized_pl']}")

# Step 2: Close short leg first (buy to close)
if wmt_short:
    print("\nClosing SHORT leg (WMT260227C00129000) - buy to close...")
    resp = requests.delete(f"{base}/v2/positions/WMT260227C00129000", headers=headers, timeout=15)
    if resp.status_code in (200, 204):
        data = resp.json() if resp.text else {}
        print(f"  -> OK: order submitted (id={str(data.get('id', ''))[:12]})")
    else:
        print(f"  -> ERROR {resp.status_code}: {resp.text[:200]}")
        print("  Cannot proceed without closing short leg first.")
        sys.exit(1)

    # Wait for fill
    print("  Waiting for fill...", end="", flush=True)
    for i in range(30):
        time.sleep(1)
        print(".", end="", flush=True)
        pos_check = requests.get(f"{base}/v2/positions", headers=headers, timeout=10).json()
        still_open = any(p["symbol"] == "WMT260227C00129000" for p in pos_check)
        if not still_open:
            print(" FILLED!")
            break
    else:
        print(" TIMEOUT (may still fill shortly)")

# Step 3: Close long leg (sell to close)
if wmt_long:
    print("\nClosing LONG leg (WMT260227C00128000) - sell to close...")
    resp = requests.delete(f"{base}/v2/positions/WMT260227C00128000", headers=headers, timeout=15)
    if resp.status_code in (200, 204):
        data = resp.json() if resp.text else {}
        print(f"  -> OK: order submitted (id={str(data.get('id', ''))[:12]})")
    else:
        print(f"  -> ERROR {resp.status_code}: {resp.text[:200]}")

    # Wait for fill
    print("  Waiting for fill...", end="", flush=True)
    for i in range(30):
        time.sleep(1)
        print(".", end="", flush=True)
        pos_check = requests.get(f"{base}/v2/positions", headers=headers, timeout=10).json()
        still_open = any(p["symbol"] == "WMT260227C00128000" for p in pos_check)
        if not still_open:
            print(" FILLED!")
            break
    else:
        print(" TIMEOUT (may still fill shortly)")

# Step 4: Verify
print("\nVerifying final state...")
positions = requests.get(f"{base}/v2/positions", headers=headers, timeout=10).json()
wmt_remaining = [p for p in positions if "WMT26022" in p["symbol"]]
if not wmt_remaining:
    print("SUCCESS: Both WMT spread legs closed.")
else:
    print("WARNING: Some WMT positions still open:")
    for p in wmt_remaining:
        print(f"  {p['symbol']}  qty={p['qty']}  uPnL=${p['unrealized_pl']}")
