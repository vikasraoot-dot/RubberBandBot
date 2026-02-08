"""Verify Alpaca API connectivity before running the bot."""
import os
import sys
import requests

key = os.getenv("APCA_API_KEY_ID", "")
secret = os.getenv("APCA_API_SECRET_KEY", "")

if not key or not secret:
    print("ERROR: Alpaca API keys not set in GitHub Secrets")
    sys.exit(1)

headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
r = requests.get(
    "https://paper-api.alpaca.markets/v2/account",
    headers=headers,
    timeout=10,
)
r.raise_for_status()
acct = r.json()
equity = float(acct.get("equity", 0))
status = acct.get("status", "unknown")
print(f"Account: {status} | Equity: ${equity:,.2f}")
