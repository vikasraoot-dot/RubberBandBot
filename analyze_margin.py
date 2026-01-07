"""
Investigate margin usage - how did we get to -$25K cash?
"""
import os
import requests
from datetime import datetime

def _alpaca_creds():
    base = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    key = os.environ.get("APCA_API_KEY_ID", "")
    secret = os.environ.get("APCA_API_SECRET_KEY", "")
    return base, key, secret

base, key, secret = _alpaca_creds()
headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

# Get account info
acct = requests.get(f"{base}/v2/account", headers=headers).json()

print("=" * 70)
print("ACCOUNT ANALYSIS")
print("=" * 70)
print(f"\nEquity:          ${float(acct['equity']):>12,.2f}")
print(f"Cash:            ${float(acct['cash']):>12,.2f}")
print(f"Buying Power:    ${float(acct['buying_power']):>12,.2f}")
print(f"Portfolio Value: ${float(acct['portfolio_value']):>12,.2f}")
print(f"Last Equity:     ${float(acct['last_equity']):>12,.2f}")

# Get positions and calculate total cost
positions = requests.get(f"{base}/v2/positions", headers=headers).json()

stock_cost = 0
stock_value = 0
option_cost = 0
option_value = 0

print("\n" + "=" * 70)
print("POSITION BREAKDOWN")
print("=" * 70)

print("\n--- STOCKS ---")
for pos in positions:
    if pos.get("asset_class") == "us_equity":
        qty = int(pos["qty"])
        cost_basis = float(pos["cost_basis"])
        market_value = float(pos["market_value"])
        stock_cost += cost_basis
        stock_value += market_value
        print(f"{pos['symbol']:8} qty={qty:4} cost=${cost_basis:>10,.2f} value=${market_value:>10,.2f}")

print(f"\nStock Total Cost: ${stock_cost:,.2f}")
print(f"Stock Market Value: ${stock_value:,.2f}")

print("\n--- OPTIONS ---")
for pos in positions:
    if pos.get("asset_class") == "us_option":
        qty = int(pos["qty"])
        cost_basis = float(pos["cost_basis"])
        market_value = float(pos["market_value"])
        option_cost += cost_basis
        option_value += market_value
        print(f"{pos['symbol']:25} qty={qty:3} cost=${cost_basis:>8,.2f} value=${market_value:>8,.2f}")

print(f"\nOption Total Cost: ${option_cost:,.2f}")
print(f"Option Market Value: ${option_value:,.2f}")

print("\n" + "=" * 70)
print("MARGIN ANALYSIS")
print("=" * 70)
total_invested = stock_cost + option_cost
print(f"\nTotal Cost Basis:    ${total_invested:>12,.2f}")
print(f"Original Capital:    ${100000:>12,.2f}")
print(f"Amount Over Capital: ${total_invested - 100000:>12,.2f}")
print(f"Margin Used:         ${abs(float(acct['cash'])):>12,.2f}")
print(f"\nMargin as % of Capital: {abs(float(acct['cash'])) / 100000 * 100:.1f}%")

print("\n" + "=" * 70)
print("WHY DID THIS HAPPEN?")
print("=" * 70)
print("""
The bots accumulated positions over multiple days without exiting.
Each day, new entries were added while old positions remained open.

Kill switch checks DAILY P&L, not total invested capital.
Bots have per-trade position size limits, but not aggregate limits.
""")
