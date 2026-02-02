import os
import requests

# Test direct API call
key = os.environ.get('APCA_API_KEY_ID') or os.environ.get('ALPACA_KEY_ID')
secret = os.environ.get('APCA_API_SECRET_KEY') or os.environ.get('ALPACA_SECRET_KEY')

if key and secret:
    print(f'API Key found: {key[:8]}...')
    headers = {'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': secret}
    
    # Test account endpoint
    r = requests.get('https://paper-api.alpaca.markets/v2/account', headers=headers, timeout=10)
    print(f'Account API: {r.status_code}')
    if r.status_code == 200:
        acct = r.json()
        print(f"  Buying Power: {acct.get('buying_power')}")
        
    # Test data endpoint
    r2 = requests.get('https://data.alpaca.markets/v2/stocks/AAPL/bars', 
                      headers=headers,
                      params={'timeframe': '1Day', 'start': '2026-01-20', 'limit': 5},
                      timeout=10)
    print(f'Data API: {r2.status_code}')
    if r2.status_code == 200:
        data = r2.json()
        bars = data.get('bars', [])
        print(f'  Bars returned: {len(bars)}')
        if bars:
            for b in bars[:3]:
                print(f"    {b.get('t')}: O={b.get('o')} H={b.get('h')} L={b.get('l')} C={b.get('c')}")
    else:
        print(f'  Error: {r2.text[:200]}')
else:
    print('No API keys found in environment')
