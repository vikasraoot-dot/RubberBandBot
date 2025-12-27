
import os
import requests
import sys

# Add repo root to path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.options_data import _resolve_creds, _headers

def test_snapshot_single():
    base, key, secret = _resolve_creds()
    print(f"Testing with Base: {base}")
    # Don't print full keys for security
    print(f"Key present: {bool(key)}")
    
    symbol = "CVNA251226C00455000"
    url = f"{base}/v1beta1/options/snapshots/{symbol}"
    
    print(f"Requesting Single: {url}")
    try:
        resp = requests.get(url, headers=_headers(key, secret))
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text}")
    except Exception as e:
        print(f"Error: {e}")

def test_snapshot_multi_endpoint():
    base, key, secret = _resolve_creds()
    symbol = "CVNA251226C00455000"
    # Alternative endpoint: /v1beta1/options/snapshots?symbols=...
    url = f"{base}/v1beta1/options/snapshots"
    params = {"symbols": symbol}
    
    print(f"Requesting Multi: {url} with params {params}")
    try:
        resp = requests.get(url, headers=_headers(key, secret), params=params)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("--- Single Snapshot Endpoint ---")
    test_snapshot_single()
    print("\n--- Multi Snapshot Endpoint ---")
    test_snapshot_multi_endpoint()
