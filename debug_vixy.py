
import sys
import os
import pandas as pd

# Ensure we can import from src
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RubberBand.src.data import fetch_latest_bars

def test_feed(feed_name):
    print(f"\nTesting feed: {feed_name}")
    try:
        bars, meta = fetch_latest_bars(["VIXY"], "1Day", 35, feed=feed_name, verbose=True)
        if "VIXY" in bars and not bars["VIXY"].empty:
            print(f"SUCCESS: {feed_name} returned {len(bars['VIXY'])} bars.")
            print(bars["VIXY"].tail())
        else:
            print(f"FAILURE: {feed_name} returned no data.")
            if meta:
                print("Meta:", meta)
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    print("Checking VIXY data availability...")
    test_feed("iex")
    test_feed("sip")
