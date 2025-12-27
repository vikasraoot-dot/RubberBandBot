from RubberBand.src.data import fetch_latest_bars
import pandas as pd

def check_vixy():
    print("Fetching VIXY...")
    bars_map, _ = fetch_latest_bars(["VIXY"], "1Day", 10, feed="iex", verbose=True)
    df = bars_map.get("VIXY")
    if df is not None and not df.empty:
        print(f"Success! Rows: {len(df)}")
        print(df.tail())
    else:
        print("Failed: DF is empty or None")

if __name__ == "__main__":
    check_vixy()
