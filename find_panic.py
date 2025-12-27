from RubberBand.src.data import fetch_latest_bars
import pandas as pd

def find_panic_dates():
    print("Fetching VIXY history (2 years)...")
    bars_map, _ = fetch_latest_bars(["VIXY"], "1Day", 500, feed="iex", verbose=False)
    df = bars_map.get("VIXY")
    
    if df is None or df.empty:
        print("Failed to fetch VIXY.")
        return

    print(f"Loaded {len(df)} days.")
    
    # Filter for VIXY > 55
    panic_df = df[df["close"] > 55]
    
    if panic_df.empty:
        print("No days found with VIXY > 55 in last 2 years.")
        max_vix = df["close"].max()
        print(f"Max VIXY was {max_vix:.2f} on {df['close'].idxmax()}")
        
        # Check > 35 (Normal/Elevated)
        elevated = df[df["close"] > 35]
        print(f"Days > 35: {len(elevated)}")
        if not elevated.empty:
            print("Recent examples > 35:")
            print(elevated.tail())
    else:
        print("Found Panic Days (VIXY > 55):")
        print(panic_df.head())
        print(panic_df.tail())

if __name__ == "__main__":
    find_panic_dates()
