
import pandas as pd
import os

def load_set(path):
    if not os.path.exists(path):
        return set()
    with open(path, 'r') as f:
        return set(line.strip() for line in f if line.strip() and not line.startswith('#'))

def compare(name, live_path, scan_df, bot_type):
    print(f"\n──────────────────────────────────────────────────────────────")
    print(f"ANALYSIS: {name}")
    print(f"──────────────────────────────────────────────────────────────")
    
    # 1. Load Live List
    live_set = load_set(live_path)
    
    # 2. Extract Scan List
    if 'bot_type' in scan_df.columns:
        scan_set = set(scan_df[scan_df['bot_type'] == bot_type]['symbol'].unique())
    else:
        # Fallback if no bot_type column (unlikely for unified scan)
        scan_set = set(scan_df['symbol'].unique())
        
    # 3. Compute Delta
    added = scan_set - live_set
    removed = live_set - scan_set
    kept = live_set & scan_set
    
    # 4. Report
    print(f"Live Count: {len(live_set)}")
    print(f"Scan Count: {len(scan_set)}")
    print(f"Net Change: {len(scan_set) - len(live_set):+d}")
    print(f"Retained:   {len(kept)}")
    print(f"Removed:    {len(removed)}")
    print(f"Added:      {len(added)}")
    
    if len(added) > 0:
        print(f"\n[+] NEW ADDITIONS (Top 10 of {len(added)}):")
        print(", ".join(sorted(list(added))[:10]) + ("..." if len(added)>10 else ""))
        
    if len(removed) > 0:
        print(f"\n[-] REMOVALS (Top 10 of {len(removed)}):")
        print(", ".join(sorted(list(removed))[:10]) + ("..." if len(removed)>10 else ""))

def main():
    csv_path = "scan_artifacts_40/candidates-csv/candidates.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Loading Daily Scan Results: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total Rows: {len(df)}")
    print(f"Bot Types Found: {df['bot_type'].unique() if 'bot_type' in df.columns else 'Unknown'}")

    # Compare 15M Stock
    compare("15M Stock Bot (RubberBand/tickers.txt)", 
            "RubberBand/tickers.txt", 
            df, 
            "15M_STK")

    # Compare 15M Options
    compare("15M Options Bot (RubberBand/tickers_options.txt)", 
            "RubberBand/tickers_options.txt", 
            df, 
            "15M_OPT")

if __name__ == "__main__":
    main()
