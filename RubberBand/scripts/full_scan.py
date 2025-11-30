#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import time
from typing import List

def read_tickers(path: str) -> List[str]:
    with open(path, "r") as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]

def main():
    # Config
    full_list_path = "tickers_full_list.txt"
    chunk_size = 20 # Smaller chunk size to avoid timeouts/rate limits
    days_arg = "30,90,120,240,350"
    output_file = "full_scan_results.json"
    
    if not os.path.exists(full_list_path):
        print(f"Error: {full_list_path} not found.")
        return

    all_tickers = read_tickers(full_list_path)
    print(f"Loaded {len(all_tickers)} tickers from {full_list_path}")
    
    # Resume capability?
    # For now, just overwrite or append? Let's overwrite master list but maybe skip if we implement resume later.
    master_results = []
    
    # Chunking
    total_chunks = (len(all_tickers) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(all_tickers), chunk_size):
        chunk = all_tickers[i:i+chunk_size]
        chunk_idx = i // chunk_size + 1
        print(f"\nProcessing Chunk {chunk_idx}/{total_chunks} ({len(chunk)} tickers)...")
        
        # Write temp ticker file
        tmp_file = f"temp_chunk_{chunk_idx}.txt"
        with open(tmp_file, "w") as f:
            f.write("\n".join(chunk))
            
        # Run backtest
        cmd = [
            sys.executable, "RubberBand/scripts/backtest.py",
            "--tickers", tmp_file,
            "--days", days_arg,
            "--config", "RubberBand/config.yaml"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # Read results
            if os.path.exists("backtest_summary.json"):
                with open("backtest_summary.json", "r") as f:
                    chunk_results = json.load(f)
                    master_results.extend(chunk_results)
            else:
                print(f"Warning: No results for chunk {chunk_idx}")
                
        except subprocess.CalledProcessError as e:
            print(f"Error running chunk {chunk_idx}: {e}")
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
                
        # Save intermediate progress
        with open(output_file, "w") as f:
            json.dump(master_results, f, indent=2)
            
        print(f"Saved {len(master_results)} records so far to {output_file}")
        
        # Sleep to be nice to API?
        time.sleep(1)

    print("\nScan Complete!")
    print(f"Total records: {len(master_results)}")

if __name__ == "__main__":
    main()
