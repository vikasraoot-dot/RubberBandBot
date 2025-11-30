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
    
    # Resume capability
    master_results = []
    processed_tickers = set()
    
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                master_results = json.load(f)
                # Build set of already processed tickers to skip
                # Note: Each ticker has multiple entries (one per timeframe), so we just check if it exists at all
                for r in master_results:
                    if "symbol" in r:
                        processed_tickers.add(r["symbol"])
            print(f"Resuming scan. Found {len(master_results)} records for {len(processed_tickers)} tickers.")
        except json.JSONDecodeError:
            print("Warning: Corrupt results file. Starting fresh.")
            master_results = []

    # Chunking
    total_chunks = (len(all_tickers) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(all_tickers), chunk_size):
        chunk = all_tickers[i:i+chunk_size]
        
        # Filter out already processed tickers
        chunk_to_run = [t for t in chunk if t not in processed_tickers]
        
        if not chunk_to_run:
            print(f"Skipping Chunk {i//chunk_size + 1} (all tickers processed)")
            continue
            
        chunk_idx = i // chunk_size + 1
        print(f"\nProcessing Chunk {chunk_idx}/{total_chunks} ({len(chunk_to_run)} tickers)...")
        
        # Write temp ticker file
        tmp_file = f"temp_chunk_{chunk_idx}.txt"
        with open(tmp_file, "w") as f:
            f.write("\n".join(chunk_to_run))
            
        # Clean up previous backtest summary to avoid stale data
        if os.path.exists("backtest_summary.json"):
            os.remove("backtest_summary.json")
            
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
                    
                    # Update processed set
                    for r in chunk_results:
                        processed_tickers.add(r.get("symbol"))
            else:
                print(f"Warning: No results generated for chunk {chunk_idx}")
                
        except subprocess.CalledProcessError as e:
            print(f"Error running chunk {chunk_idx}: {e}")
            # Don't crash, just continue to next chunk
        except Exception as e:
            print(f"Unexpected error in chunk {chunk_idx}: {e}")
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
                
        # Save intermediate progress (Atomic write if possible, but simple overwrite is okay for now)
        try:
            with open(output_file, "w") as f:
                json.dump(master_results, f, indent=2)
            print(f"Saved {len(master_results)} records so far to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
        
        # Sleep to be nice to API?
        time.sleep(1)

    print("\nScan Complete!")
    print(f"Total records: {len(master_results)}")

if __name__ == "__main__":
    main()
