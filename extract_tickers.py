import pandas as pd
import os

def extract():
    path = "temp_scan/bot-scan-results-20122220584/scan_results.csv"
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return

    df = pd.read_csv(path)
    tickers = sorted(df['symbol'].unique())
    
    with open("tickers_scanned.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(tickers))
        
    print(f"Extracted {len(tickers)} tickers to tickers_scanned.txt")

if __name__ == "__main__":
    extract()
