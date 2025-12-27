import pandas as pd
import os

def extract():
    path = "temp_scan/bot-scan-results-20122220584/scan_results.csv"
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return

    df = pd.read_csv(path)
    # Filter for options bot
    opt_tickers = df[df['bot_type'] == '15M_OPT']['symbol'].unique()
    tickers = sorted(opt_tickers)
    
    with open("tickers_options.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(tickers))
        
    print(f"Extracted {len(tickers)} Options tickers to tickers_options.txt")

if __name__ == "__main__":
    extract()
