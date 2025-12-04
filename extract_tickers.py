import pandas as pd
import os

# Path to candidates file
csv_path = r"C:\Users\vraoo\GitHub\RubberBandBot\RubberBandBot\latest runs\candidates.csv"
output_path = "candidates_tickers.txt"

if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit(1)

try:
    df = pd.read_csv(csv_path)
    tickers = df['symbol'].tolist()
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(tickers))
        
    print(f"Successfully extracted {len(tickers)} tickers to {output_path}")

except Exception as e:
    print(f"Error: {e}")
