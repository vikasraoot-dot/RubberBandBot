import yfinance as yf
import pandas as pd
import numpy as np
import os

def load_tickers(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return []
    with open(path, 'r') as f:
        lines = f.readlines()
    tickers = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        tickers.append(line.split()[0]) # Take first word if there are comments
    return list(set(tickers))

def calculate_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def analyze_tickers(tickers, chunk_size=500):
    results = []
    
    # Chunking
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} tickers)...")
        try:
            # Fetch data
            data = yf.download(chunk, period="1mo", interval="1d", progress=False, threads=True)
            
            if data.empty:
                continue
                
            # Handle MultiIndex if multiple tickers, or Single Index if one
            # yfinance returns MultiIndex (Price, Ticker) if > 1 ticker
            if isinstance(data.columns, pd.MultiIndex):
                # Iterate through tickers
                # columns are (PriceType, Ticker)
                # We need to pivot or just access per ticker
                # Easier to iterate tickers in the chunk
                valid_tickers = data.columns.levels[1]
                
                for ticker in chunk:
                    if ticker not in valid_tickers:
                        continue
                        
                    try:
                        df = data.xs(ticker, axis=1, level=1)
                        if df.empty or len(df) < 14:
                            continue
                            
                        last_close = df['Close'].iloc[-1]
                        avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
                        dollar_vol = last_close * avg_vol
                        
                        atr = calculate_atr(df).iloc[-1]
                        atr_pct = (atr / last_close) * 100
                        
                        results.append({
                            'Ticker': ticker,
                            'Price': last_close,
                            'Volume': avg_vol,
                            'DollarVol': dollar_vol,
                            'ATR': atr,
                            'ATR%': atr_pct
                        })
                    except Exception as e:
                        # print(f"Error processing {ticker}: {e}")
                        pass
            else:
                # Single ticker case (unlikely with chunking but possible)
                pass

        except Exception as e:
            print(f"Error fetching chunk {i}: {e}")
            
    return pd.DataFrame(results)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__)) # RubberBand/scripts
    rubberband_dir = os.path.dirname(script_dir) # RubberBand
    root_dir = os.path.dirname(rubberband_dir) # RubberBandBot
    
    tickers_file = os.path.join(rubberband_dir, 'tickers.txt')
    full_list_file = os.path.join(root_dir, 'tickers_full_list.txt')
    
    print(f"Script Dir: {script_dir}")
    print(f"RubberBand Dir: {rubberband_dir}")
    print(f"Root Dir: {root_dir}")
    
    print(f"Reading {tickers_file}...")
    elite_tickers = load_tickers(tickers_file)
    print(f"Found {len(elite_tickers)} elite tickers. Sample: {elite_tickers[:5]}")
    
    print(f"Reading {full_list_file}...")
    full_tickers = load_tickers(full_list_file)
    print(f"Found {len(full_tickers)} candidate tickers. Sample: {full_tickers[:5]}")
    
    # Connectivity Test
    print("\nTesting yfinance connectivity with 'SPY'...")
    try:
        test_data = yf.download("SPY", period="1d", progress=False)
        if test_data.empty:
            print("Connectivity Test FAILED: No data returned for SPY.")
            # return # Try to proceed anyway? No, pointless.
        else:
            print("Connectivity Test PASSED.")
    except Exception as e:
        print(f"Connectivity Test FAILED: {e}")
        return

    # 1. Analyze Elite Tickers to get baseline
    print("\nAnalyzing Elite Tickers...")
    elite_df = analyze_tickers(elite_tickers)
    if elite_df.empty:
        print("Failed to fetch data for elite tickers.")
        return

    print("\n=== Elite Tickers Stats ===")
    print(elite_df.describe())
    
    median_atr_pct = elite_df['ATR%'].median()
    min_atr_pct = elite_df['ATR%'].min()
    print(f"\nMedian ATR%: {median_atr_pct:.2f}%")
    print(f"Min ATR%: {min_atr_pct:.2f}%")
    
    # 2. Analyze Full List
    print("\nAnalyzing Full List...")
    full_df = analyze_tickers(full_tickers)
    
    if full_df.empty:
        print("Failed to fetch data for full list.")
        return
        
    # 3. Filter
    print("\nFiltering Candidates...")
    # Criteria:
    # Price > 5
    # Dollar Vol > 1M
    # ATR% >= Min Elite ATR% (or maybe 75% of it to be inclusive)
    
    min_price = 5.0
    min_dollar_vol = 1_000_000
    target_atr_pct = min_atr_pct * 0.8 # Allow slightly less volatile than the least volatile elite
    
    filtered = full_df[
        (full_df['Price'] >= min_price) &
        (full_df['DollarVol'] >= min_dollar_vol) &
        (full_df['ATR%'] >= target_atr_pct)
    ].copy()
    
    # Remove existing elite tickers from candidates
    filtered = filtered[~filtered['Ticker'].isin(elite_tickers)]
    
    # Sort by ATR% descending (most volatile first)
    filtered = filtered.sort_values('ATR%', ascending=False)
    
    print(f"\nFound {len(filtered)} matches.")
    
    output_file = os.path.join(rubberband_dir, 'similar_tickers.csv')
    filtered.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
    
    print("\nTop 20 Matches:")
    print(filtered[['Ticker', 'Price', 'ATR%', 'DollarVol']].head(20).to_string())

if __name__ == "__main__":
    main()
