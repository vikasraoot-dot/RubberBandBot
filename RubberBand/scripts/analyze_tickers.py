
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def analyze_tickers(tickers_file):
    with open(tickers_file, 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]

    print(f"Analyzing {len(tickers)} tickers...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365) # Analyze last year of data
    
    results = []

    # Download data in batches to be efficient
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', progress=False)

    for ticker in tickers:
        try:
            # Handle multi-index columns if downloading multiple tickers
            if len(tickers) > 1:
                df = data[ticker].copy()
            else:
                df = data.copy()
            
            if df.empty:
                print(f"No data for {ticker}")
                continue

            # Calculate metrics
            df['Close'] = df['Close'].fillna(method='ffill')
            current_price = df['Close'].iloc[-1]
            
            # 1. Average Daily Volume (Dollar Volume)
            avg_volume = df['Volume'].mean()
            avg_dollar_volume = (df['Volume'] * df['Close']).mean()

            # 2. Volatility (ATR %)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr_14 = true_range.rolling(14).mean().iloc[-1]
            atr_pct = (atr_14 / current_price) * 100

            # 3. Weekly Volatility (Standard Deviation of Weekly Returns)
            weekly_df = df['Close'].resample('W').last()
            weekly_returns = weekly_df.pct_change().dropna()
            weekly_volatility = weekly_returns.std() * np.sqrt(52) # Annualized

            # 4. Drawdown profile
            rolling_max = df['Close'].cummax()
            drawdown = (df['Close'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            # 5. Beta (Correlation to SPY - approximating with simple calculation here or just volatility ratio)
            # For simplicity in this script without SPY data, we'll use "Relative Volatility" vs the group mean
            # Ideally we'd download SPY but let's stick to intrinsic metrics for now
            
            results.append({
                'Ticker': ticker,
                'Price': round(current_price, 2),
                'ATR%': round(atr_pct, 2),
                'Avg_Dollar_Vol_M': round(avg_dollar_volume / 1_000_000, 1),
                'Weekly_Vol_Ann': round(weekly_volatility * 100, 1),
                'Max_Drawdown': round(max_drawdown, 1)
            })

        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")

    # Create DataFrame and sort by ATR% (often a key factor for rubber band strats)
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv('RubberBand/ticker_analysis.csv', index=False)
    print("\nAnalysis complete. Saved full analysis to RubberBand/ticker_analysis.csv")

if __name__ == "__main__":
    analyze_tickers("RubberBand/tickers.txt")
