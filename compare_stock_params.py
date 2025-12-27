import pandas as pd
import os

def summarize(path, label):
    """Load trades and return stats"""
    csv_path = None
    for root, dirs, files in os.walk(path):
        for f in files:
            if f == "detailed_trades.csv":
                csv_path = os.path.join(root, f)
                break
    
    if not csv_path or not os.path.exists(csv_path):
        print(f"  {label}: NO DATA FOUND")
        return None
    
    df = pd.read_csv(csv_path)
    total = len(df)
    if total == 0:
        return None
    
    winners = len(df[df['pnl'] > 0])
    losers = total - winners
    win_rate = winners / total * 100
    total_pnl = df['pnl'].sum()
    avg_pnl = df['pnl'].mean()
    avg_win = df[df['pnl'] > 0]['pnl'].mean() if winners > 0 else 0
    avg_loss = df[df['pnl'] <= 0]['pnl'].mean() if losers > 0 else 0
    
    return {
        'label': label,
        'trades': total,
        'winners': winners,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss
    }

def main():
    print("\n" + "="*90)
    print("STOCK BOT PARAMETER OPTIMIZATION COMPARISON (90 Days)")
    print("="*90)
    
    configs = [
        ("artifacts/stock-90d", "DUAL-SLOPE (Current: RSI=30, TP=2R, SL=1.5ATR)"),
        ("artifacts/stock-baseline-no-slope", "NO SLOPE FILTER (RSI=30, TP=2R, SL=1.5ATR)"),
        ("artifacts/stock-rsi25", "RSI 25 (Slope On, TP=2R, SL=1.5ATR)"),
        ("artifacts/stock-tp3r", "TP 3R (Slope On, RSI=30, SL=1.5ATR)"),
        ("artifacts/stock-sl1atr", "SL 1.0ATR (Slope On, RSI=30, TP=2R)"),
    ]
    
    results = []
    for path, label in configs:
        stats = summarize(path, label)
        if stats:
            results.append(stats)
    
    if not results:
        print("No results found!")
        return
    
    # Print table
    print(f"\n{'Configuration':<50} | {'Trades':>8} | {'WinRate':>8} | {'Total PnL':>12} | {'Avg Win':>10} | {'Avg Loss':>10}")
    print("-"*110)
    
    for r in results:
        print(f"{r['label']:<50} | {r['trades']:>8} | {r['win_rate']:>7.1f}% | ${r['total_pnl']:>10,.2f} | ${r['avg_win']:>9.2f} | ${r['avg_loss']:>9.2f}")
    
    print("-"*110)
    
    # Find best
    best = max(results, key=lambda x: x['total_pnl'])
    print(f"\n✅ BEST CONFIGURATION: {best['label']}")
    print(f"   Total PnL: ${best['total_pnl']:,.2f}")
    print(f"   Win Rate: {best['win_rate']:.1f}%")

if __name__ == "__main__":
    main()
