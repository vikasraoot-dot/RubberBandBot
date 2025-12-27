
import csv
import sys
from collections import defaultdict
import datetime

def load_trades(path):
    trades = []
    try:
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades.append(row)
    except FileNotFoundError:
        pass
    return trades

protected = load_trades("trades_protected.csv")
unprotected = load_trades("trades_unprotected.csv")

# Group by Date
p_by_date = defaultdict(list)
u_by_date = defaultdict(list)

all_dates = set()

for t in protected:
    d = t["date"]
    p_by_date[d].append(t)
    all_dates.add(d)

for t in unprotected:
    d = t["date"]
    u_by_date[d].append(t)
    all_dates.add(d)

sorted_dates = sorted(list(all_dates))

print("=== 5-DAY TRADE BREAKDOWN ===")

for d in sorted_dates:
    print(f"\n📅 DATE: {d}")
    
    # Unprotected Trades (What we missed + what we took)
    u_trades = u_by_date.get(d, [])
    p_trades = p_by_date.get(d, [])
    
    # Find missed trades (in Unprotected but not in Protected)
    # Match by symbol and entry_time
    p_keys = set((t["symbol"], t["entry_time"]) for t in p_trades)
    
    for t in u_trades:
        key = (t["symbol"], t["entry_time"])
        status = "✅ TAKEN" if key in p_keys else "❌ MISSED (Filtered)"
        pnl = float(t["pnl"])
        pnl_str = f"${pnl:.2f}"
        if pnl > 0: pnl_str = f"+{pnl_str}"
        
        print(f"  {status} | {t['symbol']:<5} | PnL: {pnl_str:>8} | Slope Info: RSI={t.get('entry_rsi','?')} Entry={t.get('entry_price')}")

