import os
import re
import json
from datetime import datetime

LOG_DIR = r"C:\Users\vraoo\GitHub\RubberBandBot\RubberBandBot\latest runs\12_17_2025"

def analyze_log(log_file):
    """Analyze a single log file and extract trades"""
    
    entries = []
    exits = []
    bot_type = None
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Detect bot type from startup messages
    if "15M Options Spreads Loop" in content:
        bot_type = "15M Options"
    elif "15M Stock" in content or "market_loop.py" in content:
        bot_type = "15M Stock"
    elif "Weekly Options" in content:
        bot_type = "Weekly Options"
    elif "Weekly Stock" in content:
        bot_type = "Weekly Stock"
    else:
        # Check for other indicators
        if "live_spreads_loop.py" in content:
            bot_type = "15M Options"
        elif "live_paper_loop.py" in content:
            bot_type = "15M Stock"
    
    # Extract JSON trade events
    for line in content.split('\n'):
        # Find SPREAD_ENTRY
        if '"type":"SPREAD_ENTRY"' in line:
            try:
                match = re.search(r'\{.*?"type":"SPREAD_ENTRY".*?\}', line)
                if match:
                    data = json.loads(match.group())
                    entries.append(data)
            except:
                pass
        
        # Find SPREAD_EXIT
        if '"type":"SPREAD_EXIT"' in line:
            try:
                match = re.search(r'\{.*?"type":"SPREAD_EXIT".*?\}', line)
                if match:
                    data = json.loads(match.group())
                    exits.append(data)
            except:
                pass
        
        # Find TRADE_ENTRY (stock bot)
        if '"type":"TRADE_ENTRY"' in line or 'Entry' in line:
            # Different format for stock bot
            pass
        
        # Find TRADE_EXIT (stock bot) 
        if '"type":"TRADE_EXIT"' in line or ('Exit' in line and 'pnl' in line):
            pass
    
    return {
        'bot_type': bot_type,
        'entries': entries,
        'exits': exits,
        'file': log_file
    }

def main():
    print("\n" + "="*80)
    print("LIVE TRADING PnL ANALYSIS - December 17, 2024")
    print("="*80)
    
    # Find all log directories
    log_dirs = []
    for item in os.listdir(LOG_DIR):
        item_path = os.path.join(LOG_DIR, item)
        if os.path.isdir(item_path) and item.startswith("logs_"):
            log_file = os.path.join(item_path, "0_trade.txt")
            if os.path.exists(log_file):
                log_dirs.append((item, log_file))
    
    print(f"\nFound {len(log_dirs)} log files")
    
    total_pnl = 0
    all_entries = []
    all_exits = []
    
    for log_name, log_file in sorted(log_dirs):
        result = analyze_log(log_file)
        bot_type = result['bot_type'] or "Unknown"
        entries = result['entries']
        exits = result['exits']
        
        print(f"\n{'='*60}")
        print(f"Log: {log_name}")
        print(f"Bot Type: {bot_type}")
        print(f"Entries: {len(entries)}, Exits: {len(exits)}")
        
        session_pnl = 0
        for exit in exits:
            pnl = float(exit.get('pnl', 0))
            session_pnl += pnl
            sym = exit.get('underlying', exit.get('symbol', 'N/A'))
            reason = exit.get('exit_reason', 'N/A')
            print(f"  EXIT: {sym} | PnL: ${pnl:+.2f} | Reason: {reason}")
        
        print(f"  Session PnL: ${session_pnl:+.2f}")
        total_pnl += session_pnl
        all_entries.extend(entries)
        all_exits.extend(exits)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total Trades Entered: {len(all_entries)}")
    print(f"Total Trades Exited: {len(all_exits)}")
    print(f"TOTAL PnL: ${total_pnl:+.2f}")
    
    # Breakdown by bot
    print(f"\n{'='*80}")
    print("By Trade:")
    print(f"{'='*80}")
    for i, exit in enumerate(all_exits, 1):
        pnl = float(exit.get('pnl', 0))
        sym = exit.get('underlying', exit.get('symbol', 'N/A'))
        reason = exit.get('exit_reason', 'N/A')
        print(f"{i}. {sym}: ${pnl:+.2f} ({reason})")

if __name__ == "__main__":
    main()
