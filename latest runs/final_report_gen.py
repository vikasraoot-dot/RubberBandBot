
import os
import re
from pathlib import Path

ROOT = Path("latest runs/12_18_2025")

def parse_log_for_stats(path):
    stats = {
        "bot_name": path.parent.name,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "total_pnl": 0.0,
        "slope_skips_3": 0,
        "slope_skips_10": 0,
        "trades_made": 0,
        "errors": []
    }
    
    content = ""
    for enc in ['utf-16', 'utf-8', 'cp1252']:
        try:
            with open(path, 'r', encoding=enc) as f:
                content = f.read()
            if content: break
        except: continue
        
    if not content:
        return stats

    # Errors
    stats["errors"] = re.findall(r"(AttributeError|UnboundLocalError|Exception:.*)", content)

    # Slope Skips
    stats["slope_skips_3"] = len(re.findall(r"SKIP_SLOPE3", content))
    stats["slope_skips_10"] = len(re.findall(r"SKIP_SLOPE10", content))

    # PnL (Stock Bot / Weekly Stock uses [KILL SWITCH DEBUG] format usually)
    # "[KILL SWITCH DEBUG] WK_STK: realized_pnl=$0.00, unrealized_pnl=$-352.85"
    ks_matches = re.findall(r"realized_pnl=\$([-\d\.]+), unrealized_pnl=\$([-\d\.]+)", content)
    if ks_matches:
        last_match = ks_matches[-1]
        stats["realized_pnl"] = float(last_match[0])
        stats["unrealized_pnl"] = float(last_match[1])

    # Options Bot (JSONL EOD Summary or heartbeat)
    # "total_pnl":0
    eod_matches = re.findall(r"\"total_pnl\":([-\d\.]+)", content)
    if eod_matches:
        stats["realized_pnl"] = float(eod_matches[-1])
    
    # Trade counts
    # SPREAD_ENTRY
    stats["trades_made"] = len(re.findall(r"SPREAD_ENTRY", content)) + len(re.findall(r"\[INFO\] ENTRY:", content))

    return stats

def main():
    print("=== LIVE BOT PERFORMANCE REPORT (Dec 18, 2025) ===")
    
    dirs = sorted([d for d in os.listdir(ROOT) if (ROOT/d).is_dir()])
    
    for d in dirs:
        if "15m_Stock" not in d: continue # Filter for Stock Bot only
        log_path = ROOT / d / "console.log"
        if not log_path.exists():
            continue
            
        s = parse_log_for_stats(log_path)
        
        print(f"\nBOT: {s['bot_name']}")
        print(f"  Realized PnL:   ${s['realized_pnl']:.2f}")
        print(f"  Unrealized PnL: ${s['unrealized_pnl']:.2f}")
        print(f"  Trades Made:    {s['trades_made']}")
        print(f"  Slope Filter:   3-bar skipped {s['slope_skips_3']} | 10-bar skipped {s['slope_skips_10']}")
        if s['errors']:
            print(f"  ERRORS:         {len(s['errors'])} (First: {s['errors'][0][:50]}...)")
        else:
            print("  ERRORS:         None")

if __name__ == "__main__":
    main()
