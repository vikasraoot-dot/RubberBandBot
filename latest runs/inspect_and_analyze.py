
import os
import zipfile
import json
import re
from pathlib import Path

ROOT = Path("latest runs/12_18_2025")

def unzip_all(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".zip"):
                path = os.path.join(root, f)
                print(f"[INFO] Unzipping {path}...")
                try:
                    with zipfile.ZipFile(path, 'r') as zip_ref:
                        zip_ref.extractall(root)
                    print(f"[INFO] Unzipped {path}")
                except Exception as e:
                    print(f"[ERROR] Failed to unzip {path}: {e}")

def parse_console_log(path):
    content = ""
    # Try encodings
    for enc in ['utf-16', 'utf-8', 'cp1252']:
        try:
            with open(path, 'r', encoding=enc) as f:
                content = f.read()
            if content: 
                break
        except:
            continue
    
    analysis = {
        "file": str(path),
        "errors": [],
        "slope_activity": [],
        "pnl_info": [],
        "trades": []
    }

    # Find Errors
    errors = re.findall(r"(Error|Exception|Traceback).*", content, re.IGNORECASE)
    analysis["errors"] = errors[:5] # Top 5

    # Find Slope Activity
    slope_skips = re.findall(r"SKIP_SLOPE.*", content)
    analysis["slope_activity"] = slope_skips

    # Find PnL (from standard output if available)
    pnl_lines = re.findall(r".*PnL.*", content, re.IGNORECASE)
    analysis["pnl_info"] = pnl_lines

    return analysis

def parse_jsonl(path):
    trades = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                trades.append(obj)
            except:
                pass
    return trades

def main():
    if not ROOT.exists():
        print(f"ROOT {ROOT} does not exist!")
        return

    # 1. Unzip everything
    unzip_all(ROOT)

    report = []

    # 2. Walk and Analyze
    for root, dirs, files in os.walk(ROOT):
        for f in files:
            path = Path(root) / f
            
            if f == "console.log" or f.endswith(".log") or f.endswith(".txt"):
                print(f"Analyzing Log: {f}")
                data = parse_console_log(path)
                report.append(f"\n--- LOG: {f} ({path.parent.name}) ---")
                if data["errors"]:
                    report.append(f"ERRORS Found: {len(data['errors'])}")
                    for e in data["errors"]:
                        report.append(f"  - {e.strip()[:200]}")
                else:
                    report.append("No Errors Found.")
                
                if data["slope_activity"]:
                    report.append(f"Slope Activity: {len(data['slope_activity'])} events")
                    for s in data["slope_activity"]:
                        report.append(f"  - {s.strip()}")
                
                if data["pnl_info"]:
                    report.append("PnL / Balance Info:")
                    for p in data["pnl_info"]:
                        report.append(f"  - {p.strip()}")

            elif f.endswith(".jsonl"):
                print(f"Analyzing JSONL: {f}")
                trades = parse_jsonl(path)
                entries = [t for t in trades if t.get("type") == "SPREAD_ENTRY"]
                exits = [t for t in trades if t.get("type") == "SPREAD_EXIT"]
                summary = [t for t in trades if t.get("type") == "EOD_SUMMARY"]
                
                report.append(f"\n--- ARTIFACT: {f} ---")
                report.append(f"Entries: {len(entries)}")
                report.append(f"Exits: {len(exits)}")
                if summary:
                    s = summary[0]
                    report.append(f"EOD Result: PnL {s.get('total_pnl')} | WinRate {s.get('win_rate_pct')}%")
                    report.append(f"Total Trades: {s.get('total_trades')} ({s.get('closed_trades')} closed)")

    print("\n".join(report))

if __name__ == "__main__":
    main()
