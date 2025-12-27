
import os
import re
import json

LOG_DIRS = {
    "15m_Options": "live_logs_dec22/options",
    "Weekly_Options": "live_logs_dec22/weekly_options",
    "Weekly_Stock": "live_logs_dec22/weekly_stock",
    # "15m_Stock": "live_logs_dec22/stock" # Missing
}

def analyze_log(bot_name, file_path):
    print(f"--- analyzing {bot_name} ({file_path}) ---")
    
    if not os.path.exists(file_path):
        print("  [File not found]")
        return

    params = {"slope": None, "slope10": None, "dte": None}
    scanned = 0
    signals = 0
    entries = 0
    skips = {}
    
    encodings = ['utf-8', 'utf-16', 'cp1252']
    content = ""
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                content = f.read()
            break
        except UnicodeError:
            continue
            
    if not content:
        print(f"  [Error parsing file: unknown encoding]")
        return

    for line in content.splitlines():
            # Config sniffing
            if "Slope Threshold" in line:
                params["slope"] = line.strip()
            if "Slope Threshold 10-bar" in line:
                params["slope10"] = line.strip()
            if "Starting Options Spreads Loop" in line:
                 if "slope=" in line:
                     params["slope_arg"] = line.split("slope=")[1].split(",")[0].strip()

            # Json parsing
            if line.strip().startswith("{"):
                try:
                    data = json.loads(line.strip())
                    evt = data.get("type", "") or data.get("event", "")
                    
                    if evt == "BARS_FETCH_SUMMARY":
                        scanned = data.get("with_data", 0)
                    
                    if evt == "SKIP_SLOPE3":
                        sym = data.get("symbol")
                        slope = data.get("slope")
                        thresh = data.get("threshold")
                        k = f"SKIP_SLOPE3 ({thresh})"
                        if k not in skips: skips[k] = []
                        skips[k].append(f"{sym}:{slope}")

                    if evt == "spread_skip":
                        reason = data.get("skip_reason")
                        k = f"SKIP: {reason}"
                        if k not in skips: skips[k] = 0
                        skips[k] += 1
                        
                    if evt == "scan_complete":
                        signals += data.get("signals", 0)
                        entries += data.get("new_entries", 0)

                except:
                    pass

    print(f"  Params detected: {params}")
    print(f"  Scanned Tickers: {scanned}")
    print(f"  Signals Found:   {signals}")
    print(f"  Trades Entered:  {entries}")
    print(f"  Skips/Filters:")
    for k, v in skips.items():
        if isinstance(v, list):
            print(f"    {k}: {len(v)} count. Items: {v[:5]}...")
        else:
            print(f"    {k}: {v}")
    print("\n")

def main():
    root = "c:/Users/vraoo/GitHub/RubberBandBot/RubberBandBot"
    for bot, rel_dir in LOG_DIRS.items():
        abs_dir = os.path.join(root, rel_dir)
        # Find console log (starts with 15m or weekly or just console.log)
        if not os.path.exists(abs_dir):
            print(f"Skipping {bot} - dir not found")
            continue
            
        found = False
        for root_dir, dirs, files in os.walk(abs_dir):
            for fname in files:
                if "console" in fname and fname.endswith(".log"):
                    analyze_log(bot, os.path.join(root_dir, fname))
                    found = True
        
        if not found:
             print(f"{bot}: No console log found in {abs_dir}")

if __name__ == "__main__":
    main()
