
import json
import re
import sys

LOG_PATH = "latest runs/12_18_2025/20339985237_15m_Stock/console.log"
TARGET = "IBM"

print(f"Scanning {LOG_PATH} for {TARGET}...")

try:
    with open(LOG_PATH, 'r', encoding='utf-16') as f:
        lines = f.readlines()
except:
    try:
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

# Sort lines by timestamp if possible? No, just print in order.
found_universe = False
found_entry = False

for line in lines:
    line = line.strip()
    if not line: continue
    
    # Check for Universe loading
    if "UNIVERSE" in line and "loaded" in line:
        if TARGET in line:
            print(f"[UNIVERSE] {TARGET} was loaded.")
            found_universe = True
    
    # Check for direct mentions
    if TARGET in line:
        # Try to parse timestamp
        # Format: 2025-12-18T...
        ts_match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', line)
        ts = ts_match.group(0) if ts_match else "???"
        
        if "SKIP_SLOPE" in line:
            print(f"[{ts}] SKIP_SLOPE: {line[:100]}...")
        else:
            print(f"[{ts}] LOG: {line[:100]}...")
            
    # Check for Health/Pause
    if "PAUSED" in line or "RESUMED" in line:
        print(f"[HEALTH] {line[:100]}")

if not found_universe:
    print(f"[WARNING] {TARGET} was NOT found in any UNIVERSE log line (or list truncated in log).")
