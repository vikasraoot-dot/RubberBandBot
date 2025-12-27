
import re
import sys

LOG_PATH = "latest runs/12_18_2025/20339985237_15m_Stock/console.log"

print(f"Scanning {LOG_PATH} for any activity between 15:00 and 16:00 UTC...")

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

count = 0
for line in lines:
    # Match 2025-12-18T15:XX:XX
    if "2025-12-18T15:" in line:
        count += 1
        if count <= 5:
            print(f"Sample Log: {line.strip()[:100]}...")

print(f"\nTotal lines found between 15:00 and 15:59: {count}")
