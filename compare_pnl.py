
import re

def get_stats(path):
    for enc in ['utf-16', 'utf-8', 'cp1252']:
        try:
            with open(path, 'r', encoding=enc) as f:
                content = f.read()
                matches = re.findall(r"TOTAL trades=([0-9]+) net=([-0-9\.]+) win_rate=([-0-9\.]+)", content)
                if matches:
                    return matches[-1]
        except:
            continue
    return ("0", "0.00", "0.0")

p_stats = get_stats("protected_5d.txt")
u_stats = get_stats("unprotected_5d.txt")

print("=== BACKTEST COMPARISON (Last 5 Days) ===")
print(f"Protected (Dual-Slope ON):   Trades={p_stats[0]} | Net=${p_stats[1]}")
print(f"Unprotected (Filters OFF):   Trades={u_stats[0]} | Net=${u_stats[1]}")
print("==========================================")
print(f"DIFFERENCE (Saved Capital):  ${float(p_stats[1]) - float(u_stats[1]):.2f}")
