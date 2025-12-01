import sys
import os

# Add project root to path if running as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RubberBand.src.data import get_daily_fills

def main():
    print("Fetching daily fills...", flush=True)
    fills = get_daily_fills()
    
    if not fills:
        print("No trades filled today.")
        return

    # Aggregate per symbol
    stats = {}
    for f in fills:
        sym = f.get("symbol")
        side = f.get("side")
        qty = float(f.get("filled_qty", 0))
        px = float(f.get("filled_avg_price", 0))
        
        if sym not in stats:
            stats[sym] = {"buy_qty": 0, "buy_val": 0.0, "sell_qty": 0, "sell_val": 0.0}
        
        if side == "buy":
            stats[sym]["buy_qty"] += qty
            stats[sym]["buy_val"] += (qty * px)
        elif side == "sell":
            stats[sym]["sell_qty"] += qty
            stats[sym]["sell_val"] += (qty * px)

    # Print Table
    header = f"{'Ticker':<8} {'Bought':<8} {'Avg Ent':<10} {'Basis':<12} {'Sold':<8} {'Avg Ex':<10} {'Day PnL':<10}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    total_pnl = 0.0
    total_vol = 0.0

    for sym in sorted(stats.keys()):
        s = stats[sym]
        b_qty = s["buy_qty"]
        b_val = s["buy_val"]
        s_qty = s["sell_qty"]
        s_val = s["sell_val"]

        avg_ent = (b_val / b_qty) if b_qty > 0 else 0.0
        avg_ex = (s_val / s_qty) if s_qty > 0 else 0.0
        
        # Calculate PnL only on matched intraday quantity
        matched_qty = min(b_qty, s_qty)
        if matched_qty > 0:
            realized_pnl = (avg_ex - avg_ent) * matched_qty
        else:
            realized_pnl = 0.0
        
        total_pnl += realized_pnl
        total_vol += (b_val + s_val)

        pnl_str = f"{realized_pnl:,.2f}" if matched_qty > 0 else "-"

        print(f"{sym:<8} {int(b_qty):<8} {avg_ent:<10.2f} {b_val:<12.2f} {int(s_qty):<8} {avg_ex:<10.2f} {pnl_str:<10}")

    print("-" * len(header))
    print(f"TOTAL Day PnL: ${total_pnl:,.2f} | TOTAL VOL: ${total_vol:,.2f}")
    print("(Note: Day PnL calculated on matched intraday buy/sell quantity)")

if __name__ == "__main__":
    main()
