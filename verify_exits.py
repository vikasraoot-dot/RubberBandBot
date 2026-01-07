
import os
import sys
import pandas as pd
from datetime import datetime, timezone
import alpaca_trade_api as tradeapi

# Add Repo Root to Path
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))

def main():
    api_key = os.environ.get("APCA_API_KEY_ID")
    api_secret = os.environ.get("APCA_API_SECRET_KEY")
    base_url = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key:
        print("Error: APCA_API_KEY_ID not set!")
        return

    # Redirect stdout to file
    with open("verification_results.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        
        print(f"Connecting to Alpaca ({base_url})...")
        api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

        # 1. Fetch Positions
        try:
            positions = api.list_positions()
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return

        print(f"\n=== OPEN POSITIONS: {len(positions)} ===")
        
        pos_data = []
        
        print(f"{'Symbol':<25} {'Qty':<5} {'Entry':<10} {'Current':<10} {'PnL %':<10} {'MarketVal':<12} {'AssetClass':<10}")
        print("-" * 95)
        
        total_invested = 0
        total_value = 0
        
        for p in positions:
            symbol = p.symbol
            qty = int(p.qty)
            entry = float(p.avg_entry_price)
            curr = float(p.current_price)
            mkt_val = float(p.market_value)
            pnl_pct = (curr - entry) / entry * 100 if entry > 0 else 0
            asset_class = p.asset_class
            
            pos_data.append({
                "symbol": symbol,
                "qty": qty,
                "entry": entry,
                "current": curr,
                "pnl_pct": pnl_pct,
                "market_value": mkt_val,
                "asset_class": asset_class,
                "has_exit": False,  # To be filled
                "exit_orders": []
            })

            if asset_class != 'crypto': # Exclude crypto from "invested" if any
                 total_invested += float(p.cost_basis)
                 total_value += mkt_val
            
            print(f"{symbol:<25} {qty:<5} {entry:<10.2f} {curr:<10.2f} {pnl_pct:<10.2f}% ${mkt_val:<11.2f} {asset_class:<10}")

        print("-" * 95)
        print(f"TOTAL INVESTED: ${total_invested:.2f}")
        print(f"TOTAL VALUE:    ${total_value:.2f}")
        print(f"TOTAL PnL:      ${total_value - total_invested:.2f} ({(total_value - total_invested)/total_invested*100:.2f}%)")

        # 2. Fetch Open Orders
        orders = api.list_orders(status='open', limit=500)
        print(f"\n=== OPEN ORDERS: {len(orders)} ===")
        
        # Map Orders to Symbols
        orders_by_symbol = {}
        for o in orders:
            if o.symbol not in orders_by_symbol:
                orders_by_symbol[o.symbol] = []
            orders_by_symbol[o.symbol].append(o)
            
        # 3. Match Positions to Exits
        print(f"\n=== EXIT COVERAGE CHECK ===")
        naked_positions = []
        covered_positions = []
        
        for p in pos_data:
            sym = p["symbol"]
            qty = p["qty"]
            side = "long" if qty > 0 else "short" # Options/Stock usually long here
            
            # Skip checking exits for Options if logic assumes expiry or explicit management
            # BUT user asked for confirmation.
            
            related_orders = orders_by_symbol.get(sym, [])
            closing_orders = []
            
            # Check for closing orders
            # For Long Position, we need Sell Orders
            # For Short Position, we need Buy Orders
            required_side = "sell" if qty > 0 else "buy"
            
            covered_qty = 0
            
            for o in related_orders:
                if o.side == required_side:
                    closing_orders.append(o)
                    # Parse qty? 
                    try: 
                        o_qty = int(o.qty) if o.qty else 0
                        covered_qty += o_qty
                    except: pass
            
            p["exit_orders"] = closing_orders
            
            status = "NAKED"
            if covered_qty >= abs(qty):
                status = "COVERED"
                p["has_exit"] = True
                covered_positions.append(sym)
            elif covered_qty > 0:
                status = "PARTIAL"
                naked_positions.append(sym)
            else:
                naked_positions.append(sym)
                
            print(f"[{status:<7}] {sym:<25} (Pos: {qty}, Orders: {covered_qty})")
            for co in closing_orders:
                print(f"    -> {co.type.upper()} {co.qty} @ {co.limit_price or co.stop_price or 'MKT'} (ID: {co.id})")
                
        print(f"\nSummary: {len(covered_positions)} Covered, {len(naked_positions)} Naked/Partial.")
        if naked_positions:
            print(f"⚠️  WARNING: The following positions have no full exit orders: {naked_positions}")
        else:
            print("✅  All positions are covered by exit orders.")
        
        # Reset stdout
        sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
