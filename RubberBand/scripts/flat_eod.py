# RubberBand/scripts/flat_eod.py
from __future__ import annotations
import os, sys

# Add project root to path if running as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RubberBand.src.data import cancel_all_orders, close_all_positions

def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return v if v is not None else default

def _creds():
    # Try to resolve using the same logic as data.py, or fall back to env vars
    # We want to support APCA_API_KEY_ID as well.
    key = _env("APCA_API_KEY_ID") or _env("ALPACA_API_KEY") or _env("ALPACA_KEY") or _env("ALPACA_KEY_ID")
    sec = _env("APCA_API_SECRET_KEY") or _env("ALPACA_API_SECRET") or _env("ALPACA_SECRET") or _env("ALPACA_SECRET_KEY")
    
    # Base URL
    base = _env("APCA_API_BASE_URL") or _env("APCA_BASE_URL") or _env("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets"
    base = base.rstrip("/")
    
    if not key or not sec:
        # If we can't find them here, we'll return None and let data.py try to find them (or fail there)
        # But data.py functions take optional args.
        pass
        
    return base, key, sec

def main():
    base, key, sec = _creds()
    
    # 1. First, close 15M Options spreads using the registry (proper spread closing)
    print("[EOD] Closing 15M_OPT spreads from registry (DTE <= 1)...", flush=True)
    try:
        from RubberBand.src.options_execution import flatten_bot_spreads
        results = flatten_bot_spreads(bot_tag="15M_OPT", max_dte_to_close=1)
        for r in results:
            sym = r.get("symbol", "?")
            short_sym = r.get("short_symbol", "")
            err = r.get("error")
            if err:
                print(f"[EOD] Error closing {sym}: {err}", flush=True)
            else:
                print(f"[EOD] Closed spread: {sym} / {short_sym}", flush=True)
    except Exception as e:
        print(f"[EOD] Error in 15M_OPT flatten: {e}", flush=True)
    
    # 1b. Close WK_OPT options that are expiring soon (DTE <= 1)
    # Weekly options should close before expiration to avoid exercise/assignment issues
    print("[EOD] Closing WK_OPT options expiring soon (DTE <= 1)...", flush=True)
    try:
        from RubberBand.src.options_execution import flatten_bot_spreads
        # WK_OPT uses single-leg options, not spreads, but flatten_bot_spreads handles both
        results = flatten_bot_spreads(bot_tag="WK_OPT", max_dte_to_close=1)
        for r in results:
            sym = r.get("symbol", "?")
            err = r.get("error")
            if err:
                print(f"[EOD] Error closing WK_OPT {sym}: {err}", flush=True)
            else:
                print(f"[EOD] Closed WK_OPT position: {sym}", flush=True)
        if not results:
            print("[EOD] No WK_OPT positions with DTE <= 1", flush=True)
    except Exception as e:
        print(f"[EOD] Error in WK_OPT flatten: {e}", flush=True)
    
    # 2. Cancel all open orders
    try:
        cancel_all_orders(base, key, sec)
        print("[EOD] canceled all open orders", flush=True)
    except Exception as e:
        print(f"[EOD] cancel orders error: {e}", flush=True)
    
    # 3. Close ONLY 15M_STK bot's stock positions (using registry)
    # CRITICAL: Do NOT close all stocks - WK_STK positions must be preserved!
    # Also preserve all options - WK_OPT has its own exit logic
    try:
        from RubberBand.src.data import get_positions, close_position
        from RubberBand.src.position_registry import PositionRegistry
        
        all_positions = get_positions(base, key, sec)
        
        # Categorize positions
        stock_positions = [p for p in all_positions if p.get("asset_class") == "us_equity"]
        option_positions = [p for p in all_positions if p.get("asset_class") != "us_equity"]
        
        print(f"[EOD] Found {len(stock_positions)} stock positions, {len(option_positions)} option positions", flush=True)
        
        # Use 15M_STK registry to identify ONLY this bot's positions
        registry_15m = PositionRegistry("15M_STK")
        my_15m_stocks = registry_15m.filter_positions(stock_positions)
        
        print(f"[EOD] 15M_STK registry has {len(registry_15m.positions)} positions, matched {len(my_15m_stocks)} from broker", flush=True)
        
        # Close only 15M_STK positions
        for pos in my_15m_stocks:
            symbol = pos.get("symbol", "")
            try:
                result = close_position(base, key, sec, symbol)
                print(f"[EOD] Closed 15M_STK position: {symbol}", flush=True)
            except Exception as e:
                print(f"[EOD] Error closing {symbol}: {e}", flush=True)
        
        # Log what we're preserving
        other_stocks = [p for p in stock_positions if p.get("symbol") not in registry_15m.get_my_symbols()]
        if other_stocks:
            print(f"[EOD] Preserved {len(other_stocks)} other stock positions (WK_STK?):", flush=True)
            for s in other_stocks:
                print(f"[EOD]   - {s.get('symbol')}", flush=True)
        
        if option_positions:
            print(f"[EOD] Preserved {len(option_positions)} option positions:", flush=True)
            for opt in option_positions:
                print(f"[EOD]   - {opt.get('symbol')}", flush=True)
        
        # Clear the 15M_STK registry (positions just closed)
        if registry_15m.positions:
            print(f"[EOD] Clearing 15M_STK registry...", flush=True)
            registry_15m.sync_with_alpaca([])  # Pass empty = all MY positions closed
            print("[EOD] 15M_STK registry cleared", flush=True)
            
    except Exception as e:
        print(f"[EOD] close positions error: {e}", flush=True)

if __name__ == "__main__":
    main()

