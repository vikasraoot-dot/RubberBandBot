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
    print("[EOD] Closing 15M_OPT spreads from registry...", flush=True)
    try:
        from RubberBand.src.options_execution import flatten_bot_spreads
        results = flatten_bot_spreads(bot_tag="15M_OPT")
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
    
    # 2. Cancel all open orders
    try:
        cancel_all_orders(base, key, sec)
        print("[EOD] canceled all open orders", flush=True)
    except Exception as e:
        print(f"[EOD] cancel orders error: {e}", flush=True)
    
    # 3. Close remaining stock positions (from 15M_STK bot)
    try:
        close_all_positions(base, key, sec)
        print("[EOD] submitted flatten all remaining positions", flush=True)
        
        # 4. Clear the 15M_STK registry (sync with Alpaca's empty positions)
        try:
            from RubberBand.src.position_registry import PositionRegistry
            registry = PositionRegistry("15M_STK")
            if registry.positions:
                print(f"[EOD] Clearing {len(registry.positions)} stale 15M_STK positions from registry...", flush=True)
                registry.sync_with_alpaca([])  # Pass empty list = all positions are closed
                print("[EOD] 15M_STK registry cleared", flush=True)
        except Exception as e:
            print(f"[EOD] Error clearing 15M_STK registry: {e}", flush=True)
            
    except Exception as e:
        print(f"[EOD] close positions error: {e}", flush=True)

if __name__ == "__main__":
    main()

