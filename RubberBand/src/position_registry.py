"""
Position Registry: Track which positions belong to which trading bot.

Provides client_order_id generation and position tracking to allow
multiple bots to share a single Alpaca account while only reporting
their own trades.

Bot Tags:
- 15M_STK: 15-minute stock trading
- 15M_OPT: 15-minute options/spreads trading  
- WK_STK: Weekly stock trading
- WK_OPT: Weekly options trading
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from zoneinfo import ZoneInfo

ET = ZoneInfo("US/Eastern")

# Valid bot tags
BOT_TAGS = {"15M_STK", "15M_OPT", "WK_STK", "WK_OPT", "SL_ETF"}

# Default registry directory
DEFAULT_REGISTRY_DIR = ".position_registry"


def generate_client_order_id(bot_tag: str, symbol: str) -> str:
    """
    Generate a unique client_order_id with bot tag prefix.
    
    Format: {BOT_TAG}_{SYMBOL}_{TIMESTAMP}
    Example: WK_OPT_NVDA250117C00140000_1733636400
    
    Args:
        bot_tag: Bot identifier (15M_STK, 15M_OPT, WK_STK, WK_OPT)
        symbol: Trading symbol (stock or option)
    
    Returns:
        Unique client order ID string
    """
    if bot_tag not in BOT_TAGS:
        raise ValueError(f"Invalid bot_tag: {bot_tag}. Must be one of {BOT_TAGS}")
    
    timestamp = int(time.time())
    # Alpaca client_order_id max length is 48 chars
    # Truncate symbol if needed: 7 (tag) + 1 (_) + symbol + 1 (_) + 10 (ts) = 19 + symbol
    max_symbol_len = 48 - 19
    symbol_clean = symbol[:max_symbol_len].replace(" ", "_")
    
    return f"{bot_tag}_{symbol_clean}_{timestamp}"


def parse_client_order_id(client_order_id: str) -> Dict[str, str]:
    """
    Parse a client_order_id to extract bot tag and symbol.
    
    Returns:
        {"bot_tag": str, "symbol": str, "timestamp": str} or empty dict if invalid
    """
    if not client_order_id:
        return {}
    
    parts = client_order_id.split("_", 2)  # Split into max 3 parts
    if len(parts) < 3:
        return {}
    
    # First two parts form the bot tag (e.g., "15M" + "STK" = "15M_STK")
    potential_tag = f"{parts[0]}_{parts[1]}"
    if potential_tag not in BOT_TAGS:
        return {}
    
    # Rest is symbol + timestamp
    rest = parts[2] if len(parts) > 2 else ""
    # Find the last underscore to separate symbol from timestamp
    last_underscore = rest.rfind("_")
    if last_underscore == -1:
        return {"bot_tag": potential_tag, "symbol": rest, "timestamp": ""}
    
    return {
        "bot_tag": potential_tag,
        "symbol": rest[:last_underscore],
        "timestamp": rest[last_underscore + 1:],
    }


class PositionRegistry:
    """
    Track positions for a specific bot.
    
    Provides:
    - Record entry/exit of positions
    - Filter Alpaca positions to only those owned by this bot
    - Persist to JSON file for cross-run tracking
    """
    
    def __init__(self, bot_tag: str, registry_dir: str = DEFAULT_REGISTRY_DIR):
        """
        Initialize registry for a specific bot.
        
        Args:
            bot_tag: Bot identifier (15M_STK, 15M_OPT, WK_STK, WK_OPT)
            registry_dir: Directory to store registry files
        """
        if bot_tag not in BOT_TAGS:
            raise ValueError(f"Invalid bot_tag: {bot_tag}. Must be one of {BOT_TAGS}")
        
        self.bot_tag = bot_tag
        self.registry_dir = registry_dir
        self.registry_path = os.path.join(registry_dir, f"{bot_tag}_positions.json")
        
        # Position data
        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position info
        self.closed_positions: List[Dict[str, Any]] = []
        
        # Load existing registry if available
        self.load()
    
    def load(self) -> bool:
        """Load registry from file. Returns True if loaded, False if new."""
        if not os.path.exists(self.registry_path):
            return False
        
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if data.get("bot_tag") != self.bot_tag:
                print(f"[registry] Warning: bot_tag mismatch in {self.registry_path}")
                return False
            
            self.positions = data.get("positions", {})
            self.closed_positions = data.get("closed_positions", [])
            print(f"[registry] Loaded {len(self.positions)} open positions for {self.bot_tag}")
            return True
        except (json.JSONDecodeError, IOError) as e:
            print(f"[registry] Error loading {self.registry_path}: {e}")
            return False
    
    def save(self) -> bool:
        """Save registry to file. Returns True on success."""
        os.makedirs(self.registry_dir, exist_ok=True)
        
        data = {
            "bot_tag": self.bot_tag,
            "updated_at": datetime.now(ET).isoformat(),
            "positions": self.positions,
            "closed_positions": self.closed_positions[-100:],  # Keep last 100
        }
        
        try:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except IOError as e:
            print(f"[registry] Error saving {self.registry_path}: {e}")
            return False
    
    def generate_order_id(self, symbol: str) -> str:
        """Generate client_order_id for this bot and symbol."""
        return generate_client_order_id(self.bot_tag, symbol)
    
    def record_entry(
        self,
        symbol: str,
        client_order_id: str,
        qty: int = 1,
        entry_price: float = 0.0,
        underlying: str = "",
        order_id: str = "",
        **extra
    ):
        """
        Record a new position entry.
        
        Args:
            symbol: Trading symbol (stock or option symbol)
            client_order_id: The client_order_id used for the order
            qty: Position quantity
            entry_price: Entry price per share/contract
            underlying: For options, the underlying stock symbol
            order_id: Alpaca order ID
            extra: Additional metadata
        """
        self.positions[symbol] = {
            "symbol": symbol,
            "underlying": underlying or symbol,
            "client_order_id": client_order_id,
            "order_id": order_id,
            "qty": qty,
            "entry_price": entry_price,
            "entry_date": datetime.now(ET).isoformat(),
            "status": "open",
            **extra,
        }
        self.save()
        print(f"[registry] Recorded entry: {symbol} (order_id={client_order_id[:30]}...)")
    
    def record_exit(
        self,
        symbol: str,
        exit_price: float = 0.0,
        exit_reason: str = "",
        pnl: float = 0.0,
    ):
        """
        Record a position exit.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price per share/contract
            exit_reason: Reason for exit (TP, SL, etc.)
            pnl: Realized P&L
        """
        if symbol not in self.positions:
            print(f"[registry] Warning: Exiting unknown position {symbol}")
            return
        
        pos = self.positions.pop(symbol)
        pos.update({
            "exit_price": exit_price,
            "exit_date": datetime.now(ET).isoformat(),
            "exit_reason": exit_reason,
            "pnl": pnl,
            "status": "closed",
        })
        self.closed_positions.append(pos)
        self.save()
        print(f"[registry] Recorded exit: {symbol} ({exit_reason})")
    
    def get_my_symbols(self) -> Set[str]:
        """Get set of symbols currently owned by this bot."""
        return set(self.positions.keys())
    
    def get_my_underlyings(self) -> Set[str]:
        """Get set of underlying symbols currently owned by this bot."""
        return {pos.get("underlying", pos.get("symbol", "")) for pos in self.positions.values()}
    
    def is_my_position(self, symbol: str) -> bool:
        """Check if a symbol is owned by this bot."""
        return symbol in self.positions
    
    def find_by_symbol(self, symbol: str) -> Optional[str]:
        """
        Find a position by either long symbol (primary key) or short_symbol.
        
        Issue #5 fix: For spreads, we need to find entries by either leg.
        
        Args:
            symbol: Either the long symbol (primary key) or short_symbol
            
        Returns:
            The primary key (long symbol) if found, or None
        """
        # Check if it's a primary key (long symbol)
        if symbol in self.positions:
            return symbol
        
        # Search through positions for matching short_symbol
        for long_symbol, pos in self.positions.items():
            short_sym = pos.get("short_symbol", "")
            if short_sym == symbol:
                return long_symbol
        
        return None
    
    def filter_positions(self, alpaca_positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter Alpaca positions to only those owned by this bot.
        
        Args:
            alpaca_positions: List of position dicts from Alpaca API
        
        Returns:
            Filtered list containing only this bot's positions
        """
        my_symbols = self.get_my_symbols()
        return [pos for pos in alpaca_positions if pos.get("symbol", "") in my_symbols]
    
    def sync_with_alpaca(self, alpaca_positions: List[Dict[str, Any]]):
        """
        Sync registry with current Alpaca positions.
        
        Removes positions from registry that no longer exist in Alpaca
        (closed by other means, expired, etc.)
        """
        alpaca_symbols = {pos.get("symbol", "") for pos in alpaca_positions}
        
        # Find positions in registry but not in Alpaca
        orphaned = [sym for sym in self.positions if sym not in alpaca_symbols]
        
        for sym in orphaned:
            print(f"[registry] Removing orphaned position: {sym}")
            pos = self.positions.pop(sym)
            pos.update({
                "exit_date": datetime.now(ET).isoformat(),
                "exit_reason": "orphaned_sync",
                "status": "closed",
            })
            self.closed_positions.append(pos)
        
        if orphaned:
            self.save()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary stats for this bot's positions."""
        return {
            "bot_tag": self.bot_tag,
            "open_positions": len(self.positions),
            "closed_today": sum(
                1 for p in self.closed_positions
                if p.get("exit_date", "").startswith(datetime.now(ET).strftime("%Y-%m-%d"))
            ),
            "symbols": list(self.positions.keys()),
        }
    


