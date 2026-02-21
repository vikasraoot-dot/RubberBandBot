# Code Patterns

Reference doc for Claude Code. Loaded on demand when writing new modules or modifying existing patterns.

These are patterns SPECIFIC to this codebase. Follow them for consistency.

## Broker API Calls

All Alpaca API calls go through `core/broker.py` → `core/http_client.py`:

```python
# CORRECT — use BrokerClient
from RubberBand.src.core.broker import BrokerClient, create_broker_from_env

broker = create_broker_from_env()
positions = broker.get_positions()
broker.submit_order("AAPL", qty=10, side="buy", order_type="limit", limit_price=150.0)

# INCORRECT — direct requests to Alpaca
import requests
requests.get("https://paper-api.alpaca.markets/v2/positions", headers=...)
```

### HTTP Client Configuration (http_client.py)
- Timeout: 15 seconds (default)
- Max retries: 5
- Backoff: exponential with 0.5s base (0.5s, 1s, 2s, 4s, 8s)
- Retry status codes: {429, 500, 502, 503, 504}
- Errors raised: `AlpacaHttpError(message, status_code, response_body)`

## Credential Resolution (alpaca_creds.py)

```python
# Resolution order for API key:
# 1. Explicit argument
# 2. APCA_API_KEY_ID env var
# 3. ALPACA_KEY_ID env var
# 4. Empty string

# Resolution order for base URL:
# 1. Explicit argument
# 2. APCA_API_BASE_URL env var
# 3. APCA_BASE_URL env var
# 4. ALPACA_BASE_URL env var
# 5. Default: "https://paper-api.alpaca.markets"
```

## Trade Logging (trade_logger.py)

```python
# Use TradeLogger methods — never write raw JSON to log files
logger = TradeLogger(path="logs/15M_OPT_2026-02-20.jsonl")

# Signal detected
logger.signal(symbol="AAPL", side="buy", price=150.0, reason="RSI oversold + EMA cross")

# Risk gate decision
logger.gate(symbol="AAPL", passed=True, checks={"regime": "NORMAL", "circuit_breaker": "ok"})

# Entry/exit lifecycle
logger.entry_submit(symbol="AAPL", qty=10, side="buy", order_id="...")
logger.entry_fill(symbol="AAPL", qty=10, price=150.05, order_id="...")
logger.exit_fill(symbol="AAPL", qty=10, price=152.10, pnl=20.50, reason="take_profit")

# End of day
logger.eod_summary()  # Auto-generates trade stats
```

### Event Types
`HEARTBEAT`, `SIGNAL`, `GATE`, `SCAN_CONTEXT`, `ENTRY_SUBMIT`, `ENTRY_ACK`, `ENTRY_REJECT`, `ENTRY_FILL`, `OCO_SUBMIT`, `OCO_ACK`, `EXIT_FILL`, `CANCEL`, `ERROR`, `SNAPSHOT`, `EOD_SUMMARY`

## Position Registry (position_registry.py)

```python
from RubberBand.src.position_registry import PositionRegistry

registry = PositionRegistry(bot_tag="15M_OPT")
registry.load()

# Generate order ID (format: {BOT_TAG}_{SYMBOL}_{TIMESTAMP}, max 48 chars)
order_id = registry.generate_order_id("AAPL250221C00190000")

# Track position lifecycle
registry.record_entry(symbol="AAPL250221C00190000", client_order_id=order_id, qty=1, entry_price=5.50, underlying="AAPL")
registry.record_exit(symbol="AAPL250221C00190000", exit_price=6.20, exit_reason="take_profit", pnl=70.0)

# Reconciliation (compare local state vs Alpaca)
is_clean, orphans, untracked = registry.reconcile_or_halt(alpaca_positions)
# If not is_clean → HALT, do NOT auto-trade to reconcile
```

## Decimal Usage

```python
from decimal import Decimal

# CORRECT — monetary calculations
equity = Decimal(str(raw_equity))
drawdown = (peak - equity) / peak
pnl = Decimal(str(exit_price)) - Decimal(str(entry_price))

# ACCEPTABLE — non-monetary math
rsi = talib.RSI(close_prices, timeperiod=14)  # float64 from pandas
atr = df["high"] - df["low"]  # DataFrame operations
```

## Circuit Breaker Integration

```python
from RubberBand.src.circuit_breaker import PortfolioGuard, ConnectivityGuard, CircuitBreakerExc

portfolio_guard = PortfolioGuard(state_file=".portfolio_guard.json", max_drawdown_pct=0.10)
connectivity_guard = ConnectivityGuard(max_errors=5)

try:
    portfolio_guard.update(current_equity)
    # ... place order ...
    connectivity_guard.record_success()
except CircuitBreakerExc as e:
    logger.critical(f"Circuit breaker: {e}")
    # HALT — do not continue trading
except Exception as e:
    connectivity_guard.record_error(e)
    raise
```

## Config Loading

```python
from RubberBand.src.utils import load_config

# 15-minute bots
cfg = load_config("RubberBand/config.yaml")

# Weekly bots
cfg = load_config("RubberBand/config_weekly.yaml")

# Access nested values
max_notional = cfg["max_notional_per_trade"]  # 2000
bracket_sl = cfg["bracket"]["atr_mult_sl"]    # 1.5
```
