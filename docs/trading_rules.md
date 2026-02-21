# Trading Rules & Safety Mechanisms

Reference doc for Claude Code. Loaded on demand when working on order flow, risk management, or circuit breakers.

## Circuit Breakers

### PortfolioGuard (`circuit_breaker.py`)
- Tracks `peak_equity` (Decimal) and compares against current equity
- Triggers `CircuitBreakerExc` when drawdown exceeds `max_drawdown_pct` (default 10%)
- State persisted to JSON file — survives restarts
- When triggered: HALT all trading, require manual reset

### ConnectivityGuard (`circuit_breaker.py`)
- Counts consecutive API errors
- Triggers `CircuitBreakerExc` after `max_errors` (default 5) consecutive failures
- `record_success()` resets counter; `record_error()` increments
- When triggered: HALT trading until connection restored

### ResilienceGuard (config.yaml → `resilience:`)
- `max_consecutive_losses`: 3 (pause after 3 consecutive losing trades)
- `drawdown_threshold_usd`: -100.0 (pause if daily drawdown exceeds)
- `lookback_trades`: 5
- `probation_period_days`: 7

## Order Validation Checklist

Every order must pass ALL of these before submission:

- [ ] Symbol is in scanned candidate list (not arbitrary)
- [ ] Market is open (`BrokerClient.is_market_open()`)
- [ ] Within entry window (`config.yaml → entry_windows`)
- [ ] Not within `flatten_minutes_before_close` (15 min) of close
- [ ] Position size within `max_notional_per_trade` ($2,000)
- [ ] Position size within `max_shares_per_trade` (10,000 for 15m, 100 for weekly)
- [ ] No existing position in same ticker (`max_open_trades_per_ticker`: 1)
- [ ] Circuit breakers not triggered (PortfolioGuard, ConnectivityGuard, ResilienceGuard)
- [ ] Regime allows entry (RegimeManager — PANIC may restrict)
- [ ] Price sanity: `min_price` ($5), `min_dollar_vol` ($1M)
- [ ] RSI within bounds (`rsi_min`: 15, `rsi_oversold`: 25, `rsi_overbought`: 70)
- [ ] For spreads: probability filter score passes thresholds (when mode = "filter")
- [ ] Bracket SL/TP calculated: SL = 1.5x ATR, TP = 2.0 R:R (15m) or 1.5 R:R (weekly)

IF ANY CHECK FAILS → reject, log reason via `trade_logger.gate()`, do NOT proceed.

## Position Limits

### 15-Minute Bots (config.yaml)
- `max_notional_per_trade`: $2,000
- `max_shares_per_trade`: 10,000
- `max_open_trades_per_ticker`: 1
- `allow_shorts`: false

### Weekly Bots (config_weekly.yaml)
- `max_notional_per_trade`: $2,000
- `max_shares_per_trade`: 100
- `qty`: 100

## Risk Limits

- **Drawdown halt**: 10% portfolio drawdown (PortfolioGuard)
- **Consecutive loss pause**: 3 losses (ResilienceGuard)
- **Daily drawdown pause**: -$100 (ResilienceGuard)
- **API error halt**: 5 consecutive errors (ConnectivityGuard)
- **Bracket SL**: 1.5x ATR (hard stop on every position)

## Multi-Leg Order Safety (options_execution.py)

- Mleg fill verification: polls order status every 1s for up to 15s
- Position confirmation: 5 retries with 2s delays after broker reports "filled"
- Naked leg emergency: `_close_naked_legs()` closes orphaned legs on partial fills
- Trust hierarchy: broker "filled" status > position API (accounts for settlement latency)

## Regime Behavior (regime_manager.py)

| Regime | VIXY Signal | Entry Filters | Effect |
|--------|-------------|---------------|--------|
| CALM | Price < SMA_20 for 3 days | Relaxed (RSI<50 weekly) | Aggressive entry |
| NORMAL | Default | Baseline (RSI<45 weekly) | Standard entry |
| PANIC | Price > Upper Band + Vol > 1.5x avg | Strict (RSI<30 weekly, dead_knife=true) | Defensive — may block entries |
