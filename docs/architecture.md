# Architecture Reference

Reference doc for Claude Code. Loaded on demand when working on structural changes.

## Signal Pipeline

```
SCAN (scan_for_bot.py)
  → Identify candidate tickers from universe
  → Apply volume, price, dollar-volume filters

SIGNAL (strategy.py + filters.py)
  → EMA crossover detection (9/21 for 15m, 10-week for weekly)
  → RSI oversold confirmation (via config thresholds)
  → Keltner Channel lower band touch
  → Regime-adjusted thresholds (regime_manager.py)

RISK GATE (circuit_breaker.py + probability_filter.py)
  → PortfolioGuard: drawdown check
  → ConnectivityGuard: API health check
  → ResilienceGuard: consecutive loss check
  → ProbabilityFilter: BSM expected value check (shadow/filter mode)
  → Position registry: duplicate position check
  → ALL can VETO — execution only proceeds if all pass

EXECUTION (options_execution.py / core/broker.py)
  → Submit order via BrokerClient
  → Verify mleg fills (options)
  → Place bracket SL/TP (stocks)
  → Log via trade_logger

AUDIT (trade_logger.py + position_registry.py)
  → JSONL event log (immutable append)
  → Position registry update
  → EOD summary generation
```

## Module Responsibilities

### Signal Generation
- `strategy.py` — EMA crossover signals (RSI confirmation is config-driven, not in strategy.py directly)
- `filters.py` — Indicator computation (EMA, RSI, ADX, ATR, Keltner, MACD)
- `regime_manager.py` — VIXY-based regime detection, threshold adjustment

### Risk Management
- `circuit_breaker.py` — PortfolioGuard, ConnectivityGuard
- `probability_filter.py` — BSM-based spread filtering
- `weekly_probability_filter.py` — BSM filtering adapted for weekly ITM calls
- `position_registry.py` — Cross-bot position awareness, duplicate prevention

### Execution
- `core/broker.py` — BrokerClient: unified API gateway for ALL Alpaca calls
- `core/http_client.py` — AlpacaHttpClient: HTTP with retry/backoff/timeout
- `options_execution.py` — Mleg order submission, fill verification, naked leg cleanup
- `alpaca_creds.py` — Env var credential resolution

### Audit & Monitoring
- `trade_logger.py` — JSONL trade events (thread-safe, line-buffered)
- `watchdog/intraday_monitor.py` — Real-time trade monitoring
- `watchdog/post_day_analyzer.py` — EOD analysis
- `watchdog/ml_gate.py` — ML-based signal gating (shadow mode)
- `watchdog/performance_db.py` — Performance tracking

### Configuration
- `config.yaml` — 15-minute bot parameters (indicators, filters, brackets, risk)
- `config_weekly.yaml` — Weekly bot parameters
- `alpaca_creds.py` — Credential resolution order: explicit arg → `APCA_API_KEY_ID` → `ALPACA_KEY_ID`
- Base URL default: `https://paper-api.alpaca.markets`

## State Management

### Position Registry (`.position_registry/*.json`)
- One JSON file per bot tag (15M_STK.json, 15M_OPT.json, etc.)
- Tracks: symbol, client_order_id, qty, entry_price, entry_time, underlying
- `client_order_id` format: `{BOT_TAG}_{SYMBOL}_{TIMESTAMP}` (max 48 chars)
- `reconcile_or_halt()` compares local state vs Alpaca positions
- On mismatch: HALT and alert — do NOT auto-reconcile with trades

### Circuit Breaker State
- `PortfolioGuard` persists `peak_equity` to JSON file
- Survives bot restarts — reloaded on init

### Trade Log (JSONL)
- Append-only, one JSON object per line
- Event types: HEARTBEAT, SIGNAL, GATE, SCAN_CONTEXT, ENTRY_SUBMIT, ENTRY_ACK, ENTRY_REJECT, ENTRY_FILL, OCO_SUBMIT, OCO_ACK, EXIT_FILL, CANCEL, ERROR, SNAPSHOT, EOD_SUMMARY
- All events include: `ts` (UTC ISO8601), `ts_et` (Eastern), `type`
- EOD_SUMMARY includes: total_trades, closed_trades, open_trades, total_pnl, win_rate_pct, exit_reasons

## GitHub Actions Deployment

### Live Trading Workflows
- `rubberband-live-loop-am.yml` — 15M_STK (9:15 AM ET trigger)
- `rubberband-options-spreads.yml` — 15M_OPT (10:00 AM ET trigger)
- `weekly-stock-live.yml` — WK_STK
- `weekly-options-live.yml` — WK_OPT

### Monitoring
- `watchdog-eod.yml` — EOD analysis (4:30 PM ET, runs persist_daily_results, market_classifier, post_day_analyzer)
- `watchdog-monitor.yml` — Intraday health checks (3x daily)
- `watchdog-weekly.yml` — Weekend summary

### CI
- `unit-tests.yml` — pytest on push + schedule
- `safety-check.yml` — Pre-market preflight (validates TP/SL on all positions)

### Runner
- Self-hosted: Windows 11 (PowerShell paths matter in workflow YAML)
- `WK_STK` runs on `ubuntu-latest` (GitHub-hosted)
- Env vars injected: `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, `APCA_API_BASE_URL`, `PYTHONUNBUFFERED=1`
