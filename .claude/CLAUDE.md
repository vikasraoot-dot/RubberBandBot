# RubberBandBot — Claude Code Instructions
## v2.0 | 2026-02-20

# IDENTITY

Production multi-bot automated trading system on Alpaca (paper account).
Python 3 · Alpaca REST API · GitHub Actions (self-hosted Windows 11 runner) · JSONL audit trail.

# BOTS

| Tag | Script | Config | Schedule |
|-----|--------|--------|----------|
| 15M_STK | `RubberBand/scripts/live_paper_loop.py` | `RubberBand/config.yaml` | Mon-Fri 9:00 AM ET |
| 15M_OPT | `RubberBand/scripts/live_spreads_loop.py` | `RubberBand/config.yaml` | Mon-Fri 9:00 AM ET |
| WK_STK | `RubberBand/scripts/live_weekly_loop.py` | `RubberBand/config_weekly.yaml` | Mon-Fri hourly 9:35 AM-3:35 PM ET |
| WK_OPT | `RubberBand/scripts/live_weekly_options_loop.py` | `RubberBand/config_weekly.yaml` | Mon 9:35 AM (scan+entry), Mon-Fri checks |
| EMA_SCALP | `ScalpingBots/scripts/live_ema_scalp.py` | — | **Disabled** (workflow off, lifetime -$1,300) |

Bot tags defined in `RubberBand/src/position_registry.py` → `BOT_TAGS`.

# CODEBASE MAP

```
RubberBand/
├── src/                          # Core modules
│   ├── strategy.py               # Signal generation (EMA crossover; RSI filter via config)
│   ├── filters.py                # Indicator computation (EMA, RSI, ADX, ATR, Keltner)
│   ├── data.py                   # Alpaca market data fetching (SIP feed)
│   ├── circuit_breaker.py        # PortfolioGuard (drawdown), ConnectivityGuard (API errors)
│   ├── regime_manager.py         # VIXY-based regime detection (NORMAL/CALM/PANIC)
│   ├── trade_logger.py           # JSONL audit trail (thread-safe, mirrors to stdout)
│   ├── position_registry.py      # Per-bot position tracking, reconciliation
│   ├── probability_filter.py     # BSM probability filter for spreads (shadow mode)
│   ├── weekly_probability_filter.py
│   ├── bsm.py                    # Black-Scholes-Merton model
│   ├── options_execution.py      # Mleg order submission + fill verification
│   ├── options_data.py           # Options chain fetching
│   ├── alpaca_creds.py           # Credential resolution from env vars
│   ├── core/
│   │   ├── broker.py             # BrokerClient — unified Alpaca API gateway
│   │   └── http_client.py        # AlpacaHttpClient — retries, backoff, timeouts
│   └── watchdog/                 # Monitoring: intraday, EOD analysis, ML gate, market classifier
├── scripts/                      # Live loops, backtests, scanners, safety checks
├── config.yaml                   # 15-minute bots config
└── config_weekly.yaml            # Weekly bots config

tests/                            # 540 tests (27 modules)
├── conftest.py                   # Shared fixtures (mock VIXY data, RegimeManager)
├── unit/                         # 22 unit test modules
└── integration/                  # 2 integration test modules

.github/workflows/                # 22 workflows (live trading, watchdog, backtests, CI)
.position_registry/               # Runtime state — DO NOT delete or commit
```

# COMMANDS

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/unit/test_probability_filter.py -v

# Safety preflight (validates all positions have TP/SL)
python RubberBand/scripts/safety_check.py

# Reconcile positions with broker
python RubberBand/scripts/reconcile_broker.py

# Audit P&L from GH Actions logs
python RubberBand/scripts/audit_bot_pnl.py
```

# INVIOLABLE RULES

These override all other considerations. Violating any of these is a critical failure.

## Capital Preservation (Priority 1)
- When uncertain, do NOT trade. Fail-safe = stay flat.
- Risk management layer (`circuit_breaker.py`) can VETO any order. It CANNOT be bypassed.
- Signal generation (`strategy.py`) NEVER executes orders directly.
- Only the execution layer (`options_execution.py`, `core/broker.py`) places orders, and only after risk approval.

## Money Arithmetic
- Use `Decimal(str(value))` for all monetary calculations (P&L, equity, drawdown, price rounding).
- Float is acceptable only for: OHLCV DataFrame columns, indicator math, display/logging.
- All percentage calculations must handle division by zero.

## Error Handling
- NEVER use bare `except:` or silent `except Exception: pass`.
- All broker API calls go through `core/http_client.py` (15s timeout, 5 retries, exponential backoff on {429,500,502,503,504}).
- Unhandled exceptions in order flow MUST halt trading.

## Security
- Credentials from env vars only (`APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`). Resolution logic in `alpaca_creds.py`.
- NEVER hardcode keys, NEVER log credentials, NEVER use `eval()`/`exec()` on external data.

## Safety
- NEVER remove or weaken existing validation, circuit breakers, or risk checks.
- NEVER increase position limits without explicit user approval.
- NEVER disable logging or the audit trail (`trade_logger.py`).
- Ask the user before any change that touches live trading logic or risk parameters.
- When using multi-agent teams, subagents may gather data in parallel, but all final trading decisions must be made by a single agent to prevent contradictions.
- When referencing positions, P&L, or account state, always verify against live broker data rather than relying on earlier conversation context that may be stale.
- Market data discussed earlier in the session may be hours old. Re-fetch before making trading decisions.

# CONVENTIONS

- **Commit messages**: `feat: description`, `fix(scope): description`, `[WATCHDOG] automated msg`
- **Line endings**: LF enforced via `.gitattributes` — avoid rewriting entire files with Write tool
- **Type hints**: Required on all new/modified functions
- **Docstrings**: Required on all new/modified public functions
- **Logging**: Use `trade_logger.py` event methods for trade events; `logging.getLogger(__name__)` for module logs
- **Position IDs**: Format `{BOT_TAG}_{SYMBOL}_{TIMESTAMP}` (max 48 chars, Alpaca limit)
- **Config values**: Externalized in `config.yaml`/`config_weekly.yaml` — no magic numbers in code

# REFERENCE DOCS (load on demand)

For detailed rules beyond the inviolable set above, read these files when the task requires them:

- **Safety & risk rules**: `docs/trading_rules.md` — circuit breakers, order validation checklist, position/risk limits
- **Architecture**: `docs/architecture.md` — module responsibilities, signal pipeline, state management
- **Testing guide**: `docs/testing_guide.md` — coverage requirements, test patterns, CI pipeline
- **Code patterns**: `docs/code_patterns.md` — project-specific patterns (broker calls, logging, registry)
