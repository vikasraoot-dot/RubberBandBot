# Testing Guide

Reference doc for Claude Code. Loaded on demand when writing or modifying tests.

## Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Specific module
pytest tests/unit/test_probability_filter.py -v

# With coverage (if pytest-cov installed)
pytest tests/ --cov=RubberBand/src --cov-report=term-missing
```

## Test Structure

```
tests/
├── conftest.py                          # Shared fixtures
│   ├── mock_vixy_calm()                 # CALM regime test data
│   ├── mock_vixy_normal()               # NORMAL regime test data
│   ├── mock_vixy_panic()                # PANIC regime + volume data
│   └── regime_manager()                 # RegimeManager(verbose=False)
├── test_resilience.py                   # ResilienceGuard logic
├── test_regime_scenarios.py             # Regime transitions
├── test_audit_cutoff.py                 # Audit tool edge cases
├── unit/                                # 22 modules
│   ├── test_bsm.py                      # Black-Scholes-Merton calculations
│   ├── test_probability_filter.py       # BSM probability scoring
│   ├── test_weekly_probability_filter.py
│   ├── test_regime_manager.py           # Regime detection
│   ├── test_options_execution.py        # Spread orders, mleg verification
│   ├── test_signals.py                  # Signal generation
│   ├── test_critical_paths.py           # Core trading paths
│   ├── test_config_validation.py        # YAML config loading
│   ├── test_mleg_fill_verification.py   # Multi-leg order fills
│   ├── test_spread_pricing.py           # Spread price calculations
│   ├── test_cross_bot_awareness.py      # Position registry coordination
│   ├── test_order_rejection_tracking.py # Rejection handling
│   ├── test_profit_lock_reset.py        # Profit locking logic
│   ├── test_quote_inversion_guard.py    # Quote validation
│   ├── test_eod_selective_cancel.py     # EOD order cancellation
│   ├── test_ema_scalper_fixes.py        # EMA scalp bot fixes
│   ├── test_persist_and_registry.py     # Position persistence
│   ├── test_scan_context.py             # Scan event logging
│   ├── test_sip_feed.py                 # SIP data handling
│   ├── test_yf_fallback.py             # Yahoo Finance fallback
│   ├── test_weekly_logic.py             # Weekly strategy
│   └── test_weekly_probability_filter.py # Weekly probability filter
└── integration/
    ├── test_live_loops.py               # End-to-end loop tests
    └── test_spread_exit_flow.py         # Spread exit flow
```

## What Must Be Tested

### Critical paths (aim for 100% branch coverage)
- Order validation logic (all rejection reasons)
- Circuit breaker triggers and resets
- Position registry operations (record_entry, record_exit, reconcile_or_halt)
- Mleg fill verification (filled, pending, failed paths)
- Probability filter scoring (pass, fail, shadow vs filter mode)

### Standard coverage
- Signal generation (EMA crossover conditions)
- Indicator calculations (RSI, ATR, Keltner)
- Config loading and validation
- Regime transitions (NORMAL → PANIC → CALM)

### Test cases must include
- Happy path (valid order, successful fill)
- Edge cases (zero quantity, boundary values at limits)
- Invalid inputs (wrong types, None, negative values)
- Error conditions (API timeout, connection failure, rate limit)
- Regime-dependent behavior (same signal, different regime → different outcome)

## CI Pipeline

- `unit-tests.yml` runs `pytest tests/ -v` on every push and on schedule
- Tests run with `PYTHONUNBUFFERED=1` for real-time CI log output
- All 540 tests must pass before merge
- No coverage enforcement in CI currently (manual review)
