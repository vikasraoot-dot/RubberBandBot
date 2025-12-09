---
description: Code review checklist for RubberBandBot trading bots - reference this before approving any changes
---

# RubberBandBot Code Review Checklist

This checklist is compiled from issues discovered during code reviews and **issues reported after live trading** that passed code review. Use this to prevent regressions.

---

## 🔴 CRITICAL - Must Check Every Time

### 1. Python Version Compatibility
- [ ] **No `datetime.UTC`** - Use `datetime.timezone.utc` instead (Python 3.10 compatibility)
- [ ] **No `zoneinfo` without fallback** - GitHub runners may need `backports.zoneinfo`

### 2. Data Fetching
- [ ] **Weekly bars: Use daily bars + resample** - Direct `1Week` timeframe fails on IEX/Basic plans
- [ ] **Sufficient history_days** - When resampling daily→weekly, need 400+ days for 52+ weeks
- [ ] **Feed parameter** - Verify `feed="iex"` or correct feed is specified

### 3. Position Attribution
- [ ] **client_order_id generated** - All order submissions must include `client_order_id`
- [ ] **client_order_id passed to API** - Verify it's actually in the payload
- [ ] **Registry recorded** - After successful order, call `registry.record_entry()`
- [ ] **Fills filtered by bot_tag** - Summaries/reports must filter by `client_order_id` prefix

### 4. GitHub Actions Workflows
- [ ] **`overwrite: true`** - All `upload-artifact` steps must have this
- [ ] **`continue-on-error: true`** - All `download-artifact` steps must have this
- [ ] **`if: always()`** - Upload steps should run even if job fails
- [ ] **Correct artifact name** - Must match bot tag (e.g., `position-registry-15M_STK`)
- [ ] **Correct path** - Download path must match where code expects registry

---

## 🟡 MEDIUM - Check for Relevant Changes

### 5. API Error Handling
- [ ] **Check response status** - Don't assume API call succeeded
- [ ] **Log error details** - Include response body in error logs
- [ ] **Graceful degradation** - Bot should continue if one ticker fails

### 6. Division by Zero
- [ ] **Guards on averages** - `if qty > 0` before dividing
- [ ] **Guards on PnL calculations** - Check matched_qty > 0

### 7. Time/Timezone Handling
- [ ] **Consistent timezone** - Use `ET` (Eastern Time) for market hours logic
- [ ] **UTC for API calls** - Alpaca expects UTC timestamps
- [ ] **Correct DST handling** - Use `ZoneInfo("US/Eastern")` not hardcoded offsets

### 8. Order Parameters
- [ ] **TIF (Time in Force)** - `day` for intraday, appropriate for swing trades
- [ ] **Bracket orders** - Verify TP/SL prices are correctly calculated
- [ ] **Min tick compliance** - Prices rounded to valid ticks (0.01 for stocks)

---

## 🟢 LOW - Best Practices

### 9. Logging
- [ ] **Structured logging** - Use JSONL for trade logs
- [ ] **Consistent format** - Match existing logging patterns in file
- [ ] **No sensitive data** - Don't log API keys or secrets

### 10. Configuration
- [ ] **Config defaults** - All config values should have sensible defaults
- [ ] **Config validation** - Check for missing required keys

### 11. Code Style
- [ ] **Imports at top** - No inline imports in functions (except for conditional imports)
- [ ] **Type hints** - Functions should have type annotations
- [ ] **Docstrings** - Public functions should be documented

---

## 📋 Issues Discovered AFTER Live Trading (Escaped Code Review)

These issues passed code review but were found during live trading:

| Issue | Root Cause | How to Catch |
|:---|:---|:---|
| Options tickers in stock bot summary | `get_daily_fills()` returned ALL account fills | Check if filtering by bot_tag when showing summaries |
| Weekly bot failed to fetch data | `1Week` bars not available on IEX | Test with actual data feed before deploying |
| Registry not persisting | Missing `overwrite: true` in workflow | Check all upload-artifact steps |

---

## 🔄 Pre-Commit Verification Steps

Before approving any commit:

1. **Run locally** - If possible, test the specific change
2. **Check all modified files** - Not just the main target file
3. **Verify imports** - New imports must be at correct locations
4. **Check related files** - If modifying a shared function, check all callers
5. **Review workflow changes** - YAML syntax is strict, verify format

---

## 📝 Bot-Specific Checks

### 15M Stock Bot (`live_paper_loop.py`)
- [ ] EOD flatten logic in `market_loop.py` will close positions
- [ ] `BOT_TAG = "15M_STK"`

### 15M Options Bot (`live_spreads_loop.py`)
- [ ] Options trading cutoff (3:00 PM ET) enforced
- [ ] `BOT_TAG = "15M_OPT"`

### Weekly Stock Bot (`live_weekly_loop.py`)
- [ ] Uses daily→weekly resampling (not direct `1Week`)
- [ ] `BOT_TAG = "WK_STK"`

### Weekly Options Bot (`live_weekly_options_loop.py`)
- [ ] Uses daily→weekly resampling (not direct `1Week`)
- [ ] Options values multiplied by 100 for contract calculations
- [ ] `BOT_TAG = "WK_OPT"`

---

*Last Updated: 2024-12-08*
*Based on: Session d02a1ffc-f661-4828-873b-ea1b913c9ce2*
