---
description: Code review checklist for RubberBandBot trading bots - reference this before approving any changes
---

# RubberBandBot Code Review Checklist

Use this checklist to identify **patterns of issues**, not just specific bugs. Each pattern is derived from real issues found during reviews or discovered in production.

---

## 🔴 CRITICAL PATTERNS

### Pattern 1: Cross-Version API Compatibility
**Root Cause:** Using Python features not available in target runtime version.
**Example:** `datetime.UTC` only exists in Python 3.11+, but runner uses 3.10.

**Checklist:**
- [ ] Any new `datetime` usage - verify compatibility with Python 3.10
- [ ] Any new stdlib imports - verify the module exists in target version
- [ ] Any new syntax features (walrus operator, match/case) - verify version support

---

### Pattern 2: Data Availability Assumptions
**Root Cause:** Assuming data is available in a specific format/granularity when it may not be.
**Example:** `1Week` bars not available on IEX Basic plan.

**Checklist:**
- [ ] Any hardcoded timeframes - can the data feed actually provide this?
- [ ] Any assumptions about data history availability - is there enough data?
- [ ] Any aggregation that depends on upstream availability - have fallbacks?
- [ ] When resampling, is there sufficient source data for indicator warm-up?

---

### Pattern 3: Shared Resource Pollution
**Root Cause:** Multiple components sharing a resource without proper isolation.
**Example:** All bots sharing one Alpaca account, summaries showing all trades instead of just one bot's.

**Checklist:**
- [ ] Any function that fetches "all" of something - does it need filtering?
- [ ] Any summary/report generation - is it scoped to the correct context?
- [ ] Any shared state (files, databases, APIs) - are components properly isolated?
- [ ] Any identifier that should be unique - is it actually unique across all contexts?

---

### Pattern 4: Workflow State Persistence
**Root Cause:** CI/CD workflows not properly saving/restoring state between runs.
**Example:** Missing `overwrite: true` causing artifact upload failures.

**Checklist:**
- [ ] Any `upload-artifact` step - has `overwrite: true` for named artifacts?
- [ ] Any `download-artifact` step - has `continue-on-error: true`?
- [ ] Any state files the bot expects - are they in the download path?
- [ ] Any cleanup steps - do they run `if: always()` to capture state on failure?

---

### Pattern 5: Silent Failures
**Root Cause:** Errors caught but not properly surfaced, leading to hidden failures.
**Example:** API returning error JSON that gets ignored because we only check status code.

**Checklist:**
- [ ] Any try/except blocks - are errors logged with details?
- [ ] Any API calls - is the response body checked for error fields?
- [ ] Any empty return values - is the caller handling empty gracefully?
- [ ] Any "continue on error" logic - is it too permissive?

---

## 🟡 MEDIUM PATTERNS

### Pattern 6: Numeric Edge Cases
**Root Cause:** Math operations without proper guards.
**Example:** Division by zero when no trades occurred.

**Checklist:**
- [ ] Any division - is the denominator checked for zero?
- [ ] Any percentage calculations - are bounds checked?
- [ ] Any financial calculations - are rounding rules correct for the asset type?
- [ ] Any option contract math - is the *100 multiplier applied correctly?

---

### Pattern 7: Timezone Mismatches
**Root Cause:** Mixing timezones without proper conversion.
**Example:** Using local time for market hour checks instead of Eastern time.

**Checklist:**
- [ ] Any time comparisons - are both sides in the same timezone?
- [ ] Any market-hour logic - is it using US/Eastern?
- [ ] Any API timestamps - are they being parsed with correct timezone?
- [ ] Any log timestamps - are they human-readable in the expected zone?

---

### Pattern 8: Configuration Override Gaps
**Root Cause:** Config values not propagating to all code paths.
**Example:** Hardcoded defaults that override user config.

**Checklist:**
- [ ] Any default values in code - does config override them?
- [ ] Any new config parameters - are they documented?
- [ ] Any split between config files - are they consistent?
- [ ] Any env vars that override config - is precedence correct?

---

### Pattern 9: Order Parameter Validation
**Root Cause:** Invalid order parameters causing API rejections.
**Example:** Take profit price not meeting minimum tick requirements.

**Checklist:**
- [ ] Any limit prices - are they rounded to valid ticks?
- [ ] Any bracket orders - is TP meaningfully different from entry?
- [ ] Any quantity calculations - are they positive integers?
- [ ] Any symbol transformations - is the format correct for the API?

---

## 🟢 LOW PATTERNS

### Pattern 10: Logging Consistency
**Root Cause:** Inconsistent logging making debugging difficult.

**Checklist:**
- [ ] Any new log statements - do they follow the file's existing pattern?
- [ ] Any structured logs (JSONL) - are all required fields present?
- [ ] Any print statements - should they be proper logging instead?

---

### Pattern 11: Import Organization
**Root Cause:** Import errors or circular dependencies.

**Checklist:**
- [ ] Any new imports - are they at the file top (unless conditional)?
- [ ] Any cross-module imports - could they cause circular dependencies?
- [ ] Any optional dependencies - are they guarded with try/except?

---

## 📋 Anti-Patterns That Escaped to Production

| Pattern | What Happened | How to Catch |
|:---|:---|:---|
| **Shared Resource Pollution** | Options tickers showed in stock bot summary | Check if any "get all" function is filtered by context |
| **Data Availability Assumptions** | Weekly bot couldn't fetch 1Week bars | Ask: "Does this API call work on the actual data plan?" |
| **Workflow State Persistence** | Registry lost between runs | Check every upload-artifact has overwrite: true |
| **Cross-Version Compatibility** | dt.UTC broke on Python 3.10 | Search for any Python 3.11+ only features |
| **Variable Rename Orphan** | `dte` renamed to `target_dte` but return dict still used `dte` | Search for ALL uses of old name when renaming |
| **Default Value Drift** | DEFAULT_OPTS had `dte:2` but CLI had `default=3` | Compare all code defaults vs config/CLI defaults |

---

## 🔍 Review Methodology

1. **Identify the pattern category** - Which of the above patterns does this change touch?
2. **Check the specific items** - Go through the relevant checklist items
3. **Test boundary conditions** - What happens with zero, empty, or error cases?
4. **Follow the data flow** - Trace from source to destination, check each transformation
5. **Consider multi-bot context** - How does this behave when 4 bots share the account?

---

## 🟢 LOW PATTERNS (Continued)

### Pattern 12: Variable Rename Orphans (NEW)
**Root Cause:** Renaming a variable but missing some references.
**Example:** Changed `dte` to `target_dte` but left `"dte": dte` in return dict.

**Checklist:**
- [ ] Any variable renames - search for ALL uses of the old name
- [ ] Any function renames - update all callers
- [ ] Any dict key renames - check all places that read the key
- [ ] Any parameter renames - update all keyword arguments

---

### Pattern 13: Default Value Drift
**Root Cause:** Multiple places define defaults for the same parameter, values diverge over time.
**Example:** `DEFAULT_OPTS["dte"]=2` but `argparse default=3`.

**Checklist:**
- [ ] Any hardcoded defaults in code - do they match config defaults?
- [ ] Any CLI arg defaults - do they match code defaults?
- [ ] Any workflow input defaults - do they match CLI defaults?
- [ ] When changing a default - search for and update ALL locations

---

## 📅 Issues Found by Date

### December 14, 2025
| Issue | Pattern | File | Fix |
|:---|:---|:---|:---|
| Negative time value not logged | Pattern 6 | live_spreads_loop.py | Added logging |
| DEFAULT_OPTS dte=2 vs CLI dte=3 | Pattern 13 | backtest_spreads.py | Unified to 3 |
| calculate_actual_dte no type guard | Pattern 5 | backtest_spreads.py | Added hasattr check |
| stock_price=0 not warned | Pattern 5 | live_spreads_loop.py | Added warning log |
| max_debit default 1.00 vs 2.00 | Pattern 13 | backtest_spreads.py | Changed to 2.00 |
| Docstring said "1-3 DTE" | Docs | backtest_spreads.py | Updated for variable DTE |
| dte renamed but missed in return | Pattern 12 | backtest_spreads.py | Fixed in a734fe1 |

---

*Last Updated: 2024-12-14*
*Derived from: Issues found in sessions a535f5b4, d02a1ffc, 82d17c66, and live trading observations*
