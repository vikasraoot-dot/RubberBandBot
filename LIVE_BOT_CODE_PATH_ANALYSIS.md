# RubberBandBot Live Bot Code Path Analysis

> **Analysis Date**: January 27, 2026
> **Method**: Agent trace + manual cross-verification
> **Scope**: LIVE trading bots ONLY (not backtest)

---

## Executive Summary

The live bots use a **completely different code path** than the backtest code. Key findings:

| Claim from Original Analysis | Verification Result | Impact |
|------------------------------|---------------------|--------|
| ADX=80 blocks trades | **FALSE for live** | ADX not used by any live bot |
| `stop_loss_pct: -0.80` used | **FALSE** | Live uses ATR-based: `atr_mult_sl: 2.5` |
| `take_profit_pct: 0.50` used | **FALSE** | Live uses ATR-based: `take_profit_r: 1.5` |
| `filters.py` controls entry | **FALSE** | Live uses `strategy.py` |
| 31.9% win rate applies | **MISLEADING** | That was from different backtest config |

---

## Cross-Verification: Agent Findings vs Manual Check

### Finding 1: `filters.py` NOT Used by Live Bots

**Agent Claim**: No live bot imports `filters.py`

**Manual Verification**:
```bash
# grep "^from|^import" live_paper_loop.py
Line 34: from RubberBand.strategy import attach_verifiers, check_slope_filter, check_bearish_bar_filter
# NO IMPORT FROM filters.py
```

**VERIFIED**: Live bots import from `strategy.py`, NOT `filters.py`. The `explain_long_gate()` function with ADX=80 check is **never called**.

---

### Finding 2: ADX Not Used by Live Bots

**Agent Claim**: ADX is calculated but never filtered on

**Manual Verification**:
```bash
# grep -n "adx|ADX" live_paper_loop.py
# Result: No matches found
```

**strategy.py** Line 32 calculates ADX:
```python
df = ta_add_adx_di(df, period=14)  # Add ADX for trend strength filtering
```

But Line 76 signal logic does NOT include ADX:
```python
df["long_signal"] = df["below_lower_band"] & df["rsi_oversold"] & df["trend_ok"] & df["time_ok"]
```

**VERIFIED**: ADX is calculated but **never used in signal logic**. The `adx_threshold: 80` config is completely ignored.

---

### Finding 3: ATR-Based Brackets, NOT Percentage-Based

**Agent Claim**: Live bots use `atr_mult_sl` and `take_profit_r`, not `stop_loss_pct` and `take_profit_pct`

**Manual Verification** (live_paper_loop.py):
```python
# Line 393-394
sl_mult = float(brackets.get("atr_mult_sl", 2.5))
tp_r = float(brackets.get("take_profit_r", 1.5))

# Line 677-679
if side == "buy":
    stop_price = round(entry - sl_mult * atr_val, 2)
    take_profit = round(entry + tp_r * atr_val, 2)
```

**VERIFIED**:
- Stop Loss = Entry - 2.5 × ATR
- Take Profit = Entry + 1.5 × ATR
- The `stop_loss_pct: -0.80` and `take_profit_pct: 0.50` in config are **NEVER READ**

---

### Finding 4: Weekly Bots Use Different Code Path

**Agent Claim**: Weekly bots use `backtest_weekly.py` → `attach_indicators()`, not `strategy.py`

**Manual Verification** (live_weekly_loop.py):
```python
# Line 28
from RubberBand.scripts.backtest_weekly import attach_indicators
# NO import from strategy.py
```

**VERIFIED**: Weekly bots have **NO filters** (no slope, no bearish bar). Only RSI + price stretch conditions.

---

## Actual Live Bot Architecture

### 15M Stock Bot (`live_paper_loop.py`)

```
IMPORTS:
  ├─ RubberBand/strategy.py → attach_verifiers(), check_slope_filter(), check_bearish_bar_filter()
  └─ RubberBand/src/regime_manager.py → RegimeManager

ENTRY FILTERS ACTUALLY APPLIED:
  1. Slope Filter (Line 524) - REGIME-DRIVEN
     - PANIC: Skip if slope < -0.20 (falling knife)
     - CALM: Skip if slope > -0.08 (too flat)
     - NORMAL: Skip if slope > -0.12

  2. Bearish Bar Filter (Line 538)
     - Skip if Close < Open
     - ENABLED for stocks

  3. Trend Filter (Line 576)
     - Skip if price < SMA_200

  4. Dead Knife Filter (Line 585)
     - Skip if RSI < 20 AND had loss today
     - Only in PANIC regime

  5. Entry Windows (Line 271)
     - Default: 09:45-15:45 ET

EXIT LOGIC:
  - Stop Loss: Entry - 2.5 × ATR
  - Take Profit: Entry + 1.5 × ATR
  - R:R Ratio: 2.5:1.5 = 1.67:1 (still unfavorable)
  - Required win rate: 62.5%
```

### 15M Options Bot (`live_spreads_loop.py`)

```
IMPORTS:
  ├─ RubberBand/strategy.py → attach_verifiers(), check_slope_filter()
  └─ RubberBand/src/regime_manager.py → RegimeManager

ENTRY FILTERS ACTUALLY APPLIED:
  1. Slope Filter (Line 388) - REGIME-DRIVEN
  2. Trend Filter (Line 381-385) - SMA_100
  3. Bearish Bar - **DISABLED** (too many volatile tickers have red bars)
  4. Min DTE: 3 days
  5. Max Debit: $3.00

EXIT LOGIC:
  - Take Profit: +80% spread profit
  - Stop Loss: -80% spread loss
  - R:R Ratio: 80:80 = 1:1 (neutral)
  - Required win rate: 50%
```

### Weekly Stock Bot (`live_weekly_loop.py`)

```
IMPORTS:
  ├─ RubberBand/scripts/backtest_weekly.py → attach_indicators()
  └─ RubberBand/src/regime_manager.py → RegimeManager

ENTRY FILTERS ACTUALLY APPLIED:
  **NONE** - Only two conditions:
  1. RSI < threshold (45 default, regime-adjusted)
  2. Price < 5% below SMA_20

EXIT LOGIC:
  - Stop Loss: Entry - 2.0 × ATR
  - Take Profit: Entry + 2.5 × Risk (where Risk = Entry - SL)
  - R:R Ratio: 1:2.5 (FAVORABLE)
  - Required win rate: 28.6%
  - Time Stop: 20 weeks max hold
```

### Weekly Options Bot (`live_weekly_options_loop.py`)

```
IMPORTS:
  ├─ RubberBand/scripts/backtest_weekly.py → attach_indicators()
  └─ RubberBand/src/regime_manager.py → RegimeManager

ENTRY FILTERS ACTUALLY APPLIED:
  **NONE** - Only two conditions:
  1. Previous week RSI < threshold
  2. Previous week price < 5% below SMA_20

EXIT LOGIC:
  - Take Profit: +100% option profit
  - Stop Loss: -50% option loss
  - R:R Ratio: 1:2 (FAVORABLE)
  - Required win rate: 33.3%
```

---

## Config Keys: Used vs Ignored

### USED by Live Bots

| Key | Bot | Value | Location |
|-----|-----|-------|----------|
| `brackets.atr_mult_sl` | 15M Stock | 2.5 | Line 393 |
| `brackets.take_profit_r` | 15M Stock | 1.5 | Line 394 |
| `trend_filter.sma_period` | 15M Stock | 200 | Line 316 |
| `bearish_bar_filter` | 15M Stock | true | Line 538 |
| `entry_windows` | 15M bots | [...] | Line 270 |
| `feed` | All | "iex" | Line 282 |

### IGNORED by Live Bots (Dead Config)

| Key | Value | Reason |
|-----|-------|--------|
| `filters.adx_threshold` | 80 | Only in `filters.py`, never imported |
| `filters.rsi_oversold` | 25 | Hardcoded as 30 in `attach_verifiers()` |
| `filters.min_dollar_vol` | 1000000 | Never read |
| `filters.min_price` | 5.0 | Never read |
| `brackets.stop_loss_pct` | -0.80 | ATR-based used instead |
| `brackets.take_profit_pct` | 0.50 | ATR-based used instead |
| `slope_threshold_10` | -0.15 | Legacy, passes through |

---

## Corrected Risk/Reward Analysis

### 15M Stock Bot (PROBLEM)

```
Stop Loss:    Entry - 2.5 × ATR
Take Profit:  Entry + 1.5 × ATR
Risk:         2.5 ATR units
Reward:       1.5 ATR units
R:R:          2.5:1.5 = 1.67:1 (unfavorable)

Required win rate for profitability:
= Risk / (Risk + Reward)
= 2.5 / (2.5 + 1.5)
= 2.5 / 4.0
= 62.5%
```

**Issue**: The bot needs to win 62.5% of trades to break even. This is aggressive for mean reversion.

### 15M Options Bot (NEUTRAL)

```
Take Profit:  +80%
Stop Loss:    -80%
R:R:          1:1 (neutral)

Required win rate: 50%
```

### Weekly Stock Bot (FAVORABLE)

```
Stop Loss:    Entry - 2.0 × ATR = Risk
Take Profit:  Entry + 2.5 × Risk
R:R:          1:2.5 (favorable)

Required win rate:
= 1 / (1 + 2.5)
= 1 / 3.5
= 28.6%
```

### Weekly Options Bot (FAVORABLE)

```
Take Profit:  +100%
Stop Loss:    -50%
R:R:          1:2 (favorable)

Required win rate: 33.3%
```

---

## Corrected Recommendations

### Priority 1: FIX 15M Stock R:R Ratio

The 15M stock bot has unfavorable R:R (1.67:1). Options:

**Option A**: Reduce risk
```yaml
brackets:
  atr_mult_sl: 1.5    # Was 2.5
  take_profit_r: 1.5  # Keep same
# New R:R = 1:1, Required win rate = 50%
```

**Option B**: Increase reward
```yaml
brackets:
  atr_mult_sl: 2.5    # Keep same
  take_profit_r: 2.5  # Was 1.5
# New R:R = 1:1, Required win rate = 50%
```

**Option C**: Match weekly strategy
```yaml
brackets:
  atr_mult_sl: 2.0    # Reduce from 2.5
  take_profit_r: 2.5  # Increase from 1.5 (2.5 × risk)
# New R:R = 1:1.25, Required win rate = 44%
```

### Priority 2: Clean Up Dead Config

Remove or document unused config keys to prevent future confusion:
- `filters.adx_threshold`
- `filters.min_dollar_vol`
- `filters.min_price`
- `brackets.stop_loss_pct`
- `brackets.take_profit_pct`

### Priority 3: Do NOT Change

- `adx_threshold: 80` - Irrelevant (not used by live bots)
- Weekly bot settings - Already have favorable R:R
- Options bot 80/80 settings - Intentional for 3-DTE volatility

---

## Summary: What Actually Controls Live Trading

| Bot | Entry Filters | Exit Logic | R:R | Win Rate Needed |
|-----|---------------|------------|-----|-----------------|
| 15M Stock | Slope, Bearish, Trend, DKF | ATR-based | 1.67:1 | 62.5% |
| 15M Options | Slope, Trend | 80%/-80% | 1:1 | 50% |
| Weekly Stock | None (RSI+Stretch) | ATR-based | 1:2.5 | 28.6% |
| Weekly Options | None (RSI+Stretch) | 100%/-50% | 1:2 | 33.3% |

**Regime Manager** controls:
- Slope thresholds per regime (PANIC/CALM/NORMAL)
- Dead Knife Filter enable/disable
- Weekly RSI thresholds
- Weekly mean deviation thresholds

---

## Verification Commands Used

```bash
# Imports check
grep "^from\|^import" live_paper_loop.py
grep "^from\|^import" live_weekly_loop.py

# ADX check
grep -n "adx\|ADX" live_paper_loop.py

# Bracket config check
grep -n "stop_loss_pct\|take_profit_pct\|atr_mult_sl\|take_profit_r" live_paper_loop.py
```

---

**Document Version**: 1.0
**Cross-Verified**: January 27, 2026
