---
description: Audit per-bot P&L from GitHub Actions logs (source of truth)
argument-hint: "[--days 10] [--bot 15M_STK] [--date 2026-02-19]"
allowed-tools: [Read, Glob, Grep, Bash]
---

# Bot P&L Audit

Audit each bot's realized P&L by pulling structured log events (EOD_SUMMARY / BOT_STOP) directly from GitHub Actions run logs. This avoids the cross-wiring problem of Alpaca order matching, where trades from one bot get mis-attributed to another.

## Arguments

The user invoked this command with: $ARGUMENTS

## Instructions

### Step 1: Run the audit script

```bash
python RubberBand/scripts/audit_bot_pnl.py $ARGUMENTS
```

If no arguments were provided, run with `--days 10` to get the last 10 calendar days.

Capture both stdout (JSON report) and stderr (progress + summary).

### Step 2: Parse the JSON output

Read the JSON output and extract:
- Per-bot per-day P&L from `bots.{tag}.days[]`
- Lifetime summaries from `bots.{tag}.summary`
- Any discrepancies from `discrepancies[]`

### Step 3: Present a summary table

Format the results as a daily P&L matrix:

```
| Date       | 15M_STK | 15M_OPT | WK_STK | WK_OPT | Daily Total |
|------------|---------|---------|--------|--------|-------------|
| 2026-02-19 |   $0.00 | -$41.00 |  $0.00 |  $0.00 |     -$41.00 |
| ...        |         |         |        |        |             |
| TOTAL      |         |         |        |        |             |
```

Use color/formatting:
- Positive P&L: show as-is
- Negative P&L: highlight
- No data: show "N/A"
- Discrepancies vs watchdog: flag with asterisk

### Step 4: Per-bot analysis

For each bot, report:
- **Lifetime P&L** over the date range
- **Win rate** and total trade count
- **Best and worst days** with details
- **Average daily P&L**
- **Open positions** still held (from most recent day)

### Step 5: Discrepancy analysis

If any discrepancies found (bot P&L != watchdog P&L by > $1):
1. Flag each discrepancy with date, bot, and delta
2. Investigate likely root causes:
   - Bracket order exits attributed to wrong bot
   - Multi-leg orders double-counted
   - Another bot trading the same symbol on the same day
3. State which number to trust (bot's own log = source of truth)

### Step 6: Cross-reference with existing reports

Optionally read:
- `results/watchdog/performance.jsonl` for the watchdog's recorded metrics
- `results/daily/{date}.json` for the daily reconciliation view
- Compare and note any structural issues in the reporting pipeline

### Step 7: Present final analysis

Provide:
- **Confidence level**: "Verified from bot's own logs" (high) vs "Estimated" (low)
- **Key insights**: Which bots are profitable, which are bleeding money
- **Recommendations**: Any bots that should be paused, investigated, or tuned
- **Market context**: Note if losses align with broad market conditions
