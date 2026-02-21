# HMM Regime Terminal: Gap Analysis & Implementation Roadmap

## Date: 2026-02-18
## Source: YouTube video analysis ("AI Pathways" — Regime Terminal architecture)
## Reference: `Youtube video analysis.txt` in repo root

---

## Gap Analysis: Current Bot vs Regime Terminal Architecture

### Where We Are Now

The bot uses a **deterministic, rule-based** two-layer system:
- **Layer 1 (RegimeManager):** VIXY price vs Bollinger Band + volume confirmation -> 3 states (PANIC/CALM/NORMAL)
- **Layer 2 (MarketConditionClassifier):** SPY ADX/ATR/breadth -> 5 conditions (CHOPPY/TRENDING_UP/DOWN/RANGE/BREAKOUT)
- **Filters:** Slope filter, dead knife filter, bearish bar filter -- all boolean gates
- **Cooldown:** 90-minute hardcoded intraday cooldown after PANIC

### What the Regime Terminal Proposes

A **probabilistic, ML-driven** two-factor system:
- **Layer 1 (HMM Engine):** Gaussian HMM trained on returns + range + volume change -> 7 hidden states with confidence scores
- **Layer 2 (Strategy Confirmation):** 7-out-of-8 indicator checklist gated by regime
- **Cooldown:** 48-hour hard cooldown after any exit

### The 7 Key Gaps

| # | Gap | Current | Proposed |
|---|-----|---------|----------|
| 1 | **State detection method** | Hard thresholds on VIXY | Gaussian HMM with learned parameters |
| 2 | **Number of states** | 3 (PANIC/CALM/NORMAL) | 7 (auto-labeled from data) |
| 3 | **Input features** | Single indicator (VIXY) | 3 engineered features (returns, range, vol change) |
| 4 | **Confidence scoring** | None -- binary pass/fail | Probability distribution across all states |
| 5 | **Entry confirmation** | 3 sequential boolean filters | 7-out-of-8 modular checklist (RSI, momentum, volatility, volume, ADX, price action, MACD) |
| 6 | **Exit logic** | Bracket orders (ATR-based SL/TP) | Regime flip = immediate close |
| 7 | **Post-exit cooldown** | 90 minutes (intraday only) | 48 hours (hard, cross-session) |

### What to Keep (Infrastructure the Video Doesn't Address)

- Circuit breakers, profit locks, daily loss limits
- Position registry with broker reconciliation
- Structured JSONL audit logging
- Watchdog monitoring and EOD analysis
- Market breadth filter (% above SMA-100)
- Dynamic position sizing via overrides

These are **complementary** to HMM -- not replaced by it.

---

## Current Architecture Deep Dive

### RegimeManager (`RubberBand/src/regime_manager.py`)

**Daily Regime Logic (Lines 82-199):**
- Input: 35 days of VIXY daily bars
- Indicators: SMA_20, STD_20, Bollinger upper band, volume SMA_20, daily % change
- Classification rules (priority order):
  - Price > UpperBand AND Volume > 1.5x AvgVol -> PANIC
  - Daily % Change > +8% AND Volume > 1.5x AvgVol -> PANIC
  - Price > UpperBand BUT Volume <= 1.5x AvgVol -> NORMAL (fake breakout)
  - Price < SMA_20 for 3 consecutive days -> CALM (hysteresis buffer)
  - Default -> NORMAL

**Intraday Panic Detection (Lines 205-342):**
- Fetches VIXY 5-min bars (IEX feed, real-time)
- Triggers: >8% intraday spike, Bollinger breakout with volume, >5% above upper band
- 90-minute cooldown after trigger

**Config Overrides per Regime:**
- CALM: dead_knife_filter=False, weekly_rsi_oversold=50
- NORMAL: dead_knife_filter=False, weekly_rsi_oversold=45
- PANIC: dead_knife_filter=True, weekly_rsi_oversold=30

### MarketConditionClassifier (`RubberBand/src/watchdog/market_classifier.py`)

**Five Market Conditions (SPY-based, Lines 329-400):**

| Priority | Condition | Result | Position Size |
|----------|-----------|--------|--------------|
| 1 | ATR_5/ATR_20 > 1.5 | BREAKOUT | 0.5x |
| 2 | ATR ratio < 0.8 AND reversals >= 12/20 | CHOPPY | 0.5x |
| 3 | Close > SMA_20 AND ADX > 25 AND +DI > -DI | TRENDING_UP | 1.0x |
| 4 | Close < SMA_20 AND ADX > 25 AND -DI > +DI | TRENDING_DOWN | 0.5x |
| 5 | ADX < 20 | RANGE | 1.0x |

**Market Breadth (Lines 236-311):**
- pct_above_sma100 < 30% -> BEARISH -> size x 0.5
- pct_above_sma100 < 70% -> CAUTIOUS
- pct_above_sma100 >= 70% -> NORMAL

### Strategy Filters (`RubberBand/strategy.py`)

**Gate chain (sequential boolean):**
1. Slope filter (Keltner middle band 3-bar slope, regime-aware threshold)
2. Dead knife filter (PANIC-only: skip symbols with losses today)
3. Bearish bar filter (skip if close < open)
4. Long signal gate (RSI oversold entry)

### How Regime Feeds Into Live Trading (`live_paper_loop.py`)

```
RegimeManager.update()           -> daily regime (PANIC/CALM/NORMAL)
RegimeManager.check_intraday()   -> intraday override
MarketConditionClassifier        -> dynamic_overrides.json
  |
  v
Per-symbol scan:
  1. Slope filter (uses regime threshold)
  2. Dead knife filter (enabled in PANIC only)
  3. Bearish bar filter
  4. Signal check -> bracket order
  Position size *= market_condition multiplier
  TP R-multiple += market_condition adjustment
```

---

## Step-by-Step Refactoring Plan

### Phase 1: Build the HMM Regime Engine (new module, no production changes)

**New file: `RubberBand/src/hmm_regime.py`**

1. Train a 7-state Gaussian HMM on hourly SPY data (730 days / ~11,000 samples)
   - Feature 1: Log returns (`np.log(close / close.shift(1))`)
   - Feature 2: Normalized range (`(high - low) / close`)
   - Feature 3: Volume change (`volume / volume.rolling(20).mean()`)
   - Library: `hmmlearn.GaussianHMM`

2. Auto-label the 7 states by sorting state means on the returns feature:
   - Highest mean return -> "BULL_RUN"
   - Lowest mean return -> "CRASH"
   - Middle states -> "RECOVERY", "DISTRIBUTION", "ACCUMULATION", "CHOPPY", "NEUTRAL"

3. Expose a simple API:
   ```python
   class HMMRegimeDetector:
       def fit(self, hourly_df: pd.DataFrame) -> None
       def predict(self, hourly_df: pd.DataFrame) -> RegimeResult
       def is_bullish(self) -> bool
       def confidence(self) -> float
   ```

4. Persist trained model to disk (`results/hmm/model.pkl`) -- retrain weekly or on-demand.

### Phase 2: Build the 7-out-of-8 Confirmation Layer (new module)

**New file: `RubberBand/src/signal_checklist.py`**

| # | Condition | Current Equivalent | Status |
|---|-----------|-------------------|--------|
| 1 | RSI < 90 | RSI in strategy.py | Have it |
| 2 | Favorable Momentum | Keltner slope / slope filter | Have it |
| 3 | Favorable Volatility | ATR ratio (market_classifier.py) | Have it |
| 4 | Favorable Volume | dollar_vol_avg in data.py | Have it |
| 5 | ADX alignment | ADX in market_classifier.py | Have it -- need to pass to signal layer |
| 6 | Price Action confirmation | Bullish bar close > open | Easy to add |
| 7 | MACD alignment | Not currently computed | Need to add |
| 8 | Trend alignment | Daily SMA trend filter | Have it |

Build as a scoring function:
```python
def check_entry_conditions(df, regime_result, min_score=7) -> (bool, int, list):
    """Returns (should_enter, score, details). Requires min_score/8 to pass."""
```

Replaces the current sequential boolean gate chain with a scored checklist.

### Phase 3: Implement 48-Hour Signal Hysteresis

**Modify: `RubberBand/scripts/live_paper_loop.py`**

1. Track last exit timestamp in persisted JSON (`results/last_exit.json`)
2. At scan time: if `now - last_exit < 48 hours` -> skip all entries
3. Log `HYSTERESIS_COOLDOWN` event when entries blocked

**Key concern:** 48-hour cooldown is very conservative for 15-minute mean-reversion trading. May need to tune to 4-8 hours or make regime-dependent (48h after CRASH, 4h after normal exit).

### Phase 4: Integrate HMM into Live Loop (shadow mode first)

**Modify: `RubberBand/scripts/live_paper_loop.py`**

1. Run HMMRegimeDetector alongside RegimeManager -- log both, act on existing system
2. Shadow mode for 2+ weeks: compare HMM calls against actual P&L outcomes
3. Once validated: gate entries on `hmm.is_bullish() and hmm.confidence() > 0.6`
4. Keep RegimeManager as safety backstop (VIXY PANIC still overrides)

### Phase 5: Regime-Triggered Exits (new behavior)

**Modify: `RubberBand/scripts/live_paper_loop.py`**

1. Each scan cycle: check if HMM regime flipped from bullish to bearish since position opened
2. If yes -> submit market sell for all open positions
3. Log as `REGIME_EXIT` with old/new regime and confidence scores
4. Triggers the 48-hour cooldown

**Highest risk change.** Noisy HMM could whipsaw. Confidence threshold is the safety valve.

### Phase 6: Backtest Validation

**Modify: `RubberBand/scripts/backtest.py`**

1. Walk-forward: train HMM on T-730d to T, test on T forward
2. Viterbi decode full history for optimal state path
3. Compare trade outcomes by HMM regime vs current VIXY regime
4. Metrics to beat: filter effectiveness, win rate by regime, max drawdown

---

## Dependencies to Add

```
hmmlearn>=0.3,<0.4      # Gaussian HMM implementation
scikit-learn>=1.4        # StandardScaler for feature normalization
```

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| HMM overfits to training period | Walk-forward validation, retrain weekly |
| 7 states too many for our universe | Start with 3-4 states, increase if backtest shows value |
| 48-hour cooldown kills trade frequency | Make regime-dependent or tune to 4-8h |
| Regime-triggered exits cause whipsaw | Confidence threshold (>0.6) + 2 consecutive readings |
| HMM adds latency to scan loop | Pre-compute regime once per hour, cache result |
| Model file corruption | Fallback to current rule-based regime if model load fails |

---

## What NOT to Change

- Circuit breakers, profit locks, daily loss limits
- Bracket order execution (add regime exit as 3rd exit path, don't remove SL/TP)
- Position registry / reconciliation
- Watchdog / auditor pipeline (just add HMM regime to daily analysis JSON)
- Market breadth filter (complementary to HMM)

---

## Open Questions for Deep Dive

1. Should HMM train on SPY hourly data (as proposed) or VIXY (as current system uses)?
2. Is 7 states optimal, or should we start with 3-4 and expand?
3. 48-hour cooldown vs adaptive cooldown based on regime confidence?
4. Should regime-triggered exits apply to all bots (15M_STK, WK_STK, options) or just stock bots?
5. Walk-forward retraining cadence: weekly? monthly? on drawdown trigger?
6. How to handle the transition period: run both systems in parallel for how long?
7. Should the 7-out-of-8 checklist replace existing filters entirely or layer on top?
