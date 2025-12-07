# Weekly RubberBand Strategy - Development Context & History

**Date:** December 2025
**Objective:** Develop a standalone weekly mean reversion strategy to improve upon the 15-minute timeframe performance.

---

## 1. Strategy Evolution & Experiments

We iterated through several versions of the strategy to find the optimal configuration. Here is the chronological history of our experiments:

### A. Initial Baseline (RSI Only)
- **Logic:** Buy when RSI < 45.
- **Result:** $48,120 Profit, 341 Trades, 66.8% Win Rate.
- **Verdict:** Good profit, but too many trades and signal quality was mediocre ($141/trade).

### B. Adding "RubberBand" Physics (Mean Deviation)
- **Logic:** RSI < 45 AND Price must be stretched below the mean.
- **Experiment 1 (-10% Stretch):** Price < 10% below 20-week SMA.
    - Result: $3,403 Profit, 4 Trades.
    - Verdict: Too restrictive.
- **Experiment 2 (-5% Stretch):** Price < 5% below 20-week SMA.
    - Result: **$23,492 Profit**, 110 Trades, **80% Win Rate**.
    - Verdict: **The Sweet Spot.** Excellent balance of frequency and quality.

### C. Adding Trend Filters (The 50-Week SMA)
- **Logic:** Only buy if 20-week SMA > 50-week SMA (Golden Cross).
- **Result:** Trades dropped to nearly zero (8 trades).
- **Verdict:** Failed. Weekly mean reversion often happens *during* corrections where the specific SMA order might be messy. The "Mean Deviation" check is sufficient safety.

### D. The "Early Exit" Experiment
- **Logic:** If trade is down 3% after holding for 2 weeks, exit early to prevent deep losses.
- **Result:** Profit dropped to $19,173 (Lost $4.3k compared to baseline).
- **Why it failed:** Many weekly trades dip initially before ripping higher. Cutting them early locked in losses that turned into wins.
- **Verdict:** Reverted. Standard Stop Loss (2x ATR) is superior.

### E. MSTR (MicroStrategy) Volatility
- **Observation:** MSTR was responsible for the biggest losses (-$2,345 total).
- **Decision:** We considered excluding it, but since the overall portfolio profit was $23.5k, we kept it to maintain reduced complexity. The big winners outweigh the specific MSTR volatility.

---

## 2. Final Strategy Configuration (The Winner)

This is the state of the code as of the end of this session.

### Core Logic
*   **Timeframe:** Weekly (1Week bars)
*   **Entry:**
    1.  **RSI < 45** (Oversold momentum)
    2.  **Price < (20-week SMA * 0.95)** (Stretched 5% below mean)
*   **Exit:**
    1.  **Mean Reversion:** Close > Keltner Channel Middle (Reverted to mean)
    2.  **Stop Loss:** Entry - (2.0 * ATR)
    3.  **Take Profit:** Entry + (2.5 * Risk)

### Performance Benchmark (365 Days)
*   **Net Profit:** $23,492
*   **Win Rate:** 80%
*   **Trades:** 110
*   **Avg Profit Per Trade:** $214
*   **Comparison:** Beats the 15-minute strategy by ~8x in total profit.

---

## 3. Key Files

*   `RubberBand/scripts/backtest_weekly.py`: The standalone backtesting engine. Contains all the custom logic for the 3-factor checks (though currently simplified to RSI + MeanDev).
*   `RubberBand/config_weekly.yaml`: Configuration file with the optimized parameters (RSI 45, MeanDev Threshold -5).

## 4. Next Steps (For Future)

*   **Live Implementation:** The strategy currently only exists as a backtest script. Needs to be ported to a live trading loop (`live_weekly_loop.py`).
*   **Alerting:** Set up weekly cron job (e.g., Monday morning) to scan for these setups.
*   **Position Sizing:** Currently fixed at $5k/trade. Could be dynamic based on portfolio equity.
