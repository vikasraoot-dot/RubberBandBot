# RubberBandBot Configuration Quick Reference

## 15-Minute Stock Bot (15M_STK)

| Parameter | Value | Source |
|:----------|:------|:-------|
| **Timeframe** | 15m | config.yaml |
| **History** | 10 days | config.yaml |
| **Keltner EMA** | 20 | config.yaml |
| **Keltner Mult** | 2.0 | config.yaml |
| **ATR Length** | 14 | config.yaml |
| **RSI Length** | 14 | config.yaml |
| **RSI Oversold** | < 25 | config.yaml |
| **RSI Min** | 15 | config.yaml |
| **Slope Threshold** | -0.20 (all regimes) | RegimeManager |
| **Stop Loss** | 1.5 x ATR | config.yaml (brackets.atr_mult_sl) |
| **Take Profit** | 2.0 x ATR | config.yaml (brackets.take_profit_r) |
| **R:R Ratio** | 1.33:1 | Derived (TP/SL) |
| **Max Notional** | $2,000 (full) / $667 (weak bull) | config.yaml + Dual-SMA sizing |
| **Entry Window** | 09:45 - 15:45 ET | config.yaml |
| **Kill Switch** | 25% daily loss | Hardcoded |
| **Bearish Bar Filter** | Disabled | RegimeManager (all regimes) |

---

## 15-Minute Options Bot (15M_OPT)

| Parameter | Value | Source |
|:----------|:------|:-------|
| **DTE** | 3 days | CLI (--dte) |
| **Min DTE** | 3 days | Hardcoded |
| **Max DTE** | 14 days | Hardcoded |
| **Long Delta** | 0.55 | Hardcoded |
| **Short Delta** | 0.35 | Hardcoded |
| **Max Debit** | $3.00/share | CLI (--max-debit) |
| **Take Profit** | +50% | Hardcoded |
| **Stop Loss** | -50% | Hardcoded |
| **Slope Threshold** | -0.20 (all regimes) | RegimeManager |
| **Scan Interval** | 15 minutes | Hardcoded |

---

## Weekly Stock Bot (WK_STK)

| Parameter | Value | Source |
|:----------|:------|:-------|
| **Timeframe** | 1 Week | config_weekly.yaml |
| **Keltner EMA** | 10 weeks | config_weekly.yaml |
| **Keltner ATR** | 10 weeks | config_weekly.yaml |
| **ATR Mult** | 2.0 | config_weekly.yaml |
| **RSI Period** | 14 | config_weekly.yaml |
| **RSI Oversold** | < 45 | config_weekly.yaml |
| **Mean Deviation** | -5% | config_weekly.yaml |
| **Stop Loss** | 2.0 x ATR | config_weekly.yaml |
| **Take Profit** | 2.5 x Risk | config_weekly.yaml |
| **Max Notional** | $2,000 | config_weekly.yaml |
| **Max Positions** | 5 | config_weekly.yaml |
| **Time Stop** | 12 weeks | config_weekly.yaml (time_stop_weeks) |

---

## Weekly Options Bot (WK_OPT)

| Parameter | Value | Source |
|:----------|:------|:-------|
| **Target Delta** | 0.65 | Hardcoded |
| **Target DTE** | 45 days | Hardcoded |
| **Max Premium** | $1,000 | CLI (--max-premium) |
| **Tickers File** | tickers_weekly.txt | CLI default |

---

## Key Architecture Notes

- **Slope thresholds** in config.yaml are LEGACY values. RegimeManager overrides them (all regimes use -0.20 since Dec 2025).
- **Dual-SMA sizing**: When trend_filter_sma is disabled (set to 0), positions use full notional. When enabled, is_strong_bull determines full ($2,000) vs reduced ($667) sizing.
- **RegimeManager** classifies market as PANIC/NORMAL/CALM using VIXY Bollinger Bands with volume confirmation and 90-minute cooldown.

---

## Environment Variables (All Bots)

| Variable | Purpose |
|:---------|:--------|
| `APCA_API_KEY_ID` | Alpaca API Key |
| `APCA_API_SECRET_KEY` | Alpaca API Secret |
| `APCA_API_BASE_URL` | Alpaca Base URL (paper) |

---

*Last Updated: February 2026*
