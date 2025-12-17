# 🤖 RubberBandBot Configuration Quick Reference

## 📈 15-Minute Stock Bot (15M_STK)

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
| **Slope Threshold** | -0.20 | CLI/YAML |
| **Stop Loss** | 2.5 × ATR | config.yaml |
| **Take Profit** | 1.5 × ATR | config.yaml |
| **Max Notional** | $2,000 | config.yaml |
| **Entry Window** | 09:45 - 15:45 ET | config.yaml |
| **Kill Switch** | 25% daily loss | Hardcoded |

---

## 📊 15-Minute Options Bot (15M_OPT)

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
| **Slope Threshold** | -0.20 | CLI/Workflow |
| **Scan Interval** | 15 minutes | Hardcoded |

---

## 📅 Weekly Stock Bot (WK_STK)

| Parameter | Value | Source |
|:----------|:------|:-------|
| **Timeframe** | 1 Week | config_weekly.yaml |
| **Keltner EMA** | 10 weeks | config_weekly.yaml |
| **Keltner ATR** | 10 weeks | config_weekly.yaml |
| **ATR Mult** | 2.0 | config_weekly.yaml |
| **RSI Period** | 14 | config_weekly.yaml |
| **RSI Oversold** | < 45 | config_weekly.yaml |
| **Mean Deviation** | -5% | config_weekly.yaml |
| **Stop Loss** | 2.0 × ATR | config_weekly.yaml |
| **Take Profit** | 2.5 × Risk | config_weekly.yaml |
| **Max Notional** | $2,000 | config_weekly.yaml |
| **Max Positions** | 5 | config_weekly.yaml |
| **Time Stop** | 20 weeks | Hardcoded |

---

## 📆 Weekly Options Bot (WK_OPT)

| Parameter | Value | Source |
|:----------|:------|:-------|
| **Target Delta** | 0.65 | Hardcoded |
| **Target DTE** | 45 days | Hardcoded |
| **Max Premium** | $1,000 | CLI (--max-premium) |
| **Tickers File** | tickers_weekly.txt | CLI default |

---

## 🔑 Environment Variables (All Bots)

| Variable | Purpose |
|:---------|:--------|
| `APCA_API_KEY_ID` | Alpaca API Key |
| `APCA_API_SECRET_KEY` | Alpaca API Secret |
| `APCA_API_BASE_URL` | Alpaca Base URL (paper) |

---

*Last Updated: December 2025*
