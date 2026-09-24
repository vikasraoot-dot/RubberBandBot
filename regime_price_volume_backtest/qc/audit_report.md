# Manual trade audit (independent recomputation)

**Overall: ALL CHECKS PASSED** (8 trades).

Each sampled FINAL trade is recomputed with plain-Python loops from the cached bars (`src/audit.py`) and compared with the engine output (`trades.csv`). Tolerance: 1e-6 relative on prices/returns; breadth within 0.5 pp (the engine uses pandas rolling means over the same members).

## GILD — signal 2020-04-16 (dev)

EMA20(signal) = 58.8866, ATR14(signal) = 2.9875

| check | engine | independent | ok |
|---|---|---|---|
| stock trend qualifies | True | True | ✅ |
| pullback qualifies (low within 2% of EMA20 in 5 sessions, close > EMA20) | True | True | ✅ |
| entry date (next session) | 2020-04-17 | 2020-04-17 | ✅ |
| entry price (next open) | 66.9739 | 66.9739 | ✅ |
| initial stop (entry - 1.5 ATR14) | 62.4926 | 62.4926 | ✅ |
| exit date | 2020-04-21 | 2020-04-21 | ✅ |
| exit price | 62.4926 | 62.4926 | ✅ |
| exit reason | stop | stop | ✅ |
| net return | -0.0678432 | -0.0678432 | ✅ |
| MFE | 0.0072796 | 0.0072796 | ✅ |
| MAE | -0.0669106 | -0.0669106 | ✅ |
| peak close return | -0.0138546 | -0.0138546 | ✅ |
| giveback (pp) | nan | nan | ✅ |
| SPY > SMA200 at signal | False | False | ✅ |
| SPY SMA50 > SMA200 at signal | False | False | ✅ |
| VIX close at signal | 40.11 | 40.11 | ✅ |
| breadth % > SMA50 at signal | 27.6549 | 27.6549 | ✅ |

## GILD — signal 2020-03-18 (dev)

EMA20(signal) = 56.7001, ATR14(signal) = 3.6156

| check | engine | independent | ok |
|---|---|---|---|
| stock trend qualifies | True | True | ✅ |
| pullback qualifies (low within 2% of EMA20 in 5 sessions, close > EMA20) | True | True | ✅ |
| entry date (next session) | 2020-03-19 | 2020-03-19 | ✅ |
| entry price (next open) | 63.6791 | 63.6791 | ✅ |
| initial stop (entry - 1.5 ATR14) | 58.2557 | 58.2557 | ✅ |
| exit date | 2020-03-20 | 2020-03-20 | ✅ |
| exit price | 58.2557 | 58.2557 | ✅ |
| exit reason | stop | stop | ✅ |
| net return | -0.0860811 | -0.0860811 | ✅ |
| MFE | 0.0616201 | 0.0616201 | ✅ |
| MAE | -0.0851668 | -0.0851668 | ✅ |
| peak close return | -0.0300074 | -0.0300074 | ✅ |
| giveback (pp) | nan | nan | ✅ |
| SPY > SMA200 at signal | False | False | ✅ |
| SPY SMA50 > SMA200 at signal | True | True | ✅ |
| VIX close at signal | 76.45 | 76.45 | ✅ |
| breadth % > SMA50 at signal | 3.53982 | 3.53982 | ✅ |

## JPM — signal 2021-03-25 (dev)

EMA20(signal) = 132.1925, ATR14(signal) = 3.4471

| check | engine | independent | ok |
|---|---|---|---|
| stock trend qualifies | True | True | ✅ |
| pullback qualifies (low within 2% of EMA20 in 5 sessions, close > EMA20) | True | True | ✅ |
| entry date (next session) | 2021-03-26 | 2021-03-26 | ✅ |
| entry price (next open) | 134.583 | 134.583 | ✅ |
| initial stop (entry - 1.5 ATR14) | 129.412 | 129.412 | ✅ |
| exit date | 2021-04-15 | 2021-04-15 | ✅ |
| exit price | 132.699 | 132.699 | ✅ |
| exit reason | signal | signal | ✅ |
| net return | -0.0149789 | -0.0149789 | ✅ |
| MFE | 0.0251208 | 0.0251208 | ✅ |
| MAE | -0.0211276 | -0.0211276 | ✅ |
| peak close return | 0.0187973 | 0.0187973 | ✅ |
| giveback (pp) | 0.0327907 | 0.0327907 | ✅ |
| SPY > SMA200 at signal | True | True | ✅ |
| SPY SMA50 > SMA200 at signal | True | True | ✅ |
| VIX close at signal | 19.81 | 19.81 | ✅ |
| breadth % > SMA50 at signal | 76.5217 | 76.5217 | ✅ |

## PM — signal 2021-06-01 (dev)

EMA20(signal) = 75.5037, ATR14(signal) = 1.0422

| check | engine | independent | ok |
|---|---|---|---|
| stock trend qualifies | True | True | ✅ |
| pullback qualifies (low within 2% of EMA20 in 5 sessions, close > EMA20) | True | True | ✅ |
| entry date (next session) | 2021-06-02 | 2021-06-02 | ✅ |
| entry price (next open) | 76.0313 | 76.0313 | ✅ |
| initial stop (entry - 1.5 ATR14) | 74.468 | 74.468 | ✅ |
| exit date | 2021-06-09 | 2021-06-09 | ✅ |
| exit price | 75.4056 | 75.4056 | ✅ |
| exit reason | signal | signal | ✅ |
| net return | -0.00922077 | -0.00922077 | ✅ |
| MFE | 0.0178996 | 0.0178996 | ✅ |
| MAE | -0.0116245 | -0.0116245 | ✅ |
| peak close return | 0.0138875 | 0.0138875 | ✅ |
| giveback (pp) | 0.0221169 | 0.0221169 | ✅ |
| SPY > SMA200 at signal | True | True | ✅ |
| SPY SMA50 > SMA200 at signal | True | True | ✅ |
| VIX close at signal | 17.9 | 17.9 | ✅ |
| breadth % > SMA50 at signal | 73.3766 | 73.3766 | ✅ |

## DD — signal 2023-02-07 (oos)

EMA20(signal) = 85.8315, ATR14(signal) = 2.1191

| check | engine | independent | ok |
|---|---|---|---|
| stock trend qualifies | True | True | ✅ |
| pullback qualifies (low within 2% of EMA20 in 5 sessions, close > EMA20) | True | True | ✅ |
| entry date (next session) | 2023-02-08 | 2023-02-08 | ✅ |
| entry price (next open) | 90.0983 | 90.0983 | ✅ |
| initial stop (entry - 1.5 ATR14) | 86.9197 | 86.9197 | ✅ |
| exit date | 2023-02-17 | 2023-02-17 | ✅ |
| exit price | 86.9197 | 86.9197 | ✅ |
| exit reason | stop | stop | ✅ |
| net return | -0.0362441 | -0.0362441 | ✅ |
| MFE | 0.0154124 | 0.0154124 | ✅ |
| MAE | -0.0352798 | -0.0352798 | ✅ |
| peak close return | -0.00401503 | -0.00401503 | ✅ |
| giveback (pp) | nan | nan | ✅ |
| SPY > SMA200 at signal | True | True | ✅ |
| SPY SMA50 > SMA200 at signal | True | True | ✅ |
| VIX close at signal | 18.66 | 18.66 | ✅ |
| breadth % > SMA50 at signal | 72.4211 | 72.4211 | ✅ |

## CDNS — signal 2025-08-28 (oos)

EMA20(signal) = 347.4725, ATR14(signal) = 7.7548

| check | engine | independent | ok |
|---|---|---|---|
| stock trend qualifies | True | True | ✅ |
| pullback qualifies (low within 2% of EMA20 in 5 sessions, close > EMA20) | True | True | ✅ |
| entry date (next session) | 2025-08-29 | 2025-08-29 | ✅ |
| entry price (next open) | 353.6 | 353.6 | ✅ |
| initial stop (entry - 1.5 ATR14) | 341.968 | 341.968 | ✅ |
| exit date | 2025-09-02 | 2025-09-02 | ✅ |
| exit price | 341.968 | 341.968 | ✅ |
| exit reason | stop | stop | ✅ |
| net return | -0.0338629 | -0.0338629 | ✅ |
| MFE | 0.00460974 | 0.00460974 | ✅ |
| MAE | -0.0328963 | -0.0328963 | ✅ |
| peak close return | -0.00896497 | -0.00896497 | ✅ |
| giveback (pp) | nan | nan | ✅ |
| SPY > SMA200 at signal | True | True | ✅ |
| SPY SMA50 > SMA200 at signal | True | True | ✅ |
| VIX close at signal | 14.43 | 14.43 | ✅ |
| breadth % > SMA50 at signal | 65.7841 | 65.7841 | ✅ |

## ADSK — signal 2023-07-11 (oos)

EMA20(signal) = 205.5844, ATR14(signal) = 5.4238

| check | engine | independent | ok |
|---|---|---|---|
| stock trend qualifies | True | True | ✅ |
| pullback qualifies (low within 2% of EMA20 in 5 sessions, close > EMA20) | True | True | ✅ |
| entry date (next session) | 2023-07-12 | 2023-07-12 | ✅ |
| entry price (next open) | 216.67 | 216.67 | ✅ |
| initial stop (entry - 1.5 ATR14) | 208.534 | 208.534 | ✅ |
| exit date | 2023-07-24 | 2023-07-24 | ✅ |
| exit price | 210.82 | 210.82 | ✅ |
| exit reason | signal | signal | ✅ |
| net return | -0.0279721 | -0.0279721 | ✅ |
| MFE | 0.0275073 | 0.0275073 | ✅ |
| MAE | -0.0342456 | -0.0342456 | ✅ |
| peak close return | 0.0110768 | 0.0110768 | ✅ |
| giveback (pp) | 0.0380763 | 0.0380763 | ✅ |
| SPY > SMA200 at signal | True | True | ✅ |
| SPY SMA50 > SMA200 at signal | True | True | ✅ |
| VIX close at signal | 14.84 | 14.84 | ✅ |
| breadth % > SMA50 at signal | 80.167 | 80.167 | ✅ |

## HD — signal 2024-06-28 (oos)

EMA20(signal) = 323.1459, ATR14(signal) = 6.3572

| check | engine | independent | ok |
|---|---|---|---|
| stock trend qualifies | True | True | ✅ |
| pullback qualifies (low within 2% of EMA20 in 5 sessions, close > EMA20) | True | True | ✅ |
| entry date (next session) | 2024-07-01 | 2024-07-01 | ✅ |
| entry price (next open) | 324.731 | 324.731 | ✅ |
| initial stop (entry - 1.5 ATR14) | 315.195 | 315.195 | ✅ |
| exit date | 2024-07-02 | 2024-07-02 | ✅ |
| exit price | 316.309 | 316.309 | ✅ |
| exit reason | signal | signal | ✅ |
| net return | -0.0269097 | -0.0269097 | ✅ |
| MFE | 0.00212253 | 0.00212253 | ✅ |
| MAE | -0.0259362 | -0.0259362 | ✅ |
| peak close return | -0.0224762 | -0.0224762 | ✅ |
| giveback (pp) | nan | nan | ✅ |
| SPY > SMA200 at signal | True | True | ✅ |
| SPY SMA50 > SMA200 at signal | True | True | ✅ |
| VIX close at signal | 12.44 | 12.44 | ✅ |
| breadth % > SMA50 at signal | 48.6542 | 48.6542 | ✅ |
