# G1 diagnostics (WP-S1.7)

Selected v1 trial T09 (min_step 0.03, spacing 1.5 sigma), BTCUSDT, SOLUSDT, XRPUSDT, DOGEUSDT, LINKUSDT, 10,000 USDT, base costs. These are diagnostics, not trials: no gate or pre-registered setting changed, and the v1 FAIL stands.

## 1. Exposure audit (2021)

**Conclusion: the reported 1.1% is a correct holdings-only number (trace says 1.1%) but not the capital the grid ties up: with resting buys exposure is 19.5%, so the 2021 return is 88.4% on average deployed capital, not 1544.4% on held capital.**

| measure (2021, T09) | value |
|---|---|
| reported `avg_exposure` (decision-sampled, holdings, fraction of equity) | 1.12% |
| holdings only, time-weighted per minute | 1.12% |
| holdings + resting buys, time-weighted per minute | 19.49% |
| holdings + resting buys, sampled after each decision (`exposure_total_post`) | 19.50% |
| average held / deployed (USDT) | 126 / 2,195 |
| net P&L (USDT) and return | 1,941 (19.4%) |
| return on average held capital | 1544.4% |
| **return on average deployed capital** (held + resting) | **88.4%** |

`avg_exposure` is a fraction of equity (0.0112, shown as 1.1%): mean over decisions of holdings at mark, before that decision's orders, **without resting buys**. The arithmetic is right, so it stays; additive fields `avg_total_exposure`, `avg_deployed_usdt` and `return_on_avg_deployed` (`summarize`, `deployed_from_decisions`) now cover holdings plus resting buys, the figure the 38% cap bounds.

Checks: holdings rebuilt from the fills alone match the exchange trace to 7e-12 USDT and the replay's decision rows to 2e-13 (67 decisions in no-bar minutes skipped); the traced run equals the plain run. Halted on 0% of decisions. `record_trace=True` only records which feature bar each decision used, so exposure is rebuilt from the exchange state.

## 2. Fill-depth sweep

**Conclusion: a 0.05% fill-depth requirement cuts the median split return from 4.2% to 2.1% (1.3% at 0.15%; median Sharpe 1.53 / 0.79 / 0.53), hitting 2019-2020 hardest (13.3% to 2.6%); 2021, 2023, 2024 keep at least 75% of their return; positive splits 4/4/4 of 6; at 1 tick the 10 best days carry 38%-158% of P&L in the winning splits.**

| split | pierce 0 (1 tick) | pierce 0.0005 | pierce 0.0015 |
|---|---|---|---|
| 2019-2020 (IS) | 13.3% / 2.03 / 17.5 / 63% | 2.6% / 0.50 / 6.1 / 224% | 1.6% / 0.31 / 4.4 / 341% |
| 2021 (IS) | 19.4% / 5.84 / 21.4 / 38% | 15.9% / 4.62 / 18.7 / 44% | 13.8% / 4.17 / 18.3 / 47% |
| 2022 (IS) | -0.3% / -0.39 / -3.1 / n/a | -0.4% / -0.45 / -3.7 / n/a | -1.2% / -1.99 / -34.3 / n/a |
| 2023 (IS) | 1.6% / 1.03 / 9.4 / 158% | 1.6% / 1.09 / 10.0 / 145% | 1.1% / 0.75 / 8.2 / 201% |
| 2024 (OOS) | 6.9% / 2.91 / 16.3 / 74% | 6.5% / 2.89 / 16.1 / 75% | 6.2% / 2.74 / 16.0 / 77% |
| 2025-2026 (OOS) | -1.3% / -0.76 / -14.5 / n/a | -1.3% / -0.81 / -16.0 / n/a | -1.5% / -0.88 / -20.1 / n/a |

Cells: return / Sharpe / capture (gross P&L over fees) / share of total P&L earned on the 10 best days (daily equity changes; can pass 100% when other days lose; n/a if the split lost money). Pierce 0 reproduces the v1 table.

## 3. VR attribution

**Conclusion: P&L sits below VR rather than tracking it. VR explains 3% of coin-year P&L (t -1.2, Spearman -0.30); the drift of VR toward 1 accounts for only 18% of the early-to-late P&L fall, so the decay is mostly capture (less P&L at the same VR), with borderline significance (year term t -2.0).**

One wallet per coin, T09 params, calendar years, 47 coin-years (BTC, ETH, SOL, XRP, DOGE, LINK). P&L (USDT) on 1h VR (G0 baseline yearly table):

| regression | slope per 1.0 VR | t | R2 | n |
|---|---|---|---|---|
| P&L ~ VR (all) | -1,087 | -1.2 | 0.03 | 47 |
| P&L ~ VR, without DOGE 2019-20 | -2,474 | -1.7 | 0.07 | 45 |

Spearman -0.30; robust-VR table slope -694 (t -0.8). Mean VR rose from 0.857 (2019-21, n=17) to 0.928 (2024-26, n=18) while mean P&L fell from +478 to +50 USDT: the VR line explains -77 of that -428, the rest (-351) is lower P&L at the same VR (year term t = -2.0; VR t = -0.4 once year is in). 2021 (DOGE, SOL) alone lifts the early mean and n is small: borderline evidence.

Scatter (cells are `VR / P&L USDT`; blank = no VR row):

| coin | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| BTC | 0.93 / +177 | 0.88 / -71 | 0.84 / +358 | 0.94 / -36 | 0.90 / +86 | 0.96 / +41 | 0.93 / -31 | 0.95 / +0 |
| ETH | 0.94 / +319 | 0.90 / +24 | 0.85 / +704 | 0.98 / -36 | 0.93 / +97 | 1.00 / +49 | 0.98 / -99 | 0.98 / -47 |
| SOL |  | 0.91 / +154 | 0.92 / +1,156 | 0.89 / -25 | 0.88 / +126 | 0.90 / +280 | 0.96 / -28 | 0.93 / -46 |
| XRP | 0.90 / +5 | 0.93 / -179 | 0.85 / -244 | 0.93 / +96 | 0.89 / +36 | 0.93 / +647 | 0.91 / -58 | 0.95 / +110 |
| DOGE | 0.55 / +41 | 0.73 / +757 | 0.88 / +2,265 | 0.90 / -74 | 0.91 / +146 | 0.93 / +161 | 0.76 / -67 | 0.94 / -25 |
| LINK | 0.84 / +707 | 0.86 / +958 | 0.88 / +1,000 | 0.94 / +22 | 0.90 / +10 | 0.96 / -67 | 0.83 / +2 | 0.91 / +86 |

## 4. Cash benchmark (4% a year in USDT)

**Conclusion: T09 beats 4% cash in 3 of 6 splits (2019-2020, 2021, 2024); the best excess is +15.4 pp and the worst -8.7 pp.**

| split | T09 return | 4% cash | excess | beats cash |
|---|---|---|---|---|
| 2019-2020 (IS) | 13.3% | 8.2% | +5.1 pp | yes |
| 2021 (IS) | 19.4% | 4.0% | +15.4 pp | yes |
| 2022 (IS) | -0.3% | 4.0% | -4.3 pp | no |
| 2023 (IS) | 1.6% | 4.0% | -2.4 pp | no |
| 2024 (OOS) | 6.9% | 4.0% | +2.9 pp | yes |
| 2025-2026 (OOS) | -1.3% | 7.5% | -8.7 pp | no |

Runtime 1274s on 1 process. `cash_yield_return` and `excess_over_cash` are now in `vv.sim.metrics.summarize`.
