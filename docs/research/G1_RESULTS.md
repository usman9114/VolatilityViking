# G1 results

Universe: BTCUSDT, SOLUSDT, XRPUSDT, DOGEUSDT, LINKUSDT (one shared wallet, 10,000 USDT). Pre-registration v1; decisions every 15 min; trial count = 10. Runtime 846s.

## Trials (in-sample only; selection = highest median net Sharpe)

| trial | median Sharpe | median return | worst MDD | round trips | capture/fees |
|---|---|---|---|---|---|
| T01 step 0.0075 / 1.5s | 0.37 | 3.0% | 8.3% | 1602 | 5.29 |
| T02 step 0.0075 / 3s | 0.48 | 1.8% | 4.1% | 401 | 8.22 |
| T03 step 0.0100 / 1.5s | 0.31 | 2.4% | 8.3% | 1480 | 5.78 |
| T04 step 0.0100 / 3s | 0.49 | 1.7% | 4.1% | 398 | 8.24 |
| T05 step 0.0150 / 1.5s | 0.23 | 1.9% | 8.2% | 1281 | 8.05 |
| T06 step 0.0150 / 3s | 0.44 | 1.4% | 4.1% | 375 | 8.38 |
| T07 step 0.0200 / 1.5s | 1.10 | 8.0% | 5.2% | 1154 | 12.16 |
| T08 step 0.0200 / 3s | 0.72 | 1.7% | 3.9% | 348 | 10.57 |
| T09 step 0.0300 / 1.5s **selected** | 1.53 | 7.4% | 3.2% | 727 | 17.80 |
| T10 step 0.0300 / 3s | 0.68 | 1.8% | 2.0% | 277 | 14.15 |

## Selected trial T09 step 0.0300 / 1.5s vs baselines (base costs)

| split | return | Sharpe | MDD | capture | avg exp | B&H ret | B&H MDD | B&H@exp ret | cash |
|---|---|---|---|---|---|---|---|---|---|
| 2019-2020 (IS) | 13.3% | 2.03 | 3.2% | 17.53 | 1.6% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2021 (IS) | 19.4% | 5.84 | 1.2% | 21.43 | 1.1% | 3027.0% | 70.1% | 33.9% | 0.0% |
| 2022 (IS) | -0.3% | -0.39 | 0.8% | -3.07 | 0.4% | -69.5% | 71.1% | -0.3% | 0.0% |
| 2023 (IS) | 1.6% | 1.03 | 1.2% | 9.38 | 1.6% | 270.6% | 29.5% | 4.4% | 0.0% |
| 2024 (OOS) | 6.9% | 2.91 | 2.2% | 16.25 | 1.5% | 146.5% | 40.4% | 2.1% | 0.0% |
| 2025-2026 (OOS) | -1.3% | -0.76 | 2.3% | -14.49 | 0.6% | -44.9% | 67.9% | -0.3% | 0.0% |

## Stress (selected trial)

| variant | split | return | Sharpe | MDD | capture |
|---|---|---|---|---|---|
| fees x2 | 2019-2020 | 13.0% | 1.98 | 3.2% | 8.98 |
| fees x2 | 2021 | 18.8% | 5.73 | 1.2% | 10.96 |
| fees x2 | 2022 | -0.3% | -0.41 | 0.8% | -1.08 |
| fees x2 | 2023 | 1.5% | 0.93 | 1.2% | 4.92 |
| fees x2 | 2024 | 6.5% | 2.85 | 2.2% | 8.38 |
| fees x2 | 2025-2026 | -1.3% | -0.78 | 2.4% | -6.83 |
| fill_haircut 0.7 | 2019-2020 | 12.0% | 2.08 | 2.6% | 17.78 |
| fill_haircut 0.7 | 2021 | 14.6% | 3.98 | 1.7% | 18.66 |
| fill_haircut 0.7 | 2022 | -1.2% | -1.96 | 1.4% | -32.10 |
| fill_haircut 0.7 | 2023 | 1.1% | 0.72 | 1.2% | 7.64 |
| fill_haircut 0.7 | 2024 | 5.9% | 2.50 | 2.2% | 15.66 |
| fill_haircut 0.7 | 2025-2026 | -1.5% | -0.90 | 2.3% | -20.48 |

## Shadow control (ETHUSDT, own portfolio, selected trial)

| split | return | Sharpe | MDD | capture |
|---|---|---|---|---|
| 2019-2020 | 3.2% | 0.89 | 1.7% | 18.08 |
| 2021 | 7.0% | 4.28 | 0.6% | 22.05 |
| 2022 | -0.4% | -0.71 | 0.5% | -41.83 |
| 2023 | 1.0% | 1.97 | 0.4% | 20.21 |
| 2024 | 0.5% | 0.63 | 0.5% | 10.32 |
| 2025-2026 | -1.0% | -1.05 | 1.0% | -195.42 |

ETH gate: FAIL (the control behaves as expected).

| criterion | ETH value | result |
|---|---|---|
| Net capture over fees | 17.34 | PASS |
| OOS Sharpe CI lower bound | -1.44 (CI -1.44..0.97, point -0.33) | FAIL |
| Positive splits | 4/6 (2019-2020, 2021, 2023, 2024); 2022 not in | FAIL |
| Max drawdown | 1.7% | PASS |

## Gate G1

| criterion | rule | value | result |
|---|---|---|---|
| Net capture over fees | >= 2.0 | 16.49 | PASS |
| OOS Sharpe CI lower bound | > 0 | -0.62 (CI -0.62..2.81, point 1.17) | FAIL |
| Positive splits | majority and 2022 among them | 4/6 (2019-2020, 2021, 2023, 2024); 2022 not in | FAIL |
| Max drawdown | <= 15% | 3.2% | PASS |

**Verdict.** FAIL. Criteria not met: OOS Sharpe CI lower bound, Positive splits. The strategy does not clear G1 as pre-registered.

## Notes

- Trials run: 10 x 6 splits = 60 base runs; stress and shadow runs reuse the selected trial.
- Order rejections, selected trial base runs: none; all 60 base runs: none.
- Churn (selected, base): 397034 placed, 394569 cancelled over 274656 decisions; halts 25; peak exposure incl. resting buys 38.0% of equity.
- Interpretations: capture = pooled gross realised / fees (all splits); drawdown = worst split; OOS CI = pooled OOS daily returns at fees x2; stress changes exchange fees only, not the strategy's maker_fee input; baselines: equal-weight B&H of symbols listed at split start, B&H@exp = that basket at the run's average exposure, rest cash.
- Filters are assumed constant over history (tick/step per symbol in `research/g1_run.py`).
- Cost: 0.018 CPU-s per symbol-day over 84 runs (3215 CPU-s).
