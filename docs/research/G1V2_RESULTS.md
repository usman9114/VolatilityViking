# G1 v2 results

Universe: BTCUSDT, SOLUSDT, XRPUSDT, DOGEUSDT, LINKUSDT (one shared wallet, 10,000 USDT). Pre-registration v2 (inherits v1: fees, risk caps, halts, splits, stress, shadow control, gate); decisions every 15 min. **Total trials across variants = 16** (A DCA+TP 10, B paired grid 3, C rebalance 3); every trial is reported below. Runtime 1852s.

## Out-of-sample, side by side (selected trial per variant, base costs)

| split | variant: selected trial | return | Sharpe | MDD | capture/fees | avg exp | B&H at same exposure | B&H fully invested |
|---|---|---|---|---|---|---|---|---|
| 2024 | A DCA+TP: T09 step 0.0300 / 1.5s | 6.9% | 2.91 | 2.2% | 16.25 | 1.5% | 2.1% | 146.5% |
| 2024 | B paired grid: B01 step 1.00% x 10 levels | 0.9% | 0.17 | 6.7% | 1.56 | 7.6% | 11.1% | 146.5% |
| 2024 | C rebalance: C03 band 40% | 17.6% | 0.97 | 13.7% | 25.72 | 28.8% | 42.2% | 146.5% |
| 2025-2026 | A DCA+TP: T09 step 0.0300 / 1.5s | -1.3% | -0.76 | 2.3% | -14.49 | 0.6% | -0.3% | -44.9% |
| 2025-2026 | B paired grid: B01 step 1.00% x 10 levels | -3.5% | -0.62 | 6.8% | -3.31 | 1.5% | -0.7% | -44.9% |
| 2025-2026 | C rebalance: C03 band 40% | -5.9% | -0.31 | 14.3% | -52.94 | 4.3% | -1.9% | -44.9% |

B&H = equal-weight buy-and-hold of the symbols listed at split start, no rebalancing; "same exposure" holds that basket at the variant's own average exposure, rest cash.

## Gate G1 per variant (each on its own)

| criterion | rule | A DCA+TP | B paired grid | C rebalance |
|---|---|---|---|---|
| Net capture over fees | >= 2.0 | PASS (16.49) | FAIL (1.98) | PASS (113.18) |
| OOS Sharpe CI lower bound | > 0 | FAIL (-0.62 (CI -0.62..2.81, point 1.17)) | FAIL (-1.52 (CI -1.52..0.85, point -0.37)) | FAIL (-0.83 (CI -0.83..1.48, point 0.33)) |
| Positive splits | majority and 2022 among them | FAIL (4/6 (2019-2020, 2021, 2023, 2024); 2022 not in) | FAIL (4/6 (2019-2020, 2021, 2023, 2024); 2022 not in) | FAIL (4/6 (2019-2020, 2021, 2023, 2024); 2022 not in) |
| Max drawdown | <= 15% | PASS (3.2%) | PASS (7.1%) | FAIL (15.2%) |
| ETH shadow control | should FAIL | FAIL (expected) | FAIL (expected) | FAIL (expected) |
| **Deployable** | all criteria (+ haircut rerun if ETH passes) | **NO** | **NO** | **NO** |

## A DCA+TP

### Trials (in-sample only; selection = highest median net Sharpe)

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

### Selected trial T09 step 0.0300 / 1.5s: all splits vs baselines (base costs)

| split | return | Sharpe | MDD | capture | avg exp | B&H ret | B&H MDD | B&H@exp ret |
|---|---|---|---|---|---|---|---|---|
| 2019-2020 (IS) | 13.3% | 2.03 | 3.2% | 17.53 | 1.6% | 0.0% | 0.0% | 0.0% |
| 2021 (IS) | 19.4% | 5.84 | 1.2% | 21.43 | 1.1% | 3027.0% | 70.1% | 33.9% |
| 2022 (IS) | -0.3% | -0.39 | 0.8% | -3.07 | 0.4% | -69.5% | 71.1% | -0.3% |
| 2023 (IS) | 1.6% | 1.03 | 1.2% | 9.38 | 1.6% | 270.6% | 29.5% | 4.4% |
| 2024 (OOS) | 6.9% | 2.91 | 2.2% | 16.25 | 1.5% | 146.5% | 40.4% | 2.1% |
| 2025-2026 (OOS) | -1.3% | -0.76 | 2.3% | -14.49 | 0.6% | -44.9% | 67.9% | -0.3% |

### Stress (selected trial)

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

### Shadow control (ETHUSDT, own portfolio, selected trial)

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

### Gate G1

| criterion | rule | value | result |
|---|---|---|---|
| Net capture over fees | >= 2.0 | 16.49 | PASS |
| OOS Sharpe CI lower bound | > 0 | -0.62 (CI -0.62..2.81, point 1.17) | FAIL |
| Positive splits | majority and 2022 among them | 4/6 (2019-2020, 2021, 2023, 2024); 2022 not in | FAIL |
| Max drawdown | <= 15% | 3.2% | PASS |

**Verdict (A DCA+TP).** FAIL. Criteria not met: OOS Sharpe CI lower bound, Positive splits. Not deployable as pre-registered.

## B paired grid

### Trials (in-sample only; selection = highest median net Sharpe)

| trial | median Sharpe | median return | worst MDD | round trips | capture/fees |
|---|---|---|---|---|---|
| B01 step 1.00% x 10 levels **selected** | 0.62 | 6.3% | 7.1% | 838 | 2.46 |
| B02 step 2.00% x 10 levels | 0.58 | 3.9% | 6.6% | 451 | 3.74 |
| B03 step 3.00% x 10 levels | 0.51 | 2.7% | 7.1% | 403 | 8.96 |

### Selected trial B01 step 1.00% x 10 levels: all splits vs baselines (base costs)

| split | return | Sharpe | MDD | capture | avg exp | B&H ret | B&H MDD | B&H@exp ret |
|---|---|---|---|---|---|---|---|---|
| 2019-2020 (IS) | 7.7% | 0.56 | 7.0% | 2.89 | 5.6% | 0.0% | 0.0% | 0.0% |
| 2021 (IS) | 5.6% | 0.67 | 6.7% | 2.96 | 3.9% | 3027.0% | 70.1% | 118.2% |
| 2022 (IS) | -5.3% | -0.88 | 7.1% | -5.01 | 4.2% | -69.5% | 71.1% | -2.9% |
| 2023 (IS) | 7.0% | 1.04 | 4.7% | 3.83 | 12.3% | 270.6% | 29.5% | 33.4% |
| 2024 (OOS) | 0.9% | 0.17 | 6.7% | 1.56 | 7.6% | 146.5% | 40.4% | 11.1% |
| 2025-2026 (OOS) | -3.5% | -0.62 | 6.8% | -3.31 | 1.5% | -44.9% | 67.9% | -0.7% |

### Stress (selected trial)

| variant | split | return | Sharpe | MDD | capture |
|---|---|---|---|---|---|
| fees x2 | 2019-2020 | 3.6% | 0.29 | 7.3% | 1.45 |
| fees x2 | 2021 | 4.5% | 0.51 | 6.8% | 1.71 |
| fees x2 | 2022 | -6.1% | -1.03 | 7.1% | -2.51 |
| fees x2 | 2023 | 4.5% | 0.69 | 4.8% | 1.92 |
| fees x2 | 2024 | -0.7% | -0.07 | 6.9% | 0.79 |
| fees x2 | 2025-2026 | -4.3% | -0.77 | 6.9% | -1.66 |
| fill_haircut 0.7 | 2019-2020 | 4.8% | 0.38 | 6.7% | 2.46 |
| fill_haircut 0.7 | 2021 | 4.7% | 0.58 | 6.7% | 2.75 |
| fill_haircut 0.7 | 2022 | -5.5% | -0.93 | 7.1% | -5.71 |
| fill_haircut 0.7 | 2023 | 6.0% | 0.91 | 4.7% | 3.62 |
| fill_haircut 0.7 | 2024 | 0.3% | 0.08 | 6.7% | 1.20 |
| fill_haircut 0.7 | 2025-2026 | -3.8% | -0.68 | 6.9% | -3.97 |

### Shadow control (ETHUSDT, own portfolio, selected trial)

| split | return | Sharpe | MDD | capture |
|---|---|---|---|---|
| 2019-2020 | 6.0% | 0.90 | 3.8% | 4.52 |
| 2021 | 5.1% | 1.03 | 3.9% | 4.27 |
| 2022 | -2.3% | -0.62 | 4.0% | -5.56 |
| 2023 | 6.1% | 1.48 | 2.8% | 6.54 |
| 2024 | -0.0% | 0.01 | 2.7% | 0.97 |
| 2025-2026 | -3.1% | -1.09 | 3.2% | -93.16 |

ETH gate: FAIL (the control behaves as expected).

| criterion | ETH value | result |
|---|---|---|
| Net capture over fees | 3.24 | PASS |
| OOS Sharpe CI lower bound | -1.55 (CI -1.55..0.63, point -0.60) | FAIL |
| Positive splits | 3/6 (2019-2020, 2021, 2023); 2022 not in | FAIL |
| Max drawdown | 4.0% | PASS |

### Gate G1

| criterion | rule | value | result |
|---|---|---|---|
| Net capture over fees | >= 2.0 | 1.98 | FAIL |
| OOS Sharpe CI lower bound | > 0 | -1.52 (CI -1.52..0.85, point -0.37) | FAIL |
| Positive splits | majority and 2022 among them | 4/6 (2019-2020, 2021, 2023, 2024); 2022 not in | FAIL |
| Max drawdown | <= 15% | 7.1% | PASS |

**Verdict (B paired grid).** FAIL. Criteria not met: Net capture over fees, OOS Sharpe CI lower bound, Positive splits. Not deployable as pre-registered.

## C rebalance

### Trials (in-sample only; selection = highest median net Sharpe)

| trial | median Sharpe | median return | worst MDD | round trips | capture/fees |
|---|---|---|---|---|---|
| C01 band 10% | 1.03 | 24.6% | 16.0% | 17 | 65.10 |
| C02 band 20% | 1.03 | 24.6% | 15.1% | 17 | 86.23 |
| C03 band 40% **selected** | 1.44 | 31.8% | 15.2% | 17 | 145.49 |

### Selected trial C03 band 40%: all splits vs baselines (base costs)

| split | return | Sharpe | MDD | capture | avg exp | B&H ret | B&H MDD | B&H@exp ret |
|---|---|---|---|---|---|---|---|---|
| 2019-2020 (IS) | 28.7% | 1.04 | 14.7% | 127.99 | 10.1% | 0.0% | 0.0% | 0.0% |
| 2021 (IS) | 102.1% | 2.87 | 15.2% | 240.73 | 12.9% | 3027.0% | 70.1% | 390.9% |
| 2022 (IS) | -13.3% | -1.45 | 14.4% | -138.59 | 8.1% | -69.5% | 71.1% | -5.6% |
| 2023 (IS) | 34.9% | 1.83 | 11.5% | 102.18 | 31.2% | 270.6% | 29.5% | 84.4% |
| 2024 (OOS) | 17.6% | 0.97 | 13.7% | 25.72 | 28.8% | 146.5% | 40.4% | 42.2% |
| 2025-2026 (OOS) | -5.9% | -0.31 | 14.3% | -52.94 | 4.3% | -44.9% | 67.9% | -1.9% |

### Stress (selected trial)

| variant | split | return | Sharpe | MDD | capture |
|---|---|---|---|---|---|
| fees x2 | 2019-2020 | 28.5% | 1.03 | 14.8% | 64.04 |
| fees x2 | 2021 | 101.8% | 2.87 | 15.3% | 120.51 |
| fees x2 | 2022 | -13.4% | -1.46 | 14.5% | -69.29 |
| fees x2 | 2023 | 34.6% | 1.82 | 11.5% | 51.10 |
| fees x2 | 2024 | 17.4% | 0.96 | 13.8% | 12.69 |
| fees x2 | 2025-2026 | -6.0% | -0.31 | 14.4% | -26.45 |
| fill_haircut 0.7 | 2019-2020 | 27.0% | 0.99 | 14.9% | 122.27 |
| fill_haircut 0.7 | 2021 | 101.2% | 2.85 | 14.9% | 230.45 |
| fill_haircut 0.7 | 2022 | -13.3% | -1.46 | 14.3% | -138.59 |
| fill_haircut 0.7 | 2023 | 34.5% | 1.81 | 11.5% | 101.21 |
| fill_haircut 0.7 | 2024 | 17.6% | 0.97 | 13.7% | 25.74 |
| fill_haircut 0.7 | 2025-2026 | -5.8% | -0.33 | 14.2% | -58.84 |

### Shadow control (ETHUSDT, own portfolio, selected trial)

| split | return | Sharpe | MDD | capture |
|---|---|---|---|---|
| 2019-2020 | 4.9% | 0.28 | 14.6% | 100.73 |
| 2021 | 38.0% | 1.94 | 10.1% | 512.20 |
| 2022 | -3.8% | -1.64 | 4.4% | -166.98 |
| 2023 | 11.9% | 1.48 | 4.7% | 140.08 |
| 2024 | 7.2% | 0.81 | 7.7% | 141.50 |
| 2025-2026 | -3.8% | -0.80 | 5.3% | -169.45 |

ETH gate: FAIL (the control behaves as expected).

| criterion | ETH value | result |
|---|---|---|
| Net capture over fees | 146.50 | PASS |
| OOS Sharpe CI lower bound | -0.96 (CI -0.96..1.42, point 0.21) | FAIL |
| Positive splits | 4/6 (2019-2020, 2021, 2023, 2024); 2022 not in | FAIL |
| Max drawdown | 14.6% | PASS |

### Gate G1

| criterion | rule | value | result |
|---|---|---|---|
| Net capture over fees | >= 2.0 | 113.18 | PASS |
| OOS Sharpe CI lower bound | > 0 | -0.83 (CI -0.83..1.48, point 0.33) | FAIL |
| Positive splits | majority and 2022 among them | 4/6 (2019-2020, 2021, 2023, 2024); 2022 not in | FAIL |
| Max drawdown | <= 15% | 15.2% | FAIL |

**Verdict (C rebalance).** FAIL. Criteria not met: OOS Sharpe CI lower bound, Positive splits, Max drawdown. Not deployable as pre-registered.

## Notes

- A DCA+TP (selected, base): 397034 placed, 394569 cancelled over 274656 decisions; halts 25; peak exposure incl. resting buys 38.0%; rejections none.
- B paired grid (selected, base): 21618 placed, 4783 cancelled over 274656 decisions; halts 28; peak exposure incl. resting buys 37.9%; rejections none.
- C rebalance (selected, base): 349 placed, 11 cancelled over 274656 decisions; halts 18; peak exposure incl. resting buys 40.0%; rejections none.
- Trials run: 16 x 6 splits = 96 base runs; stress and shadow runs reuse each variant's selected trial. Variant A was re-run with the production strategy (no v1 output file is read).
- Interpretations (v1 addendum, applied to every variant): capture = pooled gross realised / fees (all splits); drawdown = worst split; OOS CI = pooled OOS daily returns at fees x2; stress changes exchange fees only.
- Open choices of B and C (documented in `vv/core/variants/paired_grid.py` and `rebalance.py`): B anchors at the mid when a symbol first warms up, quote per level = symbol budget / levels, sells are lifted to ask + 1 tick when price already passed the level; C needs warm features (sigma) like the others and marks positions at mid.
- Filters are assumed constant over history (see `research/g1_run.py`).
- Cost per symbol-day: A DCA+TP 0.022s, B paired grid 0.024s, C rebalance 0.018s (7083 CPU-s executed; wall time inside run_replay, so inflated if the machine was busy).
