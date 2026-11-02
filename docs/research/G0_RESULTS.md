# G0 results: which coins mean-revert at grid horizons?

- BTCUSDT: data 2019-01-01 to 2026-10-31, analysed from 2019-01-31, 4,072,551 1m bars (814 empty 5m bins forward-filled); robust run dropped 576 5m bars in excluded windows and winsorised 10,450 of 814,751.
- ETHUSDT: data 2019-01-01 to 2026-10-31, analysed from 2019-01-31, 4,072,550 1m bars (814 empty 5m bins forward-filled); robust run dropped 576 5m bars in excluded windows and winsorised 9,905 of 814,751.
- SOLUSDT: data 2020-08-11 to 2026-10-31, analysed from 2020-09-10, 3,228,135 1m bars (283 empty 5m bins forward-filled); robust run dropped 576 5m bars in excluded windows and winsorised 6,200 of 645,335.
- BNBUSDT: data 2019-01-01 to 2026-10-31, analysed from 2019-01-31, 4,072,550 1m bars (814 empty 5m bins forward-filled); robust run dropped 576 5m bars in excluded windows and winsorised 8,850 of 814,751.
- XRPUSDT: data 2019-01-01 to 2026-10-31, analysed from 2019-01-31, 4,072,551 1m bars (814 empty 5m bins forward-filled); robust run dropped 576 5m bars in excluded windows and winsorised 11,000 of 814,751.
- DOGEUSDT: data 2019-07-05 to 2026-10-31, analysed from 2019-08-04, 3,806,451 1m bars (610 empty 5m bins forward-filled); robust run dropped 576 5m bars in excluded windows and winsorised 12,382 of 761,327.
- LINKUSDT: data 2019-01-16 to 2026-10-31, analysed from 2019-02-15, 4,050,350 1m bars (814 empty 5m bins forward-filled); robust run dropped 576 5m bars in excluded windows and winsorised 6,728 of 810,311.
- AVAXUSDT: data 2020-09-22 to 2026-10-31, analysed from 2020-10-22, 3,167,626 1m bars (283 empty 5m bins forward-filled); robust run dropped 576 5m bars in excluded windows and winsorised 4,751 of 633,233.

Method: Lo-MacKinlay VR on log returns, robust z*, stationary block bootstrap (mean block 1 day, 500 reps, seed 20260101), 95% CI. The bootstrap percentile interval is re-centred on the point estimate: CI = VR + (q_lo - median(draws), q_hi - median(draws)), because plain percentile intervals drift toward 1 on short, heavy-tailed samples. Each symbol's first 30 days of data are excluded in both versions. **Baseline**: the above on 5m returns as-is; pooled and yearly CI must exclude 1. **Robust**: 5m returns winsorised at +-5 sigma (sigma = 1.4826 x MAD, per calendar month) and 2025-10-10 00:00 to 2025-10-12 00:00 UTC dropped; pooled checks need the Holm-adjusted z* p < 0.05 (family = 8 symbols x {1h, 4h} = 16 tests) and a CI excluding 1; yearly checks need a CI excluding 1 and unadjusted z* p < 0.05 for that year. Both use a majority-of-years rule and the median-4h-range check. Year rows need at least 30 days of data. **Final** = passes both.

## Final G1 universe (baseline and robust)

| Coin | Baseline | Robust | Final (both) | Robust failures |
|---|---|---|---|---|
| BTCUSDT | **PASS** | **PASS** | **PASS** | - |
| ETHUSDT | fail | **PASS** | fail | - |
| SOLUSDT | **PASS** | **PASS** | **PASS** | - |
| BNBUSDT | fail | **PASS** | fail | - |
| XRPUSDT | **PASS** | **PASS** | **PASS** | - |
| DOGEUSDT | **PASS** | **PASS** | **PASS** | - |
| LINKUSDT | **PASS** | **PASS** | **PASS** | - |
| AVAXUSDT | fail | **PASS** | fail | - |

Final list: BTCUSDT, SOLUSDT, XRPUSDT, DOGEUSDT, LINKUSDT. Baseline passes 5 of 8, robust passes 8 of 8.

## Verdict per coin: baseline

Each coin is gated on its own. Only coins that pass both versions go on to the G1 replay.

| Coin | G0 | Pooled 1h | Pooled 4h | Years 1h | Years 4h | 1h since 2024 | Median 4h range |
|---|---|---|---|---|---|---|---|
| BTCUSDT | **PASS** | VR=0.899, CI [0.873, 0.920] | VR=0.858, CI [0.819, 0.894] | 8 of 8 years | 6 of 8 years | 3 of 3 | median 1.352% |
| ETHUSDT | fail | VR=0.922, CI [0.886, 0.945] | VR=0.887, CI [0.838, 0.923] | 4 of 8 years | 3 of 8 years | 0 of 3 | median 1.825% |
| SOLUSDT | **PASS** | VR=0.908, CI [0.883, 0.930] | VR=0.862, CI [0.824, 0.902] | 6 of 7 years | 6 of 7 years | 2 of 3 | median 2.634% |
| BNBUSDT | fail | VR=0.907, CI [0.869, 0.939] | VR=0.850, CI [0.806, 0.892] | 4 of 8 years | 6 of 8 years | 0 of 3 | median 1.668% |
| XRPUSDT | **PASS** | VR=0.896, CI [0.860, 0.927] | VR=0.819, CI [0.771, 0.878] | 5 of 8 years | 7 of 8 years | 2 of 3 | median 1.927% |
| DOGEUSDT | **PASS** | VR=0.853, CI [0.793, 0.912] | VR=0.771, CI [0.692, 0.859] | 7 of 8 years | 8 of 8 years | 3 of 3 | median 2.173% |
| LINKUSDT | **PASS** | VR=0.877, CI [0.846, 0.912] | VR=0.798, CI [0.753, 0.849] | 7 of 8 years | 7 of 8 years | 2 of 3 | median 2.589% |
| AVAXUSDT | fail | VR=0.803, CI [0.638, 0.933] | VR=0.742, CI [0.574, 0.876] | 3 of 7 years | 5 of 7 years | 0 of 3 | median 2.654% |

With 8 coins tested, expect occasional false passes from chance; the majority-of-years rule limits this, and G1 (replay with fees) is the real test.

## Verdict per coin: robust

Winsorised, event windows excluded. Pooled cells show the Holm-adjusted z* p.

| Coin | G0 | Pooled 1h | Pooled 4h | Years 1h | Years 4h | 1h since 2024 | Median 4h range |
|---|---|---|---|---|---|---|---|
| BTCUSDT | **PASS** | VR=0.874, CI [0.862, 0.886], Holm p=0.0000 | VR=0.840, CI [0.818, 0.860], Holm p=0.0000 | 8 of 8 years | 8 of 8 years | 3 of 3 | median 1.352% |
| ETHUSDT | **PASS** | VR=0.910, CI [0.899, 0.921], Holm p=0.0000 | VR=0.871, CI [0.849, 0.895], Holm p=0.0000 | 8 of 8 years | 6 of 8 years | 3 of 3 | median 1.824% |
| SOLUSDT | **PASS** | VR=0.926, CI [0.909, 0.943], Holm p=0.0000 | VR=0.881, CI [0.850, 0.911], Holm p=0.0000 | 7 of 7 years | 6 of 7 years | 3 of 3 | median 2.633% |
| BNBUSDT | **PASS** | VR=0.914, CI [0.896, 0.932], Holm p=0.0000 | VR=0.873, CI [0.845, 0.904], Holm p=0.0000 | 6 of 8 years | 6 of 8 years | 2 of 3 | median 1.667% |
| XRPUSDT | **PASS** | VR=0.870, CI [0.858, 0.884], Holm p=0.0000 | VR=0.775, CI [0.750, 0.799], Holm p=0.0000 | 8 of 8 years | 7 of 8 years | 3 of 3 | median 1.927% |
| DOGEUSDT | **PASS** | VR=0.849, CI [0.828, 0.870], Holm p=0.0000 | VR=0.760, CI [0.725, 0.800], Holm p=0.0000 | 8 of 8 years | 8 of 8 years | 3 of 3 | median 2.172% |
| LINKUSDT | **PASS** | VR=0.882, CI [0.870, 0.894], Holm p=0.0000 | VR=0.807, CI [0.787, 0.832], Holm p=0.0000 | 8 of 8 years | 8 of 8 years | 3 of 3 | median 2.588% |
| AVAXUSDT | **PASS** | VR=0.929, CI [0.914, 0.943], Holm p=0.0000 | VR=0.868, CI [0.842, 0.893], Holm p=0.0000 | 5 of 7 years | 4 of 7 years | 1 of 3 | median 2.651% |

Robust pooled and year counts:

| Coin | Pooled 1h | Pooled 4h | Years 1h | Years 4h |
|---|---|---|---|---|
| BTCUSDT | pass: VR=0.874, CI [0.862, 0.886] | pass: VR=0.840, CI [0.818, 0.860] | 8 of 8 years | 8 of 8 years |
| ETHUSDT | pass: VR=0.910, CI [0.899, 0.921] | pass: VR=0.871, CI [0.849, 0.895] | 8 of 8 years | 6 of 8 years |
| SOLUSDT | pass: VR=0.926, CI [0.909, 0.943] | pass: VR=0.881, CI [0.850, 0.911] | 7 of 7 years | 6 of 7 years |
| BNBUSDT | pass: VR=0.914, CI [0.896, 0.932] | pass: VR=0.873, CI [0.845, 0.904] | 6 of 8 years | 6 of 8 years |
| XRPUSDT | pass: VR=0.870, CI [0.858, 0.884] | pass: VR=0.775, CI [0.750, 0.799] | 8 of 8 years | 7 of 8 years |
| DOGEUSDT | pass: VR=0.849, CI [0.828, 0.870] | pass: VR=0.760, CI [0.725, 0.800] | 8 of 8 years | 8 of 8 years |
| LINKUSDT | pass: VR=0.882, CI [0.870, 0.894] | pass: VR=0.807, CI [0.787, 0.832] | 8 of 8 years | 8 of 8 years |
| AVAXUSDT | pass: VR=0.929, CI [0.914, 0.943] | pass: VR=0.868, CI [0.842, 0.893] | 5 of 7 years | 4 of 7 years |

## Overall: baseline (all coins together)

**FAIL.** 4 of 40 checks fail, e.g. ETHUSDT 1h: VR<1 with CI excluding 1 in a majority of years.

| Check | Result | Detail |
|---|---|---|
| BTCUSDT pooled 1h: VR<1 and CI excludes 1 | pass | VR=0.899, CI [0.873, 0.920] |
| BTCUSDT 1h: VR<1 with CI excluding 1 in a majority of years | pass | 8 of 8 years |
| BTCUSDT pooled 4h: VR<1 and CI excludes 1 | pass | VR=0.858, CI [0.819, 0.894] |
| BTCUSDT 4h: VR<1 with CI excluding 1 in a majority of years | pass | 6 of 8 years |
| BTCUSDT median 4h range > 0.375% | pass | median 1.352% |
| ETHUSDT pooled 1h: VR<1 and CI excludes 1 | pass | VR=0.922, CI [0.886, 0.945] |
| ETHUSDT 1h: VR<1 with CI excluding 1 in a majority of years | fail | 4 of 8 years |
| ETHUSDT pooled 4h: VR<1 and CI excludes 1 | pass | VR=0.887, CI [0.838, 0.923] |
| ETHUSDT 4h: VR<1 with CI excluding 1 in a majority of years | fail | 3 of 8 years |
| ETHUSDT median 4h range > 0.375% | pass | median 1.825% |
| SOLUSDT pooled 1h: VR<1 and CI excludes 1 | pass | VR=0.908, CI [0.883, 0.930] |
| SOLUSDT 1h: VR<1 with CI excluding 1 in a majority of years | pass | 6 of 7 years |
| SOLUSDT pooled 4h: VR<1 and CI excludes 1 | pass | VR=0.862, CI [0.824, 0.902] |
| SOLUSDT 4h: VR<1 with CI excluding 1 in a majority of years | pass | 6 of 7 years |
| SOLUSDT median 4h range > 0.375% | pass | median 2.634% |
| BNBUSDT pooled 1h: VR<1 and CI excludes 1 | pass | VR=0.907, CI [0.869, 0.939] |
| BNBUSDT 1h: VR<1 with CI excluding 1 in a majority of years | fail | 4 of 8 years |
| BNBUSDT pooled 4h: VR<1 and CI excludes 1 | pass | VR=0.850, CI [0.806, 0.892] |
| BNBUSDT 4h: VR<1 with CI excluding 1 in a majority of years | pass | 6 of 8 years |
| BNBUSDT median 4h range > 0.375% | pass | median 1.668% |
| XRPUSDT pooled 1h: VR<1 and CI excludes 1 | pass | VR=0.896, CI [0.860, 0.927] |
| XRPUSDT 1h: VR<1 with CI excluding 1 in a majority of years | pass | 5 of 8 years |
| XRPUSDT pooled 4h: VR<1 and CI excludes 1 | pass | VR=0.819, CI [0.771, 0.878] |
| XRPUSDT 4h: VR<1 with CI excluding 1 in a majority of years | pass | 7 of 8 years |
| XRPUSDT median 4h range > 0.375% | pass | median 1.927% |
| DOGEUSDT pooled 1h: VR<1 and CI excludes 1 | pass | VR=0.853, CI [0.793, 0.912] |
| DOGEUSDT 1h: VR<1 with CI excluding 1 in a majority of years | pass | 7 of 8 years |
| DOGEUSDT pooled 4h: VR<1 and CI excludes 1 | pass | VR=0.771, CI [0.692, 0.859] |
| DOGEUSDT 4h: VR<1 with CI excluding 1 in a majority of years | pass | 8 of 8 years |
| DOGEUSDT median 4h range > 0.375% | pass | median 2.173% |
| LINKUSDT pooled 1h: VR<1 and CI excludes 1 | pass | VR=0.877, CI [0.846, 0.912] |
| LINKUSDT 1h: VR<1 with CI excluding 1 in a majority of years | pass | 7 of 8 years |
| LINKUSDT pooled 4h: VR<1 and CI excludes 1 | pass | VR=0.798, CI [0.753, 0.849] |
| LINKUSDT 4h: VR<1 with CI excluding 1 in a majority of years | pass | 7 of 8 years |
| LINKUSDT median 4h range > 0.375% | pass | median 2.589% |
| AVAXUSDT pooled 1h: VR<1 and CI excludes 1 | pass | VR=0.803, CI [0.638, 0.933] |
| AVAXUSDT 1h: VR<1 with CI excluding 1 in a majority of years | fail | 3 of 7 years |
| AVAXUSDT pooled 4h: VR<1 and CI excludes 1 | pass | VR=0.742, CI [0.574, 0.876] |
| AVAXUSDT 4h: VR<1 with CI excluding 1 in a majority of years | pass | 5 of 7 years |
| AVAXUSDT median 4h range > 0.375% | pass | median 2.654% |

Rule applied literally: PASS only if VR<1 with the CI excluding 1 at 1h and 4h for every symbol, in a majority of years, and the median 4h range exceeds 0.375%.

## Overall: robust (all coins together)

**PASS.** Every robust check below holds.

| Check | Result | Detail |
|---|---|---|
| BTCUSDT pooled 1h: VR<1, CI excludes 1 and Holm p<0.05 | pass | VR=0.874, CI [0.862, 0.886], Holm p=0.0000 |
| BTCUSDT 1h: VR<1 with CI excluding 1 and p<0.05 in a majority of years | pass | 8 of 8 years |
| BTCUSDT pooled 4h: VR<1, CI excludes 1 and Holm p<0.05 | pass | VR=0.840, CI [0.818, 0.860], Holm p=0.0000 |
| BTCUSDT 4h: VR<1 with CI excluding 1 and p<0.05 in a majority of years | pass | 8 of 8 years |
| BTCUSDT median 4h range > 0.375% | pass | median 1.352% |
| ETHUSDT pooled 1h: VR<1, CI excludes 1 and Holm p<0.05 | pass | VR=0.910, CI [0.899, 0.921], Holm p=0.0000 |
| ETHUSDT 1h: VR<1 with CI excluding 1 and p<0.05 in a majority of years | pass | 8 of 8 years |
| ETHUSDT pooled 4h: VR<1, CI excludes 1 and Holm p<0.05 | pass | VR=0.871, CI [0.849, 0.895], Holm p=0.0000 |
| ETHUSDT 4h: VR<1 with CI excluding 1 and p<0.05 in a majority of years | pass | 6 of 8 years |
| ETHUSDT median 4h range > 0.375% | pass | median 1.824% |
| SOLUSDT pooled 1h: VR<1, CI excludes 1 and Holm p<0.05 | pass | VR=0.926, CI [0.909, 0.943], Holm p=0.0000 |
| SOLUSDT 1h: VR<1 with CI excluding 1 and p<0.05 in a majority of years | pass | 7 of 7 years |
| SOLUSDT pooled 4h: VR<1, CI excludes 1 and Holm p<0.05 | pass | VR=0.881, CI [0.850, 0.911], Holm p=0.0000 |
| SOLUSDT 4h: VR<1 with CI excluding 1 and p<0.05 in a majority of years | pass | 6 of 7 years |
| SOLUSDT median 4h range > 0.375% | pass | median 2.633% |
| BNBUSDT pooled 1h: VR<1, CI excludes 1 and Holm p<0.05 | pass | VR=0.914, CI [0.896, 0.932], Holm p=0.0000 |
| BNBUSDT 1h: VR<1 with CI excluding 1 and p<0.05 in a majority of years | pass | 6 of 8 years |
| BNBUSDT pooled 4h: VR<1, CI excludes 1 and Holm p<0.05 | pass | VR=0.873, CI [0.845, 0.904], Holm p=0.0000 |
| BNBUSDT 4h: VR<1 with CI excluding 1 and p<0.05 in a majority of years | pass | 6 of 8 years |
| BNBUSDT median 4h range > 0.375% | pass | median 1.667% |
| XRPUSDT pooled 1h: VR<1, CI excludes 1 and Holm p<0.05 | pass | VR=0.870, CI [0.858, 0.884], Holm p=0.0000 |
| XRPUSDT 1h: VR<1 with CI excluding 1 and p<0.05 in a majority of years | pass | 8 of 8 years |
| XRPUSDT pooled 4h: VR<1, CI excludes 1 and Holm p<0.05 | pass | VR=0.775, CI [0.750, 0.799], Holm p=0.0000 |
| XRPUSDT 4h: VR<1 with CI excluding 1 and p<0.05 in a majority of years | pass | 7 of 8 years |
| XRPUSDT median 4h range > 0.375% | pass | median 1.927% |
| DOGEUSDT pooled 1h: VR<1, CI excludes 1 and Holm p<0.05 | pass | VR=0.849, CI [0.828, 0.870], Holm p=0.0000 |
| DOGEUSDT 1h: VR<1 with CI excluding 1 and p<0.05 in a majority of years | pass | 8 of 8 years |
| DOGEUSDT pooled 4h: VR<1, CI excludes 1 and Holm p<0.05 | pass | VR=0.760, CI [0.725, 0.800], Holm p=0.0000 |
| DOGEUSDT 4h: VR<1 with CI excluding 1 and p<0.05 in a majority of years | pass | 8 of 8 years |
| DOGEUSDT median 4h range > 0.375% | pass | median 2.172% |
| LINKUSDT pooled 1h: VR<1, CI excludes 1 and Holm p<0.05 | pass | VR=0.882, CI [0.870, 0.894], Holm p=0.0000 |
| LINKUSDT 1h: VR<1 with CI excluding 1 and p<0.05 in a majority of years | pass | 8 of 8 years |
| LINKUSDT pooled 4h: VR<1, CI excludes 1 and Holm p<0.05 | pass | VR=0.807, CI [0.787, 0.832], Holm p=0.0000 |
| LINKUSDT 4h: VR<1 with CI excluding 1 and p<0.05 in a majority of years | pass | 8 of 8 years |
| LINKUSDT median 4h range > 0.375% | pass | median 2.588% |
| AVAXUSDT pooled 1h: VR<1, CI excludes 1 and Holm p<0.05 | pass | VR=0.929, CI [0.914, 0.943], Holm p=0.0000 |
| AVAXUSDT 1h: VR<1 with CI excluding 1 and p<0.05 in a majority of years | pass | 5 of 7 years |
| AVAXUSDT pooled 4h: VR<1, CI excludes 1 and Holm p<0.05 | pass | VR=0.868, CI [0.842, 0.893], Holm p=0.0000 |
| AVAXUSDT 4h: VR<1 with CI excluding 1 and p<0.05 in a majority of years | pass | 4 of 7 years |
| AVAXUSDT median 4h range > 0.375% | pass | median 2.651% |

## BTCUSDT: variance ratio, pooled (5m base)

Baseline:

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.949 | -4.51 | 0.000 | [0.930, 0.963] | yes (<1) |
| 30m | 6 | 0.927 | -4.08 | 0.000 | [0.908, 0.945] | yes (<1) |
| 1h | 12 | 0.899 | -4.16 | 0.000 | [0.873, 0.920] | yes (<1) |
| 2h | 24 | 0.876 | -3.99 | 0.000 | [0.845, 0.903] | yes (<1) |
| 4h | 48 | 0.858 | -3.73 | 0.000 | [0.819, 0.894] | yes (<1) |
| 8h | 96 | 0.836 | -3.56 | 0.000 | [0.783, 0.882] | yes (<1) |
| 1d | 288 | 0.867 | -2.10 | 0.036 | [0.793, 0.940] | yes (<1) |

Robust (winsorised, event windows excluded):

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.946 | -15.28 | 0.000 | [0.940, 0.953] | yes (<1) |
| 30m | 6 | 0.912 | -15.35 | 0.000 | [0.903, 0.920] | yes (<1) |
| 1h | 12 | 0.874 | -14.69 | 0.000 | [0.862, 0.886] | yes (<1) |
| 2h | 24 | 0.852 | -12.12 | 0.000 | [0.837, 0.867] | yes (<1) |
| 4h | 48 | 0.840 | -9.42 | 0.000 | [0.818, 0.860] | yes (<1) |
| 8h | 96 | 0.825 | -7.45 | 0.000 | [0.794, 0.857] | yes (<1) |
| 1d | 288 | 0.873 | -3.26 | 0.001 | [0.820, 0.932] | yes (<1) |

1m base, labelled contaminated by bid-ask bounce (kline closes are trade prices, so VR<1 here is partly bounce, which a maker grid cannot harvest); baseline only:

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 5m (1m base) | 5 | 0.981 | -2.20 | 0.028 | [0.974, 0.989] | yes (<1) |

## ETHUSDT: variance ratio, pooled (5m base)

Baseline:

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.958 | -3.35 | 0.001 | [0.928, 0.977] | yes (<1) |
| 30m | 6 | 0.948 | -2.64 | 0.008 | [0.921, 0.968] | yes (<1) |
| 1h | 12 | 0.922 | -2.93 | 0.003 | [0.886, 0.945] | yes (<1) |
| 2h | 24 | 0.904 | -2.91 | 0.004 | [0.863, 0.933] | yes (<1) |
| 4h | 48 | 0.887 | -2.85 | 0.004 | [0.838, 0.923] | yes (<1) |
| 8h | 96 | 0.865 | -2.88 | 0.004 | [0.802, 0.909] | yes (<1) |
| 1d | 288 | 0.902 | -1.59 | 0.112 | [0.829, 0.970] | yes (<1) |

Robust (winsorised, event windows excluded):

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.963 | -10.50 | 0.000 | [0.957, 0.970] | yes (<1) |
| 30m | 6 | 0.940 | -10.42 | 0.000 | [0.931, 0.949] | yes (<1) |
| 1h | 12 | 0.910 | -10.54 | 0.000 | [0.899, 0.921] | yes (<1) |
| 2h | 24 | 0.889 | -9.05 | 0.000 | [0.875, 0.906] | yes (<1) |
| 4h | 48 | 0.871 | -7.56 | 0.000 | [0.849, 0.895] | yes (<1) |
| 8h | 96 | 0.848 | -6.41 | 0.000 | [0.819, 0.882] | yes (<1) |
| 1d | 288 | 0.897 | -2.65 | 0.008 | [0.838, 0.953] | yes (<1) |

1m base, labelled contaminated by bid-ask bounce (kline closes are trade prices, so VR<1 here is partly bounce, which a maker grid cannot harvest); baseline only:

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 5m (1m base) | 5 | 0.986 | -1.63 | 0.103 | [0.973, 0.996] | yes (<1) |

## SOLUSDT: variance ratio, pooled (5m base)

Baseline:

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.959 | -4.84 | 0.000 | [0.944, 0.972] | yes (<1) |
| 30m | 6 | 0.941 | -4.54 | 0.000 | [0.920, 0.958] | yes (<1) |
| 1h | 12 | 0.908 | -5.13 | 0.000 | [0.883, 0.930] | yes (<1) |
| 2h | 24 | 0.885 | -4.96 | 0.000 | [0.854, 0.914] | yes (<1) |
| 4h | 48 | 0.862 | -4.66 | 0.000 | [0.824, 0.902] | yes (<1) |
| 8h | 96 | 0.838 | -4.30 | 0.000 | [0.793, 0.886] | yes (<1) |
| 1d | 288 | 0.843 | -2.80 | 0.005 | [0.787, 0.902] | yes (<1) |

Robust (winsorised, event windows excluded):

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.973 | -7.08 | 0.000 | [0.965, 0.981] | yes (<1) |
| 30m | 6 | 0.956 | -7.15 | 0.000 | [0.942, 0.967] | yes (<1) |
| 1h | 12 | 0.926 | -7.98 | 0.000 | [0.909, 0.943] | yes (<1) |
| 2h | 24 | 0.905 | -7.19 | 0.000 | [0.882, 0.928] | yes (<1) |
| 4h | 48 | 0.881 | -6.43 | 0.000 | [0.850, 0.911] | yes (<1) |
| 8h | 96 | 0.856 | -5.62 | 0.000 | [0.816, 0.892] | yes (<1) |
| 1d | 288 | 0.872 | -3.00 | 0.003 | [0.810, 0.928] | yes (<1) |

1m base, labelled contaminated by bid-ask bounce (kline closes are trade prices, so VR<1 here is partly bounce, which a maker grid cannot harvest); baseline only:

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 5m (1m base) | 5 | 0.965 | -5.93 | 0.000 | [0.954, 0.977] | yes (<1) |

## BNBUSDT: variance ratio, pooled (5m base)

Baseline:

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.960 | -2.27 | 0.023 | [0.930, 0.988] | yes (<1) |
| 30m | 6 | 0.948 | -2.02 | 0.043 | [0.914, 0.982] | yes (<1) |
| 1h | 12 | 0.907 | -2.83 | 0.005 | [0.869, 0.939] | yes (<1) |
| 2h | 24 | 0.872 | -3.21 | 0.001 | [0.832, 0.908] | yes (<1) |
| 4h | 48 | 0.850 | -3.21 | 0.001 | [0.806, 0.892] | yes (<1) |
| 8h | 96 | 0.827 | -3.15 | 0.002 | [0.781, 0.885] | yes (<1) |
| 1d | 288 | 0.881 | -1.65 | 0.100 | [0.802, 0.984] | yes (<1) |

Robust (winsorised, event windows excluded):

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.971 | -6.90 | 0.000 | [0.962, 0.979] | yes (<1) |
| 30m | 6 | 0.951 | -7.12 | 0.000 | [0.937, 0.963] | yes (<1) |
| 1h | 12 | 0.914 | -8.32 | 0.000 | [0.896, 0.932] | yes (<1) |
| 2h | 24 | 0.889 | -7.53 | 0.000 | [0.870, 0.913] | yes (<1) |
| 4h | 48 | 0.873 | -6.20 | 0.000 | [0.845, 0.904] | yes (<1) |
| 8h | 96 | 0.858 | -5.04 | 0.000 | [0.822, 0.902] | yes (<1) |
| 1d | 288 | 0.930 | -1.52 | 0.130 | [0.856, 1.018] | no |

1m base, labelled contaminated by bid-ask bounce (kline closes are trade prices, so VR<1 here is partly bounce, which a maker grid cannot harvest); baseline only:

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 5m (1m base) | 5 | 0.985 | -1.76 | 0.079 | [0.972, 0.999] | yes (<1) |

## XRPUSDT: variance ratio, pooled (5m base)

Baseline:

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.961 | -2.87 | 0.004 | [0.938, 0.980] | yes (<1) |
| 30m | 6 | 0.940 | -2.92 | 0.004 | [0.916, 0.963] | yes (<1) |
| 1h | 12 | 0.896 | -3.76 | 0.000 | [0.860, 0.927] | yes (<1) |
| 2h | 24 | 0.853 | -4.25 | 0.000 | [0.812, 0.895] | yes (<1) |
| 4h | 48 | 0.819 | -4.33 | 0.000 | [0.771, 0.878] | yes (<1) |
| 8h | 96 | 0.791 | -4.23 | 0.000 | [0.736, 0.865] | yes (<1) |
| 1d | 288 | 0.811 | -2.88 | 0.004 | [0.748, 0.901] | yes (<1) |

Robust (winsorised, event windows excluded):

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.956 | -10.69 | 0.000 | [0.949, 0.963] | yes (<1) |
| 30m | 6 | 0.918 | -12.25 | 0.000 | [0.908, 0.927] | yes (<1) |
| 1h | 12 | 0.870 | -13.04 | 0.000 | [0.858, 0.884] | yes (<1) |
| 2h | 24 | 0.818 | -12.87 | 0.000 | [0.801, 0.837] | yes (<1) |
| 4h | 48 | 0.775 | -11.45 | 0.000 | [0.750, 0.799] | yes (<1) |
| 8h | 96 | 0.750 | -9.24 | 0.000 | [0.715, 0.786] | yes (<1) |
| 1d | 288 | 0.810 | -4.24 | 0.000 | [0.742, 0.881] | yes (<1) |

1m base, labelled contaminated by bid-ask bounce (kline closes are trade prices, so VR<1 here is partly bounce, which a maker grid cannot harvest); baseline only:

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 5m (1m base) | 5 | 0.989 | -0.78 | 0.435 | [0.976, 1.003] | no |

## DOGEUSDT: variance ratio, pooled (5m base)

Baseline:

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.923 | -3.55 | 0.000 | [0.896, 0.948] | yes (<1) |
| 30m | 6 | 0.893 | -3.40 | 0.001 | [0.853, 0.928] | yes (<1) |
| 1h | 12 | 0.853 | -3.58 | 0.000 | [0.793, 0.912] | yes (<1) |
| 2h | 24 | 0.812 | -3.71 | 0.000 | [0.737, 0.883] | yes (<1) |
| 4h | 48 | 0.771 | -3.66 | 0.000 | [0.692, 0.859] | yes (<1) |
| 8h | 96 | 0.753 | -3.18 | 0.001 | [0.655, 0.883] | yes (<1) |
| 1d | 288 | 0.883 | -1.03 | 0.303 | [0.709, 1.162] | no |

Robust (winsorised, event windows excluded):

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.939 | -12.37 | 0.000 | [0.930, 0.947] | yes (<1) |
| 30m | 6 | 0.896 | -12.69 | 0.000 | [0.885, 0.911] | yes (<1) |
| 1h | 12 | 0.849 | -12.38 | 0.000 | [0.828, 0.870] | yes (<1) |
| 2h | 24 | 0.805 | -11.17 | 0.000 | [0.776, 0.835] | yes (<1) |
| 4h | 48 | 0.760 | -9.81 | 0.000 | [0.725, 0.800] | yes (<1) |
| 8h | 96 | 0.748 | -7.43 | 0.000 | [0.694, 0.811] | yes (<1) |
| 1d | 288 | 0.854 | -2.59 | 0.010 | [0.754, 0.991] | yes (<1) |

1m base, labelled contaminated by bid-ask bounce (kline closes are trade prices, so VR<1 here is partly bounce, which a maker grid cannot harvest); baseline only:

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 5m (1m base) | 5 | 0.949 | -3.66 | 0.000 | [0.917, 0.985] | yes (<1) |

## LINKUSDT: variance ratio, pooled (5m base)

Baseline:

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.952 | -4.31 | 0.000 | [0.937, 0.964] | yes (<1) |
| 30m | 6 | 0.921 | -4.53 | 0.000 | [0.906, 0.940] | yes (<1) |
| 1h | 12 | 0.877 | -4.87 | 0.000 | [0.846, 0.912] | yes (<1) |
| 2h | 24 | 0.840 | -5.00 | 0.000 | [0.801, 0.887] | yes (<1) |
| 4h | 48 | 0.798 | -5.30 | 0.000 | [0.753, 0.849] | yes (<1) |
| 8h | 96 | 0.774 | -5.08 | 0.000 | [0.723, 0.833] | yes (<1) |
| 1d | 288 | 0.765 | -4.03 | 0.000 | [0.704, 0.833] | yes (<1) |

Robust (winsorised, event windows excluded):

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.948 | -16.08 | 0.000 | [0.941, 0.954] | yes (<1) |
| 30m | 6 | 0.916 | -15.74 | 0.000 | [0.907, 0.927] | yes (<1) |
| 1h | 12 | 0.882 | -14.88 | 0.000 | [0.870, 0.894] | yes (<1) |
| 2h | 24 | 0.847 | -13.54 | 0.000 | [0.832, 0.866] | yes (<1) |
| 4h | 48 | 0.807 | -12.23 | 0.000 | [0.787, 0.832] | yes (<1) |
| 8h | 96 | 0.786 | -9.88 | 0.000 | [0.759, 0.817] | yes (<1) |
| 1d | 288 | 0.786 | -5.96 | 0.000 | [0.739, 0.845] | yes (<1) |

1m base, labelled contaminated by bid-ask bounce (kline closes are trade prices, so VR<1 here is partly bounce, which a maker grid cannot harvest); baseline only:

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 5m (1m base) | 5 | 0.951 | -4.43 | 0.000 | [0.923, 0.996] | yes (<1) |

## AVAXUSDT: variance ratio, pooled (5m base)

Baseline:

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.873 | -1.35 | 0.177 | [0.734, 0.984] | yes (<1) |
| 30m | 6 | 0.847 | -1.26 | 0.208 | [0.686, 0.968] | yes (<1) |
| 1h | 12 | 0.803 | -1.43 | 0.152 | [0.638, 0.933] | yes (<1) |
| 2h | 24 | 0.769 | -1.58 | 0.114 | [0.602, 0.900] | yes (<1) |
| 4h | 48 | 0.742 | -1.70 | 0.089 | [0.574, 0.876] | yes (<1) |
| 8h | 96 | 0.719 | -1.81 | 0.071 | [0.548, 0.850] | yes (<1) |
| 1d | 288 | 0.731 | -1.67 | 0.094 | [0.538, 0.873] | yes (<1) |

Robust (winsorised, event windows excluded):

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.981 | -4.86 | 0.000 | [0.973, 0.990] | yes (<1) |
| 30m | 6 | 0.963 | -5.80 | 0.000 | [0.952, 0.975] | yes (<1) |
| 1h | 12 | 0.929 | -7.52 | 0.000 | [0.914, 0.943] | yes (<1) |
| 2h | 24 | 0.897 | -7.64 | 0.000 | [0.880, 0.916] | yes (<1) |
| 4h | 48 | 0.868 | -7.02 | 0.000 | [0.842, 0.893] | yes (<1) |
| 8h | 96 | 0.836 | -6.34 | 0.000 | [0.803, 0.873] | yes (<1) |
| 1d | 288 | 0.837 | -3.80 | 0.000 | [0.782, 0.897] | yes (<1) |

1m base, labelled contaminated by bid-ask bounce (kline closes are trade prices, so VR<1 here is partly bounce, which a maker grid cannot harvest); baseline only:

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 5m (1m base) | 5 | 0.937 | -0.88 | 0.378 | [0.875, 0.979] | yes (<1) |

## Stability: per calendar year (VR, 95% CI)

| Symbol | Year | 1h VR | 1h CI | 4h VR | 4h CI | DFA Hurst |
|---|---|---|---|---|---|---|
| BTCUSDT | 2019 | 0.934 | [0.894, 0.982] | 0.896 | [0.817, 0.986] | 0.491 |
| BTCUSDT | 2020 | 0.875 | [0.822, 0.934] | 0.795 | [0.706, 0.915] | 0.460 |
| BTCUSDT | 2021 | 0.840 | [0.784, 0.894] | 0.797 | [0.720, 0.871] | 0.473 |
| BTCUSDT | 2022 | 0.941 | [0.910, 0.974] | 0.914 | [0.849, 0.974] | 0.497 |
| BTCUSDT | 2023 | 0.904 | [0.833, 0.978] | 0.914 | [0.813, 1.031] | 0.488 |
| BTCUSDT | 2024 | 0.958 | [0.922, 0.999] | 0.925 | [0.869, 0.988] | 0.495 |
| BTCUSDT | 2025 | 0.934 | [0.874, 0.992] | 0.927 | [0.831, 1.019] | 0.494 |
| BTCUSDT | 2026 | 0.953 | [0.915, 0.997] | 0.910 | [0.846, 0.989] | 0.486 |
| ETHUSDT | 2019 | 0.936 | [0.901, 0.983] | 0.880 | [0.814, 0.957] | 0.482 |
| ETHUSDT | 2020 | 0.902 | [0.852, 0.961] | 0.834 | [0.751, 0.931] | 0.466 |
| ETHUSDT | 2021 | 0.847 | [0.768, 0.910] | 0.795 | [0.698, 0.881] | 0.475 |
| ETHUSDT | 2022 | 0.978 | [0.946, 1.012] | 0.964 | [0.909, 1.018] | 0.505 |
| ETHUSDT | 2023 | 0.927 | [0.867, 0.985] | 0.926 | [0.820, 1.006] | 0.491 |
| ETHUSDT | 2024 | 0.998 | [0.951, 1.046] | 0.981 | [0.899, 1.092] | 0.503 |
| ETHUSDT | 2025 | 0.980 | [0.937, 1.027] | 1.017 | [0.940, 1.094] | 0.506 |
| ETHUSDT | 2026 | 0.984 | [0.935, 1.040] | 0.946 | [0.872, 1.033] | 0.493 |
| SOLUSDT | 2020 | 0.914 | [0.862, 0.973] | 0.806 | [0.715, 0.896] | 0.482 |
| SOLUSDT | 2021 | 0.916 | [0.870, 0.963] | 0.869 | [0.788, 0.949] | 0.490 |
| SOLUSDT | 2022 | 0.887 | [0.842, 0.934] | 0.858 | [0.805, 0.919] | 0.488 |
| SOLUSDT | 2023 | 0.875 | [0.810, 0.924] | 0.850 | [0.767, 0.921] | 0.481 |
| SOLUSDT | 2024 | 0.900 | [0.844, 0.947] | 0.824 | [0.757, 0.881] | 0.484 |
| SOLUSDT | 2025 | 0.956 | [0.911, 1.007] | 0.945 | [0.861, 1.033] | 0.492 |
| SOLUSDT | 2026 | 0.933 | [0.903, 0.970] | 0.869 | [0.794, 0.950] | 0.475 |
| BNBUSDT | 2019 | 0.905 | [0.874, 0.940] | 0.814 | [0.761, 0.868] | 0.475 |
| BNBUSDT | 2020 | 0.916 | [0.866, 0.991] | 0.821 | [0.737, 0.925] | 0.453 |
| BNBUSDT | 2021 | 0.876 | [0.837, 0.917] | 0.834 | [0.776, 0.903] | 0.475 |
| BNBUSDT | 2022 | 1.028 | [0.968, 1.093] | 1.021 | [0.960, 1.108] | 0.515 |
| BNBUSDT | 2023 | 0.925 | [0.879, 0.974] | 0.902 | [0.837, 0.964] | 0.485 |
| BNBUSDT | 2024 | 0.970 | [0.932, 1.014] | 0.908 | [0.832, 0.996] | 0.496 |
| BNBUSDT | 2025 | 0.770 | [0.547, 1.032] | 0.715 | [0.455, 1.038] | 0.464 |
| BNBUSDT | 2026 | 0.962 | [0.928, 1.002] | 0.912 | [0.848, 0.985] | 0.483 |
| XRPUSDT | 2019 | 0.895 | [0.843, 0.938] | 0.790 | [0.715, 0.862] | 0.467 |
| XRPUSDT | 2020 | 0.933 | [0.870, 1.004] | 0.822 | [0.753, 0.908] | 0.468 |
| XRPUSDT | 2021 | 0.847 | [0.778, 0.910] | 0.752 | [0.661, 0.865] | 0.459 |
| XRPUSDT | 2022 | 0.933 | [0.891, 0.973] | 0.893 | [0.828, 0.968] | 0.488 |
| XRPUSDT | 2023 | 0.892 | [0.709, 1.109] | 0.952 | [0.618, 1.432] | 0.508 |
| XRPUSDT | 2024 | 0.927 | [0.868, 1.009] | 0.855 | [0.770, 0.963] | 0.487 |
| XRPUSDT | 2025 | 0.913 | [0.852, 0.985] | 0.830 | [0.692, 0.962] | 0.457 |
| XRPUSDT | 2026 | 0.947 | [0.903, 0.998] | 0.907 | [0.819, 0.997] | 0.487 |
| DOGEUSDT | 2019 | 0.547 | [0.498, 0.592] | 0.450 | [0.391, 0.515] | 0.401 |
| DOGEUSDT | 2020 | 0.733 | [0.647, 0.831] | 0.627 | [0.528, 0.753] | 0.432 |
| DOGEUSDT | 2021 | 0.879 | [0.792, 0.987] | 0.790 | [0.649, 0.938] | 0.449 |
| DOGEUSDT | 2022 | 0.903 | [0.866, 0.943] | 0.846 | [0.775, 0.934] | 0.477 |
| DOGEUSDT | 2023 | 0.909 | [0.821, 1.033] | 0.827 | [0.736, 0.945] | 0.481 |
| DOGEUSDT | 2024 | 0.925 | [0.886, 0.977] | 0.866 | [0.809, 0.934] | 0.488 |
| DOGEUSDT | 2025 | 0.761 | [0.579, 0.986] | 0.682 | [0.445, 0.981] | 0.438 |
| DOGEUSDT | 2026 | 0.937 | [0.903, 0.973] | 0.891 | [0.820, 0.971] | 0.483 |
| LINKUSDT | 2019 | 0.843 | [0.770, 0.971] | 0.732 | [0.631, 0.899] | 0.471 |
| LINKUSDT | 2020 | 0.856 | [0.825, 0.890] | 0.734 | [0.677, 0.793] | 0.451 |
| LINKUSDT | 2021 | 0.878 | [0.831, 0.927] | 0.809 | [0.755, 0.873] | 0.473 |
| LINKUSDT | 2022 | 0.940 | [0.908, 0.974] | 0.893 | [0.847, 0.939] | 0.497 |
| LINKUSDT | 2023 | 0.897 | [0.852, 0.943] | 0.875 | [0.795, 0.965] | 0.485 |
| LINKUSDT | 2024 | 0.957 | [0.907, 1.020] | 0.922 | [0.848, 1.007] | 0.501 |
| LINKUSDT | 2025 | 0.828 | [0.708, 0.951] | 0.758 | [0.555, 0.967] | 0.441 |
| LINKUSDT | 2026 | 0.912 | [0.874, 0.949] | 0.855 | [0.793, 0.931] | 0.482 |
| AVAXUSDT | 2020 | 0.911 | [0.786, 1.059] | 0.752 | [0.620, 0.876] | 0.467 |
| AVAXUSDT | 2021 | 0.884 | [0.848, 0.919] | 0.784 | [0.742, 0.840] | 0.463 |
| AVAXUSDT | 2022 | 0.928 | [0.879, 0.977] | 0.883 | [0.786, 0.968] | 0.489 |
| AVAXUSDT | 2023 | 0.901 | [0.853, 0.950] | 0.894 | [0.822, 0.978] | 0.493 |
| AVAXUSDT | 2024 | 0.970 | [0.929, 1.017] | 0.938 | [0.874, 0.999] | 0.500 |
| AVAXUSDT | 2025 | 0.417 | [0.227, 1.020] | 0.387 | [0.182, 1.049] | 0.423 |
| AVAXUSDT | 2026 | 0.963 | [0.919, 1.014] | 0.923 | [0.842, 1.011] | 0.494 |

## Stability: per calendar year, robust (VR, 95% CI, unadjusted z* p)

| Symbol | Year | 1h VR | 1h CI | 1h p | 4h VR | 4h CI | 4h p |
|---|---|---|---|---|---|---|---|
| BTCUSDT | 2019 | 0.849 | [0.821, 0.878] | 0.000 | 0.783 | [0.733, 0.838] | 0.000 |
| BTCUSDT | 2020 | 0.816 | [0.787, 0.845] | 0.000 | 0.775 | [0.710, 0.851] | 0.000 |
| BTCUSDT | 2021 | 0.857 | [0.832, 0.880] | 0.000 | 0.827 | [0.788, 0.868] | 0.000 |
| BTCUSDT | 2022 | 0.917 | [0.888, 0.943] | 0.000 | 0.902 | [0.848, 0.961] | 0.008 |
| BTCUSDT | 2023 | 0.835 | [0.807, 0.865] | 0.000 | 0.798 | [0.743, 0.860] | 0.000 |
| BTCUSDT | 2024 | 0.944 | [0.917, 0.973] | 0.002 | 0.907 | [0.856, 0.965] | 0.009 |
| BTCUSDT | 2025 | 0.939 | [0.907, 0.966] | 0.001 | 0.924 | [0.861, 0.985] | 0.040 |
| BTCUSDT | 2026 | 0.917 | [0.883, 0.954] | 0.000 | 0.867 | [0.802, 0.942] | 0.001 |
| ETHUSDT | 2019 | 0.865 | [0.840, 0.891] | 0.000 | 0.767 | [0.716, 0.824] | 0.000 |
| ETHUSDT | 2020 | 0.862 | [0.836, 0.890] | 0.000 | 0.805 | [0.755, 0.860] | 0.000 |
| ETHUSDT | 2021 | 0.911 | [0.890, 0.937] | 0.000 | 0.880 | [0.831, 0.932] | 0.005 |
| ETHUSDT | 2022 | 0.952 | [0.924, 0.978] | 0.008 | 0.944 | [0.888, 0.994] | 0.122 |
| ETHUSDT | 2023 | 0.860 | [0.828, 0.894] | 0.000 | 0.818 | [0.767, 0.867] | 0.000 |
| ETHUSDT | 2024 | 0.959 | [0.928, 0.990] | 0.020 | 0.887 | [0.833, 0.945] | 0.001 |
| ETHUSDT | 2025 | 0.941 | [0.912, 0.967] | 0.001 | 0.952 | [0.892, 1.003] | 0.151 |
| ETHUSDT | 2026 | 0.908 | [0.878, 0.941] | 0.000 | 0.860 | [0.802, 0.921] | 0.001 |
| SOLUSDT | 2020 | 0.918 | [0.861, 0.978] | 0.007 | 0.806 | [0.720, 0.911] | 0.001 |
| SOLUSDT | 2021 | 0.937 | [0.899, 0.978] | 0.001 | 0.896 | [0.831, 0.963] | 0.007 |
| SOLUSDT | 2022 | 0.919 | [0.897, 0.945] | 0.000 | 0.922 | [0.864, 0.977] | 0.055 |
| SOLUSDT | 2023 | 0.874 | [0.851, 0.897] | 0.000 | 0.796 | [0.749, 0.841] | 0.000 |
| SOLUSDT | 2024 | 0.941 | [0.916, 0.965] | 0.001 | 0.860 | [0.816, 0.906] | 0.000 |
| SOLUSDT | 2025 | 0.947 | [0.919, 0.974] | 0.003 | 0.926 | [0.868, 0.985] | 0.038 |
| SOLUSDT | 2026 | 0.928 | [0.900, 0.962] | 0.000 | 0.884 | [0.826, 0.955] | 0.003 |
| BNBUSDT | 2019 | 0.878 | [0.853, 0.907] | 0.000 | 0.776 | [0.737, 0.818] | 0.000 |
| BNBUSDT | 2020 | 0.874 | [0.839, 0.907] | 0.000 | 0.798 | [0.749, 0.859] | 0.000 |
| BNBUSDT | 2021 | 0.911 | [0.877, 0.949] | 0.000 | 0.888 | [0.828, 0.953] | 0.017 |
| BNBUSDT | 2022 | 0.977 | [0.940, 1.018] | 0.217 | 1.017 | [0.939, 1.121] | 0.644 |
| BNBUSDT | 2023 | 0.855 | [0.825, 0.885] | 0.000 | 0.814 | [0.765, 0.871] | 0.000 |
| BNBUSDT | 2024 | 0.959 | [0.924, 0.992] | 0.028 | 0.889 | [0.826, 0.958] | 0.003 |
| BNBUSDT | 2025 | 0.997 | [0.963, 1.031] | 0.883 | 0.982 | [0.918, 1.046] | 0.639 |
| BNBUSDT | 2026 | 0.945 | [0.919, 0.976] | 0.007 | 0.900 | [0.849, 0.958] | 0.014 |
| XRPUSDT | 2019 | 0.815 | [0.791, 0.839] | 0.000 | 0.655 | [0.614, 0.699] | 0.000 |
| XRPUSDT | 2020 | 0.858 | [0.819, 0.898] | 0.000 | 0.757 | [0.690, 0.817] | 0.000 |
| XRPUSDT | 2021 | 0.841 | [0.816, 0.869] | 0.000 | 0.718 | [0.676, 0.760] | 0.000 |
| XRPUSDT | 2022 | 0.922 | [0.893, 0.951] | 0.000 | 0.887 | [0.825, 0.952] | 0.002 |
| XRPUSDT | 2023 | 0.831 | [0.800, 0.869] | 0.000 | 0.724 | [0.665, 0.803] | 0.000 |
| XRPUSDT | 2024 | 0.917 | [0.887, 0.951] | 0.001 | 0.848 | [0.775, 0.926] | 0.002 |
| XRPUSDT | 2025 | 0.913 | [0.883, 0.942] | 0.000 | 0.850 | [0.799, 0.899] | 0.000 |
| XRPUSDT | 2026 | 0.937 | [0.901, 0.975] | 0.002 | 0.919 | [0.837, 1.011] | 0.048 |
| DOGEUSDT | 2019 | 0.511 | [0.469, 0.554] | 0.000 | 0.424 | [0.368, 0.482] | 0.000 |
| DOGEUSDT | 2020 | 0.690 | [0.638, 0.738] | 0.000 | 0.594 | [0.518, 0.679] | 0.000 |
| DOGEUSDT | 2021 | 0.856 | [0.812, 0.898] | 0.000 | 0.750 | [0.668, 0.827] | 0.000 |
| DOGEUSDT | 2022 | 0.867 | [0.836, 0.900] | 0.000 | 0.784 | [0.726, 0.844] | 0.000 |
| DOGEUSDT | 2023 | 0.841 | [0.818, 0.869] | 0.000 | 0.724 | [0.687, 0.765] | 0.000 |
| DOGEUSDT | 2024 | 0.926 | [0.893, 0.963] | 0.001 | 0.854 | [0.802, 0.915] | 0.001 |
| DOGEUSDT | 2025 | 0.956 | [0.930, 0.984] | 0.012 | 0.920 | [0.871, 0.967] | 0.019 |
| DOGEUSDT | 2026 | 0.913 | [0.886, 0.943] | 0.000 | 0.876 | [0.816, 0.947] | 0.001 |
| LINKUSDT | 2019 | 0.784 | [0.752, 0.820] | 0.000 | 0.648 | [0.596, 0.704] | 0.000 |
| LINKUSDT | 2020 | 0.863 | [0.837, 0.889] | 0.000 | 0.742 | [0.694, 0.799] | 0.000 |
| LINKUSDT | 2021 | 0.907 | [0.882, 0.936] | 0.000 | 0.849 | [0.802, 0.895] | 0.000 |
| LINKUSDT | 2022 | 0.934 | [0.902, 0.966] | 0.000 | 0.912 | [0.860, 0.966] | 0.018 |
| LINKUSDT | 2023 | 0.893 | [0.866, 0.922] | 0.000 | 0.839 | [0.786, 0.910] | 0.000 |
| LINKUSDT | 2024 | 0.954 | [0.920, 0.990] | 0.013 | 0.900 | [0.844, 0.968] | 0.007 |
| LINKUSDT | 2025 | 0.922 | [0.893, 0.953] | 0.000 | 0.913 | [0.858, 0.974] | 0.011 |
| LINKUSDT | 2026 | 0.877 | [0.847, 0.908] | 0.000 | 0.832 | [0.777, 0.896] | 0.000 |
| AVAXUSDT | 2020 | 0.842 | [0.771, 0.915] | 0.000 | 0.688 | [0.605, 0.774] | 0.000 |
| AVAXUSDT | 2021 | 0.906 | [0.880, 0.931] | 0.000 | 0.803 | [0.765, 0.846] | 0.000 |
| AVAXUSDT | 2022 | 0.954 | [0.925, 0.982] | 0.025 | 0.962 | [0.905, 1.017] | 0.359 |
| AVAXUSDT | 2023 | 0.891 | [0.864, 0.920] | 0.000 | 0.857 | [0.805, 0.917] | 0.000 |
| AVAXUSDT | 2024 | 0.977 | [0.948, 1.010] | 0.199 | 0.929 | [0.877, 0.990] | 0.047 |
| AVAXUSDT | 2025 | 0.992 | [0.964, 1.020] | 0.644 | 0.985 | [0.931, 1.037] | 0.650 |
| AVAXUSDT | 2026 | 0.938 | [0.899, 0.978] | 0.001 | 0.916 | [0.831, 1.004] | 0.021 |

## DFA Hurst exponent (5m returns, 1h-1d scales; baseline)

| Symbol | Pooled H |
|---|---|
| BTCUSDT | 0.481 |
| ETHUSDT | 0.484 |
| SOLUSDT | 0.484 |
| BNBUSDT | 0.474 |
| XRPUSDT | 0.471 |
| DOGEUSDT | 0.462 |
| LINKUSDT | 0.474 |
| AVAXUSDT | 0.466 |

H < 0.5 means mean reversion, 0.5 a random walk, > 0.5 trending.

## Range vs fees: (high - low) / open

Maker round trip 0.15% (BNB); 2.5x threshold 0.375%. Robust rows exclude the event windows.

| Symbol | Bar | Median | 25th pct | Median vs 0.375% | Robust median |
|---|---|---|---|---|---|
| BTCUSDT | 1h | 0.635% | 0.397% | above | 0.635% |
| BTCUSDT | 4h | 1.352% | 0.846% | above | 1.352% |
| ETHUSDT | 1h | 0.848% | 0.539% | above | 0.848% |
| ETHUSDT | 4h | 1.825% | 1.180% | above | 1.824% |
| SOLUSDT | 1h | 1.243% | 0.809% | above | 1.243% |
| SOLUSDT | 4h | 2.634% | 1.713% | above | 2.633% |
| BNBUSDT | 1h | 0.795% | 0.496% | above | 0.795% |
| BNBUSDT | 4h | 1.668% | 1.054% | above | 1.667% |
| XRPUSDT | 1h | 0.911% | 0.589% | above | 0.911% |
| XRPUSDT | 4h | 1.927% | 1.255% | above | 1.927% |
| DOGEUSDT | 1h | 1.037% | 0.668% | above | 1.036% |
| DOGEUSDT | 4h | 2.173% | 1.406% | above | 2.172% |
| LINKUSDT | 1h | 1.248% | 0.825% | above | 1.248% |
| LINKUSDT | 4h | 2.589% | 1.739% | above | 2.588% |
| AVAXUSDT | 1h | 1.256% | 0.809% | above | 1.255% |
| AVAXUSDT | 4h | 2.654% | 1.699% | above | 2.651% |
