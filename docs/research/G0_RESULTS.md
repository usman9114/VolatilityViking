# G0 results: which coins mean-revert at grid horizons?

- BTCUSDT: 2019-01-01 to 2026-10-31, 4,115,751 1m bars (814 empty 5m bins forward-filled).
- ETHUSDT: 2019-01-01 to 2026-10-31, 4,115,750 1m bars (814 empty 5m bins forward-filled).
- SOLUSDT: 2020-08-11 to 2026-10-31, 3,271,335 1m bars (283 empty 5m bins forward-filled).
- BNBUSDT: 2019-01-01 to 2026-10-31, 4,115,750 1m bars (814 empty 5m bins forward-filled).
- XRPUSDT: 2019-01-01 to 2026-10-31, 4,115,751 1m bars (814 empty 5m bins forward-filled).
- DOGEUSDT: 2019-07-05 to 2026-10-31, 3,849,651 1m bars (610 empty 5m bins forward-filled).
- LINKUSDT: 2019-01-16 to 2026-10-31, 4,093,550 1m bars (814 empty 5m bins forward-filled).
- AVAXUSDT: 2020-09-22 to 2026-10-31, 3,210,826 1m bars (283 empty 5m bins forward-filled).

Method: Lo-MacKinlay VR on log returns, robust z*, stationary block bootstrap (mean block 1 day, 500 reps, seed 20260101), percentile 95% CI. Year rows need at least 30 days of data.

## Verdict per coin

Each coin is gated on its own. Only coins that pass go on to the G1 replay.

| Coin | G0 | Pooled 1h | Pooled 4h | Years 1h | Years 4h | 1h since 2024 | Median 4h range |
|---|---|---|---|---|---|---|---|
| BTCUSDT | **PASS** | VR=0.899, CI [0.877, 0.921] | VR=0.859, CI [0.825, 0.896] | 8 of 8 years | 6 of 8 years | 3 of 3 | median 1.349% |
| ETHUSDT | fail | VR=0.923, CI [0.894, 0.949] | VR=0.889, CI [0.850, 0.927] | 4 of 8 years | 3 of 8 years | 0 of 3 | median 1.828% |
| SOLUSDT | **PASS** | VR=0.914, CI [0.892, 0.938] | VR=0.858, CI [0.828, 0.897] | 5 of 7 years | 6 of 7 years | 2 of 3 | median 2.662% |
| BNBUSDT | fail | VR=0.909, CI [0.870, 0.944] | VR=0.851, CI [0.807, 0.896] | 4 of 8 years | 6 of 8 years | 0 of 3 | median 1.671% |
| XRPUSDT | **PASS** | VR=0.896, CI [0.864, 0.929] | VR=0.820, CI [0.769, 0.888] | 5 of 8 years | 7 of 8 years | 2 of 3 | median 1.924% |
| DOGEUSDT | **PASS** | VR=0.852, CI [0.795, 0.919] | VR=0.769, CI [0.699, 0.862] | 7 of 8 years | 8 of 8 years | 3 of 3 | median 2.174% |
| LINKUSDT | **PASS** | VR=0.874, CI [0.845, 0.908] | VR=0.794, CI [0.749, 0.845] | 7 of 8 years | 7 of 8 years | 2 of 3 | median 2.596% |
| AVAXUSDT | fail | VR=0.803, CI [0.652, 0.930] | VR=0.735, CI [0.595, 0.877] | 3 of 7 years | 4 of 7 years | 0 of 3 | median 2.673% |

With 8 coins tested, expect occasional false passes from chance; the majority-of-years rule limits this, and G1 (replay with fees) is the real test.

## Overall (all coins together)

**FAIL.** 4 of 40 checks fail, e.g. ETHUSDT 1h: VR<1 with CI excluding 1 in a majority of years.

| Check | Result | Detail |
|---|---|---|
| BTCUSDT pooled 1h: VR<1 and CI excludes 1 | pass | VR=0.899, CI [0.877, 0.921] |
| BTCUSDT 1h: VR<1 with CI excluding 1 in a majority of years | pass | 8 of 8 years |
| BTCUSDT pooled 4h: VR<1 and CI excludes 1 | pass | VR=0.859, CI [0.825, 0.896] |
| BTCUSDT 4h: VR<1 with CI excluding 1 in a majority of years | pass | 6 of 8 years |
| BTCUSDT median 4h range > 0.375% | pass | median 1.349% |
| ETHUSDT pooled 1h: VR<1 and CI excludes 1 | pass | VR=0.923, CI [0.894, 0.949] |
| ETHUSDT 1h: VR<1 with CI excluding 1 in a majority of years | fail | 4 of 8 years |
| ETHUSDT pooled 4h: VR<1 and CI excludes 1 | pass | VR=0.889, CI [0.850, 0.927] |
| ETHUSDT 4h: VR<1 with CI excluding 1 in a majority of years | fail | 3 of 8 years |
| ETHUSDT median 4h range > 0.375% | pass | median 1.828% |
| SOLUSDT pooled 1h: VR<1 and CI excludes 1 | pass | VR=0.914, CI [0.892, 0.938] |
| SOLUSDT 1h: VR<1 with CI excluding 1 in a majority of years | pass | 5 of 7 years |
| SOLUSDT pooled 4h: VR<1 and CI excludes 1 | pass | VR=0.858, CI [0.828, 0.897] |
| SOLUSDT 4h: VR<1 with CI excluding 1 in a majority of years | pass | 6 of 7 years |
| SOLUSDT median 4h range > 0.375% | pass | median 2.662% |
| BNBUSDT pooled 1h: VR<1 and CI excludes 1 | pass | VR=0.909, CI [0.870, 0.944] |
| BNBUSDT 1h: VR<1 with CI excluding 1 in a majority of years | fail | 4 of 8 years |
| BNBUSDT pooled 4h: VR<1 and CI excludes 1 | pass | VR=0.851, CI [0.807, 0.896] |
| BNBUSDT 4h: VR<1 with CI excluding 1 in a majority of years | pass | 6 of 8 years |
| BNBUSDT median 4h range > 0.375% | pass | median 1.671% |
| XRPUSDT pooled 1h: VR<1 and CI excludes 1 | pass | VR=0.896, CI [0.864, 0.929] |
| XRPUSDT 1h: VR<1 with CI excluding 1 in a majority of years | pass | 5 of 8 years |
| XRPUSDT pooled 4h: VR<1 and CI excludes 1 | pass | VR=0.820, CI [0.769, 0.888] |
| XRPUSDT 4h: VR<1 with CI excluding 1 in a majority of years | pass | 7 of 8 years |
| XRPUSDT median 4h range > 0.375% | pass | median 1.924% |
| DOGEUSDT pooled 1h: VR<1 and CI excludes 1 | pass | VR=0.852, CI [0.795, 0.919] |
| DOGEUSDT 1h: VR<1 with CI excluding 1 in a majority of years | pass | 7 of 8 years |
| DOGEUSDT pooled 4h: VR<1 and CI excludes 1 | pass | VR=0.769, CI [0.699, 0.862] |
| DOGEUSDT 4h: VR<1 with CI excluding 1 in a majority of years | pass | 8 of 8 years |
| DOGEUSDT median 4h range > 0.375% | pass | median 2.174% |
| LINKUSDT pooled 1h: VR<1 and CI excludes 1 | pass | VR=0.874, CI [0.845, 0.908] |
| LINKUSDT 1h: VR<1 with CI excluding 1 in a majority of years | pass | 7 of 8 years |
| LINKUSDT pooled 4h: VR<1 and CI excludes 1 | pass | VR=0.794, CI [0.749, 0.845] |
| LINKUSDT 4h: VR<1 with CI excluding 1 in a majority of years | pass | 7 of 8 years |
| LINKUSDT median 4h range > 0.375% | pass | median 2.596% |
| AVAXUSDT pooled 1h: VR<1 and CI excludes 1 | pass | VR=0.803, CI [0.652, 0.930] |
| AVAXUSDT 1h: VR<1 with CI excluding 1 in a majority of years | fail | 3 of 7 years |
| AVAXUSDT pooled 4h: VR<1 and CI excludes 1 | pass | VR=0.735, CI [0.595, 0.877] |
| AVAXUSDT 4h: VR<1 with CI excluding 1 in a majority of years | pass | 4 of 7 years |
| AVAXUSDT median 4h range > 0.375% | pass | median 2.673% |

Rule applied literally: PASS only if VR<1 with the CI excluding 1 at 1h and 4h for every symbol, in a majority of years, and the median 4h range exceeds 0.375%.

## BTCUSDT: variance ratio, pooled (5m base)

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.949 | -4.53 | 0.000 | [0.934, 0.967] | yes (<1) |
| 30m | 6 | 0.928 | -4.09 | 0.000 | [0.912, 0.947] | yes (<1) |
| 1h | 12 | 0.899 | -4.16 | 0.000 | [0.877, 0.921] | yes (<1) |
| 2h | 24 | 0.877 | -4.00 | 0.000 | [0.848, 0.907] | yes (<1) |
| 4h | 48 | 0.859 | -3.74 | 0.000 | [0.825, 0.896] | yes (<1) |
| 8h | 96 | 0.837 | -3.56 | 0.000 | [0.793, 0.892] | yes (<1) |
| 1d | 288 | 0.868 | -2.09 | 0.037 | [0.790, 0.930] | yes (<1) |

1m base, labelled contaminated by bid-ask bounce (kline closes are trade prices, so VR<1 here is partly bounce, which a maker grid cannot harvest):

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 5m (1m base) | 5 | 0.981 | -2.23 | 0.026 | [0.974, 0.989] | yes (<1) |

## ETHUSDT: variance ratio, pooled (5m base)

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.958 | -3.37 | 0.001 | [0.931, 0.979] | yes (<1) |
| 30m | 6 | 0.948 | -2.65 | 0.008 | [0.924, 0.971] | yes (<1) |
| 1h | 12 | 0.923 | -2.93 | 0.003 | [0.894, 0.949] | yes (<1) |
| 2h | 24 | 0.905 | -2.91 | 0.004 | [0.872, 0.936] | yes (<1) |
| 4h | 48 | 0.889 | -2.84 | 0.005 | [0.850, 0.927] | yes (<1) |
| 8h | 96 | 0.868 | -2.86 | 0.004 | [0.821, 0.922] | yes (<1) |
| 1d | 288 | 0.905 | -1.55 | 0.121 | [0.830, 0.961] | yes (<1) |

1m base, labelled contaminated by bid-ask bounce (kline closes are trade prices, so VR<1 here is partly bounce, which a maker grid cannot harvest):

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 5m (1m base) | 5 | 0.987 | -1.53 | 0.125 | [0.976, 0.998] | yes (<1) |

## SOLUSDT: variance ratio, pooled (5m base)

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.966 | -4.16 | 0.000 | [0.950, 0.980] | yes (<1) |
| 30m | 6 | 0.948 | -4.07 | 0.000 | [0.927, 0.968] | yes (<1) |
| 1h | 12 | 0.914 | -4.91 | 0.000 | [0.892, 0.938] | yes (<1) |
| 2h | 24 | 0.886 | -5.00 | 0.000 | [0.857, 0.916] | yes (<1) |
| 4h | 48 | 0.858 | -4.87 | 0.000 | [0.828, 0.897] | yes (<1) |
| 8h | 96 | 0.831 | -4.56 | 0.000 | [0.798, 0.878] | yes (<1) |
| 1d | 288 | 0.835 | -2.99 | 0.003 | [0.772, 0.886] | yes (<1) |

1m base, labelled contaminated by bid-ask bounce (kline closes are trade prices, so VR<1 here is partly bounce, which a maker grid cannot harvest):

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 5m (1m base) | 5 | 0.967 | -5.15 | 0.000 | [0.957, 0.980] | yes (<1) |

## BNBUSDT: variance ratio, pooled (5m base)

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.961 | -2.23 | 0.026 | [0.931, 0.987] | yes (<1) |
| 30m | 6 | 0.950 | -1.96 | 0.050 | [0.916, 0.984] | yes (<1) |
| 1h | 12 | 0.909 | -2.78 | 0.005 | [0.870, 0.944] | yes (<1) |
| 2h | 24 | 0.874 | -3.18 | 0.001 | [0.836, 0.912] | yes (<1) |
| 4h | 48 | 0.851 | -3.20 | 0.001 | [0.807, 0.896] | yes (<1) |
| 8h | 96 | 0.828 | -3.16 | 0.002 | [0.778, 0.883] | yes (<1) |
| 1d | 288 | 0.881 | -1.65 | 0.099 | [0.780, 0.948] | yes (<1) |

1m base, labelled contaminated by bid-ask bounce (kline closes are trade prices, so VR<1 here is partly bounce, which a maker grid cannot harvest):

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 5m (1m base) | 5 | 0.985 | -1.69 | 0.091 | [0.973, 0.998] | yes (<1) |

## XRPUSDT: variance ratio, pooled (5m base)

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.961 | -2.88 | 0.004 | [0.940, 0.981] | yes (<1) |
| 30m | 6 | 0.940 | -2.91 | 0.004 | [0.919, 0.963] | yes (<1) |
| 1h | 12 | 0.896 | -3.75 | 0.000 | [0.864, 0.929] | yes (<1) |
| 2h | 24 | 0.854 | -4.24 | 0.000 | [0.814, 0.896] | yes (<1) |
| 4h | 48 | 0.820 | -4.32 | 0.000 | [0.769, 0.888] | yes (<1) |
| 8h | 96 | 0.792 | -4.23 | 0.000 | [0.739, 0.870] | yes (<1) |
| 1d | 288 | 0.812 | -2.87 | 0.004 | [0.736, 0.890] | yes (<1) |

1m base, labelled contaminated by bid-ask bounce (kline closes are trade prices, so VR<1 here is partly bounce, which a maker grid cannot harvest):

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 5m (1m base) | 5 | 0.989 | -0.77 | 0.442 | [0.975, 1.001] | no |

## DOGEUSDT: variance ratio, pooled (5m base)

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.922 | -3.62 | 0.000 | [0.898, 0.946] | yes (<1) |
| 30m | 6 | 0.892 | -3.47 | 0.001 | [0.859, 0.929] | yes (<1) |
| 1h | 12 | 0.852 | -3.65 | 0.000 | [0.795, 0.919] | yes (<1) |
| 2h | 24 | 0.809 | -3.78 | 0.000 | [0.747, 0.892] | yes (<1) |
| 4h | 48 | 0.769 | -3.73 | 0.000 | [0.699, 0.862] | yes (<1) |
| 8h | 96 | 0.750 | -3.24 | 0.001 | [0.661, 0.880] | yes (<1) |
| 1d | 288 | 0.879 | -1.07 | 0.283 | [0.666, 1.077] | no |

1m base, labelled contaminated by bid-ask bounce (kline closes are trade prices, so VR<1 here is partly bounce, which a maker grid cannot harvest):

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 5m (1m base) | 5 | 0.947 | -3.84 | 0.000 | [0.914, 0.980] | yes (<1) |

## LINKUSDT: variance ratio, pooled (5m base)

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.950 | -4.59 | 0.000 | [0.938, 0.963] | yes (<1) |
| 30m | 6 | 0.918 | -4.77 | 0.000 | [0.904, 0.935] | yes (<1) |
| 1h | 12 | 0.874 | -5.06 | 0.000 | [0.845, 0.908] | yes (<1) |
| 2h | 24 | 0.836 | -5.20 | 0.000 | [0.796, 0.882] | yes (<1) |
| 4h | 48 | 0.794 | -5.48 | 0.000 | [0.749, 0.845] | yes (<1) |
| 8h | 96 | 0.769 | -5.26 | 0.000 | [0.724, 0.828] | yes (<1) |
| 1d | 288 | 0.760 | -4.17 | 0.000 | [0.704, 0.829] | yes (<1) |

1m base, labelled contaminated by bid-ask bounce (kline closes are trade prices, so VR<1 here is partly bounce, which a maker grid cannot harvest):

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 5m (1m base) | 5 | 0.947 | -4.84 | 0.000 | [0.920, 0.989] | yes (<1) |

## AVAXUSDT: variance ratio, pooled (5m base)

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 15m | 3 | 0.878 | -1.35 | 0.176 | [0.744, 0.989] | yes (<1) |
| 30m | 6 | 0.846 | -1.32 | 0.187 | [0.699, 0.968] | yes (<1) |
| 1h | 12 | 0.803 | -1.49 | 0.135 | [0.652, 0.930] | yes (<1) |
| 2h | 24 | 0.767 | -1.66 | 0.097 | [0.618, 0.900] | yes (<1) |
| 4h | 48 | 0.735 | -1.82 | 0.069 | [0.595, 0.877] | yes (<1) |
| 8h | 96 | 0.710 | -1.94 | 0.053 | [0.573, 0.858] | yes (<1) |
| 1d | 288 | 0.717 | -1.81 | 0.070 | [0.562, 0.878] | yes (<1) |

1m base, labelled contaminated by bid-ask bounce (kline closes are trade prices, so VR<1 here is partly bounce, which a maker grid cannot harvest):

| Horizon | q | VR(q) | z* | p | 95% boot CI | CI excl. 1 |
|---|---|---|---|---|---|---|
| 5m (1m base) | 5 | 0.939 | -0.89 | 0.371 | [0.879, 0.984] | yes (<1) |

## Stability: per calendar year (VR, 95% CI)

| Symbol | Year | 1h VR | 1h CI | 4h VR | 4h CI | DFA Hurst |
|---|---|---|---|---|---|---|
| BTCUSDT | 2019 | 0.937 | [0.899, 0.978] | 0.898 | [0.828, 0.975] | 0.492 |
| BTCUSDT | 2020 | 0.875 | [0.820, 0.932] | 0.795 | [0.703, 0.911] | 0.460 |
| BTCUSDT | 2021 | 0.840 | [0.788, 0.898] | 0.797 | [0.724, 0.875] | 0.473 |
| BTCUSDT | 2022 | 0.941 | [0.910, 0.973] | 0.914 | [0.852, 0.977] | 0.497 |
| BTCUSDT | 2023 | 0.904 | [0.833, 0.978] | 0.914 | [0.806, 1.024] | 0.488 |
| BTCUSDT | 2024 | 0.958 | [0.922, 0.998] | 0.925 | [0.871, 0.990] | 0.495 |
| BTCUSDT | 2025 | 0.934 | [0.877, 0.995] | 0.927 | [0.831, 1.019] | 0.494 |
| BTCUSDT | 2026 | 0.953 | [0.913, 0.996] | 0.910 | [0.845, 0.988] | 0.486 |
| ETHUSDT | 2019 | 0.942 | [0.903, 0.984] | 0.894 | [0.827, 0.972] | 0.492 |
| ETHUSDT | 2020 | 0.902 | [0.848, 0.957] | 0.834 | [0.749, 0.929] | 0.466 |
| ETHUSDT | 2021 | 0.847 | [0.774, 0.917] | 0.795 | [0.708, 0.891] | 0.475 |
| ETHUSDT | 2022 | 0.978 | [0.946, 1.011] | 0.964 | [0.909, 1.018] | 0.505 |
| ETHUSDT | 2023 | 0.927 | [0.866, 0.985] | 0.926 | [0.822, 1.008] | 0.491 |
| ETHUSDT | 2024 | 0.998 | [0.950, 1.045] | 0.981 | [0.895, 1.087] | 0.503 |
| ETHUSDT | 2025 | 0.980 | [0.938, 1.027] | 1.017 | [0.938, 1.091] | 0.506 |
| ETHUSDT | 2026 | 0.984 | [0.934, 1.039] | 0.946 | [0.870, 1.030] | 0.493 |
| SOLUSDT | 2020 | 0.951 | [0.896, 1.022] | 0.803 | [0.738, 0.898] | 0.464 |
| SOLUSDT | 2021 | 0.916 | [0.869, 0.963] | 0.869 | [0.788, 0.949] | 0.490 |
| SOLUSDT | 2022 | 0.887 | [0.843, 0.935] | 0.858 | [0.805, 0.919] | 0.488 |
| SOLUSDT | 2023 | 0.875 | [0.812, 0.925] | 0.850 | [0.766, 0.920] | 0.481 |
| SOLUSDT | 2024 | 0.900 | [0.846, 0.949] | 0.824 | [0.765, 0.889] | 0.484 |
| SOLUSDT | 2025 | 0.956 | [0.911, 1.008] | 0.945 | [0.863, 1.035] | 0.492 |
| SOLUSDT | 2026 | 0.933 | [0.901, 0.968] | 0.869 | [0.793, 0.949] | 0.475 |
| BNBUSDT | 2019 | 0.921 | [0.885, 0.952] | 0.829 | [0.778, 0.890] | 0.478 |
| BNBUSDT | 2020 | 0.916 | [0.864, 0.989] | 0.821 | [0.735, 0.923] | 0.453 |
| BNBUSDT | 2021 | 0.876 | [0.839, 0.919] | 0.834 | [0.777, 0.904] | 0.475 |
| BNBUSDT | 2022 | 1.028 | [0.965, 1.089] | 1.021 | [0.958, 1.106] | 0.515 |
| BNBUSDT | 2023 | 0.925 | [0.877, 0.972] | 0.902 | [0.836, 0.963] | 0.485 |
| BNBUSDT | 2024 | 0.970 | [0.932, 1.014] | 0.908 | [0.836, 1.000] | 0.496 |
| BNBUSDT | 2025 | 0.770 | [0.555, 1.039] | 0.715 | [0.469, 1.052] | 0.464 |
| BNBUSDT | 2026 | 0.962 | [0.929, 1.003] | 0.912 | [0.847, 0.984] | 0.483 |
| XRPUSDT | 2019 | 0.905 | [0.859, 0.949] | 0.803 | [0.740, 0.875] | 0.475 |
| XRPUSDT | 2020 | 0.933 | [0.871, 1.005] | 0.822 | [0.760, 0.914] | 0.468 |
| XRPUSDT | 2021 | 0.847 | [0.782, 0.914] | 0.752 | [0.665, 0.869] | 0.459 |
| XRPUSDT | 2022 | 0.933 | [0.889, 0.971] | 0.893 | [0.824, 0.964] | 0.488 |
| XRPUSDT | 2023 | 0.892 | [0.699, 1.099] | 0.952 | [0.593, 1.407] | 0.508 |
| XRPUSDT | 2024 | 0.927 | [0.867, 1.008] | 0.855 | [0.772, 0.965] | 0.487 |
| XRPUSDT | 2025 | 0.913 | [0.852, 0.985] | 0.830 | [0.703, 0.973] | 0.457 |
| XRPUSDT | 2026 | 0.947 | [0.902, 0.997] | 0.907 | [0.819, 0.997] | 0.487 |
| DOGEUSDT | 2019 | 0.566 | [0.540, 0.643] | 0.453 | [0.427, 0.573] | 0.415 |
| DOGEUSDT | 2020 | 0.733 | [0.646, 0.831] | 0.627 | [0.528, 0.752] | 0.432 |
| DOGEUSDT | 2021 | 0.879 | [0.787, 0.982] | 0.790 | [0.648, 0.936] | 0.449 |
| DOGEUSDT | 2022 | 0.903 | [0.866, 0.942] | 0.846 | [0.775, 0.933] | 0.477 |
| DOGEUSDT | 2023 | 0.909 | [0.818, 1.030] | 0.827 | [0.740, 0.948] | 0.481 |
| DOGEUSDT | 2024 | 0.925 | [0.885, 0.976] | 0.866 | [0.811, 0.935] | 0.488 |
| DOGEUSDT | 2025 | 0.761 | [0.585, 0.993] | 0.682 | [0.457, 0.992] | 0.438 |
| DOGEUSDT | 2026 | 0.937 | [0.903, 0.972] | 0.891 | [0.816, 0.968] | 0.483 |
| LINKUSDT | 2019 | 0.831 | [0.755, 0.939] | 0.719 | [0.621, 0.867] | 0.471 |
| LINKUSDT | 2020 | 0.856 | [0.828, 0.892] | 0.734 | [0.685, 0.800] | 0.451 |
| LINKUSDT | 2021 | 0.878 | [0.834, 0.930] | 0.809 | [0.758, 0.877] | 0.473 |
| LINKUSDT | 2022 | 0.940 | [0.906, 0.972] | 0.893 | [0.849, 0.941] | 0.497 |
| LINKUSDT | 2023 | 0.897 | [0.854, 0.945] | 0.875 | [0.797, 0.967] | 0.485 |
| LINKUSDT | 2024 | 0.957 | [0.905, 1.018] | 0.922 | [0.847, 1.005] | 0.501 |
| LINKUSDT | 2025 | 0.828 | [0.716, 0.958] | 0.758 | [0.571, 0.983] | 0.441 |
| LINKUSDT | 2026 | 0.912 | [0.875, 0.950] | 0.855 | [0.793, 0.931] | 0.482 |
| AVAXUSDT | 2020 | 0.847 | [0.849, 1.002] | 0.643 | [0.714, 0.972] | 0.457 |
| AVAXUSDT | 2021 | 0.884 | [0.849, 0.920] | 0.784 | [0.747, 0.844] | 0.463 |
| AVAXUSDT | 2022 | 0.928 | [0.880, 0.979] | 0.883 | [0.790, 0.971] | 0.489 |
| AVAXUSDT | 2023 | 0.901 | [0.853, 0.950] | 0.894 | [0.821, 0.976] | 0.493 |
| AVAXUSDT | 2024 | 0.970 | [0.929, 1.016] | 0.938 | [0.876, 1.002] | 0.500 |
| AVAXUSDT | 2025 | 0.417 | [0.231, 1.024] | 0.387 | [0.187, 1.054] | 0.423 |
| AVAXUSDT | 2026 | 0.963 | [0.919, 1.014] | 0.923 | [0.844, 1.014] | 0.494 |

## DFA Hurst exponent (5m returns, 1h-1d scales)

| Symbol | Pooled H |
|---|---|
| BTCUSDT | 0.482 |
| ETHUSDT | 0.487 |
| SOLUSDT | 0.482 |
| BNBUSDT | 0.476 |
| XRPUSDT | 0.472 |
| DOGEUSDT | 0.460 |
| LINKUSDT | 0.471 |
| AVAXUSDT | 0.463 |

H < 0.5 means mean reversion, 0.5 a random walk, > 0.5 trending.

## Range vs fees: (high - low) / open

Maker round trip 0.15% (BNB); 2.5x threshold 0.375%.

| Symbol | Bar | Median | 25th pct | Median vs 0.375% |
|---|---|---|---|---|
| BTCUSDT | 1h | 0.633% | 0.396% | above |
| BTCUSDT | 4h | 1.349% | 0.846% | above |
| ETHUSDT | 1h | 0.849% | 0.540% | above |
| ETHUSDT | 4h | 1.828% | 1.181% | above |
| SOLUSDT | 1h | 1.256% | 0.814% | above |
| SOLUSDT | 4h | 2.662% | 1.726% | above |
| BNBUSDT | 1h | 0.797% | 0.497% | above |
| BNBUSDT | 4h | 1.671% | 1.059% | above |
| XRPUSDT | 1h | 0.910% | 0.588% | above |
| XRPUSDT | 4h | 1.924% | 1.255% | above |
| DOGEUSDT | 1h | 1.037% | 0.669% | above |
| DOGEUSDT | 4h | 2.174% | 1.408% | above |
| LINKUSDT | 1h | 1.252% | 0.828% | above |
| LINKUSDT | 4h | 2.596% | 1.743% | above |
| AVAXUSDT | 1h | 1.263% | 0.813% | above |
| AVAXUSDT | 4h | 2.673% | 1.708% | above |
