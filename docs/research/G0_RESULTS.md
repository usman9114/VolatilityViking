# G0 results: does ETH/BTC mean-revert at grid horizons?

- ETHUSDT: 2019-01-01 to 2026-10-31, 4,115,750 1m bars (814 empty 5m bins forward-filled).
- BTCUSDT: 2019-01-01 to 2026-10-31, 4,115,751 1m bars (814 empty 5m bins forward-filled).

Method: Lo-MacKinlay VR on log returns, robust z*, stationary block bootstrap (mean block 1 day, 500 reps, seed 20260101), percentile 95% CI. Year rows need at least 30 days of data.

## Verdict

**FAIL.** 2 of 10 checks fail, e.g. ETHUSDT 1h: VR<1 with CI excluding 1 in a majority of years.

| Check | Result | Detail |
|---|---|---|
| ETHUSDT pooled 1h: VR<1 and CI excludes 1 | pass | VR=0.923, CI [0.894, 0.949] |
| ETHUSDT 1h: VR<1 with CI excluding 1 in a majority of years | fail | 4 of 8 years |
| ETHUSDT pooled 4h: VR<1 and CI excludes 1 | pass | VR=0.889, CI [0.850, 0.927] |
| ETHUSDT 4h: VR<1 with CI excluding 1 in a majority of years | fail | 3 of 8 years |
| ETHUSDT median 4h range > 0.375% | pass | median 1.828% |
| BTCUSDT pooled 1h: VR<1 and CI excludes 1 | pass | VR=0.899, CI [0.877, 0.921] |
| BTCUSDT 1h: VR<1 with CI excluding 1 in a majority of years | pass | 8 of 8 years |
| BTCUSDT pooled 4h: VR<1 and CI excludes 1 | pass | VR=0.859, CI [0.825, 0.896] |
| BTCUSDT 4h: VR<1 with CI excluding 1 in a majority of years | pass | 6 of 8 years |
| BTCUSDT median 4h range > 0.375% | pass | median 1.349% |

Rule applied literally: PASS only if VR<1 with the CI excluding 1 at 1h and 4h for both symbols, in a majority of years, and the median 4h range exceeds 0.375%.

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

## Stability: per calendar year (VR, 95% CI)

| Symbol | Year | 1h VR | 1h CI | 4h VR | 4h CI | DFA Hurst |
|---|---|---|---|---|---|---|
| ETHUSDT | 2019 | 0.942 | [0.903, 0.984] | 0.894 | [0.827, 0.972] | 0.492 |
| ETHUSDT | 2020 | 0.902 | [0.848, 0.957] | 0.834 | [0.749, 0.929] | 0.466 |
| ETHUSDT | 2021 | 0.847 | [0.774, 0.917] | 0.795 | [0.708, 0.891] | 0.475 |
| ETHUSDT | 2022 | 0.978 | [0.946, 1.011] | 0.964 | [0.909, 1.018] | 0.505 |
| ETHUSDT | 2023 | 0.927 | [0.866, 0.985] | 0.926 | [0.822, 1.008] | 0.491 |
| ETHUSDT | 2024 | 0.998 | [0.950, 1.045] | 0.981 | [0.895, 1.087] | 0.503 |
| ETHUSDT | 2025 | 0.980 | [0.938, 1.027] | 1.017 | [0.938, 1.091] | 0.506 |
| ETHUSDT | 2026 | 0.984 | [0.934, 1.039] | 0.946 | [0.870, 1.030] | 0.493 |
| BTCUSDT | 2019 | 0.937 | [0.899, 0.978] | 0.898 | [0.828, 0.975] | 0.492 |
| BTCUSDT | 2020 | 0.875 | [0.820, 0.932] | 0.795 | [0.703, 0.911] | 0.460 |
| BTCUSDT | 2021 | 0.840 | [0.788, 0.898] | 0.797 | [0.724, 0.875] | 0.473 |
| BTCUSDT | 2022 | 0.941 | [0.910, 0.973] | 0.914 | [0.852, 0.977] | 0.497 |
| BTCUSDT | 2023 | 0.904 | [0.833, 0.978] | 0.914 | [0.806, 1.024] | 0.488 |
| BTCUSDT | 2024 | 0.958 | [0.922, 0.998] | 0.925 | [0.871, 0.990] | 0.495 |
| BTCUSDT | 2025 | 0.934 | [0.877, 0.995] | 0.927 | [0.831, 1.019] | 0.494 |
| BTCUSDT | 2026 | 0.953 | [0.913, 0.996] | 0.910 | [0.845, 0.988] | 0.486 |

## DFA Hurst exponent (5m returns, 1h-1d scales)

| Symbol | Pooled H |
|---|---|
| ETHUSDT | 0.487 |
| BTCUSDT | 0.482 |

H < 0.5 means mean reversion, 0.5 a random walk, > 0.5 trending.

## Range vs fees: (high - low) / open

Maker round trip 0.15% (BNB); 2.5x threshold 0.375%.

| Symbol | Bar | Median | 25th pct | Median vs 0.375% |
|---|---|---|---|---|
| ETHUSDT | 1h | 0.849% | 0.540% | above |
| ETHUSDT | 4h | 1.828% | 1.181% | above |
| BTCUSDT | 1h | 0.633% | 0.396% | above |
| BTCUSDT | 4h | 1.349% | 0.846% | above |
