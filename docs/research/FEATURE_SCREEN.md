# Feature screen: does external data add predictive power?

*Pre-registered (`research/feature_screen_prereg.json`), run once, no tuning. Daily UTC bars, BTC and an equal-weight basket of 6 coins. In-sample 2019-2023, out-of-sample (OOS) 2024 to 15 Nov 2026.*

## Verdict

| Group | Verdict | (a) beats baseline A | (b) survives FDR with same sign |
|---|---|---|---|
| B. Binance derivatives | **contributes** | yes (single/ret_7d +0.005; single/crash_7d +0.064; basket/ret_7d +0.007) | yes |
| C. Macro (FRED) | **does not contribute** | yes (basket/ret_7d +0.005; basket/crash_7d +0.048) | no |
| D. Crypto-native | **contributes** | yes (single/crash_7d +0.089) | yes |

Rule (literal): a group contributes only if it lifts crash-7d OOS AUC by 0.03 or more, or a return target's OOS R-squared by 0.5 percentage points or more, over price-only baseline A, AND at least one of its features survives Benjamini-Hochberg FDR (10%) OOS with the same sign as in-sample. Either target or either universe counts for (a), which is generous: that is 6 chances per group.

## Does adding the group improve out-of-sample prediction?

Walk-forward, monthly refit, expanding window, standardised on training data only. R-squared is vs the training mean. Delta = group model minus baseline A on the same rows (B and all use the common sample of rows where every derivative feature exists, from 2022-05-27).

| Universe | Target | + Group | Metric | Baseline A | With group | Delta |
|---|---|---|---|---|---|---|
| single | ret_1d | B | R2 | -0.000 | -0.001 | **-0.001** |
| single | ret_1d | C | R2 | 0.001 | 0.001 | **-0.000** |
| single | ret_1d | D | R2 | 0.001 | 0.001 | **-0.001** |
| single | ret_1d | all | R2 | -0.000 | -0.001 | **-0.000** |
| single | ret_7d | B | R2 | -0.008 | -0.003 | **+0.005** |
| single | ret_7d | C | R2 | 0.003 | 0.006 | **+0.003** |
| single | ret_7d | D | R2 | 0.003 | 0.002 | **-0.000** |
| single | ret_7d | all | R2 | -0.008 | -0.015 | **-0.007** |
| single | crash_7d | B | AUC | 0.460 | 0.524 | **+0.064** |
| single | crash_7d | C | AUC | 0.520 | 0.514 | **-0.006** |
| single | crash_7d | D | AUC | 0.520 | 0.608 | **+0.089** |
| single | crash_7d | all | AUC | 0.460 | 0.518 | **+0.058** |
| basket | ret_1d | B | R2 | -0.003 | -0.004 | **-0.001** |
| basket | ret_1d | C | R2 | -0.004 | 0.000 | **+0.004** |
| basket | ret_1d | D | R2 | -0.004 | -0.005 | **-0.001** |
| basket | ret_1d | all | R2 | -0.003 | -0.004 | **-0.001** |
| basket | ret_7d | B | R2 | -0.006 | 0.001 | **+0.007** |
| basket | ret_7d | C | R2 | 0.004 | 0.009 | **+0.005** |
| basket | ret_7d | D | R2 | 0.004 | 0.002 | **-0.002** |
| basket | ret_7d | all | R2 | -0.006 | -0.007 | **-0.001** |
| basket | crash_7d | B | AUC | 0.523 | 0.524 | **+0.002** |
| basket | crash_7d | C | AUC | 0.544 | 0.593 | **+0.048** |
| basket | crash_7d | D | AUC | 0.544 | 0.514 | **-0.030** |
| basket | crash_7d | all | AUC | 0.523 | 0.526 | **+0.004** |

OOS rows up to 1049; crash-7d positives about 159 in a handful of clustered episodes, so AUC is noisy. Brier scores are in the CSV.

## Features that survive FDR OOS (9 of 208 tests)

| Universe | Feature | Target | IC in-sample | IC OOS | t OOS (HAC) | Same sign |
|---|---|---|---|---|---|---|
| basket | toptrader_ls_ratio (B) | vol_7d | -0.096 | +0.316 | 5.0 | NO |
| single | rv_30d (A) | vol_7d | +0.396 | +0.302 | 4.8 | yes |
| basket | global_ls_ratio (B) | vol_7d | +0.051 | +0.271 | 4.7 | yes |
| basket | rv_30d (A) | vol_7d | +0.379 | +0.298 | 4.6 | yes |
| single | rv_7d (A) | vol_7d | +0.369 | +0.264 | 4.5 | yes |
| basket | rv_7d (A) | vol_7d | +0.331 | +0.233 | 4.0 | yes |
| basket | stablecoin_supply_chg_30d (D) | vol_7d | +0.237 | +0.258 | 3.8 | yes |
| basket | toptrader_ls_ratio (B) | crash_7d | -0.106 | +0.184 | 3.8 | NO |
| single | global_ls_ratio (B) | vol_7d | +0.074 | +0.187 | 3.0 | yes |

All tests are in `docs/research/feature_screen_all_tests.csv`.

## Event study: BTC's 8 largest non-overlapping 7-day drawdowns

Events (base day, worst close within 7 days): 2019-07-09 (-25%), 2020-03-06 (-47%), 2021-05-14 (-26%), 2022-05-04 (-27%), 2022-06-11 (-33%), 2022-11-05 (-25%), 2026-01-29 (-26%), 2026-11-08 (-14%).

Average z-score (vs trailing 365 days) of each feature over the 7 days before the window opens. Pctile = where that sits in the distribution of 7-day average z on all other days (50% = nothing unusual). Only features with |avg z| of 0.4 or more are shown; all 26 are in the CSV.

| Feature | Events | Avg z before | Other days avg z (sd) | Pctile | Events z>0 |
|---|---|---|---|---|---|
| dist_sma200 | 7 | -0.41 | +0.06 (1.27) | 40% | 2/7 |
| toptrader_ls_ratio | 2 | +0.64 | +0.40 (1.22) | 54% | 2/2 |
| global_ls_ratio | 5 | +0.56 | -0.07 (0.84) | 70% | 3/5 |
| taker_buysell_ratio | 3 | -0.45 | +0.03 (0.41) | 13% | 0/3 |
| vix_level | 8 | +1.02 | -0.10 (1.15) | 65% | 5/8 |
| vix_chg_5d | 8 | +0.50 | +0.02 (0.78) | 58% | 4/8 |
| nasdaq_ret_5d | 8 | -0.77 | -0.02 (0.83) | 32% | 3/8 |
| stablecoin_supply_chg_7d | 8 | -0.69 | -0.07 (0.90) | 37% | 2/8 |
| stablecoin_supply_chg_30d | 8 | -0.94 | -0.03 (1.08) | 29% | 2/8 |

First date with every feature of the group (BTC): {'A': '2019-07-19', 'B': '2022-05-27', 'C': '2019-01-29', 'D': '2019-01-31'}.

## What to build

Read the verdicts with care: criterion (b) was met on volatility targets, while criterion (a) was met on crash and 7-day return targets, so the two halves of the rule are not backed by the same feature. The best OOS volatility predictors are price-only (rv_30d, rv_7d); external data adds little on top of them. Baseline A's crash AUC is at or below 0.5 on the B sample, so part of the AUC gains is a weak baseline being beaten, not a strong model. None of this is a buy/sell signal.

- **Stablecoin supply growth (30d), group D:** predicts next-week volatility OOS and fell before most large drawdowns. Use as a risk input: widen grid spacing or cut position size when supply growth is weak. Not a directional entry.
- **Long/short account ratio (global_ls_ratio), group B:** predicts volatility OOS with a stable sign. Candidate input to a size scaler or a risk-off gate. The top-trader ratio flipped sign between in-sample and OOS, so do not build on it.
- **Macro (FRED):** nothing to build; no feature survived FDR.
- Before building: paper-test each as a gate in the grid simulator against a gate on rv_30d alone; adopt only if it adds something rv_30d does not.

## Limitations

- ETF flows, on-chain exchange flows and news sentiment are unavailable (no free point-in-time source); this screen says nothing about them.
- The OOS period was seen by the team in other work, so it is not a clean holdout.
- Only a handful of independent crash episodes exist, and 7-day targets overlap; effective sample is small even with HAC errors.
- Derivatives (B) start 2021-12 and need every column present, so the B comparison sample starts later and its in-sample IC rests on under 2 years.
- Fear and Greed and stablecoin supply are lagged one extra day, an assumption the prereg does not spell out.
