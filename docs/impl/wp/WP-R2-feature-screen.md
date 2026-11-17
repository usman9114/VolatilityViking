# WP-R2 Feature screen: does external data add predictive power?
Model: sonnet · Depends: WP-R1 data (being fetched in parallel; write and test the code on synthetic data first) · Worktree: no (only touch `research/feature_screen.py`, `tests/research/test_feature_screen.py`, `docs/research/FEATURE_SCREEN.md`) · Budget: 120k · Commit: NO

## Goal
Implement `research/feature_screen_prereg.json` **exactly**. Read it and don't change it. This is the honest answer to "which external data would give the bot predictive power?".

## Inputs
- **Prices:** `data/klines/{SYM}_1m.parquet` (resample to daily UTC closes and volumes).
- **Recent prices:** 2026-11-01..15 aren't stored. Fetch them like `research/last_week.py` does, from data-api.binance.vision 1d klines.
- **External data:** `data/external/*.parquet`, schema in `docs/impl/wp/WP-R1-external-data.md`. Until those files exist, test on synthetic frames with the same schema.

## Build `research/feature_screen.py`
1. **Feature construction** per the prereg names, with the exact lagging rules. FRED daily series get +1 business day. WALCL and WTREGEN are weekly: available the day after release (Thursday for a Wednesday observation). `fed_liquidity` = WALCL − RRPONTSYD − WTREGEN, aligned with those lags; take its 4-week change. Derivative features come from BTCUSDT for the single target and the basket average for the basket target. Use trailing-only z-score windows.
2. **Targets** per the prereg, for BTC and the equal-weight basket.
3. **Univariate:** Spearman IC, in-sample and out-of-sample, with HAC t-stats (Newey-West, lags = horizon). Apply BH-FDR at 10% across all OOS tests.
4. **Incremental:**
   - Walk-forward monthly refit with an expanding window. Standardise using training data only.
   - Models: ridge (alpha chosen by inner time-series CV on training data only) for returns, logistic for `crash_7d`.
   - Compare baseline A vs A+B, A+C, A+D, and A+all.
   - Report OOS R², AUC and Brier. For B (derivatives), which only starts in 2021-12, run the comparison on the common sample.
5. **Event study:** the 8 largest 7-day BTC drawdowns since 2019 (non-overlapping), including 2026-11-09..15. Show each feature's average z-score in the 7 days before, against its distribution on all other days, as a table.
6. **Decision rule:** apply it literally per group → "contributes" or "does not contribute".

## Report (`docs/research/FEATURE_SCREEN.md`, ≤ 2.5 pages, plain language)
- The verdict per group first.
- The incremental table.
- The FDR-surviving features, with IS and OOS IC.
- The event study.
- A short "what to build" section. Recommend only features that passed, and say what each would do in the bot: e.g. a risk-off gate, position sizing, or grid widening. Not every predictor is a buy/sell signal.
- Limitations: ETF flows, on-chain data and news are unavailable; the OOS period is partly seen; few independent crash events.

## Tests first
- Lagging: a feature can never use a value dated on or after the target window start. Add a synthetic leak test that a deliberately leaked feature would fail.
- The BH-FDR procedure matches a hand-computed example.
- The walk-forward never trains on test-period rows.
- HAC t-stats are deterministic.

## Done when
Tests and ruff are green. When the WP-R1 files exist, run the real screen and write the report. If they aren't there yet, poll for them by checking `ls data/external/*.parquet` every few minutes, up to about 45 minutes.

## Reply (≤150 words)
status · test line · verdict per group · the 3 strongest OOS features with their IC

## Result
Status: done. 19 new tests (51 in tests/research) green, ruff clean. Real screen run on WP-R1 data; report in `docs/research/FEATURE_SCREEN.md`.
Extra outputs (not in the brief's file list): `docs/research/feature_screen_{all_tests,incremental,event_study}.csv`, needed for "report every test".
Choices where prereg was silent: F&G and stablecoins lagged +1 day; WALCL converted $m to $bn; logistic uses fixed C=1; vol_7d only in univariate; Nov funding left NaN (not ffilled); zero-OI rows treated as NaN; no sklearn/statsmodels (numpy/scipy implementations).
B common sample starts 2022-05-27, not 2021-12 (some metrics columns start later or have gaps).
Literal verdict: B and D "contribute", C does not. Weak: criterion (b) was met on vol_7d while (a) was met on crash/return targets; rv_30d (price-only) is the strongest OOS vol predictor.
