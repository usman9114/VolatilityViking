# G1 v1 decision (2026-11-03)

Inputs: [`G1_RESULTS.md`](G1_RESULTS.md), plus consultation with Sam (operator), Maya (quant) and the simulated Hayes lens (`docs/personas/arthur_hayes.md`).

## Verdict on record
**G1 v1: FAIL.** OOS Sharpe CI lower bound −0.62 (point 1.17), and 2022 at −0.3%. This stays on record, and nothing below changes it.

## What the advisors agreed
- **The test was weakly informative.** Average exposure was 0.4–1.6%, with 25 latched symbol halts, so the bot was mostly in cash. The OOS window (~2.8 years) gives SE(Sharpe) ≈ 0.6, so a true Sharpe of ~1.2 can't be told apart from 0. There is also weak evidence of decay: return per unit exposure ran 17.6, 1.0, 4.6, −2.2 for 2021, 2023, 2024 and 2025–26.
- **Possible fill artifact.** 19.4% in 2021 on 1.1% average exposure, with capture/fees ≈ step/fee, suggests P&L from sparse wick days. The ETH control's dose-response with VR argues against a plain bug.
- **No leverage.** Sharpe is scale-invariant, so the CI is unchanged. Kelly on a CI lower bound < 0 is zero. 5× scaling would breach the 15% drawdown limit and add liquidation and gap risk. (Hayes lens: he runs his own fund unlevered.)
- **No real money.** The benchmark is idle USDT yield (~3–5%), not 0%. Carry is ~1% net in 2025–26, roughly T-bill level.

## Decisions
1. **Run G1 v2 exactly as pre-registered** (`g1v2_preregistration.json`, frozen before v1 results). Trial count: 16.
2. **Diagnostics (not trials; they don't change any gate), WP-S1.7:**
   - Exposure-metric audit.
   - Fill-depth (pierce) sweep at 1 tick / 0.05% / 0.15% for the selected trial of each variant, with the top-10-day P&L share.
   - Coin-year P&L vs that year's VR (attribution).
   - A cash benchmark at 4% a year.
3. **Halt-latch redesign** (cooldown then re-arm, derived from production semantics and not from P&L) only as a timestamped amendment with v1 results disclosed. Any test of it is exploratory, and **2024+ OOS is now spent**, so its holdout must be paper trading.
4. **Kill rule:** if no v2 variant passes G1 with an OOS Sharpe CI lower bound > 0, stop grid development. Options then are carry (only if it beats T-bills after risk) or holding stablecoins/T-bills.
5. **Engineering continues regardless.** Paper-trade the hardened engine for the Ops gate (restarts, reconciliation, halts, alerts), with no edge claimed.
