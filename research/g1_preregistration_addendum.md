# G1 pre-registration addendum (2026-11-02)

The v1 file `g1_preregistration.json` is unchanged. Its `_note` date should read 2026-11-02 (it says 11-03 by mistake).

**Changes before the full G1 run.** The parameters, trials, splits, selection rule and gate are all unchanged.
- **WP-S1.5, ladder hysteresis:** keep the resting flat ladder while its top level is within 0.5–1.5 steps below the bid. This removes about 110k cancel/replace cycles per symbol-year that a live account couldn't sustain.
- **WP-S1.5, unique order ids:** the decision minute goes into every client id. This removes the 4–8% of placements that were silently rejected as duplicates.

**Results seen before these changes (disclosed).**
- The S1.4 smoke run (BTC + SOL, Jan–Feb 2024, 1 trial) was a plumbing check.
- One exploratory run: 5 symbols, trial T01, 2022 only. It returned −6.3%, with gross realised −583 USDT against 52 USDT in fees and 5 halt events.

No parameter was chosen using these runs. The fixes address engineering defects identified from order counts, not from P&L.

**Gate interpretations fixed now** (S1.4 proposed these; adopted as-is):
- Net capture is pooled over all splits at base costs.
- Max drawdown is the worst split at base costs.
- Positive splits are judged at base costs.
- The out-of-sample Sharpe CI uses pooled out-of-sample daily returns at fees ×2.
- The stress run changes exchange fees only.
