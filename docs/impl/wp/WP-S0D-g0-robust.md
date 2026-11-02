# WP-S0D G0-robust rerun
Model: sonnet · Depends: none · Worktree: no (only touch `research/g0_variance_ratio.py`, `tests/research/`, `docs/research/G0_RESULTS.md`) · Budget: 60k · Commit: NO

## Why
See `docs/research/G0_DECISION.md`. The final G1 coin list is the coins that pass **both** the baseline G0 and this robust version.

## Changes to `research/g0_variance_ratio.py`
1. **Listing exclusion:** drop each symbol's first 30 days of data (`--exclude-listing-days 30`, default 30).
2. **Robust variant** (computed in the same run; report both): winsorise 5m log returns at ±5σ, where σ is a robust per-calendar-month scale (1.4826 × MAD), and drop 2025-10-10 00:00 → 2025-10-12 00:00 UTC (`--exclude-window`, repeatable).
3. **Multiplicity:** for pooled checks, require Holm-adjusted p < 0.05 for the robust z*, with the family = symbols × {1h, 4h} in this run, **and** a CI excluding 1. Yearly checks require the CI to exclude 1 **and** unadjusted z* p < 0.05 for that year.
4. **CI fix:** the stationary-bootstrap percentile interval drifts toward 1 on short, heavy-tailed samples (AVAX 2020 has its point estimate outside its CI). Re-centre it on the point estimate: CI = VR̂ + (q_lo − median(draws), q_hi − median(draws)). Document this in the report's method line.
5. **Report:** add a per-coin table with columns *Baseline*, *Robust* and *Final (both)*. Keep the existing sections for the baseline. Add the robust pooled 1h/4h and years counts.

## Tests first (`tests/research/test_vr.py`, extend)
- Winsorising caps an injected 20σ spike, and VR moves back toward the no-spike value.
- An excluded window removes exactly those bars.
- Holm on a known p-vector matches a hand computation.
- After re-centring, the CI always contains the point estimate.

## Run
`.venv/bin/python -m research.g0_variance_ratio --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT DOGEUSDT LINKUSDT AVAXUSDT --out docs/research/G0_RESULTS.md` (about 30 minutes; run it in the background).

## Done when
Tests and ruff are green, and the report is regenerated with the Final column.

## Reply (≤120 words)
status · test line · Final per-coin list · any verdict that flipped vs baseline and why
