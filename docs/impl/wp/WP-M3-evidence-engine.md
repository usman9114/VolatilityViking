# WP-M3 Evidence engine: hierarchical prior, posterior, ¼-Kelly weights, factor budgets, decay monitor
Model: sonnet · Depends: none · Worktree: no (only touch `vv/evidence/`, `tests/evidence/`, `research/signals/registry.yaml`, `research/signals/ledger.csv`) · Budget: 100k · Commit: NO

## Goal
Implement `docs/AI_SIGNAL_PLAN.md` §2 exactly as pure, tested functions, plus the signal registry and the trial ledger. Read the plan first.

## Build `vv/evidence/`
- `registry.py`: load and validate `research/signals/registry.yaml` with pydantic.
  - **Fields:** id, family, role (`directional` | `risk_off`), hypothesis, sign, horizon, data, prereg_path, prereg_sha256, status (`candidate|shadow|live|retired`), created, t1 {sr_oos_stress, se, years}, notes.
  - **The ledger** is an append-only CSV of every hypothesis ever tested (id, date, outcome). `registry.verify_prereg()` checks the hash of each prereg file.
- `posterior.py`
  - `hierarchical_prior(ledger_srs, ledger_ses) -> (mu_H, tau)`: an empirical-Bayes (method-of-moments) estimate of the selection prior across ALL tested hypotheses. Fall back to (0, 0.5) when there are fewer than 5.
  - `posterior(prior, t1: (sr, se) | None, forward_daily_returns) -> (mean, sd)`: a normal-normal update.
    - Forward SR se = sqrt((1 + 0.5·SR²)/T_years).
    - Annualise with √365.
    - **Overlapping horizons:** pass the effective-n factor explicitly.
  - `p_positive(mean, sd)`.
- `weights.py`
  - `kelly_weights(posteriors: dict[id,(mean,sd)], vols: dict[id,float], min_p=0.80, frac=0.25) -> dict[id,float]`: 0 below min_p; otherwise frac·mean/σ.
  - `factor_budget(weights, factor_of: dict[id,str], budget: dict[str,float]) -> dict`: scale within each factor so the factor's total weight stays within its budget.
  - `ramp_cap(base=0.05, precision_now, precision_at_launch) -> float`: the capital-at-risk multiplier. Grows with √(precision ratio) and is capped at 1.0.
  - `incremental_alpha(signal_pnl, book_pnl) -> (alpha_sr, se)`: OLS of the signal on the book, returning the residual SR.
- `decay.py`: `CusumMonitor(expected_daily_mean, sd, k=0.5, h=5)` with `.update(r) -> state` in {ok, halve, zero}. Halve on the first breach, zero on the second.
- `report.py`: a markdown table per signal (prior, T1, forward n, posterior mean/sd, P(SR>0), weight, decay state).

## Seed the registry
Seed it with the G2 variants **S1, S2, P1, P2** as `candidate`:
- **T1 values** come from `docs/research/G2_RESULTS.md` (stress OOS SR; se = sqrt((1+0.5·SR²)/years), with years = OOS length).
- **prereg** = `research/g2_preregistration.json` and its sha256.

Seed the ledger with every tested hypothesis so far, with its outcome:
- G1 v1/v2: 16 trials
- G2: 4
- the carry study
- the feature-screen groups B, C and D

Then print the report: with the default prior, P(SR>0) and the weight for S1 should come out close to the plan's worked example (posterior mean ~0.25, P ~0.73, weight 0).

## Tests first (`tests/evidence/`)
- The normal-normal posterior matches a hand computation.
- Kelly weight is 0 below min_p.
- The factor budget caps the sum within each factor.
- The ramp cap is monotone and ≤ 1.
- The CUSUM goes ok → halve → zero on a synthetic regime break and stays ok on a stationary series (false-alarm rate < 5% over 2 years, fixed seed).
- The hierarchical prior shrinks towards the ledger mean.
- A prereg hash mismatch raises.

## Done when
`.venv/bin/pytest -q tests/evidence && .venv/bin/ruff check vv/evidence tests/evidence`, with `make test` still green, and the seeded report printed in your `## Result`.

## Reply (≤120 words)
status · test line · the seeded posterior/weight table for S1/S2/P1/P2
