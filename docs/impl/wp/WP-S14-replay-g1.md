# WP-S1.4 Replay engine, metrics, and G1 runner
Model: sonnet · Depends: S1.2b, S1.3 · Worktree: no (only touch `vv/sim/replay.py`, `vv/sim/metrics.py`, `research/g1_run.py`, `tests/sim/test_replay.py`, `tests/sim/test_metrics.py`) · Budget: 110k · Commit: NO

## Goal
Run the **production** `vv.core.decide.decide()` against `vv.sim.fake_exchange.FakeExchange` on historical 1m data for a shared-wallet portfolio, and produce G1 metrics. The settings are fixed in `research/g1_preregistration.json`. **Read it and don't change it.**

## Navigate
`graphify explain "decide"`, `graphify explain "FakeExchange"`, `graphify explain "compute_features"`. Read the `## Result` notes in `docs/impl/wp/WP-S13-fake-exchange.md` and `WP-S12b-sizing-risk-orders.md`. They describe the loop order, IOC flatten, id reuse and churn.

## `vv/sim/replay.py`
`run_replay(bars: dict[str, DataFrame], filters, grid: GridParams, risk: RiskParams, start, end, decision_minutes, start_cash, fees, fill_haircut, seed) -> ReplayResult`
- **Features:** precompute once per symbol with `compute_features` over the **full** history (warm-up needs data before `start`). At decision time t, use the latest feature row whose bar has fully closed (< t). Don't recompute per step.
- **Loop:** for each t from start to end in steps of `decision_minutes`:
  1. `ex.advance_to(t)`
  2. Build a `Snapshot` from the exchange (markets, the portfolio via `ex.portfolio_state(peak, halt)`, features, `stale=()`).
  3. `decision = decide(...)`
  4. Persist `halt = decision.halt` and update `peak = max(peak, equity)`.
  5. Apply the intents: cancels first, then places.
  6. Catch `OrderRejected`, counting each reason (especially id reuse). Never crash on one.
- Record daily equity, exposure, fills, fees, rejections, halt events, and time in market.
- **Performance target:** one symbol-year at 15-minute decisions in under 60 s. Provide `run_many(jobs, processes=N)` using multiprocessing.

## `vv/sim/metrics.py`
From daily equity:
- total return, CAGR, annualised vol, Sharpe (365-day annualisation)
- **Sortino with downside deviation** = sqrt(mean(min(r, 0)²))
- max drawdown, and longest time under water in days
- round trips, fees paid, gross grid PnL (realised before fees)
- **net capture over fees** = gross realised / fees
- average exposure

Baselines over the same window:
- cash
- equal-weight buy-and-hold of the universe
- buy-and-hold scaled to the run's average exposure

`sharpe_ci(daily_returns, mean_block=10, reps=2000, seed)` uses a stationary block bootstrap.

## `research/g1_run.py`
CLI: `--symbols ... --prereg research/g1_preregistration.json --out docs/research/G1_RESULTS.md [--smoke]`
1. For each trial × split: a fresh portfolio (cash reset, halt cleared). Run them in parallel.
2. Selection exactly as pre-registered (in-sample splits only).
3. For the selected trial, run the stress variants (fees ×2, fill_haircut 0.7) on all splits.
4. The shadow control (ETH) runs as its **own** separate portfolio with the selected trial.
5. Apply `gate_g1` literally → PASS or FAIL per criterion.
6. **Report** (≤ 2 pages markdown):
   - a trials table
   - the selected trial per split vs baselines
   - stress results
   - the shadow-control result
   - the gate table, and a plain-language verdict
   - the trial count and the rejection counts
7. `--smoke`: run 2 symbols, 1 trial, January–February 2024 only, to finish in under 2 minutes.

## Tests first
- `test_metrics.py`:
  - Sortino uses downside deviation (hand-computed series).
  - max drawdown and time under water on a known curve.
  - the Sharpe CI is deterministic under a seed.
- `test_replay.py` (synthetic 2-symbol bars, a few days):
  - **No look-ahead:** the features used at t come from bars < t.
  - A latched halt stays latched for the rest of the run and the position goes to zero within a few cycles.
  - Exposure never exceeds `total_exposure_cap`·equity at any decision.
  - Rejections are counted, not raised.
  - Same seed → identical results.

## Run
After the tests pass, run `--smoke` and paste its gate table into the WP `## Result`. **Don't run the full G1.** The orchestrator runs it once the final coin list is settled.

## Done when
`.venv/bin/pytest -q tests/sim && .venv/bin/ruff check vv research/g1_run.py tests/sim`, `make test` still green, and the smoke run completes.

## Reply (≤120 words)
status · test line · smoke runtime · projected full-G1 runtime · anything surprising in the smoke run
