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

## Result
Status: done (not committed). `pytest -q tests/sim`: 49 passed; `ruff check vv research/g1_run.py tests/sim` clean; `make test`: 222 passed, 1 xfailed (pre-existing). Files: `vv/sim/replay.py`, `vv/sim/metrics.py`, `research/g1_run.py`, `tests/sim/test_replay.py`, `tests/sim/test_metrics.py`. `g1_preregistration.json` untouched.

Smoke (`python -m research.g1_run --symbols BTCUSDT SOLUSDT --smoke --out <scratch>`): 8 s wall (4 procs). It is a plumbing check, not a G1 result; the 2022 criterion cannot pass without the 2022 split. Smoke gate table (Jan = in-sample, Feb 2024 = OOS, 1 trial):

| criterion | rule | value | result |
|---|---|---|---|
| Net capture over fees | >= 2.0 | 9.43 | PASS |
| OOS Sharpe CI lower bound | > 0 | 7.23 (CI 7.23..13.35, point 8.71) | PASS |
| Positive splits | majority and 2022 among them | 2/2; 2022 not in | FAIL |
| Max drawdown | <= 15% | 1.1% | PASS |

Runtime: one symbol-year at 15 min = about 6.6 s (5 symbols, 2022: 33 s; BTC 2023 alone about 20 s incl. exchange). Features: 12 s for 5 symbols, once. Projected full G1 (5 symbols, 10 trials + 2 stress + ETH shadow): about 0.8-1 CPU-h, about 15-20 min on 4 cores.

Design notes:
- `run_replay(..., *, features=None, record_trace=False, keep_fills=True)`; `precompute_features` shares features across runs. `run_many(jobs, processes, bars=, features=, filters=)` keeps the big inputs in a module store that fork()ed workers inherit (never pickled per job); results are keyed by `ReplayJob.key` and independent of process count.
- The exchange gets only `[start-60min, end)` bars. The run is synchronous (`_sync` drives the never-suspending exchange coroutines; no event loop).
- Daily equity = first decision of each UTC day plus a final point at `end`. `exposure_total_post` (holdings + resting buys after intents, over equity) is what the total-cap test checks.
- `gross_pnl` is realised before fees (fee-free VWAP cost); `round_trips` = sells that flatten a position.
- Gate interpretations the prereg leaves open (all stated in the report): capture pooled over all splits at base costs; max drawdown = worst split at base costs; positive splits at base costs; OOS CI on pooled OOS daily returns at fees x2; stress changes exchange fees only (the strategy's `maker_fee` input stays 0.00075, taker slippage stays 10 bps). If ETH passes the gate, the runner reruns the main gate at fill_haircut 0.7 (haircut and haircut+fees x2 runs).
- Symbol filters are assumed constant over history (table in `g1_run.py`; DOGE/LINK ticks are finer than today's so early sub-$1 prices are not distorted). Baseline "equal-weight B&H" only includes symbols listed at the split start.
- Smoke trims history to 92 days before the window (features are slightly less converged than full-history ones).

Surprises for the orchestrator (real data, base costs):
- **Heavy order churn.** The flat ladder re-anchors to the bid every cycle: about 8 place + 8 cancel per decision in smoke, about 110k places per symbol-year (BTC 2023: 495 fills). Harmless in the sim but would hit rate limits live.
- **`duplicate_client_id` rejections are frequent**: 1767 in smoke (about 4% of places), 8187 in the 5-symbol 2022 run (about 7.6%). Each is a missing ladder level for that cycle, so fills are understated (conservative) and the live id scheme needs a fix (e.g. a per-placement nonce or time bucket).
- Exposure cap is soft: single-symbol BTC 2023 peaked at 15.09% vs per_symbol_cap 15%, because marks move after placement. The 38% total cap held (peak 37.997% in the 5-symbol 2022 run).
- Exploratory single-year check, 5 symbols, trial T01, 2022 only: return -6.3%, gross realised -583 USDT vs 52 USDT fees, 5 halt events. Not a G1 result, just a hint that G1 will be hard.
- ETH shadow control passed 3 of 4 criteria in the 2-month smoke (only the 2022 criterion failed): worth watching in the full run.
- The prereg `_note` is dated 2026-11-03, a day after today's date (2026-11-02); not changed.
