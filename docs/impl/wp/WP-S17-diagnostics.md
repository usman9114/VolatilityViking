# WP-S1.7 G1 diagnostics: exposure audit, fill-depth sweep, VR attribution
Model: sonnet · Depends: S1.4–S1.6 · Worktree: no (only touch `vv/sim/fake_exchange.py` (additive param), `vv/sim/metrics.py` (additive), `research/g1_diagnostics.py`, `tests/sim/test_diagnostics.py`, `docs/research/G1_DIAGNOSTICS.md`) · Budget: 90k · Commit: NO

## Why
See `docs/research/G1_DECISION.md`. These are **diagnostics, not trials**. They don't change any gate or pre-registered setting. A full G1 v2 run is using 3 CPU cores in the background right now, so use `--processes 1` for your runs and keep them short.

## Navigate
`graphify explain "run_replay"`, `graphify explain "FakeExchange"`, `graphify explain "exposure_total_post"`.

## Tasks
1. **Exposure-metric audit.** G1 v1 reports 19.4% return in 2021 at 1.1% "average exposure". Check how average exposure is computed: whether it includes resting buys, whether it's sampled at decisions, and whether it's a fraction or a percentage. Recompute it independently from the trace (`record_trace=True`) for selected trial T09 (min_step 0.03, spacing_sigmas 1.5), BTC+SOL+XRP+DOGE+LINK, 2021. Report holdings-only exposure, holdings+resting exposure, and **return on average deployed capital**. Fix the metric if it's wrong (additive: keep the old field and add a correct one) and explain.
2. **Fill-depth sweep.** Add `min_pierce: float = 0.0` to `FakeExchange`. A resting buy fills only if `low < price·(1 − min_pierce) − tick`, and sells symmetrically. The default must stay byte-identical; prove it with the existing golden or replay tests. Rerun T09 on every v1 split at min_pierce ∈ {0 (1 tick), 0.0005, 0.0015}. For each, report the return, Sharpe and capture, plus the **share of total P&L earned on the top 10 days**.
3. **VR attribution.** For T09 at base settings, compute P&L per coin per calendar year: run single-symbol replays, one wallet per coin, with the same params. Regress it on that coin-year's 1h VR from `docs/research/G0_RESULTS.md` (parse the yearly table; ETH included as a coin). Report the slope, its t-stat, R² and a scatter table. State plainly whether P&L tracks VR (regime decay) or sits below it (capture decay).
4. **Cash benchmark.** Add a 4%-per-year USDT-yield baseline to the metrics output and show T09's excess over it per split.

## Output
`docs/research/G1_DIAGNOSTICS.md` (≤ 1.5 pages): four sections, each with a one-line conclusion. Plain language.

## Tests first
- `min_pierce=0` is identical to before.
- `min_pierce` > 0 blocks a fill that only pierces by 1 tick.
- The top-N-day P&L share is correct on a known series.
- The audited exposure metric is correct on a hand-built trace.

## Done when
`.venv/bin/pytest -q tests && .venv/bin/ruff check vv research tests`, and the report is written.

## Reply (≤120 words)
status · test line · the four one-line conclusions

## Result
Status: done. Report: `docs/research/G1_DIAGNOSTICS.md` (numbers cached in `data/g1_diagnostics_raw.json`; `python -m research.g1_diagnostics --symbols ... --reuse-raw` re-renders).

- `FakeExchange(min_pierce=0.0)`: default byte-identical, proven by a golden digest recorded from the pre-change code (`tests/sim/test_diagnostics.py`) plus the untouched fake-exchange/replay suites.
- `vv/sim/metrics.py` (additive): `cash_yield_return`, `cash_yield_curve`, `daily_pnl`, `top_n_day_pnl_share`, `exposure_audit`, `deployed_from_decisions`; `summarize` gains `cash_yield_return`, `excess_over_cash`, `avg_total_exposure`, `avg_deployed_usdt`, `return_on_avg_deployed` (old fields unchanged).
- Because `vv/sim/replay.py` is out of scope, `run_replay` gets `min_pierce` / the tracing exchange by swapping `vv.sim.replay.FakeExchange` inside `research.g1_diagnostics.pierce_exchange(...)`. A `min_pierce` argument on `run_replay` / `Fees` would be cleaner if the orchestrator wants it.
- `record_trace=True` records only the feature-bar timestamps, not exposure; the audit rebuilds exposure from a per-minute exchange trace (`TracingExchange`) and cross-checks it against the fills.
- Single-symbol windows that open before a listing (LINK 2019, DOGE 2019, SOL 2020) start one pre-roll hour after the first bar, since the exchange clock cannot start after `start`. The strategy is not warm that early, so nothing is lost.
- Runs used `--processes 1` alongside the G1 v2 job.
