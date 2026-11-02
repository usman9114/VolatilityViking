# WP-S1.6 Strategy variants B (paired grid) and C (rebalancing) for G1 v2
Model: sonnet · Depends: S1.4, S1.5 · Worktree: no (only touch `vv/core/variants/`, `vv/sim/replay.py` (additive), `research/g1v2_run.py`, `tests/core/test_variants.py`, `tests/sim/test_replay_variants.py`) · Budget: 120k · Commit: NO

## Goal
Implement the two Binance-inspired variants exactly as pre-registered in `research/g1v2_preregistration.json`. **Read it and don't change it.** Then build a runner that compares them under the same FakeExchange, risk caps, halts, splits and gate as G1 v1 (`research/g1_preregistration.json` plus the addendum).

## Navigate
`graphify explain "decide"`, `graphify explain "run_replay"`, `graphify explain "evaluate_halt"`. **Reuse** `risk.evaluate_halt` and `flatten_intents`, `sizing.symbol_budgets`, `orders.reconcile` and `order_id`, `filters`, and `features`. Don't duplicate them.

## Variant B: `vv/core/variants/paired_grid.py`
`decide_paired(snapshot, state: GridState, params: PairedGridParams, risk) -> tuple[Decision, GridState]`
- `GridState` (frozen) per symbol: `anchor`, `levels` (prices), `held: frozenset[int]` (levels whose buy has filled and whose sell is pending), `out_of_range_since`.
- Fill detection: compare `state` with the snapshot's open orders and positions. A buy at level i that's gone while the position grew → filled, so add i to `held`. A sell that's gone while the position shrank → remove i from `held`. The caller passes a fill list in the snapshot. Add `fills_since_last: tuple` to `Snapshot` (default empty) and fill it in the replay from `ex.fills`.
- Desired orders:
  - For each level i not held, with the level below the bid and buys allowed: a buy at level i.
  - For each held i: a sell at level i+1 for that level's qty.
  - Equal quote per level = budget / levels.
- Range = anchor·(1±step)^(levels/2), geometric. Re-centre only when the symbol is flat and price has been outside the range ≥ 24h.
- Stale snapshot → no intents. Halts are the same as production (latched, flatten).

## Variant C: `vv/core/variants/rebalance.py`
`decide_rebalance(snapshot, params: RebalanceParams, risk) -> Decision`
- Target value per coin = inverse-vol share of `total_exposure_cap`·equity.
- If |value − target| > band·target: one post-only order. Buy at bid − 1 tick, or sell at ask + 1 tick, sized to the difference and rounded.
- Keep at most one resting order per coin. Replace it if the target moves by more than band/2.
- Halts, stale handling and caps are the same as production.

## Replay and runner
- `run_replay(..., strategy=...)` accepts a strategy callable plus an optional state object. Keep the default production behaviour byte-identical, so existing tests stay green.
- `research/g1v2_run.py`:
  - B and C trials × splits, selection within each variant, the stress runs and the ETH shadow, as pre-registered.
  - Variant A rows come from re-running the production trials. **Don't read v1 output files.**
  - Report: `docs/research/G1V2_RESULTS.md`, with a side-by-side out-of-sample table and a per-variant gate table, plus the total trial count (16).
  - Support a `--smoke` flag.

## Tests first
- **B:**
  - A buy fill at level i → a sell at i+1 for the same qty. A sell fill → the buy re-arms at i.
  - No buy at or above the bid.
  - Re-centring only happens when flat and after 24h outside the range.
  - The regime gate blocks new buys but keeps the sells.
  - Total exposure stays ≤ the cap.
- **C:**
  - Inside the band → no order. Outside the band → one order of the right size and side.
  - Post-only prices are never marketable.
  - One resting order per coin at most.
- **Replay:** the default strategy's results are identical to before (golden comparison on the existing synthetic test). Each variant runs on the synthetic bars.

## Done when
`.venv/bin/pytest -q tests && .venv/bin/ruff check vv research tests`, then `--smoke` completes. **Don't run the full v2.** The orchestrator runs it.

## Reply (≤120 words)
status · test line · smoke runtime · projected full runtime · design choices the prereg left open
