# WP-S1.2b Sizing, risk/halt, order diffing, and the real `decide()`
Model: sonnet · Depends: S1.1, S1.2a · Worktree: no (only touch `vv/core/{sizing,risk,orders,decide}.py`, `vv/core/types.py` (add only), `tests/core/test_{sizing,risk,orders,decide}.py`) · Budget: 110k · Commit: NO

## Context
This completes the pure strategy. It replaces the old engine's money-losing defects. Each one below gets a named test:
- **C1:** buys above market
- **C2:** halt that doesn't persist or flatten
- **C3:** caps that ignore resting orders
- **C4:** take-profit below cost
- **C5:** acting on stale data

Use the S1.2a functions as they are (`features.py`, `pricing.py`, `regime.py`) and the S1.1 types and filters. Navigate with `graphify explain "buy_ladder"` and `graphify explain "PortfolioState"`.

## Signature change
`decide(snapshot, params: GridParams, risk: RiskParams) -> Decision`, where `Decision(intents: tuple[OrderIntent, ...], halt: HaltState)` is a frozen dataclass in `types.py`. The caller persists `halt`. Extend `HaltState` with `blocked_symbols: frozenset[str] = frozenset()`.

## `vv/core/risk.py`
`evaluate_halt(portfolio, marks, risk, now) -> HaltState`
- Starts from `portfolio.halt`. **Latched:** once active, it stays active (only an operator resume clears it, outside the core).
- Portfolio trip: equity < peak·(1 − `portfolio_halt_dd`).
- Symbol trip: mark < avg_entry·(1 − `symbol_halt_dd`). The symbol is added to `blocked_symbols`, which also latches.
- `flatten_intents(...)`: for the whole portfolio when the halt is active, or for blocked symbols only. Cancel all `vv-` orders for the affected symbols, then a `PlaceOrder(side="sell", post_only=False)` for the full free quantity at bid·(1 − 0.002), rounded with `filters.py`.

## `vv/core/sizing.py`
- `symbol_budgets(equity, sigmas: dict[str, float], risk) -> dict[str, float]`: an inverse-vol share of `total_exposure_cap`·equity, with each symbol capped at `per_symbol_cap`·equity.
- `ladder_quantities(prices, budget_remaining, qty_multiplier) -> list[float]`: geometric weights, with the notional summing to ≤ `budget_remaining`.
- `budget_remaining` = budget − current `exposure_by_symbol` (holdings at mark + resting buys). **Never negative.**
- A **cumulative global check** inside one decision: the running total exposure plus each new buy must stay ≤ `total_exposure_cap`·equity. Drop the rest.

## `vv/core/orders.py`
`reconcile(desired: list[PlaceOrder], open_orders, tolerance_ticks: int = 2) -> list[OrderIntent]`
- Keep an open `vv-` order if a desired order with the same symbol and side has its price within `tolerance_ticks` ticks and its qty within 10%.
- Cancel unmatched `vv-` orders. **Never touch orders whose client_id doesn't start with `vv-`.**
- Place unmatched desired orders.
- **Guards:** drop any post-only buy with price ≥ bid or post-only sell ≤ bid. Drop anything below min_notional after rounding.
- Client ids come from `ids.client_id(symbol, side, level, epoch_bucket)`. The epoch bucket changes only when the ladder anchor changes, so ids are stable across restarts.

## `vv/core/decide.py` (composition)
1. If `snapshot.stale` → `Decision((), portfolio.halt)`. **No intents at all** (C5).
2. halt = `evaluate_halt(...)`. If active → flatten everything, return.
3. For each symbol not blocked and with `features[s].warm`:
   - **Buys:** allowed only if `buys_allowed`. Otherwise desired buys are empty, so reconcile cancels the resting ones.
   - Ladder from `buy_ladder(bid, avg_entry, step, levels)`, quantities from sizing, rounded (buys round down).
   - **TP:** if the position qty > 0, one post-only sell for the full free qty at `take_profit_price(...)`, rounded up.
4. Blocked symbols get their flatten intents.
5. `reconcile`, then apply the global cumulative cap to new buys, in symbol order sorted by budget descending.

## Tests first
- C1: no post-only buy ≥ bid across randomized snapshots (property test, 200 cases, fixed seed).
- C2: the halt trips at the threshold, flattens, stays latched on the next decide with recovered prices, and blocks only the tripped symbol.
- C3: with resting buys already consuming the budget, no new buys. Total exposure never exceeds the cap across 3 symbols.
- C4: TP ≥ avg_entry·(1 + 2.5·round trip) after a falling ladder fill.
- C5: a stale snapshot gives zero intents, even with a halt condition present.
- Reconcile: keeps matching orders, ignores non-`vv-` orders, and cancels when the regime blocks.
- Determinism: same snapshot → same intents.

## Done when
`.venv/bin/pytest -q tests/core && .venv/bin/ruff check vv tests/core`, with `make test` still green.

## Reply (≤120 words)
status · test line · anything S1.4 (replay) must handle

## Result
Status: done (not committed). `pytest -q tests/core`: 119 passed; `ruff check vv tests/core` clean; `make test`: green.

Deviations and notes for S1.4:
- `reconcile(desired, open_orders, tolerance_ticks=2, *, markets)`: `markets` is a required keyword (tick size, filters and bid are needed for tolerance, rounding and guards).
- `flatten_intents(portfolio, markets, symbols, now)`. Limit is bid*(1-0.01), re-emitted every decide while free qty > 0, with a per-minute epoch bucket so the id is new each retry. A resting vv sell with matching price/qty is kept instead.
- Client ids: `orders.order_id(symbol, side, level, price, qty, epoch_bucket)`; a resized TP gets a new id. Ladder epoch bucket = anchor index on a geometric grid of one step. Residual risk: an identical order re-placed after an earlier identical one was cancelled/filled (same bucket) reuses its id and would be rejected; S1.4 should count such rejections.
- Ladder budget is net of holdings and non-vv resting buys, not of our own resting buys (they are the target ladder). Subtracting them would cancel the ladder once it fills the budget. The cumulative global/per-symbol cap counts all resting buys that survive the cycle.
- Marks are mid prices; a position with no market is marked at cost.
- Cold/no-feature symbols: resting vv buys are cancelled, sells left alone. Blocked symbols are flattened each cycle.
- Fills change qty/anchor, so ladder orders may be replaced after each fill (qty outside the 10% tolerance); expect churn in replay.
