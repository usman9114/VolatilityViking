# WP-S1.5 Ladder hysteresis and unique order ids (pre-G1 fixes)
Model: sonnet · Depends: S1.4 · Worktree: no (only touch `vv/core/decide.py`, `vv/core/orders.py`, `tests/core/test_decide.py`, `tests/core/test_orders.py`) · Budget: 50k · Commit: NO

## Why
The S1.4 smoke and exploratory runs (see `## Result` in `WP-S14-replay-g1.md`) showed two defects that would make G1 test a strategy we'd never run live:
1. **Churn.** With no position, the ladder re-anchors to the bid every cycle: about 8 cancels and 8 places per decision, about 110k places per symbol-year. Live, that hits Binance's rate limits.
2. **Duplicate ids.** `duplicate_client_id` rejections run at 4–8% of places. The ids don't change when an identical order is re-placed after a cancel or fill, so ladder levels go silently missing.

Navigate with `graphify explain "decide"` and `graphify explain "order_id"`. The relevant code is `vv/core/decide.py` around lines 80–110 and `vv/core/orders.py` lines 31–60.

## Changes
1. **Hysteresis (buy ladder, no position, regime OK).**
   - Let `ours` = this symbol's resting `vv-` buys and `top` = the highest-priced one.
   - If `ours` is non-empty and bid·(1 − 1.5·step) ≤ top ≤ bid·(1 − 0.5·step), the desired buys are **exactly the existing orders**, so reconcile keeps them all.
   - Otherwise, build a fresh ladder as now.
   - With a position (avg_entry set), keep the current behaviour (anchor = min(bid, avg_entry)). The ladder then only changes when a fill moves avg_entry or the bid drops below it.
2. **Unique ids per placement.**
   - Add the decision minute (`snapshot.ts` floored to the minute, as an int) to the hash input of **every** new `order_id`, alongside the existing fields.
   - Kept orders keep their old id, because reconcile matches on price and qty, not id.
   - This is still deterministic: same snapshot → same ids.

## Tests first
- Flat, with the bid drifting up 0.3·step per cycle over 5 cycles: zero cancels and zero places after the first cycle.
- The bid rises 0.6·step above the anchor → the ladder is replaced.
- The same desired order in two different decision minutes gets different ids. The same minute → identical ids.
- All existing C1–C5 tests still pass.

## Done when
`.venv/bin/pytest -q tests/core tests/sim && .venv/bin/ruff check vv tests`. Then run `.venv/bin/python -m research.g1_run --symbols BTCUSDT SOLUSDT --smoke --out /tmp/claude-0/-home-user-VolatilityViking/01f0721a-1d36-5a2d-8343-532333f56486/scratchpad/smoke2.md` and report places per decision and duplicate_client_id counts (both should drop sharply).

## Reply (≤120 words)
status · test line · smoke places/decision before vs after · duplicate rejections after
