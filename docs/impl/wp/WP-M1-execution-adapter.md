# WP-M1 Execution adapter: Binance USD-M perps + spot (target-position rebalancer)
Model: sonnet · Depends: vv/core/ports.py, vv/core/filters.py · Worktree: no (only touch `vv/exec/`, `tests/exec/`) · Budget: 120k · Commit: NO

## Goal
Turn **target positions** (signed coin quantities per symbol) into safe exchange orders, and reconcile real state. This replaces nothing in the old grid path; it's a new, small module for the trend/evidence book. Read `docs/AI_SIGNAL_PLAN.md` §5 (hard limits) first.

## Build `vv/exec/`
- `binance_perp.py`: `PerpAdapter` (ccxt `binanceusdm`)
  - **Modes:** `testnet` (`https://testnet.binancefuture.com`) or `live`. Keys come from env vars `VV_BINANCE_KEY` / `VV_BINANCE_SECRET`; **never log them**.
  - **On start, verify, refusing to trade on any failure:**
    - **one-way position mode:** set it if it's off and there are no positions; otherwise refuse
    - **isolated margin and leverage 1** per symbol
    - `adjustForTimeDifference`
  - **Reads:** `positions()` returns signed qty, entry, mark, liquidation price and ADL rank per symbol. `balance()` returns wallet, available and margin balance.
  - `rebalance(targets: dict[str, float], *, max_short_frac=0.5, gross_cap=1.0, dry_run=False) -> list[OrderResult]`
    - Compute deltas. Round qty down to step (`vv/core/filters.py`) and skip anything below min notional.
    - **Clip the targets first** so that Σ|notional| ≤ gross_cap × equity and Σ short notional ≤ max_short_frac × equity.
    - Exits and reductions use `reduceOnly=True`. Entries are LIMIT at best bid/ask with `postOnly` (GTX); if they're unfilled after N seconds, fall back to a marketable LIMIT IOC within a 0.2% slippage bound.
    - Deterministic `newClientOrderId` (`vv-` prefix, hash of symbol, side, qty and decision timestamp), so retries are idempotent.
    - Refuse to trade if `halt_active()` (a callable passed in) is true.
- `binance_spot.py`: `SpotAdapter` with the same interface; long-only and no margin. A live spot testnet isn't reachable here, so test against the fake only.
- `fake.py`: `FakeVenue`, an in-memory venue for both modes (fills at the quoted price ± slippage, funding hook, liquidation-price calc), used by the tests and by the dry-run runner.
- `reconcile.py`: compares the exchange's positions with the state store's expected positions and returns diffs. Any unexplained diff → halt plus alert (the caller decides).

## Tests first (`tests/exec/`, no network)
- **Clipping:** gross and short caps hold (property test).
- **Rounding:** step/min-notional rounding never increases exposure.
- **Exits:** reduce-only on every exit.
- **Idempotency:** a retry with the same decision produces the same clientOrderId, and FakeVenue rejects the duplicate.
- **Halt:** the halt blocks every order.
- **Mode guards:** one-way and isolated checks refuse to trade when unmet.
- **Logging:** a secret-redaction test checks that keys never appear in log output.

## Testnet smoke (optional, this container)
**Public endpoints only:** load markets and fetch the ticker on testnet.binancefuture.com. Private calls need testnet keys. **Don't** request or use any key in this WP. Just record that the private smoke test is pending keys.

## Done when
`.venv/bin/pytest -q tests/exec && .venv/bin/ruff check vv/exec tests/exec`, with `make test` still green.

## Reply (≤120 words)
status · test line · public testnet smoke result · interface notes for the M2 runner
