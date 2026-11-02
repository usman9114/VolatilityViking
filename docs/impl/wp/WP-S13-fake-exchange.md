# WP-S1.3 FakeExchange: multi-coin, shared wallet, honest fills
Model: sonnet · Depends: S1.1 · Worktree: no (only touch `vv/sim/`, `tests/sim/`) · Budget: 90k · Commit: NO

## Goal
`vv/sim/fake_exchange.py`: `FakeExchange` implements the `ExchangeAdapter` protocol in `vv/core/ports.py` over historical 1m bars for **several symbols sharing one USDT wallet**, on a virtual clock. It's the backbone of the G1 replay, so **fills must be pessimistic**. See `docs/research/G0_DECISION.md` §3 and §6.

## Navigate
`graphify explain "ExchangeAdapter"`, `graphify explain "PortfolioState"`. Read `vv/core/types.py`, `ports.py` and `filters.py` fully (they're small).

## Spec
- **Construction:**
  - `FakeExchange(bars: dict[str, pd.DataFrame], filters: dict[str, SymbolFilters], start_cash: float, maker_fee=0.00075, taker_fee=0.001, taker_slippage_bps=10, fill_haircut=1.0, seed=0)`
  - A `VirtualClock` (implements `Clock`), with `advance_to(ts)`.
  - Bars are aligned on a common UTC 1m index. A symbol not yet listed has no bars, and ordering it is rejected.
- **Market data:**
  - `fetch_tickers` derives bid/ask from the current bar's close ± half a synthetic spread of 1 tick.
  - `fetch_ohlcv` returns only bars ≤ now (assert no look-ahead).
- **Orders:**
  - Post-only buys at ≥ ask and post-only sells at ≤ bid are **rejected**, never filled as taker.
  - Prices and quantities must already be on tick and step (reject otherwise). Reject below min_notional.
  - A buy reserves cash (`cash_free` → `cash_reserved`) at placement. Reject if `cash_free` is insufficient. Sells need free base quantity.
  - Duplicate `client_id` is rejected (idempotency).
- **Matching** happens on `advance_to(next_ts)` for each elapsed bar:
  - A resting buy fills only if `low < price − tick` (**trade-through, not touch**). A sell fills only if `high > price + tick`.
  - Fill at the limit price with the maker fee.
  - `fill_haircut` < 1: each eligible fill happens with that probability (seeded RNG); otherwise the order stays resting.
  - **Intrabar path:** O→L→H→C if close ≥ open, else O→H→L→C. An order placed in a bar can't fill in that same bar.
  - Non-post-only orders (used for flattening) fill at the **next bar's open** ∓ slippage, with the taker fee.
  - Partial fills aren't modelled (document this). Full fills only.
- **Accounting:**
  - `Position.avg_entry` is the volume-weighted buy cost **including fees**. Realized PnL goes on sells.
  - Fees are charged in quote currency.
  - Expose `portfolio_state(peak_equity, halt) -> PortfolioState` and `fills` (a list of fill records: ts, symbol, side, price, qty, fee, client_id).
  - `fetch_my_trades` returns those records.

## Tests first (`tests/sim/test_fake_exchange.py`)
1. Touch ≠ fill (low == price − tick) and trade-through = fill.
2. Post-only crossing is rejected.
3. Cash is reserved at placement and released on cancel. A second buy beyond cash is rejected.
4. **Shared wallet:** two symbols compete for the same cash.
5. No same-bar fill. Intrabar ordering: in a down bar, a resting sell above and a buy below → the sell fills first.
6. Fees and avg_entry maths on 3 buys and 1 sell (hand-computed).
7. A taker flatten fills at the next open with slippage.
8. A duplicate client_id is rejected.
9. `fetch_ohlcv` never returns future bars.
10. `fill_haircut=0.0` → nothing fills. Deterministic under a fixed seed.

## Done when
`.venv/bin/pytest -q tests/sim && .venv/bin/ruff check vv tests/sim`, with `make test` still green.

## Reply (≤120 words)
status · test line · modelling simplifications S1.4 must know
