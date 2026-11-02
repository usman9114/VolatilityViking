# WP-S1.1 Core types and ports (multi-coin, shared wallet)
Model: sonnet · Depends: wave 0 · Worktree: no (only touch `vv/`, `tests/core/`) · Budget: 60k · Commit: NO

## Goal
Create the typed foundation every later WP builds on. The strategy becomes a pure function over a **portfolio-wide** snapshot (see `docs/research/G0_DECISION.md` §3 and §6). This WP defines types and interfaces only. Don't port strategy logic; that's S1.2.

## Navigate
`graphify explain "process_symbol"` and `graphify explain "AsyncMultiSymbolBot"` show which fields the live loop uses. Read `src/live/async_runner.py` only at the line ranges the graph gives you.

## Create
- `vv/__init__.py`, `vv/core/__init__.py`
- `vv/core/types.py` (frozen dataclasses; floats in the core):
  - `SymbolFilters(tick_size, step_size, min_qty, min_notional)`
  - `SymbolMarket(symbol, ts, bid, ask, last, filters)`
  - `Position(symbol, qty, avg_entry, realized_pnl)`
  - `OpenOrder(symbol, side, price, qty, filled, client_id, exchange_id, ts)`
  - `HaltState(active: bool, reason: str | None, since: datetime | None)`
  - `PortfolioState(ts, cash_free, cash_reserved, positions: dict, open_orders: tuple, peak_equity, halt)` with properties `equity(marks)`, `exposure(marks)` (holdings + resting buy notional), `exposure_by_symbol(marks)`
  - `Snapshot(ts, markets: dict[str, SymbolMarket], bars: dict[str, pd.DataFrame], portfolio, stale: tuple[str, ...])`. `fresh` is True when `stale` is empty.
  - `PlaceOrder(symbol, side, price, qty, client_id, post_only=True)`, `CancelOrder(symbol, client_id)`, `OrderIntent = PlaceOrder | CancelOrder`
- `vv/core/params.py`: pydantic v2 frozen `GridParams`. One shared parameter set **in σ units** (spacing_sigmas, levels, qty_multiplier, tp_fee_multiple=2.5, maker_fee) plus `vol_scalar: dict[str, float]`, and `RiskParams` (total_exposure_cap ≤ 0.6, per_symbol_cap, portfolio_halt_dd=0.15, symbol_halt_dd=0.25). Validate ranges. **No defaults for risk fields.**
- `vv/core/filters.py`: Decimal-based `round_price(side, price, tick)` (buys round down, sells round up), `round_qty(qty, step)` (always down), `validate_notional(...)`. Round first, then validate.
- `vv/core/ports.py`: `typing.Protocol`s for:
  - `ExchangeAdapter` (async: `load_markets`, `fetch_tickers`, `fetch_balance`, `fetch_open_orders`, `fetch_ohlcv`, `create_order(PlaceOrder)`, `cancel_order(CancelOrder)`, `fetch_my_trades`)
  - `Clock` (`now()`)
  - `StateStore` (`get(key)`, `put(key, value)`, `record_order`, `record_fill`, `load_halt()`, `save_halt(HaltState)`)
- `vv/core/decide.py`: only the signature `def decide(snapshot: Snapshot, params: GridParams, risk: RiskParams) -> list[OrderIntent]`, raising `NotImplementedError`, with a docstring stating the contract: pure, no I/O, returns zero intents if `not snapshot.fresh` or `portfolio.halt.active`.
- `vv/core/ids.py`: deterministic `client_id(symbol, side, level, epoch_bucket)` → `vv-` prefix, ≤ 36 chars, stable across restarts.

## Tests first (`tests/core/`)
- `test_filters.py`: tick and step rounding directions; min-notional rejected after rounding; Decimal edge cases (0.1 + 0.2).
- `test_types.py`: `equity` and `exposure` across 2 symbols, including reserved cash for resting buys; params validation rejects total cap > 0.6 and missing risk fields.
- `test_ids.py`: deterministic, unique per level, ≤ 36 chars.

## Done when
`.venv/bin/pytest -q tests/core && .venv/bin/ruff check vv tests/core`. The existing suite still passes (`make test`).

## Out of scope
Porting strategy logic, ccxt code, SQLite, replay.

## Reply (≤120 words)
status · files · test line · any interface decision S1.2/S1.3 must know
