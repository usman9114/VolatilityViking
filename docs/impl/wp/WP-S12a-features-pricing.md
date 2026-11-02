# WP-S1.2a Features, pricing, regime (pure strategy maths)
Model: sonnet · Depends: S1.1 · Worktree: no (only touch `vv/core/features.py`, `vv/core/pricing.py`, `vv/core/regime.py`, `vv/core/params.py`, `vv/core/types.py` (add fields only), `tests/core/test_features.py`, `tests/core/test_pricing.py`, `tests/core/test_regime.py`) · Budget: 80k · Commit: NO

## Context
We're implementing the **new** strategy directly. We're not porting the old `process_symbol`; its behaviour is documented as buggy in `docs/PRODUCTION_PLAN.md` §2. The design comes from `docs/research/G0_DECISION.md` and the debate: a wide grid in σ units, take-profit anchored to cost basis, and a regime gate. Another agent (S1.2b) will build sizing, risk and order diffing on top of these exact signatures. **Don't change them.**

## 1. `vv/core/features.py`
`compute_features(bars_1m: pd.DataFrame, horizon_minutes: int, ema_span_4h: int = 200) -> pd.DataFrame`
- Input: a UTC-indexed 1m OHLCV frame for one symbol.
- Output: one row per 1m timestamp, using only past data (no look-ahead).
- Columns:
  - `close`
  - `sigma_h`: EWMA vol (halflife ≈ 1 day) of 5m log returns, scaled to the horizon as σ_5m·√(h/5)
  - `ema_4h`: EMA(`ema_span_4h`) of 4h closes, forward-filled to 1m. **Only completed 4h bars count.**
  - `regime_ok`: close ≥ ema_4h
  - `warm`: enough history for every column
- Vectorised (the replay runs it over 4M rows; target < 10 s per symbol).
- `latest_features(bars_1m, …) -> SymbolFeatures` returns the last row, using the same function. Add `SymbolFeatures(close, sigma_h, ema_4h, regime_ok, warm)` to `types.py`, and add `features: dict[str, SymbolFeatures]` (default empty) to `Snapshot`.

## 2. `vv/core/params.py` (extend `GridParams`)
Add `min_step: float` (fraction, required, 0.001–0.2), `sigma_horizon_minutes: int` (required, 5–1440) and `round_trip_fee` (a property equal to 2 × `maker_fee`).

## 3. `vv/core/pricing.py`
- `step_size(features: SymbolFeatures, params, symbol) -> float` = max(`min_step`, `tp_fee_multiple` × `round_trip_fee`, `spacing_sigmas` × `sigma_h` × `vol_scalar.get(symbol, 1)`)
- `buy_ladder(bid: float, avg_entry: float | None, step: float, levels: int) -> list[float]`
  - ref = bid if there's no position, else min(bid, avg_entry).
  - Level i (1..levels) = ref·(1−step)^i.
  - Prices are strictly decreasing and every level is < bid.
- `take_profit_price(avg_entry: float, step: float, params, ask: float) -> float` = max(avg_entry·(1+step), avg_entry·(1+`tp_fee_multiple`·`round_trip_fee`), ask). It is never below breakeven plus fees.
- Prices aren't rounded here. Tick rounding happens in S1.2b using `filters.py`.

## 4. `vv/core/regime.py`
`buys_allowed(features: SymbolFeatures) -> bool` returns `features.warm and features.regime_ok`.

## Tests first
- **features:**
  - Changing the future doesn't change past rows (look-ahead test: truncate at t, recompute, compare).
  - `sigma_h` on synthetic GBM ≈ the true σ (±15%).
  - `ema_4h` ignores the in-progress 4h bar.
  - `warm` is False at the start.
- **pricing:**
  - `step_size` respects all three floors.
  - The ladder is strictly below bid and decreasing.
  - With a position the ladder references avg_entry.
  - TP ≥ breakeven + fees in a falling-price case (this was the old C4 bug).
  - TP ≥ ask.
- **regime:** blocks when not warm or below the EMA.

## Done when
`.venv/bin/pytest -q tests/core && .venv/bin/ruff check vv tests/core`, with `make test` still green.

## Reply (≤120 words)
status · test line · features runtime on the BTC 1m parquet (`data/klines/BTCUSDT_1m.parquet`) · any signature note for S1.2b
