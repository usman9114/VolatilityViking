# WP-T2 Spot and perpetual-futures portfolio simulator, and the G2 runner
Model: sonnet · Depends: WP-T1 interface (being built in parallel; code against its signatures in `docs/impl/wp/WP-T1-trend-strategy.md` and stub it in tests until it lands) · Worktree: no (only touch `vv/sim/portfolio_sim.py`, `research/g2_run.py`, `tests/sim/test_portfolio_sim.py`, `docs/research/G2_RESULTS.md`) · Budget: 120k · Commit: NO

## Goal
1. A target-weight portfolio simulator for **spot (long/flat)** and **USD-M perpetuals (long/short, cross margin, funding, liquidation)**.
2. `research/g2_run.py`, which runs the four pre-registered variants and applies gate G2 literally.

All settings come from `research/g2_preregistration.json`. **Read it and don't change it.**

## `vv/sim/portfolio_sim.py`
`simulate(weights_daily: pd.DataFrame, venue: Literal["spot","perp"], prices_1h: dict[str, pd.DataFrame], funding: dict[str, pd.Series], *, fee: float, slippage_bps: float, maintenance_margin: float = 0.005, start_cash: float = 10_000, band=None) -> SimResult`
- **Decisions:** a daily weight row dated t (decided at the close of day t) is executed at the first hourly bar after t+1 day 00:00 UTC.
  - Perp: that bar's **open**.
  - Spot: that bar's **close**. Spot only has closes.
  - Buys pay +slippage and sells pay −slippage, plus the fee on traded notional. Size against equity marked at the execution price.
  - `band`: a callable `(current_w, target_w) -> held_w` (from WP-T1 `apply_band`). Use it when given.
- **Spot accounting:**
  - Coin units plus USDT.
  - No shorting: assert that weights are ≥ 0.
  - No borrowing: gross ≤ 1.
- **Perp accounting:**
  - Cash collateral plus signed positions in contracts (coin units), marked hourly at the close.
  - Unrealised PnL counts in equity.
  - **Funding:** at each funding timestamp, `cash -= qty * mark * rate` (signed qty, so shorts receive positive funding). Use the nearest hourly mark at or before that timestamp.
  - **Liquidation:** if hourly equity, using the bar's adverse extreme (high for net shorts, low for net longs, per position), falls below the maintenance margin × gross notional, close everything at that extreme, charge the taker fee, and record the event.
- **Coverage:** coins without data at a timestamp can't be traded. Their target is treated as 0 and existing positions are held, with a warning counted.
- **`SimResult` contents:**
  - `equity_hourly`
  - `equity_daily`
  - `trades` (DataFrame)
  - `fees_paid`
  - `funding_paid` (signed)
  - `liquidations` (list)
  - `turnover`
  - `max_gross`

## `research/g2_run.py`
1. **Load data:**
   - `data/futures/{SYM}_spot_1h.parquet` for spot closes. Resample to daily closes for the signals.
   - `data/futures/{SYM}_perp_1h.parquet`
   - `data/futures/{SYM}_funding.parquet`
2. **Dial features:** build them with `research/feature_screen.py`'s feature construction (import it; don't reimplement). Columns: `stablecoin_supply_chg_30d`, `vix_level`, `global_ls_ratio` (basket mean), with that module's lagging.
3. **Run** S1, S2, P1 and P2 as pre-registered, each as one continuous run 2020-01-01 → 2026-10-31. Then run the stress versions (fee ×2, slippage ×2).
4. **Metrics:**
   - Reuse `vv/sim/metrics.py` for the per-split table, the full-run max drawdown and the stationary-bootstrap Sharpe CI (OOS, stress).
   - Also report excess over 4% cash, turnover, fees, funding, liquidations, and the share of time long, short and flat.
   - Benchmarks: an equal-weight buy-and-hold of eligible coins, and cash at 4% a year.
5. **Gate G2** per variant, literal. Then write `docs/research/G2_RESULTS.md` (≤ 2.5 pages):
   - a side-by-side OOS table first
   - a per-variant gate table
   - the per-split tables
   - a plain-language verdict
   - an explicit note on what futures added: P vs S on the same signal

## Tests first (`tests/sim/test_portfolio_sim.py`)
- **Spot:**
  - A buy at the next-bar close with fee and slippage: hand-computed equity.
  - A negative weight raises.
  - The execution timestamp is strictly after the decision.
- **Perp:**
  - A short gains when price falls. Hand-computed.
  - Funding sign: a short receives positive funding and a long pays it.
  - Liquidation triggers on an adverse wick at 3× gross in a synthetic test (G2 itself uses ≤ 1×) and is recorded.
  - Equity is continuous across a funding print with no position.
- **Band:** with `band`, small target changes produce no trades.
- **Determinism:** identical inputs → identical results.

## Done when
Tests and ruff are green. When WP-T1's `vv/core/trend.py` exists (check every few minutes, up to about 40 minutes), run the full G2 and write the report. Use `--processes` ≤ 3.

## Reply (≤150 words)
status · test line · per-variant gate verdict · OOS return and Sharpe per variant · what futures added
