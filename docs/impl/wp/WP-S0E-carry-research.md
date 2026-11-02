# WP-S0E Funding carry research (research only)
Model: sonnet · Depends: none · Worktree: no (only touch `research/carry_*.py`, `tests/research/test_carry.py`, `docs/research/CARRY_RESULTS.md`, `data/futures/`) · Budget: 80k · Commit: NO

## Why
Carry (long spot + short USDⓈ-M perpetual, delta-neutral) is the fallback and possible second sleeve. See `docs/PRODUCTION_PLAN.md` §3. The experts estimated a net of about 3–6% a year on total capital. **Measure it.** No live code.

## Data (public, `data.binance.vision`)
- Funding: `data/futures/um/monthly/fundingRate/{SYM}/{SYM}-fundingRate-{YYYY-MM}.zip`
- Perp 1h klines: `data/futures/um/monthly/klines/{SYM}/1h/{SYM}-1h-{YYYY-MM}.zip`
- Spot 1h: resample the existing `data/klines/{SYM}_1m.parquet`.

Symbols: BTC, ETH, SOL, XRP, DOGE, LINK, BNB, AVAX (USDT), 2020-01 → 2026-10. Reuse the download, checksum and timestamp logic from `scripts/fetch_klines.py` (import it; don't copy). Mind the µs timestamps from 2025 on. Write to `data/futures/`.

## Analysis (`research/carry_analysis.py`)
Per symbol and per year:
1. Annualised gross funding, % of negative funding periods, and the longest negative streak.
2. Basis (perp/spot − 1) distribution: the worst 1h and 1d adverse move for the short-perp leg.
3. **Net carry on total capital:**
   - Assume 50% of capital in spot, 50% as perp margin (1× short).
   - Entry and exit fees on both legs: spot maker 0.075%, perp maker 0.02%.
   - Monthly rebalance cost.
   - **Exit rule:** go flat when the trailing 7-day funding is < 0. Re-enter when it's > 0, paying fees.
4. The carry equity curve (funding received minus fees minus basis mark-to-market), its max drawdown, and Sharpe.
5. A portfolio of the top-3 symbols by trailing funding, rebalanced monthly, using only information available at the time (no look-ahead).

## Gate CARRY-0 (literal)
PASS if the portfolio's median annual net is ≥ 3% on total capital, its worst year is ≥ −2%, and its max drawdown is ≤ 8%. Report PASS or FAIL with numbers.

## Tests first (`tests/research/test_carry.py`)
- Funding annualisation (3 periods/day).
- The exit rule fires on a synthetic negative-funding run.
- Fees are charged on each re-entry.
- Portfolio selection uses only past data. Shift the input and check that the chosen symbols change only after that point.

## Done when
Tests and ruff are green, and `docs/research/CARRY_RESULTS.md` (≤ 1.5 pages: tables plus verdict) is written.

## Reply (≤120 words)
status · test line · CARRY-0 verdict with median/worst-year/maxDD · surprises
