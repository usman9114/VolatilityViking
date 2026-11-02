# WP-S0C G0 edge pre-check: does ETH/BTC mean-revert at grid horizons?
Model: sonnet · Depends: none to write; H0B data to run · Worktree: no (only touch `research/`, `tests/research/`, `docs/research/`) · Budget: 90k · Commit: NO

## Why
A grid is short volatility / long mean reversion. On a random walk it has zero edge before fees (see `docs/PRODUCTION_PLAN.md` §3–4, gate G0). We need the measured answer before building anything else.

## Build `research/g0_variance_ratio.py`
Input: `data/klines/{SYMBOL}_1m.parquet` (UTC index; columns open, high, low, close, volume). CLI: `--symbols ETHUSDT BTCUSDT --out docs/research/G0_RESULTS.md`.
1. **Variance ratio** (Lo–MacKinlay) on log returns.
   - Base interval 5m; horizons q giving 15m, 30m, 1h, 2h, 4h, 8h, 1d.
   - Also base 1m → 5m, labelled "contaminated by bid-ask bounce". Kline closes are trade prices, so very short-horizon VR < 1 is partly bounce, which a maker grid can't harvest.
   - Report VR(q), the heteroskedasticity-robust z*, p-value, and a **stationary block-bootstrap 95% CI** (block ≈ 1 day, ≥ 500 reps, fixed seed). Vectorise; the run should finish in a few minutes.
2. **Stability:** the same table **per calendar year** (2019 → 2026), for 1h and 4h. Pooled-only results don't count.
3. **DFA Hurst exponent** on 5m returns over 1h–1d scales (pooled and per year).
4. **Range vs fees:** the median and 25th percentile of (high − low)/open on 1h and 4h bars, compared with a 0.15% maker round trip (BNB) and the 2.5× threshold (0.375%).
5. **Verdict section**, applying the G0 rule literally: PASS only if VR < 1 with the CI excluding 1 at 1h **and** 4h for both symbols, in a majority of years, **and** the median 4h range > 0.375%. Otherwise FAIL, with a one-line reason. Use plain language and no hype.

## Tests (write first) `tests/research/test_vr.py`
- AR(1) returns with φ = −0.2 → VR(2) < 1, CI excludes 1.
- GBM returns → CI contains 1.
- Block bootstrap is deterministic under a seed.
- DFA on white noise ≈ 0.5 (±0.1).

Use the scratchpad venv (`/tmp/claude-0/-home-user-VolatilityViking/01f0721a-1d36-5a2d-8343-532333f56486/scratchpad/venv/bin/python`); pip install `scipy` and `pyarrow` there if they're missing. Don't touch `.venv`.

## Done when
The tests pass. If the parquet files exist (another agent is downloading them), run the script and write `docs/research/G0_RESULTS.md` (one page: tables plus verdict). If they don't exist yet, stop after the tests pass and say so.

## Reply (≤120 words)
status · test line · verdict (PASS/FAIL plus the key VR numbers at 1h/4h) or "awaiting data"
