# WP-M2 Daily runner: data → trend targets → evidence weights → orders
Model: sonnet · Depends: M1 (`vv/exec`), M3 (`vv/evidence`), T1 (`vv/core/trend.py`) · Worktree: no (only touch `vv/runner/`, `tests/runner/`, `config/runner.example.yaml`) · Budget: 120k · Commit: NO

## Goal
A single daily job (`python -m vv.runner --config config/runner.yaml --mode {dry|testnet|live}`) that runs the minimum viable loop from `docs/AI_SIGNAL_PLAN.md` §6. **Live mode must also require the env var `VV_LIVE_CONFIRM=yes` and refuse otherwise.**

## Steps per run (00:05 UTC daily)
1. **Data:** fetch daily closes for the configured symbols from `data-api.binance.vision` (public). Keep an append-only local parquet with ingestion timestamps. **Gap detection:** if any expected daily bar is missing or stale (> 26 h), **skip trading**, alert, and exit non-zero.
2. **Signals:** for each registry signal with status in {shadow, live}, compute target weights with `vv/core/trend.target_weights` (variant params from the registry: spot/perp, short allowed, dial on/off). Compute dial features with `research/feature_screen.py`'s builders when a variant uses the dial; if the external data is stale, apply no penalty and alert.
3. **Evidence:**
   - Update each signal's shadow P&L from yesterday's targets and today's closes. Include perp funding (from `data-api` / the funding archive; if it's unavailable, estimate 0 and flag it).
   - Feed it to `vv/evidence` (posterior, CUSUM).
   - Compute weights: ¼-Kelly, factor budget (all trend variants are factor `risk_on`), ramp cap.
4. **Book:** final target = Σ_signal weight × signal targets, then caps (gross ≤ 1, shorts ≤ 0.5, per-coin ≤ 0.25). Convert to coin quantities at the latest close.
5. **Execute:**
   - `dry`: log intended orders.
   - `testnet` / `live`: `adapter.start(symbols)`, then `reconcile()`. Any unexplained diff → persist halt, alert, exit.
   - Otherwise `adapter.rebalance(targets, decision_ts=...)`.
6. **State:** SQLite (WAL) with tables `runs`, `targets`, `orders`, `fills`, `positions`, `shadow_pnl`, `evidence`, `halt`, `ingest`. **The halt persists and never auto-resumes.** `python -m vv.runner resume --reason "..."` clears it and is logged.
7. **Alerts** go to Telegram via env `VV_TELEGRAM_TOKEN` / `VV_TELEGRAM_CHAT`; if they're unset, log only. Also post a heartbeat to the healthchecks.io URL in env `VV_HEARTBEAT_URL`, if it's set. Never log secrets.
8. **Report:** write `reports/YYYY-MM-DD.md` with the evidence table, targets vs held, orders, P&L and alerts.

## Config (`config/runner.example.yaml`)
Fields: symbols, venue, the registry path, caps, ramp base, state DB path, data dir, mode default `dry`.

## Tests first (`tests/runner/`, FakeVenue and synthetic data, no network)
- **Gaps:** a missing bar → no orders, alert, non-zero exit.
- **Halt:** a persisted halt blocks the next run, and `resume` clears it.
- **Live guard:** live mode without `VV_LIVE_CONFIRM` refuses.
- **Caps:** the combined book respects all caps.
- **Zero weight:** zero evidence weight → zero targets → no orders (today's real situation).
- **Reconcile:** a diff → halt.
- **Determinism:** the same inputs → identical orders (ids) on rerun, so it's idempotent.
- **Secrets:** a secret-redaction test.

## Run
After the tests pass, run `--mode dry` once against real public data in this container. Use `SSL_CERT_FILE=/root/.ccr/ca-bundle.crt`; ccxt doesn't use the env proxy, see `vv/exec`. Paste the report into your `## Result`.

## Done when
`.venv/bin/pytest -q tests/runner && .venv/bin/ruff check vv/runner tests/runner`, `make test` still green, and the dry run completes.

## Reply (≤120 words)
status · test line · dry-run summary (evidence weights, targets, intended orders)
