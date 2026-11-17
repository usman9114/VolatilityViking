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

## Result
status: done. Tests: `tests/runner` 20 passed; ruff clean; `make test` 397 passed, 1 xfailed.

Files: `vv/runner/{__init__,__main__,config,state,data,alerts,engine,core,report}.py`, `tests/runner/{__init__,conftest,test_runner}.py`, `config/runner.example.yaml`.

Design decisions to check:
- Only `live` signals of the configured venue enter the book; `shadow` signals get targets, shadow P&L and evidence but carry no capital. All four registry signals are `candidate`, so today the runner is a no-op (registry untouched).
- Variant params (venue, shorts, dial) come from the signal's prereg JSON `variants[id]` (the registry schema has no such fields).
- Gap checks: last bar must equal yesterday (UTC), age <= 26h, no holes in last 180 days. Exit codes: 0 ok, 1 error, 2 gap, 3 halted, 4 startup/reconcile halt, 5 live refused.
- Dial inputs (fred, stablecoins, BTC metrics) come from `data/external`, which nothing refreshes; if older than 5 days the dial is skipped and an alert is raised. Funding archive ends 2026-10-31, so perp shadow funding is 0 and flagged (`funding_missing`).
- Shadow P&L is forward-only (from the runner's stored targets), minus `shadow_cost_bps` on turnover. Spot testnet is refused (vv.exec has none).
- Dry orders are sized from `dry_equity` (10000) vs. the `positions` table.

Dry run (real public data, all four signals temporarily `shadow` via a scratchpad registry copy; the real registry, all `candidate`, gives "no signals", no orders):
# Daily run 2026-11-17 (dry, venue spot)

Status: **ok**

## Evidence
| signal | status | variant | post mean | post sd | P(SR>0) | fwd n | 1/4-Kelly | decay | final weight |
|---|---|---|---|---|---|---|---|---|---|
| S1 | shadow | spot | 0.272 | 0.402 | 0.751 | 0 | 0.000 | ok | 0.0000 |
| S2 | shadow | spot/dial | 0.259 | 0.400 | 0.741 | 0 | 0.000 | ok | 0.0000 |
| P1 | shadow | perp/short | 0.213 | 0.394 | 0.705 | 0 | 0.000 | ok | 0.0000 |
| P2 | shadow | perp/short/dial | 0.173 | 0.390 | 0.671 | 0 | 0.000 | ok | 0.0000 |

## Targets vs held
| symbol | close | target weight | target qty | held qty |
|---|---|---|---|---|
| BTC/USDT | 64882.87 | 0.0000 | 0.000000 | 0.000000 |
| ETH/USDT | 1820.67 | 0.0000 | 0.000000 | 0.000000 |
| SOL/USDT | 71.58 | 0.0000 | 0.000000 | 0.000000 |
| XRP/USDT | 0.9553 | 0.0000 | 0.000000 | 0.000000 |
| DOGE/USDT | 0.05893 | 0.0000 | 0.000000 | 0.000000 |
| LINK/USDT | 9.142 | 0.0000 | 0.000000 | 0.000000 |

## Orders
No orders.

## Shadow P&L (daily, signal at weight 1)
| signal | last date | pnl | days |
|---|---|---|---|
| S1 | - | no realised days yet | 0 |
| S2 | - | no realised days yet | 0 |
| P1 | - | no realised days yet | 0 |
| P2 | - | no realised days yet | 0 |

## Alerts
None.
