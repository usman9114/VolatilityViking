# WP-M4 Deployment kit for the operator's VM (Docker, schedule, data refresh, runbook, drills)
Model: haiku · Depends: M1, M2 · Worktree: no (only touch `Dockerfile`, `docker-compose.yml`, `deploy/`, `docs/RUNBOOK.md`, `vv/runner/data.py` (funding function only), `tests/runner/test_funding_fetch.py`, `.dockerignore`) · Budget: 70k · Commit: NO

Every step is mechanical. If something needs a design decision, stop and report it.

## 1. Image
Replace the old root `Dockerfile` (it references removed code) with:
- `python:3.12-slim`, multi-stage
- `pip install --require-hashes -r requirements-live.txt`
- copy only `vv/`, `research/` (needed for prereg and feature builders), `config/runner.example.yaml`, `research/signals/`
- a non-root user `vv`, `WORKDIR /app`
- `ENTRYPOINT ["python","-m","vv.runner"]`
- no secrets in the image

`.dockerignore` excludes `data/`, `.venv/`, `legacy/`, `.git/`, `graphify-out/`, `reports/`, `*.db`, `.env*`.

## 2. Compose and schedule
`docker-compose.yml`:
- **Service `runner`:** volumes `./state:/app/state`, `./data:/app/data`, `./reports:/app/reports`, `./config:/app/config:ro`; `env_file: .env`; read-only rootfs with a tmpfs `/tmp`; log rotation (10 MB × 5).
- **Service `refresh`:** runs `python -m research.external_fetch --incremental`. If that flag doesn't exist, add a minimal `--incremental` that only fetches dates after each file's last date. That's a small mechanical change; keep the existing behaviour as the default.

`deploy/systemd/`:
- `vv-refresh.service` + `.timer` at 23:40 UTC daily
- `vv-runner.service` + `.timer` at 00:05 UTC daily
- each runs `docker compose run --rm <svc> ...`
- `Persistent=true`

## 3. Live funding fetch (VM only)
In `vv/runner/data.py`, add `fetch_funding(symbol, since_ms)`. It calls `https://fapi.binance.com/fapi/v1/fundingRate?symbol=...&startTime=...&limit=1000` with `urllib` and returns a Series of rates indexed by funding time (UTC).
- On any error, return an empty Series. The runner already flags missing funding.
- Wire it where the runner currently sets funding to 0, only when that returns data.
- Test it with a mocked `urlopen` (no network).

## 4. Runbook (`docs/RUNBOOK.md`, plain language, ≤ 2 pages)
- **VM:** in a region where Binance USD-M futures are legal for the operator. Docker and docker compose, NTP (chrony), a firewall with no inbound ports.
- **API key:** a dedicated sub-account. **Withdrawals disabled**, futures enabled, an IP allowlist with only the VM's IP. Testnet keys come first from testnet.binancefuture.com.
- **`.env` variables:** `VV_BINANCE_KEY`, `VV_BINANCE_SECRET`, `VV_TELEGRAM_TOKEN`, `VV_TELEGRAM_CHAT`, `VV_HEARTBEAT_URL`, and `VV_LIVE_CONFIRM` (only for live). `chmod 600`.
- **Stages, each with its exit criteria from `docs/AI_SIGNAL_PLAN.md` §5:** (1) dry for 1 week, (2) testnet for 2 weeks with the ops drills, (3) live shadow, (4) tiny live **only with the owner's approval**.
- **Daily check:** read `reports/YYYY-MM-DD.md`, then the Telegram alerts.
- **Halt handling:** how to read the halt reason, `docker compose run --rm runner resume --reason "..."`, and never resuming without understanding the cause.
- **Updating:** `git pull`, rebuild, run the tests in the container.

## 5. Ops drills (`deploy/drills/*.sh`, each prints PASS/FAIL)
1. Kill the container mid-run → the next run reconciles cleanly.
2. Set a halt → the next run exits 3 with no orders.
3. A bad key → startup refuses (exit 4).
4. Delete yesterday's data bar → the run exits 2 with no orders.
5. Unreachable network → alert, no orders.

Drills 1–2 and 4 must work in dry mode here. Drills 3 and 5 just need to be written; mark them "VM only".

## Done when
- `docker build` succeeds here, if Docker is available. If not, say so and validate the Dockerfile syntax with `hadolint`, if available, or just by reading it.
- The tests pass: `make test`.
- Drills 2 and 4 pass in dry mode.

## Reply (≤120 words)
status · test line · docker build result · drill results
