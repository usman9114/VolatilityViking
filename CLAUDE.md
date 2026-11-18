# VolatilityViking: agent conventions

Binance **spot + USD-M perpetuals** bot, rebuilt to production grade around an **evidence-weighted signal engine**: trend signals, sized by evidence, long/short at ≤1× gross.
**Start here:** `docs/STATUS.md` (current state and next steps). Then `docs/AI_SIGNAL_PLAN.md` (design and testing plan) and `docs/RUNBOOK.md` (VM operations).
History: `docs/PRODUCTION_PLAN.md`, `docs/IMPLEMENTATION_PLAN.md`, `docs/research/*` (every test and decision). If you were given a work-package brief (`docs/impl/wp/WP-*.md`), do only that.

## Code map (live path)
- `vv/core/`: pure strategy maths. `trend.py` (signals, vol sizing, risk dial), plus the grid-era `decide.py`, `pricing.py`, `risk.py`, `orders.py`, `filters.py`, `types.py`
- `vv/evidence/`: registry, hierarchical prior and posterior, ¼-Kelly weights, factor budgets, ramp cap, CUSUM decay
- `vv/exec/`: Binance perp/spot adapters (one-way, isolated, reduce-only exits, caps), `FakeVenue`, reconcile
- `vv/runner/`: the daily job (`python -m vv.runner --mode dry|testnet|live`), SQLite state, persisted halt, alerts, reports
- `vv/sim/`: `FakeExchange` (grid research), `portfolio_sim.py` (spot/perp, funding, liquidation), replay, metrics
- `research/`: pre-registrations (`*prereg*.json`), study runners, data fetchers. `research/signals/registry.yaml` and `ledger.csv` hold signal status and every hypothesis tested
- `deploy/`: systemd timers, ops drills, `testnet_smoke.py`. `Dockerfile` and `docker-compose.yml` are at the root
- `legacy/`: the old grid bot and fake backtests. Never import from it. `src/live/async_runner.py` is the old grid engine (not the live path anymore)

## Navigate graph-first (saves tokens)
- Build or refresh: `make graph` (= `graphify update .`; no LLM cost, a few seconds). Output goes to `graphify-out/` (gitignored)
- `graphify explain "<symbol>"`, `graphify query "<q>" --context call --budget 1500`, `graphify path "A" "B"`
- Read only the line ranges the graph points to. Read whole files only if you are rewriting them.

## Environment
- Python 3.12 `.venv`: `bash scripts/bootstrap.sh`. Then `make test` (expect 406 passed, 1 xfailed) and `make lint`
- Market data: `scripts/fetch_klines.py` (1m spot) and `research/external_fetch.py` (macro, stablecoins, Binance metrics) write to `data/` (gitignored)
- Keys only in a VM `.env` (chmod 600): `VV_BINANCE_KEY`, `VV_BINANCE_SECRET`, `VV_BINANCE_TEST_VENUE=demo|testnet`, plus Telegram and heartbeat vars. Never in git, chat or logs
- Binance blocks US-hosted cloud containers (451). Live, testnet and demo calls must run on the operator's VM

## Rules
- **Never trade real money without the owner's explicit approval.** Live mode also requires `VV_LIVE_CONFIRM=yes`
- Never print secrets. Never weaken the hard limits: gross ≤ 1×, shorts ≤ 0.5×, isolated margin, reduce-only exits, a persisted halt that is never auto-resumed
- **Research discipline:** pre-register before looking at results. Never edit a prereg after results exist. Count every trial in `research/signals/ledger.csv`. LLM-derived signals are forward-only and risk-off only
- Tests first for any logic change. Done = the brief's "Done when" command is green. Max 2 fix cycles (Haiku: 1), then report
- Commit with explicit paths (`git commit -- <paths>`), never a bare `git commit`. Agents don't commit; the orchestrator merges
- Don't reformat files you aren't otherwise changing

## Agent reply contract (≤120 words)
status · files changed · test line · open questions. Longer notes go in `## Result` in your WP file.
