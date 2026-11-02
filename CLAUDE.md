# VolatilityViking: agent conventions

Binance **spot**, long-only grid bot being rebuilt to production grade.
Plans: `docs/PRODUCTION_PLAN.md` (what and why), `docs/IMPLEMENTATION_PLAN.md` (waves, work packages).
Your task is in the WP brief you were given (`docs/impl/wp/WP-*.md`). Do only that.

## Navigate graph-first (saves tokens)
- `graphify explain "<symbol>"`: callers, callees, tests, file:line (~400 tokens)
- `graphify query "<question>" --context call --budget 1500`
- `graphify path "A" "B"`
- Read only the line ranges the graph points to. Read whole files only if you are rewriting them.
- Rebuild after code changes: `graphify update .` (no LLM cost)

## Environment
- Python 3.12 venv at `.venv` (`bash scripts/bootstrap.sh`, or `make bootstrap`)
- `make test` · `make lint` · `make graph`
- Market data lives in `data/` (gitignored). Never commit data, `.env`, or keys.
- `api.binance.com` is geo-blocked here (451). Use `data.binance.vision` / `data-api.binance.vision` for public data.

## Rules
- Live code path: `main.py --mode live` → `src/live/async_runner.py`. Dead code lives in `legacy/` (don't import from it).
- Never set `live_mode: true`, never place real orders, never print secrets.
- Tests first for any logic change. Done = the brief's "Done when" command is green.
- Max 2 fix cycles on red tests (Haiku: 1), then stop and report.
- Don't reformat files you aren't otherwise changing.
- Don't commit or push unless the brief says so. The orchestrator merges.

## Final reply contract (≤120 words)
status (done / blocked) · files changed · test result line · open questions.
Put longer notes in a `## Result` section appended to your WP file.
