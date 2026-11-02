# WP-H0A Environment, tooling, safety, quarantine
Model: haiku · Depends: none · Worktree: no (shared tree; don't touch `research/`, `scripts/fetch_klines.py`, `data/`) · Budget: 60k · Commit: NO

Do the steps in order. Each step is mechanical. If one needs a design decision, stop and report it.

## 1. Environment
- `uv venv .venv --python 3.12`
- Create `requirements-live.in`: `ccxt`, `pandas`, `numpy<2.3`, `python-dotenv`, `pyarrow`, `pydantic>=2`
- Create `requirements-dev.in`: `-r requirements-live.in`, `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, `graphifyy==0.9.101`, `scipy`
- Compile with hashes: `uv pip compile --generate-hashes requirements-live.in -o requirements-live.txt --python-version 3.12`, and the same for dev → `requirements-dev.txt`
- `git mv requirements.txt legacy/requirements-legacy.txt` (the Dockerfile will be redone in wave 3; leave it alone)
- `scripts/bootstrap.sh`: `set -euo pipefail`; create `.venv` if missing; `uv pip install --python .venv/bin/python -r requirements-dev.txt`
- `Makefile` targets: `bootstrap`, `test` (`.venv/bin/pytest -q`), `lint` (`.venv/bin/ruff check src main.py tests`), `graph` (`.venv/bin/graphify update .`)
- `pyproject.toml` with only `[tool.ruff]` `target-version="py312"` and `[tool.ruff.lint] select=["E9","F63","F7","F82"]`. Correctness errors only; don't reformat anything.

## 2. Safety
- `config.json`: set `"live_mode": false`. Change nothing else.
- `.gitignore`: anchor `data/`, `models/`, `logs/`, `logs-test/`, `data-test/`, `mlruns/` with a leading `/` (e.g. `/data/`), so `src/data` and `src/models` are no longer ignored.
- `git rm docs/backtest_results.png`. In `README.md`, delete the `## 📈 Comprehensive Backtest Results` section (through its `---`) and the "Current Performance" bullets. Put this line directly under the title: `> ⚠️ Under rework: earlier AI/performance claims were unsubstantiated and have been removed. See docs/PRODUCTION_PLAN.md.`

## 3. Quarantine dead code → `legacy/` (use `git mv`, keep relative paths)
- `src/live/binance_bot.py`, `src/live/trader.py`, `src/live/portfolio_manager.py`, `src/live/momentum_allocator.py`, `src/live/recursive_grid.py`
- `src/backtest/` (whole dir), `src/features/news_embedding.py`, `src/utils/db.py`, `src/utils/create_notebook.py`, `optimizer.py`, `config_opt.json`, `tools/optimizer.py`, `tools/allocate_capital.py`, `notebooks/`
- `tests/test_live_logic.py`, `tests/test_trading_scenarios.py`, `tests/test_portfolio_manager.py`
- Every file in `scripts/` that imports any quarantined module or `src.models` / `src.data` (find them with grep) → `legacy/scripts/`. Keep `check_balance.py` and `fetch_klines.py` in `scripts/`.
- If `src/features/indicators.py` is imported only by quarantined code, move it too.
- `pytest.ini`: add `norecursedirs = legacy .venv graphify-out data`
- `src/live/async_runner.py`: delete the function `run_portfolio_manager_async` (it lazily imports the moved PortfolioManager). Change nothing else in that file.
- `main.py`: remove `import mlflow`, the `run_pipeline` function, and the `train` mode (choices become `live`, `dashboard`). Keep the live and dashboard behaviour identical.

## 4. Stale test fixture
`tests/test_smart_grid.py` fails because the fixture never sets `bot.adx` / `bot.ema_slope` (see `async_runner.py:606`). In the fixture only, set every attribute `process_symbol` reads that's missing (e.g. `bot.adx = 20.0`, `bot.ema_slope = 0.0`). Don't edit production code. If the tests still fail after one try, leave them failing and report the cause.

## 5. Verify
`bash scripts/bootstrap.sh && make lint && make test && make graph`

## Done when
All four commands run. Lint errors and test failures are reported, not hidden. `python main.py --help` works in `.venv`.

## Reply (≤120 words)
status · files moved/created (counts) · lint result · test pass/fail line · graph node count · anything you couldn't do mechanically
