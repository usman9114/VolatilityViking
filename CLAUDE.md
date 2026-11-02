# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

VolatilityViking is a crypto trading bot for Binance spot, written in Python. The active code is a **Passivbot-style grid bot** that trades several symbols at once (`src/live/async_runner.py`). Most of the README describes the **older ETH "Metamodel" pipeline** (LSTM + XGBoost, walk-forward training). That pipeline is now legacy, and the parts of it that live in `src/models/` and `src/data/` are **not in git** (see Gotchas).

## Commands

```bash
pip install -r requirements.txt          # Dockerfile installs CPU-only torch first, then the rest
pip install pytest pytest-asyncio        # not in requirements.txt; pytest.ini uses asyncio_mode=auto

pytest                                    # all tests (testpaths=tests, -v --tb=short)
pytest tests/test_kill_switch.py          # one file
pytest tests/test_grid_calculation.py::TestGeometricGrid::test_name   # one test

python main.py --mode live --config config.json   # async multi-symbol grid bot (the main production path)
python main.py --mode dashboard                   # streamlit run dashboard.py (port 8501)
python main.py --mode train                       # legacy Metamodel pipeline (needs src/models, data/processed)
python src/live/binance_bot.py [--live] [--config config.json]  # older single-process BinanceBot
python src/live/portfolio_manager.py             # scanner that rewrites config.json symbols
python tools/monitor.py                           # health check over bot logs
```

`main.py` reads `--env-file` (default `.env`) with `override=True`. It needs these environment variables:
- **Mainnet:** `BINANCE_API_KEY`, `BINANCE_SECRET_KEY` (or `BINANCE_SECRET`)
- **Testnet:** `BINANCE_TESTNET_API_KEY`, `BINANCE_TESTNET_SECRET_KEY`
- **Optional:** `CRYPTOPANIC_API_KEY` for news

There is no linter or formatter configured.

## Architecture

**Config-driven live loop.** `config.json` drives everything the live bot does:
- **`live_mode`** picks mainnet or testnet. The `--live` flag only applies to `binance_bot.py`; `main.py` mode reads this key.
- **Symbol lists:** `symbols` (core) and `forager.approved_symbols`.
- **Tuning sections:** `grid_settings`, `smart_grid`, `regime_filter`, `unstuck_settings`, `trailing_settings`, `ema_settings`.

`AsyncMultiSymbolBot.run()` repeats this cycle:
1. Hot-reloads `config.json` every 300s.
2. Fetches prices, positions and orders in parallel with `asyncio.gather`.
3. Recomputes indicators (EMA, ATR, RSI, ADX, EMA slope, GARCH vol) every 60s.
4. Checks the drawdown **kill switch** (`max_drawdown_pct`). If it trips, cancels every order and exits.
5. Runs `process_symbol()` for all symbols in parallel.
6. Sleeps 5s.

Changing `config.json` while the bot runs therefore changes live behaviour within 5 minutes.

**`process_symbol()`** (the core strategy, about 330 lines) runs these steps in order:
1. Adaptive anchoring.
2. Smart grid: RSI panic sell, then trend-adaptive spacing, then `aggressive_mode` overrides, which bypass the GARCH safety minimums.
3. Regime filter, which uses ADX and EMA slope to block buys in strong trends.
4. `auto_unstuck`.
5. Ideal-grid calculation: `calc_geometric_grid` for prices and `calc_martingale_qty` for sizes. The wallet-exposure target is `core_wallet_exposure` for core symbols and `forager_wallet_exposure` for forager symbols.
6. Grid maintenance with "mark & sweep". Orders within `drift_tolerance` are kept and the rest are cancelled, which prevents order churn and stuck orders.
7. Sell side: the trailing stop takes priority over grid sells.

**Portfolio manager → config.** `PortfolioManager` (`src/live/portfolio_manager.py`) ranks coins with `MultiFactorScanner` (volatility, volume, momentum, RSI, funding, OI, spread) and `NewsAnalyzer` (sentence-transformers). It then writes the chosen symbols into `config.json` through `update_config`. The bot picks them up on its next hot reload. `run_portfolio_manager_async` exists, but `main_async` does **not** start it, so it currently runs only standalone.

**Two bot implementations.** `src/live/binance_bot.py` (`BinanceBot`, about 1700 lines) is the earlier single-symbol, synchronous version. The `scripts/verify_*.py` scripts and the Docker testnet commands still use it. `async_runner.py` is the current version, and `main.py --mode live` and the tests target it. If you fix strategy logic in one, check whether the other needs the same fix.

**Backtesters** (`src/backtest/`) are separate re-implementations of the strategy, not shared code:
- `garch_backtester.py`: GARCH-scaled grid. `optimizer.py` and `tools/optimizer.py` (`GeneticOptimizer`) use it with `config_opt.json`.
- `robust_backtester.py`, `event_backtester.py`: event-driven grid.
- `smart_backtester.py`, `backtester.py`: legacy Metamodel.

Strategy changes in `async_runner.py` are **not** automatically reflected in the backtesters.

**Tests** construct `AsyncMultiSymbolBot` against a mocked exchange (`tests/conftest.py`: `mock_exchange`, `sample_config`). They cover grid maths, the kill switch, wallet exposure, the regime filter and smart grid. `conftest.py` puts the repo root on `sys.path`, so imports use `src.live...`.

## Gotchas

- **`.gitignore` has bare `data/` and `models/` entries, so they also match `src/data/` and `src/models/`.** Modules imported from there are not in the repo: `src.data.news_fetcher`, `src.models.metamodel` and others. Importing `portfolio_manager.py`, `binance_bot.py`, `trader.py` or `main.py --mode train` will fail on a fresh clone. `tests/test_portfolio_manager.py` is affected too. To add files there, anchor the ignore rules to the root (`/data/`, `/models/`) or force-add the files.
- `async_runner.py` creates `logs/async_bot_<timestamp>.log` **at import time**, including during tests.
- The README and HOW_TO_GUIDE describe the legacy ETH Metamodel. `src/dashboard/app.py` does not exist; the dashboard is `dashboard.py` in the repo root.
- The checked-in `config.json` has `"live_mode": true`, so `main.py --mode live` trades **real money** by default. Use a copy with `live_mode: false` (passed through `--config`) for testnet.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.
Install the CLI with `pip install graphifyy`. The hooks in `.claude/settings.json` do nothing when it is not installed. The graph was built with `graphify extract . --code-only`, which uses the AST only and needs no API key. Community names are kept in `graphify-out/.graphify_labels.json`.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.
