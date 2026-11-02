# Graph Report - VolatilityViking  (2026-11-02)

## Corpus Check
- 61 files · ~34,230 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: (none) 3, .ipynb 1, .ini 1)

## Summary
- 569 nodes · 1071 edges · 36 communities (15 shown, 21 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 69 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `37fd9f9e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Entry Points & Ops Scripts
- AsyncMultiSymbolBot
- Binance Grid Bot & Verifiers
- Portfolio Manager & News
- GARCH Backtest & Optimizer
- mock_exchange
- RobustBacktester
- Portfolio Manager Internals
- Data Processing & Features
- Trading Scenario Tests
- Grid Calculation Tests
- Technical Indicators
- Smart Grid / Forager Tests
- Legacy Trader & Prediction DB
- Martingale Sizing Tests
- Live Logic Tests
- Monitor Tool
- Smart Backtester
- News Embeddings
- BinanceBot Internals
- ETH Trading Bot - Docker Quick Reference
- VM Setup Script
- Cloud Migration Script
- ETH-AI: Advanced Ethereum Trading Bot
- .process_symbol
- How-To Guide
- EventBacktester
- TestKillSwitch
- TestWalletExposureLimit
- [Unreleased] - 2025-12-26
- CLAUDE.md

## God Nodes (most connected - your core abstractions)
1. `BinanceBot` - 40 edges
2. `AsyncMultiSymbolBot` - 38 edges
3. `GarchBacktester` - 21 edges
4. `PortfolioManager` - 18 edges
5. `Architecture` - 14 edges
6. `MultiFactorScanner` - 13 edges
7. `mock_exchange()` - 13 edges
8. `RobustBacktester` - 12 edges
9. `TestTradingScenarios` - 12 edges
10. `ETH Trading Bot - Docker Quick Reference` - 12 edges

## Surprising Connections (you probably didn't know these)
- `Fixed` --references--> `AsyncMultiSymbolBot`  [INFERRED]
  CHANGELOG.md → src/live/async_runner.py
- `Architecture` --references--> `run_portfolio_manager_async()`  [INFERRED]
  CLAUDE.md → src/live/async_runner.py
- `Architecture` --references--> `BinanceBot`  [INFERRED]
  CLAUDE.md → src/live/binance_bot.py
- `Architecture` --references--> `MultiFactorScanner`  [INFERRED]
  CLAUDE.md → src/live/portfolio_manager.py
- `Architecture` --references--> `NewsAnalyzer`  [INFERRED]
  CLAUDE.md → src/live/portfolio_manager.py

## Import Cycles
- None detected.

## Communities (36 total, 21 thin omitted)

### Community 0 - "Entry Points & Ops Scripts"
Cohesion: 0.07
Nodes (24): check(), calculate_realized_pnl(), fetch_all_trades(), get_exchange(), load_symbols(), main(), run_pipeline(), run_backtest_only() (+16 more)

### Community 1 - "AsyncMultiSymbolBot"
Cohesion: 0.15
Nodes (3): Architecture, AsyncMultiSymbolBot, main_async()

### Community 2 - "Binance Grid Bot & Verifiers"
Cohesion: 0.07
Nodes (4): test_forager(), test_geometric_grid(), test_recursive_grid_logic(), BinanceBot

### Community 3 - "Portfolio Manager & News"
Cohesion: 0.06
Nodes (6): run_portfolio_manager_async(), NewsAnalyzer, PortfolioManager, run_scheduler(), TestConfigUpdate, TestPositionCalculation

### Community 4 - "GARCH Backtest & Optimizer"
Cohesion: 0.09
Nodes (13): GeneticOptimizer, fetch_and_save(), get_top_50_symbols(), main(), run_backtest(), run_test(), main(), GarchBacktester (+5 more)

### Community 5 - "mock_exchange"
Cohesion: 0.12
Nodes (4): mock_exchange(), sample_config(), sample_position(), sample_prices()

### Community 6 - "RobustBacktester"
Cohesion: 0.19
Nodes (3): main(), RobustBacktester, RecursiveGrid

### Community 8 - "Data Processing & Features"
Cohesion: 0.20
Nodes (8): calculate_macd(), calculate_rsi(), create_target(), engineer_eth_features(), engineer_gas_features(), load_data(), main(), process_news()

### Community 13 - "Legacy Trader & Prediction DB"
Cohesion: 0.40
Nodes (5): job(), start_trader(), get_logs(), init_db(), log_prediction()

### Community 20 - "ETH Trading Bot - Docker Quick Reference"
Cohesion: 0.06
Nodes (31): 1️⃣ MAINNET - Conservative Mode (Default), 2️⃣ MAINNET - Aggressive Mode, 3️⃣ TESTNET - Conservative Mode, 4️⃣ TESTNET - Aggressive Mode, 5️⃣ Dashboard Only, 6️⃣ Bot + Dashboard (Recommended), 📋 All Run Commands, "Cannot connect to Docker daemon" (+23 more)

### Community 23 - "ETH-AI: Advanced Ethereum Trading Bot"
Cohesion: 0.10
Nodes (19): 1. Market Data (Binance), 1. Setup, 2. Live Dashboard (Monitoring) 📊, 2. Sentiment Analysis (News), 3. On-Chain Data (Infura), 3. Training Mode (Historical) 🏫, 4. Live Production Mode 🔴, 5. Multi-Coin Support 🌍 (+11 more)

### Community 24 - ".process_symbol"
Cohesion: 0.15
Nodes (4): bot(), test_ranging_market(), test_strong_downtrend(), test_strong_uptrend()

### Community 25 - "How-To Guide"
Cohesion: 0.15
Nodes (12): 1. Backtesting, 2. Live Trading, 3. Training, 4. Remote VM Deployment, Connection, Full Backtest (Existing Model), How-To Guide, Mainnet (REAL MONEY) (+4 more)

### Community 31 - "[Unreleased] - 2025-12-26"
Cohesion: 0.33
Nodes (5): Added, Changed, Changelog, Fixed, [Unreleased] - 2025-12-26

### Community 32 - "CLAUDE.md"
Cohesion: 0.33
Nodes (4): Commands, Gotchas, graphify, What this is

## Knowledge Gaps
- **21 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AsyncMultiSymbolBot` connect `AsyncMultiSymbolBot` to `.get_active_symbols`, `.fetch_all_data`, `.update_indicators`, `Grid Calculation Tests`, `Smart Grid / Forager Tests`, `Martingale Sizing Tests`, `.process_symbol`, `async_runner.py`, `TestKillSwitch`, `.place_limit_order`, `TestWalletExposureLimit`, `[Unreleased] - 2025-12-26`?**
  _High betweenness centrality (0.193) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `AsyncMultiSymbolBot` (e.g. with `Fixed` and `Architecture`) actually correct?**
  _`AsyncMultiSymbolBot` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Should `Entry Points & Ops Scripts` be split into smaller, more focused modules?**
  _Cohesion score 0.06882716049382716 - nodes in this community are weakly interconnected._
- **Why does `Architecture` connect `AsyncMultiSymbolBot` to `CLAUDE.md`, `Binance Grid Bot & Verifiers`, `Portfolio Manager & News`, `GARCH Backtest & Optimizer`, `mock_exchange`, `Portfolio Manager Internals`, `.process_symbol`?**
  _High betweenness centrality (0.151) - this node is a cross-community bridge._
- **Should `Binance Grid Bot & Verifiers` be split into smaller, more focused modules?**
  _Cohesion score 0.06604324956165984 - nodes in this community are weakly interconnected._
- **Why does `BinanceBot` connect `Binance Grid Bot & Verifiers` to `Entry Points & Ops Scripts`, `AsyncMultiSymbolBot`?**
  _High betweenness centrality (0.150) - this node is a cross-community bridge._
- **Should `Portfolio Manager & News` be split into smaller, more focused modules?**
  _Cohesion score 0.059800664451827246 - nodes in this community are weakly interconnected._