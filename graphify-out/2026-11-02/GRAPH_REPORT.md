# Graph Report - VolatilityViking  (2026-11-02)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 491 nodes · 984 edges · 23 communities (9 shown, 14 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 55 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `37fd9f9e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Entry Points & Ops Scripts
- Async Multi-Symbol Bot
- Binance Grid Bot & Verifiers
- Portfolio Manager & News
- GARCH Backtest & Optimizer
- Test Fixtures & Kill Switch
- Event/Robust Backtesters & Recursive Grid
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
- Notebook Generator
- VM Setup Script
- Cloud Migration Script

## God Nodes (most connected - your core abstractions)
1. `BinanceBot` - 39 edges
2. `AsyncMultiSymbolBot` - 36 edges
3. `GarchBacktester` - 21 edges
4. `PortfolioManager` - 17 edges
5. `RobustBacktester` - 12 edges
6. `MultiFactorScanner` - 12 edges
7. `TestTradingScenarios` - 12 edges
8. `mock_exchange()` - 12 edges
9. `MomentumAllocator` - 9 edges
10. `EventBacktester` - 9 edges

## Surprising Connections (you probably didn't know these)
- `TestGeometricGrid` --uses--> `AsyncMultiSymbolBot`  [INFERRED]
  tests/test_grid_calculation.py → src/live/async_runner.py
- `TestMartingaleSizing` --uses--> `AsyncMultiSymbolBot`  [INFERRED]
  tests/test_grid_calculation.py → src/live/async_runner.py
- `TestKillSwitch` --uses--> `AsyncMultiSymbolBot`  [INFERRED]
  tests/test_kill_switch.py → src/live/async_runner.py
- `TestWalletExposureLimit` --uses--> `AsyncMultiSymbolBot`  [INFERRED]
  tests/test_kill_switch.py → src/live/async_runner.py
- `TestSmartGrid` --uses--> `AsyncMultiSymbolBot`  [INFERRED]
  tests/test_smart_grid.py → src/live/async_runner.py

## Import Cycles
- None detected.

## Communities (23 total, 14 thin omitted)

### Community 0 - "Entry Points & Ops Scripts"
Cohesion: 0.09
Nodes (20): check(), get_exchange(), main(), run_pipeline(), run_backtest_only(), calculate_stats(), check_balance(), comprehensive_backtest() (+12 more)

### Community 1 - "Async Multi-Symbol Bot"
Cohesion: 0.06
Nodes (6): AsyncMultiSymbolBot, main_async(), bot(), test_ranging_market(), test_strong_downtrend(), test_strong_uptrend()

### Community 2 - "Binance Grid Bot & Verifiers"
Cohesion: 0.07
Nodes (4): test_forager(), test_geometric_grid(), test_recursive_grid_logic(), BinanceBot

### Community 3 - "Portfolio Manager & News"
Cohesion: 0.06
Nodes (6): run_portfolio_manager_async(), NewsAnalyzer, PortfolioManager, run_scheduler(), TestConfigUpdate, TestPositionCalculation

### Community 4 - "GARCH Backtest & Optimizer"
Cohesion: 0.09
Nodes (13): GeneticOptimizer, fetch_and_save(), get_top_50_symbols(), main(), run_backtest(), run_test(), main(), GarchBacktester (+5 more)

### Community 5 - "Test Fixtures & Kill Switch"
Cohesion: 0.05
Nodes (6): mock_exchange(), sample_config(), sample_position(), sample_prices(), TestKillSwitch, TestWalletExposureLimit

### Community 6 - "Event/Robust Backtesters & Recursive Grid"
Cohesion: 0.08
Nodes (8): calculate_realized_pnl(), fetch_all_trades(), load_symbols(), main(), main(), EventBacktester, RobustBacktester, RecursiveGrid

### Community 8 - "Data Processing & Features"
Cohesion: 0.20
Nodes (8): calculate_macd(), calculate_rsi(), create_target(), engineer_eth_features(), engineer_gas_features(), load_data(), main(), process_news()

### Community 13 - "Legacy Trader & Prediction DB"
Cohesion: 0.40
Nodes (5): job(), start_trader(), get_logs(), init_db(), log_prediction()

## Knowledge Gaps
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AsyncMultiSymbolBot` connect `Async Multi-Symbol Bot` to `Portfolio Manager & News`, `Test Fixtures & Kill Switch`, `Grid Calculation Tests`, `Smart Grid / Forager Tests`, `Martingale Sizing Tests`?**
  _High betweenness centrality (0.213) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `AsyncMultiSymbolBot` (e.g. with `TestGeometricGrid` and `TestMartingaleSizing`) actually correct?**
  _`AsyncMultiSymbolBot` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Should `Entry Points & Ops Scripts` be split into smaller, more focused modules?**
  _Cohesion score 0.08525149190110827 - nodes in this community are weakly interconnected._
- **Why does `BinanceBot` connect `Binance Grid Bot & Verifiers` to `Entry Points & Ops Scripts`, `Smart Grid / Forager Tests`?**
  _High betweenness centrality (0.171) - this node is a cross-community bridge._
- **Should `Async Multi-Symbol Bot` be split into smaller, more focused modules?**
  _Cohesion score 0.05639097744360902 - nodes in this community are weakly interconnected._
- **Why does `PortfolioManager` connect `Portfolio Manager & News` to `Entry Points & Ops Scripts`?**
  _High betweenness centrality (0.088) - this node is a cross-community bridge._
- **Should `Binance Grid Bot & Verifiers` be split into smaller, more focused modules?**
  _Cohesion score 0.07205387205387205 - nodes in this community are weakly interconnected._