<!-- Snapshot of graphify-out/GRAPH_REPORT.md on 2026-11-18 (2642 nodes). Regenerate with: make graph -->
# Graph Report - VolatilityViking  (2026-11-18)

## Corpus Check
- 211 files · ~165,418 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 16 file(s) not represented in the graph (top: (none) 4, .csv 4, .service 2)

## Summary
- 2642 nodes · 6591 edges · 139 communities (99 shown, 40 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 444 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `24129156`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- os
- BinanceBot
- g0_variance_ratio.py
- fetch_klines.py
- mock_exchange
- ETH Trading Bot - Docker Quick Reference
- GarchBacktester
- TestGeometricGrid
- ETH-AI: Advanced Ethereum Trading Bot
- VolatilityViking quant/backtest review (read-only)
- PortfolioManager
- .process_symbol
- VolatilityViking: Implementation Plan (multi-agent, token-conservative)
- test_filters.py
- .run
- How-To Guide
- TestTradingScenarios
- MultiFactorScanner
- PlaceOrder
- FeatureEngineer
- RobustBacktester
- EventBacktester
- TestKillSwitch
- Sam, opening position: grid vs ML
- pytest
- VolatilityViking live engine review (read-only)
- external_fetch.py
- test_risk.py
- TestMartingaleSizing
- [Unreleased] - 2025-12-26
- FakeExchange
- setup_vm.sh
- debate_maya_r1.md
- debate_maya_r2.md
- debate_maya_r3.md
- debate_sam_r3.md
- migrate_to_cloud.sh
- WP-H0A Environment, tooling, safety, quarantine
- WP-H0B Historical 1m klines downloader
- TestPositionCalculation
- WP-S0C G0 edge pre-check: does ETH/BTC mean-revert at grid horizons?
- VolatilityViking: agent conventions
- smart_backtester.py
- NewsEmbedder
- PrintLogger
- bootstrap.sh
- numpy
- test_portfolio_manager.py
- NewsAnalyzer
- test_fetch_klines.py
- G0 results: which coins mean-revert at grid horizons?
- paired_grid.py
- decide
- WP-S0E Funding carry research (research only)
- WP-S0D G0-robust rerun
- g1_run.py
- test_features.py
- dashboard.py
- TestWalletExposureLimit
- pytest
- binance_bot.py
- pandas
- argparse
- pandas
- types.py
- time
- PlaceOrder
- math
- carry_analysis.py
- G1 results
- g1_diagnostics.py
- test_evidence_engine.py
- WP-S1.3 FakeExchange: multi-coin, shared wallet, honest fills
- test_replay.py
- OrderRejected
- process_data.py
- core.py
- run_many
- engine.py
- refresh
- test_exec_adapters.py
- Sam, round 2: rebuttal to Maya
- State
- test_runner.py
- Funding carry research (WP-S0E)
- common.py
- test_trend.py
- test_portfolio_sim.py
- CcxtPerpClient
- registry.py
- CcxtSpotClient
- test_diagnostics.py
- runner.example.yaml
- SymbolFilters
- data.py
- datetime
- signals
- test_funding_fetch.py
- SpotAdapter
- explore_timing.py
- last_week.py
- client_id
- VolatilityViking runbook (operator)
- Evidence-Weighted Signal Engine (EWSE): plan and testing plan
- carry_fetch.py
- TracingExchange
- WP-R2 Feature screen: does external data add predictive power?
- VolatilityViking - Production Readiness Review (non-strategy)
- load_month
- WP-M3 Evidence engine: hierarchical prior, posterior, ¼-Kelly weights, factor budgets, decay monitor
- WP-M4 Deployment kit for the operator's VM (Docker, schedule, data refresh, runbook, drills)
- WP-S1.7 G1 diagnostics: exposure audit, fill-depth sweep, VR attribution
- Arthur Hayes: Advisory Profile
- G2 results: trend + vol targeting, spot vs perpetuals
- WP-S1.1 Core types and ports (multi-coin, shared wallet)
- WP-S1.5 Ladder hysteresis and unique order ids (pre-G1 fixes)
- WP-S1.6 Strategy variants B (paired grid) and C (rebalancing) for G1 v2
- Feature screen: does external data add predictive power?
- lib.sh
- WP-R1 External data fetchers (daily)
- WP-T1 Trend + vol-targeting strategy and risk-off dial (pure functions)
- Last week (09 Nov to 15 Nov 2026): what happened and did we see it coming
- Market timing: hindsight vs rules you could actually follow
- io
- SecretRedactingFilter
- reconcile
- OrderRejected
- 01_kill_mid_run.sh
- 02_halt_exit3.sh
- 03_bad_key_exit4.sh
- 04_missing_bar_exit2.sh
- 05_unreachable_network_alert.sh
- g1_preregistration_addendum.md

## God Nodes (most connected - your core abstractions)
1. `PlaceOrder` - 59 edges
2. `decide()` - 52 edges
3. `FakeExchange` - 50 edges
4. `CancelOrder` - 45 edges
5. `SymbolFilters` - 42 edges
6. `RiskParams` - 41 edges
7. `BinanceBot` - 39 edges
8. `AsyncMultiSymbolBot` - 39 edges
9. `run_replay()` - 39 edges
10. `State` - 36 edges

## Surprising Connections (you probably didn't know these)
- ``research/g1_run.py`` --references--> `gate_g1()`  [INFERRED]
  docs/impl/wp/WP-S14-replay-g1.md → research/g1_run.py
- `Fixed` --references--> `AsyncMultiSymbolBot`  [INFERRED]
  CHANGELOG.md → src/live/async_runner.py
- `1. Architecture` --references--> `AsyncMultiSymbolBot`  [INFERRED]
  docs/review/live_engine_review.md → src/live/async_runner.py
- `6. Code quality` --references--> `AsyncMultiSymbolBot`  [INFERRED]
  docs/review/live_engine_review.md → src/live/async_runner.py
- `Result` --references--> `client_id()`  [INFERRED]
  docs/impl/wp/WP-S13-fake-exchange.md → vv/core/ids.py

## Import Cycles
- None detected.

## Communities (139 total, 40 thin omitted)

### Community 0 - "os"
Cohesion: 0.11
Nodes (3): create_notebook(), check_balance(), comprehensive_backtest()

### Community 1 - "BinanceBot"
Cohesion: 0.07
Nodes (5): simulate(), test_forager(), test_geometric_grid(), test_recursive_grid_logic(), BinanceBot

### Community 2 - "g0_variance_ratio.py"
Cohesion: 0.05
Nodes (42): analyse_symbol(), dfa_hurst(), dfa_scales(), drop_listing_days(), evaluate_g0(), exclude_windows(), final_verdicts(), holm_adjust() (+34 more)

### Community 3 - "fetch_klines.py"
Cohesion: 0.21
Nodes (10): checksum_url(), ChecksumError, download_month(), DownloadError, _get(), month_range(), parse_checksum(), verify_checksum() (+2 more)

### Community 4 - "mock_exchange"
Cohesion: 0.12
Nodes (4): mock_exchange(), sample_config(), sample_position(), sample_prices()

### Community 5 - "ETH Trading Bot - Docker Quick Reference"
Cohesion: 0.06
Nodes (31): 1️⃣ MAINNET - Conservative Mode (Default), 2️⃣ MAINNET - Aggressive Mode, 3️⃣ TESTNET - Conservative Mode, 4️⃣ TESTNET - Aggressive Mode, 5️⃣ Dashboard Only, 6️⃣ Bot + Dashboard (Recommended), 📋 All Run Commands, "Cannot connect to Docker daemon" (+23 more)

### Community 6 - "GarchBacktester"
Cohesion: 0.10
Nodes (10): GeneticOptimizer, run_backtest(), run_test(), main(), GarchBacktester, MomentumAllocator, main(), evaluate_config() (+2 more)

### Community 8 - "ETH-AI: Advanced Ethereum Trading Bot"
Cohesion: 0.12
Nodes (16): 1. Market Data (Binance), 1. Setup, 2. Live Dashboard (Monitoring) 📊, 2. Sentiment Analysis (News), 3. On-Chain Data (Infura), 3. Training Mode (Historical) 🏫, 4. Live Production Mode 🔴, 5. Multi-Coin Support 🌍 (+8 more)

### Community 9 - "VolatilityViking quant/backtest review (read-only)"
Cohesion: 0.12
Nodes (14): 1. Fidelity: backtest vs live (async_runner.process_symbol and friends), 2. Fill model, 3. Biases and overfitting, 4. Metrics, 5. GARCH, 6. docs/backtest_results.png, 7. Is there PnL evidence?, 8. README +1,395% / LSTM + XGBoost (+6 more)

### Community 11 - ".process_symbol"
Cohesion: 0.15
Nodes (4): bot(), test_ranging_market(), test_strong_downtrend(), test_strong_uptrend()

### Community 12 - "VolatilityViking: Implementation Plan (multi-agent, token-conservative)"
Cohesion: 0.07
Nodes (28): 1. Model routing, 2. Token-saving system, 3. Wave 0: bootstrap (Haiku, ~1 session), 4. Work packages and waves, 5. Orchestration loop (per wave), 6. Token budget, 7. Templates, 8. Human tasks (can't be delegated) (+20 more)

### Community 13 - "test_filters.py"
Cohesion: 0.15
Nodes (20): test_buy_price_rounds_down(), test_invalid_side_and_nonpositive_increment_rejected(), test_min_notional_rejected_after_rounding(), test_min_qty_rejected(), test_notional_exact_boundary_passes_using_decimal(), test_notional_ok_after_rounding(), test_price_decimal_edge_0_1_plus_0_2(), test_price_float_tick_drift_does_not_move_exact_multiple() (+12 more)

### Community 15 - "How-To Guide"
Cohesion: 0.15
Nodes (12): 1. Backtesting, 2. Live Trading, 3. Training, 4. Remote VM Deployment, Connection, Full Backtest (Existing Model), How-To Guide, Mainnet (REAL MONEY) (+4 more)

### Community 16 - "TestTradingScenarios"
Cohesion: 0.10
Nodes (10): Done when, Goal, Reply (≤150 words), `research/g2_run.py`, Tests first (`tests/sim/test_portfolio_sim.py`), `vv/sim/portfolio_sim.py`, WP-T2 Spot and perpetual-futures portfolio simulator, and the G2 runner, TestBotLogic (+2 more)

### Community 18 - "PlaceOrder"
Cohesion: 0.21
Nodes (18): _open(), test_cancels_all_ours_when_nothing_desired_regime_blocks(), test_guard_allows_non_post_only_sell_below_bid(), test_guard_drops_post_only_buy_at_or_above_bid_and_sell_at_or_below_bid(), test_guard_rounds_then_drops_below_min_notional(), test_identical_order_is_kept_never_cancelled_and_replaced(), test_ignores_non_vv_orders(), test_keeps_matching_order_within_tolerance() (+10 more)

### Community 20 - "RobustBacktester"
Cohesion: 0.19
Nodes (3): main(), RobustBacktester, RecursiveGrid

### Community 23 - "Sam, opening position: grid vs ML"
Cohesion: 0.25
Nodes (7): 1. Thesis, 2. Where grids make money and where they die, 3. What "AI" was, and what ML costs, 4. What I'd build, 5. Concessions, 6. Questions for Maya, Sam, opening position: grid vs ML

### Community 24 - "pytest"
Cohesion: 0.18
Nodes (36): at(), bars(), buy(), flat(), _grid_rows(), make(), reason(), _run() (+28 more)

### Community 25 - "VolatilityViking live engine review (read-only)"
Cohesion: 0.12
Nodes (9): 1. Architecture, 2. Risk exposure, 3. Correctness bugs (async_runner.py unless noted), 4. State & recovery, 5. Observability, 6. Code quality, Ranked top 15, VolatilityViking live engine review (read-only) (+1 more)

### Community 26 - "external_fetch.py"
Cohesion: 0.06
Nodes (41): aggregate_metrics(), build(), epoch_to_utc(), _fetch_fred(), _fetch_funding(), _fetch_metrics(), _fetch_premium(), FetchError (+33 more)

### Community 27 - "test_risk.py"
Cohesion: 0.20
Nodes (18): Navigate, _mkt(), _oo(), _pf(), test_blocked_symbol_latched_and_accumulates(), test_flatten_cancels_vv_orders_and_sells_free_qty_below_bid(), test_flatten_dust_not_sold(), test_flatten_expired_ioc_is_reissued_with_new_client_id() (+10 more)

### Community 29 - "[Unreleased] - 2025-12-26"
Cohesion: 0.33
Nodes (5): Added, Changed, Changelog, Fixed, [Unreleased] - 2025-12-26

### Community 30 - "FakeExchange"
Cohesion: 0.16
Nodes (4): FakeExchange, run_replay(), _sync(), _ts()

### Community 37 - "WP-H0A Environment, tooling, safety, quarantine"
Cohesion: 0.12
Nodes (10): 1. Environment, 2. Safety, 3. Quarantine dead code → `legacy/` (use `git mv`, keep relative paths), 4. Stale test fixture, 5. Verify, Done when, Reply (≤120 words), Result (+2 more)

### Community 38 - "WP-H0B Historical 1m klines downloader"
Cohesion: 0.25
Nodes (7): Done when, Goal, Reply (≤120 words), Run, Spec, Test (write it first), WP-H0B Historical 1m klines downloader

### Community 40 - "WP-S0C G0 edge pre-check: does ETH/BTC mean-revert at grid horizons?"
Cohesion: 0.29
Nodes (6): Build `research/g0_variance_ratio.py`, Done when, Reply (≤120 words), Tests (write first) `tests/research/test_vr.py`, Why, WP-S0C G0 edge pre-check: does ETH/BTC mean-revert at grid horizons?

### Community 41 - "VolatilityViking: agent conventions"
Cohesion: 0.33
Nodes (5): Environment, Final reply contract (≤120 words), Navigate graph-first (saves tokens), Rules, VolatilityViking: agent conventions

### Community 46 - "numpy"
Cohesion: 0.08
Nodes (42): auc_score(), basket_index(), bh_fdr(), build_all(), build_features(), derivs(), _build_section(), build_targets() (+34 more)

### Community 49 - "test_fetch_klines.py"
Cohesion: 0.14
Nodes (4): _checksum_text(), TestChecksum, TestHelpers, TestNormaliseOpenTime

### Community 50 - "G0 results: which coins mean-revert at grid horizons?"
Cohesion: 0.08
Nodes (22): Caveats raised, Decision, G0 decision (2026-11-02), Result, AVAXUSDT: variance ratio, pooled (5m base), BNBUSDT: variance ratio, pooled (5m base), BTCUSDT: variance ratio, pooled (5m base), DFA Hurst exponent (5m returns, 1h-1d scales; baseline) (+14 more)

### Community 51 - "paired_grid.py"
Cohesion: 0.07
Nodes (29): Result, Variant B: `vv/core/variants/paired_grid.py`, test_budget_remaining_never_negative(), test_budgets_equal_sigma_equal_split_under_cap(), test_budgets_inverse_vol_split_sums_to_pool(), test_budgets_nonpositive_equity_or_empty(), test_budgets_per_symbol_cap_applies_to_single_symbol(), test_budgets_zero_sigma_is_finite() (+21 more)

### Community 52 - "decide"
Cohesion: 0.18
Nodes (37): _apply(), _buy_exposure(), _cancels(), _feat(), _mkt(), _oo(), _places(), _run_cycles() (+29 more)

### Community 53 - "WP-S0E Funding carry research (research only)"
Cohesion: 0.22
Nodes (8): Analysis (`research/carry_analysis.py`), Data (public, `data.binance.vision`), Done when, Gate CARRY-0 (literal), Reply (≤120 words), Tests first (`tests/research/test_carry.py`), Why, WP-S0E Funding carry research (research only)

### Community 54 - "WP-S0D G0-robust rerun"
Cohesion: 0.25
Nodes (7): Changes to `research/g0_variance_ratio.py`, Done when, Reply (≤120 words), Run, Tests first (`tests/research/test_vr.py`, extend), Why, WP-S0D G0-robust rerun

### Community 55 - "g1_run.py"
Cohesion: 0.11
Nodes (25): baseline_stats(), build_splits(), build_trials(), filters_for(), gate_g1(), gate_passes(), load_klines(), main() (+17 more)

### Community 56 - "test_features.py"
Cohesion: 0.05
Nodes (49): Alerts, Config (`config/runner.example.yaml`), Daily run 2026-11-17 (dry, venue spot), Done when, Evidence, Goal, Orders, Reply (≤120 words) (+41 more)

### Community 57 - "dashboard.py"
Cohesion: 0.12
Nodes (4): calculate_realized_pnl(), fetch_all_trades(), get_exchange(), load_symbols()

### Community 59 - "pytest"
Cohesion: 0.07
Nodes (47): 1. `vv/core/features.py`, 2. `vv/core/params.py` (extend `GridParams`), 3. `vv/core/pricing.py`, 4. `vv/core/regime.py`, Context, Done when, Reply (≤120 words), Tests first (+39 more)

### Community 60 - "binance_bot.py"
Cohesion: 0.22
Nodes (7): run_backtest_only(), fetch_live_data(), get_feature_schema(), get_model_path(), inference(), test_pipeline(), Colors

### Community 61 - "pandas"
Cohesion: 0.13
Nodes (22): check_no_lookahead(), scramble(), calculate_stats(), _cal(), _synthetic_inputs(), test_auc_hand_example(), test_bh_fdr_step_up_and_nan(), test_decision_rule_needs_both_criteria() (+14 more)

### Community 63 - "argparse"
Cohesion: 0.22
Nodes (5): check(), main(), main(), test_cryptocompare(), test_cryptopanic()

### Community 64 - "pandas"
Cohesion: 0.40
Nodes (5): job(), start_trader(), get_logs(), init_db(), log_prediction()

### Community 65 - "types.py"
Cohesion: 0.09
Nodes (7): Create, _matches(), ExchangeAdapter, StateStore, CancelOrder, OpenOrder, SymbolMarket

### Community 66 - "time"
Cohesion: 0.26
Nodes (7): fetch_and_save(), get_top_50_symbols(), main(), fetch_data(), fetch_month_data(), get_latest_log(), parse_log()

### Community 67 - "PlaceOrder"
Cohesion: 0.13
Nodes (32): csnap(), _feat(), level(), _oo(), Sim, test_b_buy_fill_places_sell_one_level_up_and_sell_fill_rearms_buy(), test_b_cold_symbol_cancels_buys_and_places_nothing(), test_b_does_not_recentre_while_holding() (+24 more)

### Community 68 - "math"
Cohesion: 0.05
Nodes (59): Result, 1. Exposure audit (2021), 2. Fill-depth sweep, 3. VR attribution, 4. Cash benchmark (4% a year in USDT), G1 diagnostics (WP-S1.7), benchmarks(), daily_close_from_spot() (+51 more)

### Community 69 - "carry_analysis.py"
Cohesion: 0.07
Nodes (38): annualised_funding(), basis_table(), exit_state(), funding_table(), funding_to_grid(), gate_carry0(), load_all(), load_perp_1h() (+30 more)

### Community 70 - "G1 results"
Cohesion: 0.05
Nodes (38): Decisions, G1 v1 decision (2026-11-03), Verdict on record, What the advisors agreed, G1 results, Gate G1, Notes, Selected trial T09 step 0.0300 / 1.5s vs baselines (base costs) (+30 more)

### Community 71 - "g1_diagnostics.py"
Cohesion: 0.09
Nodes (21): attribution_stats(), conclusions(), _f(), holdings_from_fills(), main(), ols(), ols_multi(), _p() (+13 more)

### Community 72 - "test_evidence_engine.py"
Cohesion: 0.09
Nodes (24): _run(), test_cusum_regime_break_ok_halve_zero(), test_cusum_stationary_false_alarm_rate(), test_factor_budget_caps_sum(), test_hierarchical_prior_fallback_and_shrinkage(), test_incremental_alpha_recovers_residual(), test_kelly_zero_below_min_p(), test_p_positive() (+16 more)

### Community 73 - "WP-S1.3 FakeExchange: multi-coin, shared wallet, honest fills"
Cohesion: 0.22
Nodes (8): Done when, Goal, Navigate, Reply (≤120 words), Result, Spec, Tests first (`tests/sim/test_fake_exchange.py`), WP-S1.3 FakeExchange: multi-coin, shared wallet, honest fills

### Community 74 - "test_replay.py"
Cohesion: 0.11
Nodes (23): bars(), run(), synth(), test_bad_window_raises(), test_exposure_never_exceeds_total_cap(), test_it_trades_and_accounts(), test_latched_halt_stays_latched_and_flattens(), shock() (+15 more)

### Community 75 - "OrderRejected"
Cohesion: 0.18
Nodes (10): Done when, Goal, Navigate, Reply (≤120 words), `research/g1_run.py`, Run, Tests first, `vv/sim/metrics.py` (+2 more)

### Community 76 - "process_data.py"
Cohesion: 0.22
Nodes (8): calculate_macd(), calculate_rsi(), create_target(), engineer_eth_features(), engineer_gas_features(), load_data(), main(), process_news()

### Community 77 - "core.py"
Cohesion: 0.10
Nodes (14): test_heartbeat_posted_when_url_set(), test_no_telegram_env_means_log_only(), test_redaction_everywhere(), Alerter, install_log_redaction(), redact(), load_config(), RunnerConfig (+6 more)

### Community 78 - "run_many"
Cohesion: 0.24
Nodes (8): Result, test_run_many_matches_serial_and_is_order_independent(), test_strategy_runs_are_deterministic_and_picklable_through_run_many(), precompute_features(), ReplayJob, ReplayResult, _run_job(), run_many()

### Community 79 - "engine.py"
Cohesion: 0.09
Nodes (17): Result, test_apply_caps_random(), test_combine_book_scales_big_weights_down(), Signal, Caps, fetch_funding(), active_signals(), apply_caps() (+9 more)

### Community 80 - "refresh"
Cohesion: 0.06
Nodes (34): driver, options, max-file, max-size, build, command, entrypoint, image (+26 more)

### Community 81 - "test_exec_adapters.py"
Cohesion: 0.13
Nodes (24): perp(), test_adapter_treats_duplicate_as_already_submitted(), test_below_min_notional_skipped(), test_clip_caps_hold_random(), test_dry_run_places_nothing(), test_exits_and_reductions_reduce_only_entries_not(), test_fake_liq_price_and_funding(), test_flip_splits_reduce_then_entry() (+16 more)

### Community 82 - "Sam, round 2: rebuttal to Maya"
Cohesion: 0.33
Nodes (5): (a) Your three questions, (b) Agreement and disagreement, (c) Too heavy for one operator, (d) Three go/no-go gates before real money, Sam, round 2: rebuttal to Maya

### Community 83 - "State"
Cohesion: 0.13
Nodes (9): main(), make(), _state(), test_halt_blocks_next_run_and_resume_clears(), test_resume_cli(), run(), finish(), main() (+1 more)

### Community 84 - "test_runner.py"
Cohesion: 0.18
Nodes (22): factory_for(), fake_venue(), frame(), make_cfg(), make_fetcher(), synth_close(), write_repo(), cfg() (+14 more)

### Community 86 - "common.py"
Cohesion: 0.14
Nodes (6): Balance, client_order_id(), OrderResult, PositionInfo, Rebalancer, VenueClient

### Community 87 - "test_trend.py"
Cohesion: 0.17
Nodes (20): build_weights(), run_variant(), _features(), _panel(), test_band(), test_caps_property(), test_dial_all_penalties_hit_floor(), test_dial_missing_no_penalty() (+12 more)

### Community 88 - "test_portfolio_sim.py"
Cohesion: 0.15
Nodes (17): perp_bars(), spot_bars(), test_band_suppresses_small_changes(), test_deterministic(), test_execution_strictly_after_decision(), test_perp_equity_continuous_across_funding_without_position(), test_perp_funding_sign_short_receives_long_pays(), test_perp_liquidation_on_adverse_wick_at_3x() (+9 more)

### Community 89 - "CcxtPerpClient"
Cohesion: 0.11
Nodes (7): _load_keys(), main(), _redact(), test_client_repr_has_no_keys(), test_test_venue_selects_endpoint(), CcxtPerpClient, _f()

### Community 90 - "registry.py"
Cohesion: 0.15
Nodes (14): test_ledger_append_only(), test_prereg_hash_mismatch_raises(), hierarchical_prior(), append_ledger(), ledger_srs(), load_ledger(), load_registry(), PreregMismatch (+6 more)

### Community 91 - "CcxtSpotClient"
Cohesion: 0.13
Nodes (4): CcxtSpotClient, _f(), DuplicateOrder, OrderRejected

### Community 92 - "test_diagnostics.py"
Cohesion: 0.14
Nodes (19): parse_yearly_vr(), _buy(), _golden_scenario(), _make(), _rows(), _sell(), _sync(), _synth_bars() (+11 more)

### Community 93 - "runner.example.yaml"
Cohesion: 0.09
Nodes (22): caps, gross, per_coin, short, data_dir, dry_equity, external_dir, external_max_age_days (+14 more)

### Community 95 - "data.py"
Cohesion: 0.18
Nodes (6): DailyStore, detect_gaps(), expected_last_bar(), fetch_daily(), update_store(), wide()

### Community 96 - "datetime"
Cohesion: 0.17
Nodes (3): _split_symbol(), _utc(), VirtualClock

### Community 97 - "signals"
Cohesion: 0.11
Nodes (17): signals, created, data, family, horizon, hypothesis, id, notes (+9 more)

### Community 98 - "test_funding_fetch.py"
Cohesion: 0.16
Nodes (8): _payload(), _Resp, test_any_error_returns_empty_series(), test_funding_with_live_empty_live_returns_none(), test_funding_with_live_falls_back_to_live_fetch(), test_malformed_payload_returns_empty_series(), test_parses_rates_indexed_by_utc_funding_time(), fake_urlopen()

### Community 99 - "SpotAdapter"
Cohesion: 0.17
Nodes (10): Build `vv/exec/`, Done when, Goal, Reply (≤120 words), Result, Testnet smoke (optional, this container), Tests first (`tests/exec/`, no network), WP-M1 Execution adapter: Binance USD-M perps + spot (target-position rebalancer) (+2 more)

### Community 100 - "explore_timing.py"
Cohesion: 0.27
Nodes (7): daily_close(), fmt(), main(), px(), rules(), run_rule(), stats()

### Community 101 - "last_week.py"
Cohesion: 0.21
Nodes (6): fetch(), main(), pct(), px(), stored_daily(), load()

### Community 102 - "client_id"
Cohesion: 0.24
Nodes (8): test_deterministic(), test_invalid_inputs(), test_prefix_length_and_charset(), test_stable_known_value(), test_symbol_separator_variants_do_not_collide_with_other_fields(), test_unique_per_level_side_symbol_bucket(), test_anchor_bucket_stable_and_changes_with_anchor(), client_id()

### Community 103 - "VolatilityViking runbook (operator)"
Cohesion: 0.17
Nodes (11): 1. The VM, 2. API key, 3. `.env` (never commit it), 4. Stages and exit criteria, 5. Schedule, 6. Daily check (about 5 minutes), 7. Exit codes and halts, 8. Updating (+3 more)

### Community 104 - "Evidence-Weighted Signal Engine (EWSE): plan and testing plan"
Cohesion: 0.18
Nodes (10): 0. What changed from v1, and why, 1. Core loop, 2. Evidence-proportional allocation (replaces the 0.95 gate), 3. Signal types are judged on what they're for, 4. AI and LLM rules (leak control), 5. Testing plan, 6. Minimum viable loop (build now), 7. Weekly cadence (operator: ~1 hour a week) (+2 more)

### Community 105 - "carry_fetch.py"
Cohesion: 0.33
Nodes (5): build(), _fetch_zip(), main(), _month(), parse_funding_csv()

### Community 106 - "TracingExchange"
Cohesion: 0.25
Nodes (3): TracingExchange, _Order, _Pos

### Community 107 - "WP-R2 Feature screen: does external data add predictive power?"
Cohesion: 0.20
Nodes (9): Build `research/feature_screen.py`, Done when, Goal, Inputs, Reply (≤150 words), Report (`docs/research/FEATURE_SCREEN.md`, ≤ 2.5 pages, plain language), Result, Tests first (+1 more)

### Community 108 - "VolatilityViking - Production Readiness Review (non-strategy)"
Cohesion: 0.20
Nodes (9): 0. Headline blockers (fix before real money), 1. Failure-mode table, 2. Security, 3. Operability, 4. Deployment, 5. Target architecture (single operator, passive bot), 6. Prioritised fix list (effort: S/M/L), VolatilityViking - Production Readiness Review (non-strategy) (+1 more)

### Community 109 - "load_month"
Cohesion: 0.31
Nodes (5): _atomic_parquet(), build_symbol(), count_gaps(), load_month(), read_zip_csv()

### Community 110 - "WP-M3 Evidence engine: hierarchical prior, posterior, ¼-Kelly weights, factor budgets, decay monitor"
Cohesion: 0.22
Nodes (8): Build `vv/evidence/`, Done when, Goal, Reply (≤120 words), Result, Seed the registry, Tests first (`tests/evidence/`), WP-M3 Evidence engine: hierarchical prior, posterior, ¼-Kelly weights, factor budgets, decay monitor

### Community 111 - "WP-M4 Deployment kit for the operator's VM (Docker, schedule, data refresh, runbook, drills)"
Cohesion: 0.22
Nodes (8): 1. Image, 2. Compose and schedule, 3. Live funding fetch (VM only), 4. Runbook (`docs/RUNBOOK.md`, plain language, ≤ 2 pages), 5. Ops drills (`deploy/drills/*.sh`, each prints PASS/FAIL), Done when, Reply (≤120 words), WP-M4 Deployment kit for the operator's VM (Docker, schedule, data refresh, runbook, drills)

### Community 112 - "WP-S1.7 G1 diagnostics: exposure audit, fill-depth sweep, VR attribution"
Cohesion: 0.22
Nodes (8): Done when, Navigate, Output, Reply (≤120 words), Tasks, Tests first, Why, WP-S1.7 G1 diagnostics: exposure audit, fill-depth sweep, VR attribution

### Community 113 - "Arthur Hayes: Advisory Profile"
Cohesion: 0.22
Nodes (8): 1. Who he is, 2. Core frameworks, 3. Trading style and risk, 4. Views on our topics, 5. "What would Hayes ask?" checklist, 6. Known biases and blind spots, 7. Sources, Arthur Hayes: Advisory Profile

### Community 114 - "G2 results: trend + vol targeting, spot vs perpetuals"
Cohesion: 0.22
Nodes (8): Full-run costs and exposure, G2 results: trend + vol targeting, spot vs perpetuals, Gate G2 per variant, Method notes, Out-of-sample, side by side, Per-split results (pre-registered costs), Verdict, What futures added (P vs S, same signal)

### Community 115 - "WP-S1.1 Core types and ports (multi-coin, shared wallet)"
Cohesion: 0.25
Nodes (7): Done when, Goal, Navigate, Out of scope, Reply (≤120 words), Tests first (`tests/core/`), WP-S1.1 Core types and ports (multi-coin, shared wallet)

### Community 116 - "WP-S1.5 Ladder hysteresis and unique order ids (pre-G1 fixes)"
Cohesion: 0.25
Nodes (7): Changes, Done when, Reply (≤120 words), Result, Tests first, Why, WP-S1.5 Ladder hysteresis and unique order ids (pre-G1 fixes)

### Community 117 - "WP-S1.6 Strategy variants B (paired grid) and C (rebalancing) for G1 v2"
Cohesion: 0.25
Nodes (7): Done when, Goal, Replay and runner, Reply (≤120 words), Tests first, Variant C: `vv/core/variants/rebalance.py`, WP-S1.6 Strategy variants B (paired grid) and C (rebalancing) for G1 v2

### Community 118 - "Feature screen: does external data add predictive power?"
Cohesion: 0.25
Nodes (7): Does adding the group improve out-of-sample prediction?, Event study: BTC's 8 largest non-overlapping 7-day drawdowns, Feature screen: does external data add predictive power?, Features that survive FDR OOS (9 of 208 tests), Limitations, Verdict, What to build

### Community 120 - "WP-R1 External data fetchers (daily)"
Cohesion: 0.29
Nodes (6): Done when, Goal, Reply (≤120 words), Rules, Tests first, WP-R1 External data fetchers (daily)

### Community 121 - "WP-T1 Trend + vol-targeting strategy and risk-off dial (pure functions)"
Cohesion: 0.29
Nodes (6): Done when, Goal, Interface (`vv/core/trend.py`), Reply (≤100 words), Tests first (`tests/core/test_trend.py`), WP-T1 Trend + vol-targeting strategy and risk-off dial (pure functions)

### Community 122 - "Last week (09 Nov to 15 Nov 2026): what happened and did we see it coming"
Cohesion: 0.29
Nodes (6): BTC's 5 biggest hourly moves, Did we see it coming? Signals as of the close before the week, How it happened: BTC day by day (UTC), Last week (09 Nov to 15 Nov 2026): what happened and did we see it coming, Reported causes (press, not verified against primary data), What happened

### Community 123 - "Market timing: hindsight vs rules you could actually follow"
Cohesion: 0.33
Nodes (5): 1. What the market did since 2025, 2. BTC since 1 Jan 2025: what perfect hindsight would have earned, 3a. Rules you could actually follow, 2019 to Oct 2026, 3b. Rules you could actually follow, 2025 to Oct 2026, Market timing: hindsight vs rules you could actually follow

### Community 126 - "reconcile"
Cohesion: 0.40
Nodes (3): test_reconcile(), Diff, reconcile()

## Knowledge Gaps
- **1 isolated node(s):** `HTTPS_PROXY`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1034 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **40 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AsyncMultiSymbolBot` connect `.run` to `os`, `WP-H0A Environment, tooling, safety, quarantine`, `TestGeometricGrid`, `.process_symbol`, `TestKillSwitch`, `VolatilityViking live engine review (read-only)`, `TestWalletExposureLimit`, `TestMartingaleSizing`, `[Unreleased] - 2025-12-26`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **Are the 19 inferred relationships involving `PlaceOrder` (e.g. with `_places()` and `test_ignores_non_vv_orders()`) actually correct?**
  _`PlaceOrder` has 19 INFERRED edges - model-reasoned connections that need verification._
- **What connects `HTTPS_PROXY` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `os` be split into smaller, more focused modules?**
  _Cohesion score 0.11333333333333333 - nodes in this community are weakly interconnected._
- **Why does `FakeExchange` connect `FakeExchange` to `datetime`, `types.py`, `PlaceOrder`, `g1_diagnostics.py`, `WP-S1.3 FakeExchange: multi-coin, shared wallet, honest fills`, `TracingExchange`, `OrderRejected`, `VolatilityViking: Implementation Plan (multi-agent, token-conservative)`, `WP-S1.7 G1 diagnostics: exposure audit, fill-depth sweep, VR attribution`, `Sam, round 2: rebuttal to Maya`, `decide`, `pytest`, `test_risk.py`, `test_diagnostics.py`, `SymbolFilters`, `OrderRejected`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Should `BinanceBot` be split into smaller, more focused modules?**
  _Cohesion score 0.06954887218045112 - nodes in this community are weakly interconnected._
- **Why does `G1 diagnostics (WP-S1.7)` connect `math` to `G1 results`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._