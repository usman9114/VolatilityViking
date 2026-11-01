# Maya, Opening Position: does VolatilityViking need AI/ML?

**1. Thesis.** No, not as the core. The live bot is a short-volatility, long-mean-reversion position with a martingale tail. Its P&L is set by fees and trend/crash inventory, and ML fixes neither. Fix execution and risk, measure whether the edge exists, then add at most a small, boring model (vol forecast, regime gate). Deep learning for direction is the wrong tool at 1m-1h horizons for a single operator. I put under 10% on the LSTM+XGBoost story surviving an honest test.

**2. Grid economics.** A grid sells gamma: it earns spacing x round trips and pays in trends. On a random walk it has zero edge before costs and negative after (our GBM runs: -6.3%/30d with 0.1% fees vs -0.9% without). Edge must come from real data:
- Negative autocorrelation at grid horizons. Measure VR(q) = Var(q-bar returns)/(q x Var(1-bar)) and Hurst/DFA at 5m-4h on ETH/SOL, with bootstrap CIs. VR>=1 means stop.
- Capture per round trip > fees + adverse selection. Config `grid_span` is 0.006 over 5 levels (about 0.12% steps before GARCH widening) against 0.2% taker round-trip. Even at 0.075% maker x2 the margin is thin. Measure realized net capture and post-only fill rate.
- Survivable inventory tail: distribution of episode max drawdown and time underwater (95th/99th percentile), Calmar. Never average return alone.
- Stability across years, including crash regimes.

**3. ML hierarchy (best to worst).**
1. **Vol forecasting for spacing.** Modest, real value; vol is the predictable quantity. Data: 1m bars, plentiful. Failure: recursive GARCH on 1m noise scaled by `garch*38`. Baseline: HAR-RV/EWMA. Must beat it on QLIKE out of sample; gain over HAR will be small.
2. **Regime gate (on/off/widen).** Best value per complexity because it attacks the tail. 2-3 state HMM or change-point on RV and trend strength. Failure: late detection, flip-flop churn. Baseline: a correctly wired ADX/EMA-slope filter plus a drawdown-from-anchor stop (the current one is effectively inert, so test that first).
3. **Inventory/risk control.** Mostly not ML: cost basis, hard exposure cap, vol-targeted size, stop. It beats everything below.
4. **Meta-labeling.** Label grid cycles (profit/stop), learn P(win | features) to skip or size. Needs thousands of cycles and purged CV. Baseline: the unfiltered grid. Small gain, only after 1-3.
5. **Direction prediction.** Expect AUC about 0.51-0.53, decaying. Baseline: coin flip and 200-bar momentum.
6. **News/LLM sentiment.** Latency disadvantage, leakage minefield. Skip.

**4. Critique.** README claims +1,395%, 411 features, news embeddings, walk-forward. I verified that `src/models/*` and `src/data/*` are not in git, `--mode train` imports a missing module, and the live path uses none of it. Unreproducible means unevidenced. "Better model gets more votes next year" adds a selection degree of freedom on seven test years.

Backtesting: four separate re-implementations, none running live `process_symbol`. In `robust_backtester.py`, buys fill whenever `low <= order price`: touch-fill, no queue, no adverse selection, which flatters a maker strategy. Fee is a flat 0.1% taker (`fee_rate = 0.001`) on limit fills, and the panic sell hardcodes a 0.995 haircut. `optimizer.py` searches grid_span in [0.01, 0.15] on the same history it scores, with no holdout.

Credible pipeline:
- Replay the actual live decision code on tick/L2 data, with a queue-aware maker fill model, post-only rejects, tiered fees and BNB discount.
- Walk-forward with purge/embargo, then CPCV for a path distribution.
- Log every config tried; report deflated Sharpe and PBO.
- Stress at 2x fees and 2x slippage.
- 4-8 weeks paper or tiny-size live, reconciling real fill rate to simulated.

**5. Concessions.** Sam is right that the live defects are the real leak: buys above bid, kill switch that re-arms on restart, no cost basis, duplicate ladders, 2.4x budgets. Each can erase any ML gain. A robust, observable, boring grid comes first. If vol-adaptive spacing stays, it should be HAR/EWMA, not a trained net.

**6. Questions for Sam.**
1. Can the backtester drive the real `process_symbol` against a mock exchange so research and production cannot diverge? If not, why trust any number?
2. Which maker fill model will you accept? If the sim fills at the touch but live is queue-starved, what do you do with a 2x gap?
3. What pre-committed stop (max drawdown, or rolling VR>=1) turns the bot off and keeps it off across a Docker restart?
