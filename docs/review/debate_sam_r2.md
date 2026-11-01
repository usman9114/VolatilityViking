# Sam, round 2: rebuttal to Maya

## (a) Your three questions
**1. Can the backtester drive the real `process_symbol` against a mock exchange?** Yes, and I'll make it a requirement. The bot reaches the world only through `self.exchange` (about 8 ccxt calls: ticker, balance, open orders, ohlcv, markets, create/cancel order). A `FakeExchange` with a virtual clock and a 1m-bar matcher is a few days of work. The prerequisite is extracting the decision logic from the 320-line `process_symbol` into a pure function. Then live, replay and tests run the same code and the 4 backtesters get deleted. That also fixes the tests: 14 of the 32 passing ones test a local copy of the logic.

**2. Fill model?** A post-only buy fills in replay only when price trades through the level by at least 1 tick. A touch is not a fill. That is a pessimistic bound with no L2 data. Then I calibrate with tiny-size live paper trading, comparing placed to filled. If live is 2x worse than sim, the sim is wrong and I trust live. I re-run the gates on the live-calibrated model, and if they fail the bot doesn't scale.

**3. Pre-committed stop?** A persisted `halt.json`, written atomically, with peak equity and a halted flag. At equity -15% from peak, or -25% on any symbol from cost basis, the bot cancels, flattens, alerts, and exits non-zero. On startup it refuses to trade while the flag is set, and only a manual `--resume` clears it. Compose uses `restart: on-failure`.

## (b) Agreement and disagreement
I agree that edge on real data is unmeasured, touch-fill flatters makers, and HAR/EWMA beats GARCH. I accept your ML ranking.

I still disagree that ML belongs in the first release. A regime gate must first beat a 3-line rule, and nobody has run that test.

## (c) Too heavy for one operator
Tick/L2 queue replay, CPCV and PBO are over-built for two symbols and a handful of parameters. We'd spend weeks on a fill model whose error exceeds the signal. Cheaper and still honest:
- Use 1m klines (free, years of history) with the through-tick rule, 2x fees and fixed slippage.
- Freeze 3-5 parameters up front and run walk-forward on 3-4 splits including 2022 and a bull year.
- Log configs tried and discount the result by hand instead of computing PBO.
- Let four weeks of reconciled paper fills be the real validator.

## (d) Three go/no-go gates before real money
1. **Replay:** production `process_symbol` on real 1m data, maker-only at 2x fees. Net positive in at least 3 of 4 splits, and 99th-percentile max drawdown under the halt threshold.
2. **Paper:** 4 weeks live at tiny size. Fill rate within 30% of sim, zero duplicate orders, zero taker fills, and a restart-during-halt drill passed.
3. **Engineering:** critical defects fixed with real tests (exposure cap, cost-basis TP, halt persistence, fetch-error skip), and alerts and heartbeat fire on a forced failure.

Fail any one and the bot doesn't trade.
