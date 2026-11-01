# Sam, opening position: grid vs ML

## 1. Thesis
This project does not need ML. It needs to stop losing money to its own plumbing, then run a risk-capped, maker-only strategy one operator can leave alone. A well-built grid is the right engine, but a grid alone is not a business. It earns a modest, regime-dependent yield, and survival depends on inventory limits, not prediction.

## 2. Where grids make money and where they die
**Fees (Binance spot, VIP0):** 0.10% maker/taker, 0.075% with BNB. A maker round trip costs 0.15% with BNB (0.20% without). Spacing needs to be 0.3-0.4% or more to leave margin.

**The repo fails this.** The raw first level in `config.json` is about 0.08%, so the first round trip is below fees. The buy loop (`async_runner.py:797-804`) has no `price < bid` guard and no post-only flag, so in a dump the ladder fills as taker. The RSI panic sell (`:637`) is a deliberate taker.

**Our Monte Carlo (synthetic GBM):** at zero drift the grid lost 6.3% a month with 0.1% fees versus 0.9% without, so churn costs about 5 points a month. Max drawdown hit -51% in 30 days. It beat buy-and-hold in 5 of 20 runs at zero drift and 2 of 20 at +150%/yr drift.

On a random walk a grid has no edge. It is short volatility and only pays if real ETH/SOL prices mean-revert at our spacing by more than 0.15-0.2% per cycle. Nobody has measured that.

**Where it dies (also: capped upside, uncapped downside):** a -50% trend. A long-only 1.0x spot ladder ends up near 100% invested and loses roughly 40-50% of equity. Alts with beta above 1 do worse. The kill switch (`:996-998`) cancels orders but never sells, so the bag rides the crash down.

## 3. What "AI" was, and what ML costs
The README's ETH-AI claim (LSTM+XGBoost, 411 features, +1,395%) is not in the live path. `src/models/*` and `src/data/*` aren't in git, because `.gitignore` excludes `data/` and `models/`. `--mode train` imports a module that doesn't exist. I treat the claim as unsubstantiated.

ML in production for one person means point-in-time feature pipelines (news and on-chain data leak easily), scheduled retraining, walk-forward validation, drift monitoring, and a fallback when the model dies. It also fails silently: a stale model keeps emitting confident numbers, while a broken grid throws errors. For an unattended bot that matters more than a point of Sharpe.

## 4. What I'd build
1. **Fix the engine first:** post-only orders, a persisted kill switch that flattens, an exposure cap that counts resting orders, cost-basis-anchored take-profits, and alerts.
2. **Spot grid with hard inventory caps:** 30-40% of equity per symbol, 60% overall, maker-only with BNB fees, spacing of 0.35% or more, staged de-risk at a fixed drawdown from average cost. ETH and BTC only.
3. **Funding/basis carry as the passive sleeve:** spot long plus perp short, delta-neutral. Baseline funding of 0.01% per 8h is about 11% a year, but it varies by regime (roughly 5-15% in bull markets, near zero or negative in bears). It needs no direction call. Costs: four legs of fees, margin management, short-leg liquidation risk (keep leverage at 1-2x), counterparty risk. Add it once the grid is trustworthy.

## 5. Concessions
ML could earn its place in regime detection (grid on/off or wider) and volatility forecasting for spacing, though HAR/GARCH is a strong baseline. I don't expect it to help with direction prediction.

I'd change my mind if I saw a walk-forward test on real 1m data, with fees and fills modeled, where an ML regime gate beats a simple rule (price vs EMA, realized-vol ratio) on net Sharpe and max drawdown, across at least two regimes, at under a day a month to operate.

## 6. Questions for Maya
1. What exactly are you predicting, and over what horizon? Show me a net-of-fees, out-of-sample result on real data, not synthetic data and not the README.
2. What beats a 3-line rule (price below the 4h 200-EMA means grid off) for regime gating, and how do you show it isn't overfit with only 2-3 real regimes?
3. Who notices model staleness at 3am, and what does the bot do? "Fail closed to the dumb rule" is fine. "Keep trading on the last prediction" is not.
