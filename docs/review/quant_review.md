# VolatilityViking quant/backtest review (read-only)

Repo: /home/user/VolatilityViking, 4 commits, last 2026-01-14. Paths below are relative to the repo root.

## Verdict
Positive expectancy after costs is UNDEMONSTRATED. The only headline artifacts (README +1,395%, docs/backtest_results.png) are not trading simulations or are unreproducible. The grid backtests are not the live strategy. Synthetic zero-drift runs on the repo's own backtesters lose money after 0.1% fees: robust about -6.3% per 30d vs buy-and-hold +7.5% (5/20 seeds beat it); garch about -3% per 30d. Fee drag is about 5 points per month. Any real positive number would have to come from drift (beta), fill optimism, or bugs. No live or paper PnL exists in the repo.

## 1. Fidelity: backtest vs live (async_runner.process_symbol and friends)
There are two live engines: BinanceBot (`binance_bot.py`, default) and AsyncMultiSymbolBot (`async_runner.py`, `--async-mode`). Backtests mirror neither exactly.

| Item | Live (async_runner.py) | Backtests |
|---|---|---|
| Anchor | EMA-long (1000 1m bars), EMA-short if ADX>25 (:598-608) | garch/event: current close (garch_backtester.py:144). robust: EMA anchor at robust_backtester.py:200, but the ADAPTIVE ANCHORING block (:457-491) is indented under `elif rsi > 70`, so EMA-short anchoring only runs when RSI>70 (indentation bug) |
| Exits | 3 sell levels mirrored from the first 3 buys, equal split, placed only if `price > current*1.005` (:810, :897-907). With the 0.6% span, levels sit at 0.08/0.18/0.29% and are never placed unless price is >0.5% below neutral | garch/event: 5 martingale-sized sells around the current price, not cost basis (garch_backtester.py:224-232). robust: NO sell grid, only a trailing stop (3%/1%) on true avg entry (:278-331) |
| Trailing stop | Uses `avgEntryPrice` that the position map never contains (:174-179), falling back to the EMA (:850-854). Then `sell_qty = free_amount` at :875 reads a variable that is only assigned at :634 (RSI branch, which returns) and :894, so it raises UnboundLocalError, swallowed at :917. The live trailing stop is dead code | Works in robust only |
| Unstuck | `pos.get('avg_entry', current_price)` (:552) is never populated, so distance is 0 and unstuck never fires | Not modelled anywhere |
| Kill-switch (30% DD) | async_runner.py:233-252 | Not modelled |
| Regime filter | Block buys in downtrend, block sells in uptrend; slope over 4 1m bars of EMA1000 (:93, :709-727); `adx_threshold` 30 | garch/event: none. robust: buys blocked, but "uptrend blocks sells" is not applicable (no sell grid), and slope is one 5m-bar change of EMA200 (:155-157), a different scale |
| Exposure | core WEL 1.0 per core symbol (ETH and SOL) plus forager 0.2, so resting buys can exceed equity (:739-750). `total_wallet_exposure_limit` 1.5 split per symbol (:215-231) | single symbol, 100% |
| Order validation | min amount/cost, precision (:281-318, :156-158) | none |
| Panic sell RSI>90 | limit at 0.98x (:638) | robust only: 0.995x close, evaluated every 5 min (:401-423) |
| Span | `aggressive_mode:false` in config.json means the GARCH floor overrides `grid_span` | see section 5 |

The CHANGELOG's "0.6% scalping" is overridden by the GARCH floor in non-aggressive mode (about 2x daily vol, i.e. several percent). The backtested system, the live system and the documented system are three different things.

## 2. Fill model
- Fill on touch: buy if `low <= price`, sell if `high >= price` (garch_backtester.py:112-130; event_backtester.py:267-285; robust_backtester.py:207-273). There is no queue position, so a touch counts as a fill. This is optimistic for maker orders.
- Intrabar ambiguity: buys are iterated before sells (appended first), so a buy and a sell can fill in the same 1m bar if the range covers both. Buy-then-sell is assumed. Only the garch/event backtesters allow this. With tight levels (aggressive mode) this inflates trade count and PnL.
- Stops fill exactly at `stop_price` (robust :315-318) with no gap/slippage.
- Fees: flat 0.1% for maker and taker (garch :21, robust :30). No spread, no slippage (except the 0.5% panic haircut), no latency, no partial fills, no lot or min-notional rounding (live has these). Spot, so no funding; the robust "shorts" simulation (:244-273) has no borrow or funding and does not match live.
- Fee math: with 0.6% span the first level is 0.081% from centre. First-level round trip is 0.16% gross vs 0.20% fees, so it is negative before slippage.
- Timing: robust shifts indicators one 5m bar (:77), so no look-ahead. garch/event place orders at the bar-i close and check fills from bar i+1, also fine. `sigma2[0]=np.var(returns)` uses full-sample variance (garch :50, robust :96); this decays and is minor.

## 3. Biases and overfitting
- Optimizer (`optimizer.py`): genetic search, pop 6 x 3 generations = 18 evaluations (:111). 4 genes (:26-31); only 2 are mutated (:99-100). Fitness is the raw profit of the LAST 30 days of 1m data (:51-53), in-sample only. No walk-forward, no holdout, no costs sensitivity. It writes `config_opt.json` as a side effect (:43). That file is a stale copy of config.json with grid_span 0.101, vol_scale 0.945, trend_bias 0.16, qty_mult 1.246. With `aggressive_mode:false` the GARCH floor overrides most of what is tuned.
- `tools/optimizer.py`: calls `GarchBacktester('config.json', config_overrides)` (:33) but the constructor takes only `config_path` (garch_backtester.py:8). Every call raises TypeError, caught at :71-72 as `{'error':...}`, so it can never have produced results. If it did run: 50 random configs on 7 days (:13, :42-43), ranked by raw profit though Calmar is computed (:62, :98). `drift_tolerance` is not read by any backtester, and it overrides `core_wallet_exposure`, which GarchBacktester ignores (it reads `wallet_exposure_limit`, :160). Those are pure-noise dimensions.
- Hardcoded tuned constants: `Window=96, Std=0.86` "Optimized Params" (backtester.py:42-43, also inference.py:222-223). They were selected on the same predictions they are evaluated on. `smart_backtester.py:359-366` grid-searches window and k by best return, with no split and no fees (:326-329), allowing shorts (-1).
- Survivorship and selection: `mass_backtest.py:17-37` picks TODAY's top-50 USDT pairs by 24h volume and backtests only 7 days (:15). `approved_symbols` LINK/AVAX are hand-picked. Test periods are 3 to 30 days (`run_garch_backtest.py:398` uses 3 days), a single regime. Model selection takes the lowest `test_mse` MLflow run (inference.py:133), i.e. selection on the test set.
- ML pipeline: `backtest_only.py` loads one saved model and runs it over the WHOLE 2018+ dataset with `test_size=0` (:55-60, :110-118), so the "walk_forward_results_meta.csv" is in-sample, not walk-forward. `comprehensive_backtest.py` then scores it. Fear and Greed daily values are `merge_asof` backward onto candle open times (indicators.py:277-283), a likely mild look-ahead. News is resampled with left-label 4h bins (process_data.py:101), which looks causal. `fillna(method='ffill')` (process_data.py:173) breaks on pandas 3.
- Train/serve skew: `inference.py:181-185` zero-pads the 403 news/gas features at inference. The schema lists 19 price features, while the notebook dataset has 21 columns and 0 embeddings (cell 3 output).

## 4. Metrics
- `backtester.py:133-134`: Sharpe is mean/std*sqrt(2190), correct formula but applied to the synthetic hybrid series. MaxDD (:139-141) is correct. No time-under-water, no turnover or per-trade stats, no confidence intervals, no deflated Sharpe.
- `comprehensive_backtest.py:236-238`: "Sortino" is mean / std of the negative returns (std about their own mean, not downside deviation vs 0). Win rate (:231) counts positive periods.
- `robust_backtester.py:558-561`: daily Sharpe from equity, fine. B&H is printed (:589), but there is no naive static-grid or hold-cash baseline. `run_baseline_backtest.py` just runs GarchBacktester.
- `calculate_backtest_stats.py:15-19`: long if Ensemble>0, compounded simple returns, no fees, in-sample. This is the probable source of the README figure.

## 5. GARCH
It is a fixed-parameter EWMA-like recursion (omega 1e-8, alpha 0.05, beta 0.90; never fit) in garch_backtester.py:38-57, robust_backtester.py:92-99 and async_runner.py:35-47. The sole use is a grid-span floor: `daily_vol = garch*38` (1m; *17 on 5m bars), times safety factor 2.5-4.0 times vol_scale (async_runner.py:664-682). It is not fit on past data in any statistical sense (params are constants), so there is no look-ahead except the init variance. With ETH 1m vol around 0.1%, the floor is several percent and overrides the configured span. There is no ablation of GARCH vs ATR vs constant span anywhere in the repo (ATR is computed in async_runner.py:385-389 but unused). It adds no demonstrated value.

## 6. docs/backtest_results.png
Log-scale chart titled "Performance (Log Probability Scale)", produced by `comprehensive_backtest.py:265-280`. The "Hybrid Grid" equity runs from roughly 1e4 in 2018 to about 1e37 in late 2025, i.e. around 1e33x, versus buy-and-hold of roughly 1e4-1e5. Cause: `backtester.py:104-109` and `comprehensive_backtest.py:213-219` credit `rolling_vol*0.5` as a guaranteed positive "yield" every flat 4h bar and compound it. No orders, fills, fees or price path are simulated, and the rolling std includes the current bar's return. This is an accounting artifact. It also contradicts the README's own +1,395% (14x).

## 7. Is there PnL evidence?
No. `data/`, `logs/`, `mlruns/` and `models/` are in .gitignore; `src/models` and `src/data` (imported by main.py:31 and trader.py) are missing from the repo, so the LSTM/XGBoost code and any trained model are absent. CHANGELOG has no numbers. Tests are unit tests of formulas. The notebook (Masterclass_Analysis.ipynb): cell 12 (backtest) has no output; cell 3 shows a 21-column dataset with 0 embedding columns (README claims 411 features), starting 2022-01 (README says 2019-2025; the chart says 2018). Decoding cell 10's plotly data: "Actual" has mean 0.86, std 0.17, last 500 rows 2025-09-22 to 2025-12-14. That is a price-level scale, not 4h log returns, and the prediction correlates 0.985 with it. That is level persistence, not skill.

## 8. README +1,395% / LSTM + XGBoost
Nothing substantiates it. No XGBoost anywhere in code or requirements.txt. No LSTM code (the repo mentions CNN and a "meta" model; code absent). The live `trader.py:38` "prediction" is `random.uniform(-0.01, 0.01)`. `inference.py` only prints a signal; the JSON output is commented out (:288). BinanceBot loads an MLflow model if one exists (binance_bot.py:890-905) and otherwise falls back to the grid with no AI. Where ML would plug in: `Ensemble` column to bands in `Backtester` (regime pick: trend vs grid), but that is not wired to live execution.

## 9. Network
api.binance.com, data-api.binance.vision, fapi.binance.com, data.binance.vision (plus coinbase, bybit, coingecko): all fail with CONNECT 403 from the sandbox policy proxy. Real klines are NOT reachable here. Real backtests need data supplied or a different environment.

## Synthetic evidence (my scripts in scratchpad: synth.py, GBM 1m, zero drift)
Coordinator aggregate: robust about -6.3%/30d vs B&H +7.5% (5/20 beat B&H); garch about -3%/30d; fee drag about 5 points per month.

## Top 10 issues (ranked) with fixes
1. README/chart headline is an accounting artifact (guaranteed vol yield compounded, 1e33x) and the +1,395% has no reproducible source. Fix: delete the claims; publish only event-driven results with fills and fees.
2. ML results are in-sample (`backtest_only.py` test_size=0; model picked by test_mse) and the model, LSTM/XGBoost code and data are missing. Fix: purged walk-forward or CPCV, a model registry, embargo, report OOS only.
3. Backtest is not the live strategy: three divergent implementations, no sell grid in robust, no unstuck/kill-switch, anchor and regime differ. Fix: one shared strategy core (pure function of state) used by live and the simulator; golden-trace tests.
4. Live has bugs that disable risk controls: trailing-stop UnboundLocalError (:875), unstuck never fires (:552), exposure sums above 100%. Fix: track avg entry from fills, fix, add tests with a fake exchange. Do not trade live until fixed.
5. Optimistic fills: touch=fill, same-bar buy+sell, stop at exact price, flat 0.1% fees, no slippage, lot or notional limits. Fix: require trade-through by a tick or volume-based queue model; adverse-selection haircut; tick or 1s data; BNB/VIP-tier fee matrix.
6. Optimizer is in-sample on 7-30 days, a handful of evaluations, noise dimensions, and tools/optimizer.py is broken. Fix: walk-forward with at least 1 year, multiple regimes (2022 bear, chop), fewer parameters, report distribution and a deflated Sharpe, stability plateau selection.
7. Martingale grid with no hard stop has fat left-tail risk that short tests hide. Fix: test through 2020-03, 2022-05/11 crashes, report worst-case and time-under-water, and cap the max inventory.
8. Survivorship and selection: today's top-50 list, hand-picked forager symbols, and 3-7 day windows. Fix: point-in-time universe from historical volume snapshots including delisted pairs.
9. Metrics are weak: wrong Sortino, no TuW, no baselines. Fix: compare to buy-and-hold, hold-cash, static grid with the same fees; report bootstrap CIs.
10. GARCH is untested noise: fixed params, only a span floor that overrides the config, and no ablation. Fix: fit on a rolling window (arch package), A/B vs ATR and a constant span, and make `aggressive_mode` semantics explicit.

Honest status: edge unknown, with the evidence available leaning negative (fees plus martingale tails on a driftless random walk).
