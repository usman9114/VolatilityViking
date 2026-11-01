# VolatilityViking live engine review (read-only)

Scope: main.py, config.json, src/live/{async_runner,binance_bot,portfolio_manager,recursive_grid,trader,momentum_allocator}.py, src/utils/db.py, Dockerfile. Tests could not be run (pytest not installed).

## 1. Architecture

- Entry: `main.py:89-100` `--mode live` always runs `asyncio.run(main_async(config))`. The `--async-mode`, `--aggressive`, `--symbol` flags are parsed (main.py:62-64) but ignored in live mode. The Dockerfile CMD (`--symbol ETH/USDT`) is therefore misleading.
- Live engine = `src/live/async_runner.py` `AsyncMultiSymbolBot`. REST polling (ccxt.async_support, NOT ccxt.pro) every 5 s (async_runner.py:1030): gather(tickers, balance, open orders per symbol) (977-981), indicators every 60 s (984), peak/kill-switch (991-998), then gather(process_symbol for each symbol) (1022). No websockets, so no reconnect logic and fills are inferred from balance polling.
- `src/live/binance_bot.py` is the legacy sync bot (1681 lines). It is dead and broken, see below. `trader.py` is a mock demo (random.uniform signal, trader.py:199) that does not trade. `portfolio_manager.py` is disabled in live path (async_runner.py:1058-1070, run_portfolio_manager_async never scheduled). `momentum_allocator.py` is never called by live code and writes `specific_exposures` to config.json (momentum_allocator.py:141) which nothing reads. `recursive_grid.py` is only used by backtester (robust_backtester.py:7). `db.py` is a predictions-log sqlite helper used only by trader.py (no positions/orders/equity persistence).
- Market: SPOT, long-only (`defaultType: 'spot'`, async_runner.py:98). No leverage, no shorts, no hedge/one-way mode, no futures. Passivbot terms like "short", "unstuck", "WE" are approximate analogues only. config.json has `live_mode: true` committed (config.json:3) -> mainnet by default.
- Strategy actually executed (process_symbol, async_runner.py:584-907):
  - Neutral price = EMA(1000 x 1m) (602), switched to EMA(100) if ADX>25 (607-608), skewed by RSI if smart_grid.trend_bias_strength>0 (config sets 0.0).
  - Buy ladder: `calc_geometric_grid` (434-454): n=max_orders(5) levels below neutral, geometric spacing weight 1.2, total depth = grid_span (0.6% in config; with raw config levels at 0.08/0.18/0.29/0.43/0.60%). grid_span is widened to GARCH-based `min_span = garch*38*safety*vol_scale` (671-682) unless `aggressive_mode` true (config false), so effective span is usually several %.
  - Buy sizes: `calc_martingale_qty` (456-471): q_i = q0*mult^i, q0 chosen so SUM of the 5 orders = equity * target_wel; mult=1.25 -> 12.2/15.2/19.0/23.8/29.7 % of the budget. Budget = `core_wallet_exposure` (1.0) for core symbols, `forager_wallet_exposure` (0.2) otherwise (739-743), times `total_equity`.
  - Take profit: "sells" = mirror of first 3 buy levels above neutral (810), each 1/3 of free base (897). Not tied to average entry (spot has no avg entry; `position.get('avgEntryPrice')` is always absent, 850).
  - "Trailing": after pnl vs (EMA) > 3%, sell on 1% retrace (829-889).
  - "Forager": static: first `max_positions` approved symbols not in core (118). The volatility/volume ranking only exists in dead legacy code (binance_bot.py:1473).
  - "Unstuck": inert (see bug 9).
  - Regime filter: blocks new buys when ADX>30 and EMA slope < -0.05% (723-727).
  - RSI "panic sell": 1m RSI14 > 90 -> cancel all and limit-sell free balance at -2% (626-641).

## 2. Risk exposure

- `total_wallet_exposure_limit: 1.5` is meaningless on spot (cannot exceed 1.0 of balance); it is only used to derive `per_symbol_wel = 1.5/n_active = 0.375` (async_runner.py:220-222) and block a buy if (position value + that one order)/equity > 0.4125 (225-227). It ignores resting open orders and uses a stale position snapshot, so at flat position all five grid levels (max 29.7% each) pass the check in the same tick; they are placed with no cumulative budget. Core budgets sum to 1.0+1.0+0.2+0.2 = 2.4x equity of resting buys; real cap is USDT balance (Binance rejects when insufficient), first-come across symbols.
- Budget is never reduced by what is already held; ladder is recomputed from full equity each tick and re-centred as EMA moves, so it keeps averaging down (re-placement is only blocked per-order by the 41% guard).
- BUY-ABOVE-MARKET BUG: second placement loop (797-804) has no "price < market" guard (the guard at 763-765 sits in a dead loop whose body does nothing). When price < neutral EMA by more than the span (any dump), the "limit buys" are marketable and fill immediately as taker, i.e. the bot market-buys the whole ladder into a falling knife.
- Regime filter is effectively inert: slope is computed over `iloc[-5]` (4 steps) of an EMA(1000) (426); a 0.05% move needs price roughly >6% away from EMA. Also when allow_buys is False existing buys are not cancelled (cancel loop is inside `if allow_buys`, 778).
- Worst case (spot, no leverage, no liquidation; loss <= 100%): assume the ladder fills fully (likely, because of buy-above-market): ~100% of equity long ETH/SOL/LINK/AVAX (alts beta >1). -40% market crash -> about -40% of equity (alts more); -70% -> about -70%. Kill switch fires at -30% equity drawdown but only cancels orders (async_runner.py:996-998); it never sells. Positions are held through the rest of the crash. If it is not fully invested the loss scales with the deployed fraction, but the ladder re-centres so deployment is typically high.
- max_drawdown_pct: config 0.3 read once in `__init__` (async_runner.py:68; not hot-reloaded; code default 0.2; legacy bot hardcodes 0.20 at binance_bot.py:174). Measured vs in-memory `peak_equity` (991-992) = mark-to-market of USDT + ONLY currently active symbols' holdings. Not persisted: a restart resets peak to current equity. Docker guide uses `--restart unless-stopped` (DOCKER_RUN_GUIDE.md) and a kill-switch exit is a normal return, so the container restarts and trades again with drawdown reset. Deposits raise the peak; withdrawals or a symbol dropped by hot-reload shrink equity and can false-trigger it; stale/missing ticker price gives value 0 (171-172).
- Martingale: sum is bounded per symbol by budget (geometric normalization), multiplier 1.25 is mild; the real unboundedness is the repeated re-centring and the weak/stale exposure guard above.
- Sells not anchored to cost basis: sell levels are EMA*(1+d). In a downtrend the EMA falls, so TPs re-centre below average cost and realise losses. At raw span 0.6% the first round trip (buy -0.08%, sell +0.08%) is below ~0.2% round-trip taker fees.
- RSI panic sell dumps position at -2% limit on 1m RSI>90 (not rare on a 1m RSI14) and the grid re-buys next tick: whipsaw losses and fees.

## 3. Correctness bugs (async_runner.py unless noted)

1. Trailing-stop path raises UnboundLocalError: `sell_qty = free_amount` (875) before `free_amount` is assigned (894). Caught by `except` at 917 and logged each tick; trailing never executes and the rest of the symbol logic is skipped while pnl>3%. HWM never decays when pnl falls back (stale high-water mark).
2. Auto-unstuck never runs: `avg_entry = pos.get('avg_entry', current_price)` (552) is never set -> distance 0 -> returns (556-557). If it ever fired it uses a market sell with no minNotional/precision handling (570).
3. `fetch_all_orders` swallows errors and returns `[]` (326-327), so a transient 429/network error makes the bot believe no orders exist and it re-places the full ladder (double ordering; limited only by free balance). Similarly `update_indicators` failures are silent (340-341).
4. Stale data: `last_prices` is never invalidated (146); if a ticker fails repeatedly, orders are priced off old data. Iteration 1 computes equity with price 0 because fetch_all_positions runs concurrently with fetch_all_data (977-981) -> first peak understated.
5. Order matching ignores quantity and side of drift: orders matched to grid by price only (786); drift_tol 0.2% (config) > the first level spacing (0.08-0.1%) so levels are ambiguous. Grid levels shift every 60 s with GARCH/EMA, so churn: cancel (794) then place (804) every minute; each placement calls `fetch_balance` (486, weight 20) and the result is never used to cap size. Roughly 50 weight per 5 s baseline (~600/min) plus 20 per order, plus fetch_ohlcv(1100) x symbols every 60 s; manageable against Binance's 6000/min IP weight but wasteful. No handling of -1003/418/429 backoff other than ccxt's enableRateLimit.
6. Cancel-all scope: any open order on the symbol that does not match the grid is cancelled (794, 827), including manual orders or another bot's.
7. Rounding: validation uses unrounded amount (303-306) then `amount_to_precision` truncates (490); the post-rounding amount can fall below minNotional/min qty -> exchange reject logged as error; no PERCENT_PRICE / tick re-check. float() of strings, float equality sets (754).
8. No clientOrderId anywhere; a create_order timeout is invisible until next poll (works only because state is re-read every 5 s).
9. `refresh_balance_before_order` result unused; `self.free_wallet_usdt` shared across concurrent coroutines with no lock, so symbols race for USDT (no per-symbol reservation).
10. Exception swallowing: `except Exception` + log in nearly every method (e.g. 195, 326, 340, 429, 917); legacy has bare `except:` (binance_bot.py:87, 393, 695, 783, 1584, 1593). Missing API keys are not validated in init_exchange (86-99): bot loops with equity 0 and kill switch never fires (peak 0 -> False, 238-239). `run()` only catches KeyboardInterrupt (1032); an unexpected exception exits without cancelling orders; shutdown never cancels orders.
11. Timestamp/recvWindow: ccxt default recvWindow 10 s, `adjustForTimeDifference` not enabled (94-99); clock drift -> -1021 errors swallowed.
12. Pre-existing holdings: equity and sells use total/free balance for the symbol, so the user's unrelated ETH/SOL holdings are sized into the "position" and can be sold by RSI-panic/trailing.
13. Legacy binance_bot.py is non-functional: `__init__` ends at 179, and the exchange/connection setup is unreachable code after `return qtys` in `calc_martingale_qty` (binance_bot.py:304-336) so `self.exchange` is never created; `self.config_path` never set (203); `eth_bal` undefined (1364); `auto_unstuck` defined twice (437, 813); unit `avg_entry = total_equity/total_eth` is not an entry price (666, 755, 842); hijacks `sys.stdout` at import (62); imports torch/sentence-transformers at import time. DOCKER_RUN_GUIDE.md still tells users to run it.
14. Dockerfile does not COPY config.json (Dockerfile COPY list); `load_config` swallows the error and returns `{}` (async_runner.py:80-81) -> defaults (ETH only, live_mode False -> testnet, kill switch 20%) silently.
15. `hot_reload_config` replaces `self.config` every 5 min with no validation (926); a half-written / invalid file returns {} (80-81) and wipes symbols/limits (falls back to ['ETH/USDT'] with default exposures). `momentum_allocator` rewrites config.json non-atomically relative to readers (it uses os.replace, OK) but its output is unused. `portfolio_manager.update_config` writes with plain open('w') (portfolio_manager.py:450), non-atomic.

## 4. State & recovery

- Good: positions, balances and open orders are re-read from the exchange every 5 s; no local order book to drift. A restart mid-grid simply re-matches existing orders to a freshly computed grid by price (within 0.2%) and cancels/replaces the rest.
- Lost on restart: peak equity/kill switch, trailing HWM, any cost basis, unstuck timers, `_balance_warned`. No persistence at all (no sqlite/JSON state). No clientOrderId, no trade-fill history from the exchange (`fetch_my_trades`), no PnL accounting; legacy trade CSV is only in dead bot.
- Restart policy makes the kill switch ineffective (see above).

## 5. Observability

- Python logging to file + console (async_runner.py:17-29) at INFO; emoji in logs; `logs/` and `data/` are gitignored. Status line per minute (1001-1018). No alerting (Telegram/email), no metrics endpoint, no PnL/fill logging, no structured logs, no heartbeat/watchdog, no order audit trail. Kill switch only logs CRITICAL.

## 6. Code quality

- Duplication: async_runner.py and binance_bot.py each contain geometric grid, martingale sizing, unstuck, trailing, forager; recursive_grid.py and backtesters (garch_backtester.py:152, event_backtester.py:109) re-implement them again, so backtests do not test the live code path. Live process_symbol is a ~320-line god-method (584-918) with a dead loop (757-768), mixed config/indicator/order logic and commented thoughts. `AsyncMultiSymbolBot` mixes exchange I/O, indicators, strategy, risk. Indicators computed on every 60 s with pandas over 1100 candles x symbols plus a Python GARCH loop (377-378).
- Testability: tests use `__new__` to bypass `__init__` (tests/test_kill_switch.py), no exchange fake/replay; no tests for process_symbol or order flow (the UnboundLocalError and unreachable-code bugs would have been caught).
- `main.py` ignores most CLI flags in live mode; config schema unvalidated (no pydantic/jsonschema); config keys `wallet_exposure_limit`, `specific_exposures`, `capital_limit` are partially/never used in the live path.

## Ranked top 15

| # | Sev | Issue | Fix |
|---|---|---|---|
| 1 | CRITICAL | Buys placed above market become taker fills; no `price < bid` guard in active loop (async_runner.py:797-804) | Clamp/skip buys with price >= min(bid, last)*(1-ε); use post-only (`timeInForce` GTX/`LIMIT_MAKER`); reject marketable orders |
| 2 | CRITICAL | Kill switch only cancels, never closes; holds bag through crash (996-998) | On trip: cancel all, then reduce/liquidate per policy (market sell or staged), and alert; add hard per-symbol stop-loss vs cost basis |
| 3 | CRITICAL | Peak equity not persisted; docker `--restart unless-stopped` re-arms trading after trip | Persist peak/kill flag to a state file (atomic write); on start refuse to trade while flag set; use `restart: on-failure` and non-zero exit |
| 4 | CRITICAL | Exposure guard ignores resting orders, stale snapshot, budget never decremented; total_wallet_exposure_limit meaningless on spot; core budgets sum 2.4x | Compute exposure = holdings + sum of open buy notional + new order; enforce a global and per-symbol cumulative cap; scale budgets to sum <= limit; reduce budget by position already held |
| 5 | CRITICAL | No cost basis: sells at EMA*(1+d) can realise losses; no min markup; avg entry never tracked | Track avg entry from `fetch_my_trades` (persisted); TP = avg_entry*(1+markup>=2*fee+edge); never place TP below it |
| 6 | HIGH | Open-order fetch failure returns [] -> duplicate ladders (326-327) | Fail the tick (skip processing) on any fetch error; use clientOrderId prefix and dedupe |
| 7 | HIGH | Regime/downtrend filter inert (EMA1000 5-step slope) and does not cancel resting buys | Use price vs EMA distance / return over N minutes; cancel buys when blocked |
| 8 | HIGH | Trailing stop UnboundLocalError (875) and avg-entry fallback to EMA | Fix variable order; base on tracked avg entry; reset HWM when below activation; add test |
| 9 | HIGH | Unstuck inert; legacy unstuck uses fake avg entry | Implement with real cost basis and loss allowance vs persisted balance, or remove it from docs |
| 10 | HIGH | RSI(1m)>90 panic sell with -2% limit and immediate re-buy | Use higher timeframe, cooldown, only sell if profitable or per risk policy; hysteresis |
| 11 | HIGH | Drawdown measure fragile: active-symbol-only equity, deposits/withdrawals, price 0 on ticker failure, first tick price 0 | Compute equity on all balances priced by one snapshot; skip kill-switch evaluation if any price/balance is stale; track net deposits |
| 12 | HIGH | Config handling: Dockerfile omits config.json, silent `{}` fallback to ETH/testnet or wipe on hot-reload; live_mode true committed | Fail fast on load/validation error; require explicit `--live`; schema-validate; keep previous config on bad reload |
| 13 | MEDIUM | Churn/rate limits: cancel/replace on each GARCH shift, fetch_balance per order, qty ignored in matching, drift_tol > level spacing | Use hysteresis, match by price AND qty, cache balance with local reservation, batch; handle -1003/-1021 with backoff and `adjustForTimeDifference` |
| 14 | MEDIUM | Rounding/minNotional after truncation, no clientOrderId, no PERCENT_PRICE check, float math | Round first then validate; use Decimal for qty/price; newClientOrderId with deterministic id |
| 15 | MEDIUM | Observability + legacy/duplicate/dead code (binance_bot.py broken, god-method, no alerts, no state) | Delete or quarantine binance_bot.py/trader.py; extract strategy as pure functions shared with backtester; add Telegram alerts, fill/PnL log, heartbeat, tests with fake exchange |
