# VolatilityViking: Production Plan

*Review date: 2026-11-01. Detailed reviews and the full engineering-vs-ML debate are in [`docs/review/`](review/).*

## TL;DR

1. **Do not run the bot with real money in its current state.** The live engine has five defects that can lose money on their own, independent of market conditions (see §2).
2. **The README's performance claims are not evidence.** The +1,395% figure and the LSTM/XGBoost "metamodel" have no code in the repo. The backtest chart (`docs/backtest_results.png`, ~10^33x) is an accounting artifact: `src/backtest/backtester.py:103-108` credits `rolling_vol × 0.5` as guaranteed profit on every flat bar.
3. **What actually trades is a spot, long-only DCA grid** (`src/live/async_runner.py`). Whether it has an edge after fees on real data has **never been measured**. On synthetic random-walk data the repo's own backtester loses about 6%/month after 0.1% fees.
4. **AI or grid? Decision: grid (or funding carry) on a sound engine.** ML enters last, only as a fail-closed throttle that can widen the grid or switch it off, and only if it beats a 3-line rule in every held-out regime. Both reviewers converged on this independently.
5. **The first real work is cheap.** Measure whether ETH/BTC mean-revert at grid horizons (variance ratio). If they don't, the grid has no edge, and we pivot to delta-neutral funding carry before spending weeks hardening a grid.

---

## 1. What the system actually is

| Component | Status |
|---|---|
| `main.py --mode live` → `src/live/async_runner.py` | **The real bot.** Binance spot, long-only. EMA(1000×1m) anchor, 5 geometric buy levels, ×1.25 martingale sizing, GARCH-widened span, 3 take-profits above the EMA, ADX/slope regime filter, RSI(1m)>90 panic sell. REST polling every 5 s. |
| `src/live/binance_bot.py` (1,681 lines) | Legacy, **non-functional**: exchange initialisation is unreachable code after a `return`. Still documented in `DOCKER_RUN_GUIDE.md`. |
| `src/live/trader.py` | Demo; "prediction" = `random.uniform(-0.01, 0.01)`. |
| `portfolio_manager.py`, `momentum_allocator.py` | Not wired into the live path. |
| `src/models/*`, `src/data/*` | **Not in git.** `.gitignore` patterns `data/` and `models/` swallow them; `--mode train` and 4 tests fail on import. |
| `src/backtest/*` (4 backtesters) | Each re-implements the strategy differently; **none runs the live code**. |
| `optimizer.py` / `tools/optimizer.py` | In-sample on 7–30 days; `tools/optimizer.py` raises `TypeError` on every call (swallowed). It never produced a result. |
| Tests | 39 tests: 32 pass, 7 fail; 14 of the passes test a local copy of logic, not production code. `async_runner.py` coverage 36%. |

## 2. Critical defects (fix before any real capital)

| # | Defect | Where | Fix |
|---|---|---|---|
| C1 | Buy orders can sit **above market** → taker fills; ladder buys straight into a dump | `async_runner.py:797-804` | Post-only (`LIMIT_MAKER`) and reject any buy ≥ best bid |
| C2 | Kill switch only cancels orders, never reduces position; peak equity is in memory only, and `--restart unless-stopped` re-arms trading after it trips | `:994-998`, `:67-69` | Persisted halt latch (SQLite/atomic file), flatten policy, non-zero exit, manual `--resume` only |
| C3 | Exposure check ignores resting orders and held inventory; budgets sum to 2.4× equity; `total_wallet_exposure_limit: 1.5` is meaningless on spot | `:215-227`, `:739-743` | Exposure = holdings + open buy notional + new order; global and per-symbol cumulative caps |
| C4 | No cost basis: TPs = EMA×(1+d), so they realise losses in downtrends; first round trip (±0.08%) < 0.2% fees | `:810`, `:850` | Track average entry from fills (`fetch_my_trades`), TP ≥ avg_entry × (1 + 2.5×fee) |
| C5 | Open-order fetch error returns `[]` → duplicate ladders; no `clientOrderId` | `:320-327` | Any fetch failure skips the tick; deterministic `newClientOrderId` |
| H1 | Trailing stop crashes (`UnboundLocalError`), unstuck never fires, regime filter effectively inert, RSI panic sell then re-buy | `:875`, `:552`, `:426`, `:626` | Fix or delete; each behaviour must have a test |
| H2 | Config load failure silently becomes `{}` (Dockerfile doesn't copy `config.json`); hot-reload can wipe live config; `live_mode: true` committed | `:75-81`, `:920-926` | Schema validation (pydantic), fail fast, keep last-good config on bad reload, explicit live confirmation |
| H3 | No alerting, heartbeat, order/fill audit trail, PnL accounting or SIGTERM handling | everywhere | See Phase 3 |
| H4 | Unpinned dependencies (incl. ccxt), Python 3.10 base image (EOL), root container, `check_creds.py` prints partial secrets | `requirements.txt`, `Dockerfile` | Lockfile with hashes, slim live-only image, non-root |

## 3. The engineering vs ML debate: conclusions

Two reviewers argued it over three rounds (transcripts in `docs/review/debate_*.md`):
- **Sam**, a senior SWE and crypto bot operator
- **Maya**, a quant/ML engineer

**Where they started.** Sam: "No ML. Fix the plumbing, run a capped maker-only grid, add funding carry." Maya: "ML is not the core. The grid is a short-volatility position whose P&L is set by fees and trend inventory, which ML can't fix. At most a vol forecast or regime gate, later."

**What they agreed:**
1. **The engine comes first.** Every live defect above can erase any ML gain. A boring, observable, correct grid beats a clever broken one.
2. **The grid's edge is unknown and must be measured.** On a random walk a grid has zero edge before costs and negative after. It only pays if real prices mean-revert at the grid's spacing by more than round-trip fees. Test: variance ratio VR(q) < 1 with a confidence interval excluding 1, at 5m–4h horizons.
3. **ML ranking, most to least useful:**
   1. Volatility forecast for spacing (baseline: HAR-RV / EWMA)
   2. Regime gate on/off/widen (baseline: price vs 4h 200-EMA)
   3. Inventory/risk control (mostly not ML)
   4. Meta-labeling
   5. Direction prediction (expected AUC ≈ 0.51–0.53)
   6. News/LLM sentiment (skip)
4. **ML may only reduce risk.** A model may widen spacing or switch the grid off, never add exposure. Staleness detection (model age, feature nulls, forecast error) falls back to the EMA rule. If the fallback fails, flatten and halt.
5. **Caps and spacing are volatility-targeted**, clipped by static ceilings (30–40% per symbol, 60% total). Spacing = max(2.5 × round-trip fee, c × σ at fill horizon). A static cap is a hidden volatility bet.
6. **Funding/basis carry is an equal candidate, not just an add-on.** Spot long + perp short, delta-neutral. Realistic net is about 3–6% per year on total capital, near zero or negative in bear funding regimes. Tail risks: ADL, venue custody, basis blowouts, collateral depeg. If the grid fails Gate 1, pivot to carry-only (paper-traded first).
7. **Validation rigor sized for one operator:**
   - 1m klines with through-tick fills (a touch is not a fill) plus a fill-probability haircut
   - 2× fee stress
   - Walk-forward / 6-fold CPCV, with one split being the 2022 crash
   - A written trial count
   - Gating on per-regime wins and the Sharpe CI lower bound
   - Tick/L2 queue replay is overkill here; paper trading is the fill-model calibrator.

**Verdict:** *Grid or carry on a sound engine, ML only as a fail-closed throttle, and no real money until Gates 1 and 2 pass.*

## 4. Go / no-go gates

| Gate | Pass criteria | If it fails |
|---|---|---|
| **G0: Edge pre-check** (cheap, day 1–3) | ETH/BTC VR(q) < 1 with CI excluding 1 at the grid's horizon (5m–4h) across 2019–2026, and median 4h range > 2.5 × round-trip maker fee | Don't invest in the grid; go to carry track |
| **G1: Edge** (real-data replay of the *production* decision code) | Net capture ≥ 2× fees; OOS net Sharpe CI lower bound > 0 at 2× fees; positive in ≥ 3 of 4 splits **including 2022**; 99th-pct max DD ≤ 15% with vol-targeted caps; trial count logged | No grid. Carry-only or stop. |
| **G2: Ops** (6 weeks paper / tiny live) | Fill rate within 25% of sim, per grid level; zero reconciliation breaks, duplicate orders and taker fills; restart-during-halt drill, key-revocation drill and network-cut drill all pass; alerts verified | Fix and repeat; don't scale |
| **G3: Add-on** (ML throttle or carry sleeve) | Beats the baseline (EMA rule / HAR) on net Sharpe **and** max DD in *every* held-out regime; pre-registered candidates; < 1 h/month to operate | Don't ship it |

## 5. Phased roadmap

Phases 1 and 2 can run in parallel with Phase 0.

### Phase 0: Stop the bleeding and measure the edge (week 1)
- [ ] Stop any running live instance. Set `live_mode: false` in the committed config; live requires an explicit env flag (`VV_LIVE_CONFIRM=<account-id>`).
- [ ] Fix `.gitignore` (anchor to `/data/`, `/models/`). Decide whether the missing `src/models`/`src/data` code still exists anywhere. If not, delete the code paths that import it.
- [ ] Rewrite the README to describe what the bot actually is. Remove `docs/backtest_results.png` and the +1,395% claim.
- [ ] **G0 study**: download 2019–2026 1m klines for ETH/USDT and BTC/USDT from `data.binance.vision`. This must run outside the sandbox, whose network policy blocks Binance. Compute variance ratios / Hurst at 5m–4h with bootstrap CIs, plus intraday range vs fee. One notebook, one page of results.

### Phase 1: One strategy core, one truth (weeks 1–3)
- [ ] Extract the decision logic from the 320-line `process_symbol` into a **pure function**: `decide(snapshot, state, params) -> [OrderIntent]`.
- [ ] Put all I/O behind an `ExchangeAdapter` interface (ticker, balances, open orders, OHLCV, markets, create/cancel, my trades).
- [ ] Build a `FakeExchange` with a virtual clock and a 1m-bar matcher (through-tick fills, maker/taker fee tiers, min-notional and lot rounding, partial fills).
- [ ] **Replay harness** = production `decide()` + `FakeExchange` + historical data. Delete the four legacy backtesters, `binance_bot.py`, `trader.py`, and the unused allocators.
- [ ] Golden-trace tests: the same input produces the same intents in live and replay. Remove the 14 tests that test copied logic.

### Phase 2: Fix the strategy and risk logic (weeks 2–4)
- [ ] C1–C5 and H1 from §2, each with a failing test first.
- [ ] Cost basis from fills, persisted; TPs anchored to average entry.
- [ ] Vol-targeted per-symbol caps, global cap ≤ 0.6 of equity on spot, budget reduced by inventory held.
- [ ] Spacing = max(2.5 × round-trip fee, c × σ). Replace the fixed-parameter GARCH with EWMA/HAR (A/B it against constant span in replay).
- [ ] Regime rule: grid off below the 4h 200-EMA (or measured alternative). When blocked, cancel resting buys.
- [ ] Halt policy:
  - Trip at -15% equity from peak, -25% on a symbol from cost basis, or about 3 daily σ of inventory, whichever is first.
  - Time trip: N days underwater with no completed cycle.
  - On trip: flatten per policy, alert, persist the latch, exit non-zero.
  - Only `vv resume` clears the latch.
- [ ] Equity valued over all balances from one snapshot; deposits and withdrawals adjust the peak.
- [ ] Remove the RSI(1m) panic sell (or move it to a higher timeframe with cooldown, proven in replay).
- [ ] **Run G1.**

### Phase 3: Production platform (weeks 3–5)
- [ ] **State:** SQLite (WAL) with tables `orders(client_id, exch_id, …, ctx_json)`, `fills`, `equity_snapshots`, `kv(peak, halt_latch, config_hash)`, `events`. Nightly backup.
- [ ] **Startup gates:**
  - config schema valid
  - secrets present
  - clock skew < 1 s (`adjustForTimeDifference`)
  - auth OK
  - `apiRestrictions.enableWithdrawals == false`
  - markets active
  - reconcile exchange orders/balances vs DB
  - "started" alert
- [ ] **Cycle discipline:** build one snapshot per cycle; if any part is stale or failed, take no order actions. Error taxonomy:
  - transient → backoff
  - 429/418 → global pause until ban end
  - auth → halt + alert
  - invalid order → per-symbol cooldown
  - maintenance → paused
- [ ] **Observability:**
  - JSON logs with run id
  - Telegram alerts (halt, restart, auth/ban, stale > 2 min, balance idle > 1 h, rejected config reload)
  - daily digest (equity, realised PnL, fees, drawdown)
  - healthchecks.io dead-man's switch
  - optional Prometheus `/metrics`
- [ ] **Shutdown:** SIGTERM handler finishes the cycle, flushes the DB and closes ccxt; `init: true`.
- [ ] **Security:**
  - trade-only key, withdrawals off, IP-whitelisted, dedicated sub-account
  - separate read-only key for dashboard
  - dashboard bound to localhost (SSH tunnel / Tailscale)
  - never print secrets
- [ ] **Build and release:**
  - split `requirements-live.txt` (ccxt pinned exactly, pandas, numpy, pydantic) from research deps
  - lockfile with hashes
  - Python 3.12+ slim, non-root, multi-stage image
  - CI (pytest, ruff, pip-audit, gitleaks)
  - image pushed by digest; compose pins `@sha256`, so rollback = change digest
- [ ] Replace `setup_vm.sh` / `migrate_to_cloud.sh` (which rsyncs `.env` and hard-codes a host) with docker compose + systemd.

### Phase 4: Paper, then staged capital (weeks 5–12)
- [ ] Testnet for plumbing (1 week), then **mainnet at tiny size for 6 weeks** (G2).
- [ ] Fire drills: kill the container mid-grid, restart during halt, revoke the key, cut the network, fill the disk.
- [ ] Staged capital: 5% → 25% → 100% of target, each stage ≥ 4 weeks and gated on live PnL within the replay band.

### Phase 5: Optional add-ons (only after G2)
- [ ] **Carry sleeve** (or carry-only if G0/G1 failed):
  - spot + perp short at 1–2× on the short leg
  - funding-flip exit rule
  - its own halt (basis, margin ratio, depeg)
  - paper-traded first
- [ ] **ML throttle** (2-week bounded experiment):
  - pre-registered HAR-vs-ML vol forecast for spacing, and regime gate vs EMA rule
  - may only widen or switch off
  - watchdog fallback
  - ships only if it passes G3

## 6. Realistic expectations

- A well-built spot grid on ETH/BTC is a **range-market income strategy with trend risk**. In chop it can earn modest returns; in a sustained downtrend it accumulates inventory and loses roughly in line with its capped exposure. "Steady" comes from the caps and the halt, not from prediction.
- Fees decide everything at this spacing. Maker-only with BNB (0.075%) vs 0.1% taker is the difference between positive and negative expectancy in the synthetic runs (about 5 points/month of fee drag at current churn).
- Funding carry is the steadiest passive return available (about 3–6% per year net, regime-dependent) but carries operational and venue tail risk.
- Treat no number as real until it comes from the production code replayed on real data (G1) and is then confirmed by six weeks of reconciled live fills (G2).
