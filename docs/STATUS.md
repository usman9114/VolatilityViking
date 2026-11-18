# Project status and handoff (2026-11-18)

## Where we are
The bot has been rebuilt from scratch. The old grid bot and its "AI" claims didn't survive honest testing. The live path is now an **evidence-weighted trend book** on Binance spot and USD-M perps. Everything is tested (406 passing) and ready to deploy on the owner's VM. **No signal trades real money yet**: all evidence weights are 0.

## What the research found (all pre-registered; details in `docs/research/`)
| Study | Result |
|---|---|
| G0: mean reversion screen (8 coins) | BTC, SOL, XRP, DOGE and LINK show reversion. ETH, BNB and AVAX don't since 2024 |
| G1 / G1 v2: grid designs (16 configurations: DCA+TP, paired grid, rebalance) | **All fail.** Edge is thin, depends on wick fills, decaying. Kill rule applied |
| Carry (funding) | Narrow pass, driven by 2020–21. Only ~1%/yr in 2025–26 |
| Feature screen (26 external features) | **No direction predictor.** Weak crash and vol signals. Volatility is predictable (mostly from price) |
| G2: trend + vol targeting (spot S1/S2, perp long/short P1/P2) | All fail the strict gate (low statistical power), but **~10%/yr OOS** vs B&H −0.3%. Perps were positive in all 6 years including 2022, at a 22% drawdown |
| Timing / last-week studies | Exploratory. The trend filter and regime gate would have sidestepped the Nov 2026 selloff on most coins |

## Current design (`docs/AI_SIGNAL_PLAN.md` v2)
- **Signals:** G2 trend variants S1, S2, P1 and P2 are in `research/signals/registry.yaml` with status **shadow**. They collect forward evidence and carry no capital.
- **Evidence engine:** hierarchical prior + T1 out-of-sample + forward data → posterior. Weight = ¼-Kelly when P(SR>0) ≥ 0.80, with factor budgets, a ramp cap starting at 5%, and CUSUM decay (h=8). Today P(SR>0) is 0.67–0.75, so every weight is 0.
- **Execution limits:** one-way mode, isolated margin, gross ≤ 1×, shorts ≤ 0.5×, reduce-only exits, persisted halt.
- **AI/LLM signals** (not built yet): forward-only, risk-off veto only.

## Owner decisions on record
- Binance USD-M futures are legal for the owner where the VM runs (2026-11-17).
- No leverage beyond 1× gross. Real money only with the owner's explicit approval.
- The test venue is **Binance Demo Trading** (`VV_BINANCE_TEST_VENUE=demo`). testnet.binancefuture.com now redirects there.

## Next steps (in order)
1. **On the owner's machine or VM:** `bash scripts/bootstrap.sh && make test` (expect 406 passed). Then `docker compose build`. This is the first real image build; it has never been built.
2. Put the Demo keys in `.env` and run `deploy/testnet_smoke.py`. Expect `SMOKE PASS`.
3. **Stage 1, dry run, 1 week:** enable the systemd timers in `deploy/systemd/`. Check `reports/YYYY-MM-DD.md` daily.
4. **Stage 2, demo, 2 weeks:** `--mode testnet`, plus ops drills `deploy/drills/*.sh` (drills 3 and 5 only run on the VM).
5. **Stage 3, shadow:** let forward evidence accumulate. A signal earns weight once P(SR>0) ≥ 0.80.
6. **Stage 4, tiny live:** owner approval required. Tracking band: live − shadow daily P&L mean absolute ≤ 10 bps, 4-week gap ≤ 1%.

## Known gaps / open items
- `docker compose build` hasn't run anywhere yet (the cloud container had no Docker daemon).
- The private Demo smoke test is pending (the Demo endpoint is blocked from the cloud).
- The funding archive ends 2026-10-31. On the VM the runner falls back to the live funding API.
- ETF-flow data (likely the strongest missing input) needs a paid feed.
- Deferred until something earns weight: AI hypothesis agents, the LLM risk-off veto, the full point-in-time store.
- `tests/test_smart_grid.py::test_atr_widens_grid` is an xfail on the old grid engine. Harmless.
- The cloud environment "Crypto" still holds plain-text variables `binance` and `bs_key` (Demo keys). They can't be used from the cloud; delete them.

## Advisors (simulated personas; prior debate in `docs/review/`)
- **Sam** (operator / SWE)
- **Maya** (quant / ML)
- **Hayes lens** (`docs/personas/arthur_hayes.md`; simulated from public writings, not affiliated)

Consult them at gate decisions.
