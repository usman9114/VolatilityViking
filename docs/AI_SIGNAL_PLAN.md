# Evidence-Weighted Signal Engine (EWSE): plan and testing plan

*v2, 2026-11-17. Revised after review by Sam (operator) and Maya (quant). v1 is in git history. Goal: use AI to find predictive signals for long/short (spot + USD-M perps). Capital follows evidence, strong evidence gets more, and the loop never stalls.*

## 0. What changed from v1, and why
- **v1's hard gate couldn't decide.** It promoted only at P(SR>0) ≥ 0.95 on forward data. Maya's arithmetic: a genuine Sharpe-1.0 signal needs ~7–10 years of daily data to clear that. Sam's: a Sharpe-0.5 signal needs ~11 years. The 26-week cap would have retired every real signal. That's the "getting stuck" trap.
- **v2 sizes by evidence, with no on/off gate.** Capital is proportional to the posterior (fractional Kelly on a shrunk estimate). The prior comes from pre-registered historical tests. Forward data **adds** to that evidence and watches for **decay**; it doesn't have to prove the edge from scratch.

## 1. Core loop
```
hypothesis (AI-assisted, ≤10/yr) ─► pre-registration (hashed) ─► T1 historical walk-forward
          ▲                                                           │ result = informative prior N(SR̂, se²),
          │ graveyard (all trials count)                              │ shrunk by a hierarchical selection prior
          │                                                           ▼
   post-mortems ◄── live & shadow P&L ◄── allocator: weight ∝ posterior SR (¼-Kelly), factor budgets, caps
                                              ▲                       │
                                              └── decay monitor ◄─────┘ (CUSUM vs prior; cut on decay)
```

## 2. Evidence-proportional allocation (replaces the 0.95 gate)
- **Posterior per signal.**
  - Prior: hierarchical N(μ_H, τ²), estimated from **every** hypothesis we've tested, graveyard included. Until enough exist, use μ_H = 0, τ = 0.5.
  - Then update with the T1 out-of-sample estimate (stress costs), then with forward daily returns.
- **Weight:** w_i = ¼ × posterior-mean SR_i / σ_i. It's zero if P(SR_i > 0) < 0.80.
- **Ramp cap** on capital at risk: 5% of target capital to start, scaled by the posterior precision gained since launch. No signal jumps to full size on history alone.
- **Combination:** a new signal's evidence is its **incremental** alpha over the current book, i.e. its shadow P&L regressed on the book. Weights come from a shrunk covariance (Ledoit-Wolf) of shadow P&L. One risk budget applies per factor. Trend, positioning, funding, stablecoin and VIX all load on one "risk-on" factor, so they share a budget instead of stacking.
- **Decay monitor:** a CUSUM of realised minus prior-expected P&L. A breach halves the weight; a second breach sets it to 0 and sends it for review.
- **Worked example, G2 S1 (spot trend).** Stress OOS SR 0.77, se ≈ 0.73 over 2.8 years. With the sceptical prior, the posterior mean is about 0.25 and P(SR>0) about 0.73, so **no weight yet**. One more year forward at SR ~0.8 lifts P(SR>0) above 0.80 and earns a small weight. Evidence ramps capital; it never flips it on all at once.

## 3. Signal types are judged on what they're for
| Type | Role | Judged by | Validation |
|---|---|---|---|
| Directional (trend, cross-sectional momentum) | Tilts long/short | Posterior SR, incremental | T1 prior + forward |
| Risk-off / vol (dial, event calendar, vol forecast) | **Only reduces** exposure | Drawdown reduction, QLIKE of vol forecasts, utility in vol targeting. These have far more statistical power than SR | T1 + forward |
| **AI/LLM signals** (news classifier, regime memo) | **Risk-off veto only. Can never add risk** | Drawdown utility, forward only | Forward only, with strict provenance (below) |

## 4. AI and LLM rules (leak control)
1. **Forward only.** No historical backtests of LLM-derived signals: the model's training already contains the outcomes.
2. **One prompt hash = one signal.** Changing the prompt starts a new signal with a fresh clock. No tuning on forward results.
3. **Timestamp provenance.** Store the article publish time, crawl time and revision. Use the crawl time.
4. **Post-mortem hypotheses** (from AI agents that know 2022–26) are tested **only on data after the post-mortem date**, or run forward-only.
5. **Append-only store:** model id, prompt hash, inputs, output and ingestion timestamp.

## 5. Testing plan
| Tier | What | Rule |
|---|---|---|
| T0 Unit / property | No look-ahead, leak canaries, determinism, caps, perp accounting (funding, margin, liquidation) | CI green |
| T1 Historical | Pre-registered walk-forward. Costs ×2. Trial ledger and deflation. Pierce sweep for fill realism | Produces the **prior**, not a pass/fail |
| T2 Shadow (VM) | Daily forward paper book with real data; tracking error vs simulator | Validates plumbing and decay, not Sharpe |
| T3 Tiny live | Ramp-capped capital and hard limits (below) | Live vs shadow tracking error within band; zero ops incidents for 4 weeks before any ramp-up |
| Ops drills | Restart mid-position, halt persistence, key revocation, network cut, data gap | Must pass before T3 |

**Hard limits in production:**
- one-way position mode
- **isolated margin per symbol**
- gross ≤ 1×
- **shorts capped at 0.5× equity in total**
- reduce-only exits
- a persisted halt that never auto-resumes
- a no-withdrawal API key with an IP allowlist

## 6. Minimum viable loop (build now)
| WP | What | Model |
|---|---|---|
| M1 | **Execution adapter** (USD-M perps + spot) via ccxt. One-way mode, isolated margin, reduce-only exits, clientOrderId, precision/min-notional, time sync, ADL rank logging, reconcile on restart. Tested on the futures testnet (reachable here) and a fake | Sonnet |
| M2 | **Daily runner:** data collector with gap detection → G2 trend targets → evidence allocator → orders (dry-run / testnet / live modes). SQLite state, persisted halt, Telegram alerts, heartbeat | Sonnet |
| M3 | **Evidence engine:** hierarchical prior, posterior, ¼-Kelly weights, factor budgets, CUSUM decay. Registry YAML plus trial ledger | Sonnet |
| M4 | **Deploy:** Docker compose, systemd, runbook, ops-drill scripts for the operator's VM | Haiku |

Deferred until something is ready to scale: AI hypothesis agents (≤10 hypotheses a year, quarterly), the full LLM feature store, the news veto.

## 7. Weekly cadence (operator: ~1 hour a week)
- **Mon:** an automated post-mortem plus the evidence report. Manual 15-minute review.
- **Daily:** the runner (automatic).
- **Quarterly:** 2–3 new pre-registered hypotheses through T1.
- **Every cycle ends in a weight change or a documented "no change".** The loop always moves.

## 8. Honest expectations
- **No signal has proven edge yet.** G2 trend passed 3 of 5 gates and failed on statistical power and on 2022/drawdown. It therefore starts with **zero or tiny weight** and earns more only as forward evidence arrives.
- **Even when a signal earns weight,** expect a modest Sharpe with smaller drawdowns than buy-and-hold, not a crystal ball.
- **Real-money steps** (T3 and every ramp-up) need the owner's explicit approval.
- **Futures need the owner's confirmation** that Binance USD-M futures are legal for them where the VM runs.
