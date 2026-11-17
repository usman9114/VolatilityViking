# Evidence-Weighted Signal Engine (EWSE): plan and testing plan

*Draft v1, 2026-11-17. Goal: use AI to find and act on predictive signals for long/short (spot + USD-M perps), where every signal earns its weight through evidence, only strong signals trade, and the research loop never stalls.*

## 1. Core idea
Treat AI as a **hypothesis factory**, not an oracle. Cheap AI agents propose many candidate signals. A strict, automated evidence engine kills most of them. A Bayesian allocator gives money only to the few with strong, accumulating evidence, and takes it away again as evidence decays.

```
 market week ──► post-mortem ──► AI hypothesis agents ──► auto pre-registration
                                                              │
        ┌──────────────── graveyard (never retest) ◄──fail───┤
        ▼                                                     ▼
  history-eligible? ──yes──► Tier-1 replay screen ──pass──► shadow book (forward, paper)
        │ no (AI/LLM/news)                                    │ sequential evidence, daily
        └──────────────────────────────────────────────────► │
                                                              ▼
                       Bayesian allocator (only P(SR>0) ≥ 0.95 trades) ──► vol-targeted long/short book
                                                              │
                              live P&L + decay monitor ───────┘ (demote on decay)
```

## 2. What's new here (and why it fits the AI age)
1. **AI as the researcher, not the predictor.** Agents read essays, papers, event post-mortems and market structure, then write *testable* hypotheses with an expected sign, horizon, data source and kill criterion. A pre-registration file is generated and hashed **before** any data is touched. One operator can run hundreds of honest experiments a year that used to need a quant team.
2. **No LLM backtests (look-ahead by training).** An LLM asked to judge 2022 news already "knows" how 2022 ended, so any historical backtest of an LLM-derived signal is contaminated. Every AI/LLM/news signal is therefore validated **forward only**. Its outputs are stored in an append-only, timestamped store (model id, prompt hash, inputs, output) at the moment they are produced. This is the rule most "AI trading" claims break.
3. **Evidence-weighted, abstaining predictions.**
   - Each signal outputs a direction **and a calibrated confidence**.
   - Conformal / calibrated thresholds mean a signal *abstains* unless its confidence clears a level that held up historically or in forward tests. Fewer trades, higher hit rate: "only a strong signal passes".
   - Signal weight = posterior evidence, not backtest return: w_i ∝ E[SR_i | data], shrunk toward 0 by a sceptical prior (SR ~ N(0, 0.5²) annualised). It is non-zero **only if P(SR_i > 0) ≥ 0.95** with at least 60 effective independent observations.
4. **Sequential testing, so we're never stuck.** Each shadow signal gets an always-valid confidence sequence (a mixture-martingale bound) on its Sharpe. Decisions happen as soon as the evidence allows, not at an arbitrary date:
   - **Promote** when the lower bound > 0.
   - **Retire for futility** when the upper bound < 0.3.
   - Cap shadow time at 26 weeks.

   Every cycle ends in a decision. Nothing waits forever, and failures go to the graveyard with their reason, so they're never re-tested.
5. **Risk-first composition.** The combined book uses only volatility targeting, the risk-off dial (from the feature screen), caps, and gross ≤ 1×. Shorts are allowed on perps. Predictive signals tilt the book; they never lever it.

## 3. Signal families (initial hypothesis queue)
| Family | Example hypothesis | Validation path |
|---|---|---|
| Trend / time-series momentum | 20/60/120-day trend (G2) | History (Tier 1) |
| Positioning / crowding | Crowded long (long/short ratio z > 1) plus a positive funding spike precedes higher crash risk | History (2022+) |
| Liquidity / macro regime | Stablecoin supply contraction plus a rising dollar ⇒ risk-off | History |
| Volatility events | Scheduled CPI/FOMC days ⇒ higher next-day vol, so cut size beforehand | History (calendar is point-in-time) |
| Cross-sectional | Long the strongest / short the weakest coin on 30-day momentum | History |
| **LLM event classifier** | Exchange hacks, delistings, enforcement or ETF-flow news classified daily ⇒ risk-off for N days | **Forward only** |
| **LLM daily regime memo** | Structured risk-on/off probability from that day's news and data | **Forward only** |
| ETF flows (needs a paid feed) | Multi-day outflow streak ⇒ negative next-week drift | History once the data is available |

## 4. Testing plan
| Tier | What | Pass rule | Applies to |
|---|---|---|---|
| T0 Unit / property | No look-ahead (truncation tests), leak canaries (a deliberately leaked feature must be caught), determinism, caps | All green in CI | Everything |
| T1 Historical replay | Pre-registered walk-forward 2019/20→2026: costs ×2, BH-FDR across the cycle's tests, deflated Sharpe with the trial count | OOS Sharpe CI lower > 0 and survives FDR | Point-in-time-clean data only |
| T2 Shadow / forward | Paper P&L from live data; sequential confidence sequence | Promote: lower bound > 0 (P(SR>0) ≥ 0.95). Retire: upper bound < 0.3 or 26 weeks | Everything (mandatory for AI/LLM) |
| T3 Tiny live | 1–5% of target capital, real fills, kill switches | Live vs shadow tracking error within band for 4 weeks; no ops incidents | Promoted combos |
| T4 Scale | 25% → 100% | Rolling evidence stays above threshold; decay monitor quiet | — |

Cross-cutting checks:
- A **fill-realism check** (pierce sweep).
- **Benchmark beats:** cash at 4%, buy-and-hold at the same exposure.
- A **trial ledger** (every hypothesis ever tested counts toward deflation).
- **Ops drills** before T3.

## 5. Weekly loop (cadence, budgets, never stuck)
1. **Mon: automated post-mortem** of the past week (as in `LAST_WEEK_*.md`): what moved, and what each live and shadow signal said beforehand.
2. **Mon: hypothesis agents** (Sonnet, ≤ 5 new hypotheses a week, each ≤ 1 page) write pre-registrations. Haiku checks schema and data availability.
3. **Tue–Wed: T1 screen** for history-eligible hypotheses (CPU budget ≤ 2 h). Failures → graveyard.
4. **Daily: shadow book** updates and the evidence engine runs. Promotions and retirements are automatic, with a human sign-off for anything touching capital.
5. **Fri: allocator re-weights** live signals; a weekly report goes to the team page.
6. **Budgets:** ≤ ~500k tokens a week for agents; a fixed CPU budget; a quarterly review of the whole funnel (hit rate of hypotheses, cost per promoted signal).

## 6. Build order (work packages)
| WP | What | Model |
|---|---|---|
| E1 | Signal registry (YAML: id, hypothesis, sign, horizon, data, prereg hash, status, evidence) plus graveyard | Sonnet |
| E2 | Point-in-time feature store (append-only, ingestion timestamps; LLM outputs with model id and prompt hash) | Sonnet |
| E3 | Evidence engine: Bayesian shrinkage weights, always-valid confidence sequences, promote/retire rules | Sonnet |
| E4 | Shadow book: daily forward P&L per signal and for the combined book (spot and perp, funding) reusing `portfolio_sim` | Sonnet |
| E5 | Allocator: evidence weights → vol-targeted long/short targets with dial and caps | Sonnet |
| E6 | Hypothesis-agent harness: prompt plus template that emits a pre-registration; Haiku schema check | Sonnet + Haiku |
| E7 | Weekly loop runner plus report; scheduled routine | Haiku |
| E8 | Perp/spot execution adapter (testnet first), state, alerts, Docker, for T2/T3 on the operator's VM | Sonnet + Haiku |

## 7. Honest expectations
Most hypotheses will die, and that is the system working. A realistic success is **a few weak, independent signals** combining into a modest, steady Sharpe, with smaller drawdowns than holding. Not a crystal ball. The engine's real value is that it stops us fooling ourselves while letting AI do the research grind at scale.
