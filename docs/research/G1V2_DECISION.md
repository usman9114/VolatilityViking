# G1 v2 decision: kill rule applied (2026-11-03)

Inputs: [`G1V2_RESULTS.md`](G1V2_RESULTS.md) (16 pre-registered trials, 3 variants), [`G1_DIAGNOSTICS.md`](G1_DIAGNOSTICS.md) and [`G1_DECISION.md`](G1_DECISION.md).

## Result
**No variant passes G1.**

| Variant | OOS 2024 | OOS 2025–26 | OOS Sharpe CI lower | Failed criteria |
|---|---|---|---|---|
| A DCA + take-profit (production) | +6.9% | −1.3% | −0.62 | CI, positive splits |
| B paired grid (Binance Spot Grid style) | +0.9% | −3.5% | −1.52 | capture (1.98), CI, positive splits |
| C threshold rebalance (Binance Rebalancing style) | +17.6% | −5.9% | −0.83 | CI, positive splits, max DD (15.2%) |

Supporting evidence:
- A's profits are wick-dependent: they halve at 0.05% fill depth, and the top-10 days carry 38–158% of P&L.
- A's decay is mostly capture, not the market.
- C trails buy-and-hold at the same exposure in every split, so its returns are beta, not edge.
- The ETH controls all fail, as expected.
- A reproduces G1 v1 exactly (determinism check).

## Decision (per the pre-agreed kill rule in G1_DECISION.md)
1. **Stop grid strategy development.** None of the 16 tested configurations shows an out-of-sample edge after costs.
2. **No leverage.** That was already decided; the evidence now is stronger.
3. **No real money in any grid variant.**

## Lessons recorded for any future work
- **Latched halts in a long-only spot book lock in bear-market losses.** The 15% portfolio halt flattened near the lows; C lost 13.3% in 2022 against 5.6% for buy-and-hold at the same exposure. Any future design needs re-arming halts and a drawdown budget sized to its exposure.
- The 2024+ out-of-sample period is spent. Any new hypothesis needs paper trading as its holdout.
- Measured mean reversion (G0) didn't translate into capturable grid P&L after realistic fills.

## What remains useful
The hardened engine:
- pure `decide()` with C1–C5 fixed and tested
- FakeExchange
- replay
- research tooling
- ~300 tests

It's a sound base for testing any future idea quickly and honestly.
