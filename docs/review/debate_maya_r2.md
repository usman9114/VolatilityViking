# Maya, Round 2

**(a) Sam's questions**
1. *Predict what?* Not direction. Targets: next-4h realized variance (baseline HAR/EWMA) and P(grid cycle stops out within 24h). Standard: purged walk-forward on real 1m ETH/SOL, QLIKE vs baseline, then net-of-fees grid P&L with vs without the forecast, CPCV, deflated Sharpe > 0.95. I have no such result yet, and neither does the README.
2. *Beyond the 3-line rule?* Possibly nothing; that rule is the bar. With 2-3 regimes, pre-register: fix the candidate set before looking, count trials, and require the gain in every held-out regime, not pooled. Pooled-only wins are rejected.
3. *3am staleness?* A watchdog, not a human: model age, feature null-rate, or forecast-vs-realized QLIKE ratio out of bounds means fall back to the EMA rule; fallback failure means flatten and halt, persisted across restarts. The model may only widen or switch off, never add risk, so staleness fails closed.

**(b) Carry sleeve**
The 11% is gross. Roughly half of capital earns it; the rest is perp margin buffer. After four legs, rebalancing and funding-flip months, I expect about 3-6% on total capital in an average year, near zero or negative in bears. Funding is persistent but regime-bound and compresses when crowded. Tails: ADL, single-venue custody risk, basis blowouts forcing margin top-ups at the worst moment, and stablecoin/collateral depeg (the Oct-2025 cascade, from memory). On risk-adjusted terms it likely beats an unproven grid, since it has no directional exposure, but its drawdowns are rare, fat and operational. Paper-trade it before believing any Sharpe.

**(c) Caps and spacing**
Static 30-40% caps are a hidden vol bet: stress loss = cap x crash size, and crash size scales with vol. Vol-target: cap_i = (tolerable loss, e.g. 10% equity) / (k x sigma_daily x sqrt(horizon) x beta), clipped by Sam's static ceilings. Spacing likewise: max(about 2.5x round-trip fee, c x sigma at the fill horizon). At ETH 70% annual vol, 5m sigma is about 0.23%, so 0.35% is a sane default but wrong at half or double vol.

**(d) Agree / disagree**
Agree: plumbing first, ML not the core, fail closed, ETH/BTC only, EMA rule as baseline. Coverage of 36% (14 passing tests exercise local copies) means no repo number is trusted yet.
Disagree: skipping ML tests entirely. A bounded 2-week vol-forecast/regime test is worth it after the engine is sound. Also, carry is a candidate core, not just an add-on: if the grid edge fails, go carry-only.

**Go/no-go gates**
1. **Edge:** live `process_symbol` replayed on 3+ years of real data with queue-aware maker fills. Need VR<1 (CI excludes 1), net capture >= 2x fees, OOS net Sharpe CI lower bound > 0 at 2x fee stress, max DD <= 15% at vol-targeted caps. Fail: no grid.
2. **Ops:** 6 weeks paper/tiny live, zero reconciliation breaks, fill rate within 25% of sim, kill-switch-survives-restart drill passed.
3. **Add-on (ML or carry):** beats baseline on CPCV net Sharpe and max DD in every held-out regime, DSR > 0.95, under 1 hour/month to operate.
