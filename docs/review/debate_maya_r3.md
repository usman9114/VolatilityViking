# Maya, Closing Statement

The README chart is an accounting artifact (`backtester.py:103-108`) and `tools/optimizer.py` never produced a result. Nothing in the repo is evidence yet.

**1. Minimum rigor.** Sam wins: tick/L2 replay, CPCV and PBO are over-built for two symbols and 3-5 frozen parameters. I sign his stack (1m klines, through-tick fills, 2x fees, 3-4 splits, 4 weeks reconciled paper) with three additions:
- One split must be a crash (2022). Report 99th-percentile max drawdown and time underwater; "3 of 4 positive" with one -40% split is a fail.
- A written trial count, with the headline Sharpe haircut by hand.
- Paper fills reconciled per grid level, not only in aggregate.

Data must be pulled outside this sandbox (Binance returns 403 here).

**2. Halt design.** Keep it: persisted flag, manual `--resume`, non-zero exit. One vol-aware change: -15% means different things at 40% and 120% annual vol. Trigger on the lower of the static limit and about 3 daily sigma of current inventory, and shrink caps as vol rises so the halt stays a last defense, not the sizing mechanism. Add a time trip: underwater for N days with no completed cycle.

**3. Sequencing and kill criterion.** ML enters last: only after the engine fixes and all three of Sam's gates pass, and the EMA rule is a measured baseline. Then one pre-registered test (HAR spacing, regime gate), adopted only if it beats the rule in every held-out split. Kill the grid if real-data VR is not below 1 at grid horizons (CI excluding 1), or if net replay at 2x fees fails 3-of-4 splits. Pivot to carry-only, paper-traded first, with its own halt (basis, margin, depeg).

**Concessions.** An ML gate at launch was wrong. Carry is an equal fallback. Engineering gates come first.

**Verdict:** Grid, built honestly and proven on real data first; ML is a late optional filter, and carry is the fallback if the edge isn't there.
