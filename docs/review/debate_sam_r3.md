# Sam, round 3: closing statement

**1. Bounded 2-week ML test: yes.** It starts only after the engine passes the Ops gate. The model may only widen spacing or switch the grid off. Baseline is HAR/EWMA plus the 3-line EMA rule. Candidates and trial count are pre-registered, and it must win in every held-out regime. Staleness falls back to the EMA rule, and fallback failure means flatten and halt. Your watchdog answers my 3am question.

**2. Carry: candidate core, not just an add-on.** The grid's evidence got worse: the README chart is an accounting artifact, `tools/optimizer.py` never ran, and there is no real-data edge number. Your 3-6% net on total capital is honest and deflates my 11%. I accept "if the grid fails Gate 1, go carry-only," on two conditions: paper-trade it first, and size it to what custody and ADL could cost.

**3. Merged validation I'd sign.**
- **Caps:** vol-targeted, clipped by static ceilings (30-40% per symbol, 60% total). A static cap is a hidden vol bet, as you said. Spacing is max(2.5x round-trip fee, c x sigma).
- **Fills:** we have no L2 and the sandbox can't reach Binance, so I propose through-tick fills plus a fill-probability haircut, calibrated by paper trading.
- **Rigor:** CPCV with 6 folds, DSR reported with the trial count. I gate on per-regime wins and the Sharpe CI lower bound, not DSR>0.95 alone, since 3 years holds too few independent regimes.
- **Gate 1, Edge:** on live `process_symbol`, net capture at least 2x fees, VR CI excludes 1, out-of-sample Sharpe CI lower bound above 0 at 2x fees, max DD at most 15%.
- **Gate 2, Ops:** your 6 weeks (conceded from 4), fill rate within 25% of sim, zero reconciliation breaks, restart-during-halt drill passed.
- **Gate 3, Add-on:** ML or carry beats baseline in every held-out regime, under 1 hour a month to operate.


**Verdict:** a grid or carry on a sound engine, with ML only as a fail-closed throttle, and no real money until Gates 1 and 2 pass.
