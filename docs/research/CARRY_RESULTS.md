# Funding carry research (WP-S0E)

Long spot + short USDT-M perpetual, 50% spot / 50% perp margin (1x), net of fees (spot maker 0.075%, perp maker 0.02%, both legs on entry, exit and monthly resize), idle cash earns 0. Hourly simulation, Jan 2020 to Oct 2026. Exit rule: flat when trailing 7d funding < 0, re-enter when > 0. Reproduce: `python -m research.carry_fetch && python -m research.carry_analysis`.

**Gross funding, annualised, % of perp notional**

| symbol | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|
| BTC | 17.2% | 30.6% | 4.2% | 7.9% | 11.9% | 5.1% | 3.1% |
| ETH | 27.4% | 37.5% | 0.8% | 8.3% | 13.0% | 4.9% | 2.1% |
| SOL | -12.5% | 28.6% | -38.0% | 1.3% | 13.6% | 0.4% | -0.6% |
| XRP | 24.6% | 48.1% | 0.0% | 8.2% | 14.2% | 3.5% | 0.2% |
| DOGE | 9.4% | 38.5% | 3.4% | 10.0% | 14.0% | 4.3% | 3.2% |
| LINK | 11.5% | 43.0% | 5.9% | 10.2% | 13.3% | 5.1% | 4.6% |
| BNB | 2.5% | 21.7% | -12.6% | -8.3% | -3.3% | -2.1% | 2.8% |
| AVAX | -12.7% | 31.0% | -5.2% | 7.5% | 10.5% | 0.1% | 1.3% |

**Negative funding periods, % of periods (longest streak in periods in table below)**

| symbol | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|
| BTC | 14% | 7% | 22% | 10% | 8% | 13% | 25% |
| ETH | 3% | 4% | 34% | 9% | 4% | 16% | 29% |
| SOL | 25% | 5% | 46% | 28% | 10% | 39% | 46% |
| XRP | 5% | 4% | 39% | 16% | 7% | 29% | 46% |
| DOGE | 15% | 8% | 26% | 10% | 10% | 25% | 31% |
| LINK | 17% | 1% | 17% | 7% | 10% | 21% | 21% |
| BNB | 18% | 21% | 51% | 29% | 22% | 18% | 1% |
| AVAX | 24% | 12% | 48% | 18% | 23% | 38% | 36% |

**Net carry per symbol WITH exit rule, annualised, on total capital**

| symbol | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|
| BTC | 9.9% | 19.3% | 1.4% | 3.9% | 6.4% | 2.3% | 0.8% |
| ETH | 17.5% | 26.8% | 0.5% | 4.4% | 6.9% | 2.2% | 0.2% |
| SOL | 1.8% | 25.6% | 0.5% | 3.3% | 7.4% | -1.2% | -0.2% |
| XRP | 16.6% | 35.6% | -0.4% | 4.1% | 8.1% | 0.9% | -0.8% |
| DOGE | 5.3% | 19.2% | 0.8% | 4.7% | 8.8% | 1.7% | 0.4% |
| LINK | 11.5% | 29.5% | 2.9% | 5.2% | 7.5% | 1.6% | 2.1% |
| BNB | 9.2% | 21.5% | -0.1% | 0.9% | 2.0% | 0.3% | 1.4% |
| AVAX | 5.0% | 23.3% | -0.5% | 5.1% | 6.1% | 0.7% | -0.6% |

**Per symbol, whole period** (net = with exit rule; hold = always in)

| symbol | longest neg streak | net CAGR | maxDD | Sharpe | hold CAGR | hold maxDD | worst basis 1h | worst basis 1d |
|---|---|---|---|---|---|---|---|---|
| BTC | 24 | 6.26% | 1.3% | 7.83 | 6.65% | 1.5% | 2.34% | 2.46% |
| ETH | 25 | 8.14% | 1.4% | 7.08 | 8.42% | 1.4% | 1.46% | 2.46% |
| SOL | 53 | 5.51% | 5.1% | 3.59 | 3.12% | 15.4% | 11.18% | 23.87% |
| XRP | 28 | 8.71% | 4.6% | 5.61 | 9.06% | 4.6% | 6.94% | 6.67% |
| DOGE | 14 | 5.94% | 28.6% | 1.50 | 6.64% | 27.5% | 6.43% | 11.31% |
| LINK | 19 | 8.23% | 2.1% | 5.96 | 8.15% | 3.7% | 2.70% | 2.82% |
| BNB | 46 | 4.62% | 1.4% | 3.37 | 1.75% | 14.1% | 1.64% | 2.75% |
| AVAX | 48 | 5.54% | 2.3% | 3.99 | 4.71% | 21.3% | 4.81% | 7.38% |

Basis = perp/spot - 1 (hourly closes); worst moves are the largest rise of the basis (adverse for the short perp), in percentage points of price.

**Top-3 portfolio** (rank by trailing 30d funding at each month start, equal weight 1/3, each sleeve obeys the exit rule; start 2020-02-01)

| year | active days | net (period) | net (annualised) |
|---|---|---|---|
| 2020 | 335 | 12.44% | 13.55% |
| 2021 | 365 | 32.20% | 32.20% |
| 2022 | 365 | 1.69% | 1.69% |
| 2023 | 365 | 3.99% | 3.99% |
| 2024 | 366 | 7.25% | 7.25% |
| 2025 | 365 | 1.41% | 1.41% |
| 2026 | 304 | 0.81% | 0.97% |

Max drawdown 7.93%, Sharpe 6.76, fees paid 10.18% of capital over 2465 days, funding received 82.1%, basis MTM 0.4%, 190 entries / 187 exits.

**Margin risk.** Monthly-only resizing lets notional drift when a coin rallies mid-month. Sleeve periods where the short perp lost >= 90% of its margin (no top-up, liquidation not simulated): portfolio 12; single-symbol runs BTC 0, ETH 1, SOL 5, XRP 4, DOGE 7, LINK 1, BNB 2, AVAX 6. Sensitivity, resizing also when notional drifts > 25%: margin events 1, median 3.37%, worst year 0.92%, max drawdown 2.08%, Sharpe 6.57, fees 10.03% (PASS).

## Gate CARRY-0: PASS

Median annual net 3.99% (need >= 3%), worst year 0.97% (need >= -2%), max drawdown 7.93% (need <= 8%). Years counted (>= 90 active days): 2020, 2021, 2022, 2023, 2024, 2025, 2026; partial years annualised linearly. Using un-annualised partial-year returns instead: median 3.99%, worst 0.81% (PASS).

**Reading.** The literal gate passes, but narrowly on drawdown and on the back of 2020-21 (annualised 13.6% and 32.2%). The current regime is thin: 2025-26 nets 1.4%, 1.0%, below the 3% bar, and most symbols had 25-45% negative funding periods in 2025-26. Fees consume roughly 1.5%/yr of capital through ~190 re-entries. Idle cash earns 0 here; parking it in T-bills or Earn would add to the figures. The monthly-only rebalance is not survivable on hot coins (see margin risk), so a live carry sleeve needs threshold rebalancing and margin top-ups.
