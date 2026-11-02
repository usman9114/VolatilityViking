# G0 decision (2026-11-02)

Inputs: [`G0_RESULTS.md`](G0_RESULTS.md) (8 coins, 2019–2026), plus consultation with Sam (operator) and Maya (quant).

## Result
G0 PASS: BTC, SOL, XRP, DOGE, LINK. G0 fail: ETH, BNB, AVAX (0 of 3 significant years since 2024 for each).

## Caveats raised
- **Thin edge.** BTC's 1h VR has drifted from 0.84 (2021) to 0.95 (2026). Rough break-even grid spacing is about 1–2.5% at VR 0.85–0.90 with a 0.15% round trip. At 0.35% spacing, the grid needs about 43% reversion against the 10–15% measured. G0 is necessary, not sufficient.
- **Not independent.** The majors correlate at about 0.7 and share cascades, so the effective number of independent findings is 2–3.
- **Survivorship.** All 8 coins survived to 2026. G1 must emphasise out-of-sample results from 2024 onward.
- **Single-event years.** The Oct-2025 cascade inflates 1−VR for several alts (huge CIs). A ladder suffers wick-and-snapback events rather than profiting from them.
- **CI method.** AVAX 2020 (~100 days from listing) has its point estimate outside its own bootstrap CI. Percentile intervals from the stationary bootstrap drift toward 1 on short, heavy-tailed samples. The bias is conservative, but the method needs tightening.

## Decision
1. **G0-robust rerun (WP-S0D).** Exclude each coin's first 30 days after listing. Report results both ways: as-is, and with 5m returns winsorised at 5σ plus 2025-10-10..11 excluded. Pooled checks require the Holm-adjusted z* p < 0.05 **and** a CI excluding 1. Use a bias-corrected CI. **The final G1 universe is the coins that pass both versions.**
2. **ETH is a zero-capital shadow control in G1.** It gets capital only if it passes G1 net of fees **and** its trailing-12m VR CI excludes 1. A profitable ETH replay is a red flag that the fill model is too generous.
3. **Portfolio-level risk.**
   - One shared wallet across coins on aligned timestamps.
   - Total exposure cap derived from a stressed loss of about 12% at ρ = 1 (about 35–40% of equity), split by volatility.
   - Portfolio drawdown halt.
   - One shared parameter set in σ units; the only per-coin difference is a volatility scalar.
4. **G1 spacing sweep** must include ≥ 1% steps (pre-registered set; trial count logged).
5. **Funding carry research in parallel (WP-S0E).** Research only; no live carry code before G1 reports.
6. **Wave 1 change.** The strategy core takes a `PortfolioState` (cash, reserved cash for resting buys, all positions), not a per-symbol view. FakeExchange and replay run one wallet across all coins.
