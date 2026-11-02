# Arthur Hayes: Advisory Profile

*Simulated advisory lens based on public writings; not affiliated with or endorsed by Arthur Hayes.*

**Evidence note.** Sources 1-23 are Hayes's own Substack essays (cryptohayes.substack.com), read via its public API; I grepped them for relevant passages, not full read-throughs. Claims citing them are **primary**. Sources 24-33 are press or digests seen only as search summaries; claims citing them are **secondary**. The Substack archive has no posts between June 2024 and September 2025. Labels: **[D n]** = documented in source n; **[I]** = team inference. Quotes are verbatim and machine-checked.

## 1. Who he is
- Co-founded BitMEX with Ben Delo and Samuel Reed; first employees Greg Dwyer and Jinming Shao [D 10]. The XBTUSD perpetual launched in May 2016 [D 10].
- **Perp origin.** He says the first perp copied Robert Shiller's 1993 perpetual-futures design, with help from Joseph Wang and Bhavik Patel. His own fix, in mid-June 2016, was a premium-index funding rate (the look-back basis becomes the next funding rate, capped) [D 10]. "Hayes invented funding" is therefore too strong.
- Indicted by DOJ and CFTC for Bank Secrecy Act and Commodity Exchange Act violations [D 10]. Pleaded guilty in 2022, $10M each for the three founders, Hayes two years' probation (some reports add six months' home detention) [D 27, 28]. Exact dates **not verified**.
- Pardoned by Trump, reported 28 March 2025, with Delo and Reed [D 24, 25, 26]. Whether Dwyer and the BitMEX entity were pardoned: **unverified**. Hayes frames the case as political and calls Trump's pro-crypto stance "revenge" [D 10].
- Maelstrom, his family office, launched December 2022 [D 21]; he is CIO and makes the liquid directional trading decisions himself [D 1].

## 2. Core frameworks
1. **Dollar liquidity.** His USD Liquidity Index has three parts: Fed balance sheet, TGA, RRP [D 7]. Falling RRP releases liquidity [D 8]; a rising TGA (e.g. tax day) removes it [D 9]; borrowing without spending drains it [D 14]. He tracks the net of RRP and TGA [D 8]. The explicit "balance sheet minus RRP minus TGA" arithmetic is secondary only [D 33]; the sign logic is primary.
2. **Bitcoin as liquidity barometer.** BTC price follows the price and quantity of dollars and yuan; the four-year cycle is dead, because past tops tracked monetary tightening [D 15]. After BTC peaked in October 2025 and fell about 50%, he asked what he got wrong, and blamed AI capex absorbing roughly $1.5T of newly created dollars [D 3].
3. **Central banks and Treasury.** Expects "stealth QE" through the standing repo facility and reserve-management purchases as Treasury issuance grows [D 14, 17]. Bill issuance to drain RRP was bullish [D 8].
4. **Yen and yield-curve control.** The BOJ holds huge JGB positions after YCC and cannot hike without large losses [D 12, 13]. His preferred resolution: Japan repos Treasuries to the Fed, sells dollars for yen, a dollar-liquidity injection [D 12]; Jan 2026 he also described Fed intervention in yen/JGB [D 11]. By Sept 2026 his "North Star" was EURJPY [D 4].
5. **Cycle and timing.** Macro direction from liquidity, then buys over time. He admits timing is poor and says to go slow [D 23].

## 3. Trading style and risk
- **Sizing and payoff.** Quote: "Only 33% of my trades were profitable, but my sizing was on point" [D 1]. Winners averaged 8.5x the average loser [D 1].
- **Leverage.** Maelstrom's book: "As an un-levered long-only investor, Maelstrom can let time and compound interest do their thang." [D 2]. On altcoin accumulation: "Because I use no leverage in this part of my portfolio, I don’t care if there are massive wicks down in price." [D 6]. He buys with VWAP algos over time [D 23]. Levered exposure came via Strategy and Metaplanet [D 18].
- **Direction.** "I’ll swing the portfolio long or flat on a dime; I never short." [D 4]. [I] This conflicts with secondary reports of put spreads and a 5% staked-USDe election hedge [D 31]; the primary essays describe "financing trades" feeding BTC purchases [D 1].
- **Altcoins.** Concentrated, revenue-sharing tokens (Pendle, ether.fi, HYPE, ENA, ZEC) [D 1, 17]. Entered 2026 near maximum risk with few dollar stables [D 1]. Q1 2026: almost no trading, which he called a no-trade zone [D 2].
- **Stablecoins/DeFi yield.** USDe yield = ETH staking plus perp funding [D 5]. He disclosed Maelstrom is a founding advisor to Ethena and received governance tokens [D 5].
- **Mistakes admitted.** PUMP was his biggest 2025 loss; he should stay away from memecoins [D 1]. Position-sizing lesson from an early gold bet [D 18]. "But this time I got it wrong, so back to basics." [D 3].

## 4. Views on our topics
- **Perps funding.** Expects positive funding in most periods, because ETH is deflationary and USD inflationary [D 5]. Warns funding can stay negative long enough to break USDe's peg [D 5]. Quote: "The funding rate could go so negative that the net asset value of Ethena’s assets is below the amount of issued USDe." [D 5].
- **Basis/carry.** Ran cash-and-carry on ICBIT from 2013 [D 18]. Says falling BTC shrinks basis yield to roughly T-bill, making CEX and smart-contract risk uncompensated [D 12]. Says IBIT's big holders are basis traders, so ETF inflows overstate demand [D 19]. He claims Ethena executed his 2023 synthetic-dollar idea [D 16].
- **Market-neutral, grid, mean-reversion.** No public view found (no matches in his Substack archive; secondary search also empty).
- **Binance and exchange risk.** Calls the US treatment of CZ and Binance absurd, while saying all centralisation is sin [D 20]. Acknowledges Ethena's CEX counterparty risk and argues exchanges are incentive-aligned [D 5]. No public view found on the 10 Oct 2025 Binance USDe episode.
- **Regime shifts.** His whole method: liquidity, yen, Fed reaction, AI-credit shock [D 2, 3, 11, 12].

## 5. "What would Hayes ask?" checklist
1. Which way is dollar liquidity (balance sheet, TGA, RRP) moving, and does the bot assume it is flat? [D 7, 9]
2. Is the bot long-only, unlevered and able to sit flat in a no-trade regime? [D 2, 4]
3. What does a grid do in a one-way liquidity trend or a yen-shock liquidation? [I, from D 12, 15]
4. If funding stays negative for weeks, what is the loss and when does carry exit? [D 5]
5. Does basis yield exceed T-bills by enough to pay for exchange and smart-contract risk? [D 12]
6. How much sits on one venue, and what is the counterparty-failure plan? [D 5, 20]
7. Is sizing built for few winners paying many times the losers? [D 1]
8. What would prove the thesis wrong, and how fast do you act? [D 3]
9. Are you reading flows correctly (e.g. ETF inflows that are basis trades)? [D 19]

## 6. Known biases and blind spots
- **Incentives.** Advisor and token holder in Ethena and Pendle [D 5, 22]; critics (ZachXBT, June 2026) allege followers became exit liquidity, which Hayes disputes [D 32, secondary].
- **Missed calls.** Told readers in Nov 2025 not to mistake weakness for the top [D 14]; BTC later fell about 50% [D 3]. Protos scored most of 20 calls as wrong [D 29]; Decrypt reported his self-rated accuracy as poor [D 30]. Counts **unverified**.
- **Single-factor framing.** [I] Liquidity dominates his model, and by his own account he does not study where liquidity goes [D 3].
- **Politics and directional bias.** [I] Strongly BTC-bullish and political; poor guide to neutral strategies. Price targets are forecasts, not facts.

## 7. Sources
Primary (cryptohayes.substack.com/p/…), date shown:
1. suavemente (2026-01-05)
2. no-trade-zone (2026-04-15)
3. reality-test (2026-06-08)
4. atencion (2026-09-02)
5. dust-on-crust-part-deux (2024-03-08)
6. kite-or-board (2023-08-23)
7. be-present-aff45d6421b4 (2023-02-07)
8. bad-gurl (2023-11-09)
9. heatwave (2024-04-08)
10. adapt-or-die (2025-11-27)
11. woomph (2026-01-27)
12. yen-quake (2026-08-11)
13. the-easy-button (2024-05-20)
14. hallelujah (2025-11-03)
15. long-live-the-king (2025-10-08)
16. the-trifecta (2026-10-26)
17. love-language (2025-12-19)
18. frowny-cloud (2026-01-14)
19. snow-forecast (2025-11-17)
20. panda-power (2023-11-28)
21. shapellas-show-e63d52378f7 (2023-04-11)
22. mayday (2024-05-02)
23. fungible (2023-06-15)

Secondary (search summaries, pages not opened):
24. https://www.theblock.co/post/348719/trump-pardons-bitmex-co-founders-including-arthur-hayes-report (Mar 2025)
25. https://www.bloomberg.com/news/articles/2025-03-28/trump-pardons-bitmex-co-founders-arthur-hayes-benjamin-delo-and-samuel-reed (28 Mar 2025)
26. https://decrypt.co/312169/trump-pardons-arthur-hayes-bitmex-founders (Mar 2025)
27. https://cointelegraph.com/news/arthur-hayes-to-serve-2-year-probation-owning-up-to-bitmex-s-aml-mishap (2022)
28. https://www.bnnbloomberg.ca/bitmex-co-founder-arthur-hayes-sentenced-to-two-years-probation-1.1768761 (2022)
29. https://protos.com/we-calculated-the-accuracy-of-20-market-calls-by-arthur-hayes (about May 2025)
30. https://decrypt.co/283282/bitcoin-billionaire-arthur-hayes-rates-his-market-prediction-accuracy-as-pretty-shit (2024)
31. https://hotfix.coindesk.com/markets/2024/10/28/arthur-hayes-maelstrom-hedges-us-election-uncertainty-with-staked-usde-holds-large-btc-eth-bullish-bets (28 Oct 2024)
32. https://www.kucoin.com/blog/zachxbt-accuses-arthur-hayes-exit-liquidity (Jun 2026; low reliability)
33. https://my.tradingview.com/chart/USCBBS/Giz0oXxR-Arthur-Hayes-USD-Liquidity-Index (chart page)
