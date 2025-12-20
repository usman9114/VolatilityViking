# Operational Scenarios (Starting Capital: $100 USDT)

This document outlines how the bot operates in 5 different market situations, assuming you start with **100 USDT** on Binance Mainnet.

## Core Settings
- **Capital Limit**: $100 (The bot will never use more than this).
- **Timeframe**: 4 Hours (AI predicts every 4 hours).
- **Checks**: Every 1 minute (monitors grid/price).

---

### Scenario 1: The "Boring" Start (Neutral Market)
**Context**: You start the bot. The AI predicts a small return (e.g., +0.02%) that is within the "noise" threshold.
**Signal**: `GRID (RANGE)`

1.  **Action**: The bot sees you have 100 USDT and 0 ETH.
2.  **Calculation**: It calculates market volatility (e.g., 2%).
3.  **Grid Deployment**:
    *   It calculates a "Buy Budget" of ~$90 (90% of capital).
    *   It places **5 Buy Limit Orders** spread below the current price (e.g., at -1%, -2%, -3%...).
    *   *Example*: Use $18 per order to buy ETH at $2800, $2780, $2760...
    *   **Sells**: Since you have 0 ETH, it cannot place Sell orders above the price yet. It prints: *"Not enough ETH balance to place Sell Grid"*.
4.  **Outcome**: The bot waits. If price dips to $2800, it buys some ETH. Now you have inventory to sell later.

### Scenario 2: The Bull Run (Strong Uptrend)
**Context**: A few hours later, positive news hits. The AI predicts a return of +1.5% (above the dynamic threshold).
**Signal**: `LONG (TREND)`

1.  **Action**: The bot immediately cancels any open Grid orders to free up your USDT.
2.  **Execution**: It executes a **Market Buy**.
    *   **Size**: ~$95 (95% of $100).
    *   **Price**: Buys ETH at current market price (e.g., $2850).
3.  **Outcome**: You are now "All In" on ETH (minus $5 safety dust). The bot holds this position as long as the signal remains LONG or GRID.

### Scenario 3: The Crash (Strong Downtrend)
**Context**: You are holding ETH from Scenario 2. Bad news hits. AI predicts -2.0% return.
**Signal**: `SHORT (TREND)`

1.  **Action**: The bot switches logic to "Sell Everything".
2.  **Execution**: It checks your ETH balance (acquired in Scenario 2).
3.  **Trade**: It executes a **Market Sell** for 100% of your ETH holdings.
4.  **Outcome**: You are back to USDT (hopefully with profit from the ride up, or minimizing loss from the crash). You sit safely in cash while the price potentially drops further.

### Scenario 4: The "Chop" (Volatility Harvesting)
**Context**: The market stabilizes after the crash. AI predicts flat returns again.
**Signal**: `GRID (RANGE)`

1.  **Action**: You now have USDT again.
2.  **Grid Deployment**: The bot measures volatility again.
    *   It places Buy Orders below the current price.
    *   If you had any leftover ETH dust, it might try to place a small Sell Order above, but mostly it sets up to buy dips.
3.  **Profit Mechanism**:
    *   Price drops -> Buy Order fills (You get ETH cheap).
    *   Price bounces -> The bot (in a future update or if holding inventory) would sell this ETH higher.
    *   *Note*: The current logic resets the grid every 4 hours or if price deviates >2%, helping you re-center your orders.

### Scenario 5: The "Silent" Night (Strict Mode)
**Context**: It's 3 AM. CryptoPanic API goes down, or there is no news in the last 4 hours.
**Signal**: `NONE (ABORT)`

1.  **Action**: The AI attempts to fetch data.
2.  **Strict Check**: It sees `News: []` (Empty).
3.  **Safety Trigger**: Instead of guessing with a "Zero Vector" (which looks like neutral news), the bot **aborts** the prediction cycle.
4.  **Log**: `>> ABORT: Missing Data. Skipping this cycle.`
5.  **Outcome**: No trades are made. The bot sleeps for 1 minute and tries again. Your capital is preserved from "blind" trading.
