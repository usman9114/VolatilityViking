# ETH-AI: Advanced Ethereum Trading Bot

## 🚀 Project Overview
**ETH-AI** is a sophisticated quantitative trading bot designed to trade Ethereum (ETH) using deep learning and alternative data. Unlike simple moving-average bots, it uses a **Metamodel** approach—combining specific strengths of different AI architectures to predict market moves.

**Current Performance (2019-2025):**
- **Total Return**: +1,395% (14x ROI)
- **Strategy**: Spot Long-Only (No Leverage, No Shorting)
- **Risk Profile**: Go-to-Cash during crashes.

---

## 🧠 The Brain (AI Strategy)
The bot uses an **Ensemble Model** that dynamically weights two distinct trading engines based on their recent performance:

1.  **LSTM (Deep Learning)**:
    - *Strength*: Capturing complex, non-linear time-series patterns.
    - *Role*: The "Intuition" engine that looks at price history sequences.
2.  **XGBoost (Gradient Boosting)**:
    - *Strength*: Handling regime changes and tabular features.
    - *Role*: The "Logic" engine that analyzes hard metrics (Gas Fees, RSI levels).

**How It Decides:**
The **Walk-Forward Engine** tests both models every year. If LSTM did better in 2023, it gets more voting power in 2024.

---

## 📊 The Data (Features)
We feed the AI a 360-degree view of the market using **411 distinct features**:

### 1. Market Data (Binance)
- **Price Action**: Open, High, Low, Close, Volume.
- **Technicals**: RSI, MACD, Bollinger Bands, ATR (Volatility), etc.
- *Purpose*: Teaches the bot standard technical analysis patterns.

### 2. Sentiment Analysis (News)
- **Source**: Historic crypto news headlines.
- **Tech**: `SentenceTransformer` (BERT-based) converts text into numerical vectors (Embeddings).
- *Purpose*: Allows the bot to "read" the news and detect fear/hype cycles.

### 3. On-Chain Data (Infura)
- **Metric 1: Base Fee Trends**: High fees often signal network congestion (tops/capitulation).
- **Metric 2: Gas Ratio**: Measures how full blocks are (demand proxy).
- *Purpose*: Provides fundamental signals about actual network usage, which price action can miss.

---

## 🛡️ The Strategy: "Spot Alpha"
We strictly adhere to a **Spot Long-Only** risk profile as per your requirements.

- **Buy Signal**: When the AI predicts a positive return > 0.5%.
    - *Action*: Buy ETH (100% Allocation).
- **Sell Signal**: When the AI predicts a drop or flat market.
    - *Action*: Sell to stablecoin (USDT) and wait.
    - *Benefit*: Avoids holding through massive -80% bear markets.
- **No Shorting**: We never bet against ETH with borrowed money.
- **No Leverage**: We never borrow extra money to trade larger size.

---

## 🛠️ How to Use

## 🛠️ How to Use

### 1. Setup
Ensure dependencies are installed and `.env` is configured:
```bash
pip install -r requirements.txt
# .env file:
# BINANCE_API_KEY=...
# INFURA_PROJECT_ID=...
```

### 2. Live Dashboard (Monitoring) 📊
Launch the Control Room to see the bot in action:
```bash
streamlit run src/dashboard/app.py
```

### 3. Training Mode (Historical) 🏫
Use the unified CLI to fetch data, train the AI, and generate a report:
```bash
# Standard Run (2018-2025)
python main.py --mode train --symbol ETH/USDT --timeframe 4h

# Fast Test (Debug Mode - Last 3 Months)
python main.py --mode train --symbol ETH/USDT --debug
```

### 4. Live Production Mode 🔴
Start the infinite trading loop (Fetch -> Predict -> Log -> Sleep):
```bash
python main.py --mode live --symbol ETH/USDT
```
*Note: This logs predictions to `data/predictions.db` which the Dashboard reads in real-time.*

### 5. Multi-Coin Support 🌍
Want to trade Solana?
```bash
python main.py --mode train --symbol SOL/USDT
```



---

## 📈 Why This Helps You
1.  **Automation**: Removes emotional decision-making (FOMO/Panic Selling).
2.  **Alpha**: Uses data humans can't process (400+ indicators simultaneously).
3.  **Capital Preservation**: The "Cash Mode" aims to protect gains during crypto winters.
