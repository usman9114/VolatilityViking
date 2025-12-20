# How-To Guide

## Setup
Activate the virtual environment before running any commands:
```bash
source /home/usman/eth-bot/venv/bin/activate
```

## 1. Backtesting
Run these commands to verify strategy performance using the latest model/data.

### Quick Inference Check
Runs the pipeline on the last 200 candles to verify the model is working and ready.
```bash
python3 scripts/test_pipeline.py
```

### Full Backtest (Existing Model)
Runs the model on historical data (2018-Present) and generates a performance report.
```bash
python3 scripts/backtest_only.py
```
*Results will be saved to `data/processed/comprehensive_plot.png`*

---

## 2. Live Trading
The bot supports both Testnet (Paper Trading) and Mainnet (Real Money).

### Testnet (Safe Mode)
Runs the bot in Testnet mode. Connects to Binance Testnet.
```bash
# Run with default settings (ETH/USDT, $1000 Capital)
python3 src/live/binance_bot.py

# Custom settings
python3 src/live/binance_bot.py --symbol ETH/USDT --amount 1000
```
*Check `data/logs/live_trades.csv` for trade logs.*

### Mainnet (REAL MONEY)
**WARNING**: This will execute real trades on your Binance account. Ensure you have `BINANCE_API_KEY` and `BINANCE_SECRET_KEY` set in your `.env` file.

```bash
# Enable Live Trading with the --live flag
python3 src/live/binance_bot.py --live --amount 1000
```
*Check `data/logs/live_mainnet_trades.csv` for trade logs.*

---

## 3. Training
To retrain the model from scratch (Full Pipeline):
```bash
python3 main.py --mode train
```

## 4. Remote VM Deployment
**Hostname**: `35.193.28.206`
**User**: `usman.qureshi`
**Directory**: `/home/usman.qureshi/eth`

### Connection
```bash
ssh usman.qureshi@35.193.28.206
cd eth
source venv/bin/activate
```

### Running the Bot (Remote)
Use `nohup` to keep it running after disconnecting.
```bash
# Mainnet (Real Money)
nohup python3 src/live/binance_bot.py --live --amount 1000 > bot.log 2>&1 &

# Monitor Log
tail -f bot.log
```
