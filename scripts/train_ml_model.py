
import os
import sys
import logging
import pandas as pd
import ccxt
import time
import json
from datetime import datetime
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.ai.ml_predictor import MLPredictor

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_historical_data(symbol, timeframe='1m', days=30):
    """
    Fetch historical candles from Binance (Generic/Public data).
    Mimics FreqAI data download.
    """
    logger.info(f"Fetching {days} days of {timeframe} data for {symbol}...")
    
    exchange = ccxt.binance({
        'enableRateLimit': True
    })
    
    since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
    all_ohlcv = []
    
    try:
        while since < exchange.milliseconds():
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv:
                break
            
            since = ohlcv[-1][0] + 1
            all_ohlcv += ohlcv
            
            # Simple progress log
            dt = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
            print(f"\rFetched up to {dt}", end="", flush=True)
            
            time.sleep(0.5) # Rate limit respect
            
        print("\nDownload Complete.")
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
        
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def fetch_funding_history(symbol, days=30):
    """
    Fetch funding rate history from Binance Futures.
    Crucial for identifying Long/Short squeezes.
    """
    logger.info(f"Fetching {days} days of Funding Rates for {symbol}...")
    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    
    since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
    all_funding = []
    
    try:
        while since < exchange.milliseconds():
            # Binance limits funding history to 1000 items
            rates = exchange.fetch_funding_rate_history(symbol, since, limit=1000)
            if not rates:
                break
            
            since = rates[-1]['timestamp'] + 1
            all_funding += rates
            
            # Simple progress
            time.sleep(0.5)
            
        if not all_funding:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_funding)
        df = df[['timestamp', 'fundingRate']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # Funding is every 8h usually, we forward fill it to 1m candles later
        return df.sort_values('timestamp')
        
    except Exception as e:
        logger.warning(f"Error fetching funding history (Spot symbol?): {e}")
        return pd.DataFrame()

def main():
    # Load Config
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config.json: {e}")
        return

    symbols = config.get('symbols', ['ETH/USDT'])
    
    # Add Forager symbols if enabled
    if config.get('forager', {}).get('enabled', False):
        forager_syms = config.get('forager', {}).get('approved_symbols', [])
        symbols.extend(forager_syms)
        
    # Deduplicate
    symbols = list(set(symbols))
    
    logger.info(f"Starting FreqAI-Lite Training Loop for {len(symbols)} symbols: {symbols}")
    
    # 1. Fetch Data for ALL symbols
    all_data = {}
    
    # Get lookback from config (default 30)
    lookback = config.get('ml_settings', {}).get('lookback_days', 30)
    logger.info(f"Training on {lookback} days of data per symbol.")
    
    for symbol in symbols:
        df = fetch_historical_data(symbol, days=lookback)
        if not df.empty:
            # 1.5 Fetch Funding Data
            funding_df = fetch_funding_history(symbol, days=lookback)
            
            if not funding_df.empty:
                # Merge: OHLCV is 1m, Funding is 8h.
                # We merge on timestamp (nearest/ffill)
                df = pd.merge_asof(df, funding_df, on='timestamp', direction='backward')
                df['fundingRate'] = df['fundingRate'].fillna(0) # Fill initial NaN
            else:
                 df['fundingRate'] = 0.0
            
            all_data[symbol] = df
        else:
            logger.warning(f"Skipping {symbol} due to no data.")
            
    if not all_data:
        logger.error("No data fetched for any symbol. Exiting.")
        return
        
    # 2. Combine Data for Global Training
    # We add a 'symbol_id' feature? Or just mix them? 
    # Mixing is fine for general crypto physics (Pump/Dump dynamics are similar).
    combined_df = pd.concat(all_data.values(), ignore_index=True)
    logger.info(f"Global Dataset Size: {len(combined_df)} candles")
    
    # 3. Train Model
    predictor = MLPredictor()
    logger.info("Training Global XGBoost Model...")
    
    # We modify train_model slightly to return the split data if we wanted to verify strictly,
    # but MLPredictor sorts that internally.
    # For reporting "Per Coin Accuracy", we should train on a TRAIN set, and then test on each coin's TEST set.
    # MLPredictor.train_model splits 80/20 internally.
    # Let's trust the internal split for the 'Global Score' and then run a manual validation pass.
    
    success, global_acc = predictor.train_model(combined_df)
    
    if success:
        logger.info("="*40)
        logger.info(f"✅ GLOBAL MODEL TRAINED SUCCESSFULLY")
        logger.info(f"🎯 Global Test Accuracy: {global_acc:.2%}")
        logger.info(f"💾 Model Saved to: {predictor.model_path}")
        logger.info("="*40)
        
        # 4. Per-Coin Verification (Accuracy on recent data)
        # We use the predictor (which now has the trained model) to predict on each coin's dataframe.
        # We'll take the LAST 20% of each coin's data as a "Proxy Test Set".
        
        print("\n📊 PER-COIN ACCURACY REPORT:")
        print(f"{'SYMBOL':<15} | {'ACCURACY':<10} | {'PREDICTION (Next 5m)':<25}")
        print("-" * 60)
        
        for symbol, df in all_data.items():
            # Create targets manually to check accuracy
            # predictor.prepare_features doesn't create targets, it creates X.
            # We need to recreate the target logic here locally to verify.
            
            # Feature Prep
            df_test = df.copy()
            # Recreate Target Logic from MLPredictor (DRY violation but acceptable for script)
            lookahead = 5
            threshold = 0.005
            df_test['future_close'] = df_test['close'].shift(-lookahead)
            df_test['change'] = (df_test['future_close'] - df_test['close']) / df_test['close']
            df_test['target'] = 0
            df_test.loc[df_test['change'] > threshold, 'target'] = 1
            df_test.loc[df_test['change'] < -threshold, 'target'] = 2
            df_test = df_test.dropna()
            
            # Use last 20%
            split_idx = int(len(df_test) * 0.8)
            test_subset = df_test.iloc[split_idx:]
            
            X_test = predictor.prepare_features(test_subset)
            y_test = test_subset['target']
            
            preds = predictor.model.predict(X_test)
            acc = (preds == y_test).mean()
            
            # Current Prediction
            next_move, conf = predictor.predict_next_move(df)
            move_map = {0: "CHOP 🦀", 1: "PUMP 🚀", 2: "DUMP 📉"}
            move_str = f"{move_map[next_move]} ({conf:.0%})"
            
            print(f"{symbol:<15} | {acc:.1%}    | {move_str:<25}")
            
        print("-" * 60)

    else:
        logger.error("Training Failed.")

if __name__ == "__main__":
    main()
