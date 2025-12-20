import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import torch
import mlflow.pytorch
import ccxt
from src.models.dataset import CryptoDataset
import warnings

warnings.filterwarnings('ignore')

def fetch_live_data(symbol='ETH/USDT', limit=500, strict=False):
    try:
        print(f"Connecting to Binance to fetch {symbol}...")
        # Try CCXT first
        exchange = ccxt.binance({
            'timeout': 5000,
            'enableRateLimit': True
        })
        # 4h candles
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=limit)
        if not ohlcv:
            raise ValueError("Empty response from Binance")
            
        print(f"Fetched {len(ohlcv)} candles.")
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        if strict:
            raise IOError(f"Strict Mode: Failed to fetch live price data: {e}")
            
        print(f"CCXT Error: {e}. Using local data tail for demo.")
        df = pd.read_parquet('data/processed/train_dataset.parquet').tail(limit)
        return df

def get_feature_schema():
    """
    Returns the expected feature columns for the model.
    Generates dynamically without requiring the training dataset file.
    """
    # Price features (19 columns from indicators.py)
    price_features = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower',
        'atr', 'adx', 'cci', 'stoch_k', 'stoch_d',
        'obv', 'vwap'
    ]
    
    # News embedding features (384 columns)
    news_features = [f'emb_{i}' for i in range(384)]
    
    # Combine all features
    feature_cols = price_features + news_features
    return feature_cols

def get_model_path(local_dir="models/production"):
    """
    Returns the path to the model.
    Priority:
    1. Local directory (models/production) if it exists and is not empty.
    2. MLflow Best Run (downloads to models/production if not found).
    """
    # 1. Check Local
    if os.path.exists(local_dir) and len(os.listdir(local_dir)) > 0:
        print(f"Loading model from Local Directory: {local_dir}")
        # Check for nested 'model' directory (standard MLflow artifact structure)
        nested_model_path = os.path.join(local_dir, "model")
        if os.path.exists(nested_model_path):
             return nested_model_path
        return local_dir
        
    print(f"Local model not found in {local_dir}. Searching MLflow...")
    
    # 2. MLflow Search
    # Find the best run based on test_mse
    runs = mlflow.search_runs(experiment_names=["ETH-Meta-Trader"], order_by=["metrics.test_mse ASC"])
    if len(runs) == 0:
        raise Exception("No MLflow runs found!")
    
    # Get the best run artifact uri
    best_run = runs.iloc[0]
    run_id = best_run.run_id
    uri = f"runs:/{run_id}/model"
    print(f"Found Best Model in Run: {run_id} (Test MSE: {best_run['metrics.test_mse']:.6f})")
    
    # 3. Download to Local
    print(f"Downloading model artifacts to {local_dir}...")
    os.makedirs(local_dir, exist_ok=True)
    
    # MLflow downloads the "model" directory content into the target
    # We want the content of 'model' artifact to be IN local_dir
    mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model", dst_path=local_dir)
    
    # The download_artifacts might create a subdirectory 'model' inside local_dir or properly dump contents.
    # Usually it downloads `model/*` to `local_dir/model/*` if we specify artifact_path="model"
    # Let's check struct.
    
    # If mlflow creates `models/production/model`, we should probably return that.
    # Let's verify standard behavior or adjust return path.
    full_path = os.path.join(local_dir, "model")
    if os.path.exists(full_path):
        return full_path
        
    return local_dir

def inference():
    print("--- Live Inference (Hybrid Strat) ---")
    
    # 1. Fetch Data
    symbol = 'ETH/USDT'
    print(f"Fetching last 500 candles for {symbol}...")
    df = fetch_live_data(symbol)
    current_price = df['close'].iloc[-1]
    print(f"Current Price: ${current_price:.2f}")
    
    # 2. Feature Alignment
    # The model expects specific columns (Price + News Embeddings).
    # We only have price. We must pad the rest with Zeros.
    feature_schema = get_feature_schema()
    
    # Add missing columns
    missing_cols = [c for c in feature_schema if c not in df.columns]
    if len(missing_cols) > 0:
        print(f"Padding {len(missing_cols)} missing features (News/Gas) with 0...")
        # Create a DataFrame of zeros for missing cols
        zero_data = np.zeros((len(df), len(missing_cols)), dtype=np.float32)
        zero_df = pd.DataFrame(zero_data, columns=missing_cols, index=df.index)
        df = pd.concat([df, zero_df], axis=1)
        
    # Reorder columns matches schema
    df = df[feature_schema]
    
    # 3. Prepare Dataset
    # This handles RevIN and sequences
    # Use inference_mode=True to prevent dropping last row
    ds = CryptoDataset(dataframe=df, seq_len=96, news_emb_path=None, gas_path=None, inference_mode=True)
    loader, _, (n_price, n_news) = ds.get_torch_loaders(batch_size=32, return_split=True, shuffle_train=False)
    
    # 4. Load Model
    model_uri = get_model_path()
    # Handle CPU-only environment
    map_location = None if torch.cuda.is_available() else torch.device('cpu')
    model = mlflow.pytorch.load_model(model_uri, map_location=map_location)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 5. Generate Predictions
    preds = []
    with torch.no_grad():
        for x_price, x_news, _ in loader:
            x_price, x_news = x_price.to(device), x_news.to(device)
            out = model(x_price, x_news)
            preds.extend(out.cpu().numpy().flatten())
            
    # We now have predictions for the sequence.
    # The last prediction corresponds to the forecast for the NEXT candle.
    current_prediction = preds[-1]
    
    # 6. Apply Logic (regime Switching)
    # We need statistics of past predictions (Window=96)
    pred_series = pd.Series(preds)
    
    WINDOW = 96
    K = 0.86
    
    if len(pred_series) < WINDOW:
        print("Warning: Not enough history for dynamic thresholds. Using fallback.")
        threshold_up = 0.01
        threshold_down = -0.01
    else:
        rolling_mean = pred_series.rolling(window=WINDOW).mean().iloc[-1]
        rolling_std = pred_series.rolling(window=WINDOW).std().iloc[-1]
        
        threshold_up = rolling_mean + (K * rolling_std)
        threshold_down = rolling_mean - (K * rolling_std)
        
    print(f"\n--- Strategy Decision ---")
    print(f"Raw Prediction (Next 4h Return): {current_prediction:.5f} ({current_prediction*100:.2f}%)")
    print(f"Thresholds: Low {threshold_down:.5f} | High {threshold_up:.5f}")
    
    # Decision
    signal = "NEUTRAL"
    if current_prediction > threshold_up:
        signal = "LONG (TREND)"
    elif current_prediction < threshold_down:
        signal = "SHORT (TREND)"
    else:
        signal = "GRID (RANGE)"
        
    print(f"SIGNAL: {signal}")
    
    if signal == "GRID (RANGE)":
        # Calculate Rolling Volatility Grid Params
        # Volatility of Returns (last 42 periods = 7 days)
        # We need returns logic here?
        # Actually simplest is to calculate from the Price DF directly
        # Re-calc returns on the fetched DF
        df_prices = fetch_live_data(symbol) # Re-fetch raw or use copy
        returns = np.log(df_prices['close'] / df_prices['close'].shift(1))
        rolling_vol = returns.rolling(window=42).std().iloc[-1]
        daily_vol_usd = rolling_vol * np.sqrt(6) * current_price
        
        # Grid Spacing
        # 50% of 4h volatility (approx)
        grid_step = rolling_vol * current_price * 0.5
        
        # Range (Wide)
        grid_range = 2400 # Or based on Half-Life (from previous stats)
        
        print(f"\n>> GRID CONFIG <<")
        print(f"Volatility (7d): {rolling_vol*100:.2f}%")
        print(f"Rec. Step Size:  ${grid_step:.2f}")
        print(f"Rec. Range:      ${current_price - grid_range:.0f} to ${current_price + grid_range:.0f}")
        
    elif "TREND" in signal:
        print(f"\n>> TREND CONFIG <<")
        print(f"Action: Market Buy/Sell immediately.")
        print(f"Stop Loss: Use {threshold_down if 'LONG' in signal else threshold_up} deviation?")
    
    # Output JSON for Bot
    import json
    output = {
        "timestamp": str(pd.Timestamp.now()),
        "price": current_price,
        "prediction": float(current_prediction),
        "signal": signal,
        "grid_step": float(grid_step) if signal == "GRID (RANGE)" else 0
    }
    # print(json.dumps(output)) # If needed for pipe

if __name__ == "__main__":
    inference()
