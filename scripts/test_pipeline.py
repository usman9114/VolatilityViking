import os
import sys
import pandas as pd
import numpy as np
import torch
import ccxt
from dotenv import load_dotenv

# Path setup
sys.path.append(os.getcwd())
from src.data.news_fetcher import NewsFetcher
from src.models.dataset import CryptoDataset
from sentence_transformers import SentenceTransformer
import mlflow.pytorch

# Reuse utils
from scripts.inference import get_model_path, get_feature_schema

def test_pipeline():
    print("=== T-1 Integration Test: Pre-Flight Check ===")
    
    # 1. Connectivity Check
    load_dotenv()
    api_key = os.getenv('BINANCE_TESTNET_API_KEY')
    print(f"[Check] API Key Loading: {'OK' if api_key else 'FAIL'}")
    
    # 2. Model Loading
    print("\n[Step 1] Loading AI Brain...")
    try:
        model_uri = get_model_path()
        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        model = mlflow.pytorch.load_model(model_uri, map_location=map_location)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        print(f" > Model Loaded: {model_uri}")
    except Exception as e:
        print(f" > FAIL: {e}")
        return

    # 3. Embedding Model
    print("\n[Step 2] Loading News/Sentence Transformer...")
    try:
        sent_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(" > Sentence Transformer Ready.")
    except Exception as e:
        print(f" > FAIL: {e}")
        return

    # 4. Fetch Live Data (Price)
    print("\n[Step 3] Fetching Price Data (Binance)...")
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv('ETH/USDT', timeframe='4h', limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        print(f" > Fetched {len(df)} candles. Last Close: ${df['close'].iloc[-1]}")
    except Exception as e:
        print(f" > FAIL: {e}")
        return

    # 5. Fetch Live News
    print("\n[Step 4] Fetching Live News (CryptoPanic)...")
    try:
        nf = NewsFetcher()
        news_df = nf.fetch_news('ETH', limit=5) # Just get top 5 for visual check
        print(f" > Found {len(news_df)} headlines.")
        if not news_df.empty:
            print(" > Latest Headlines:")
            for t in news_df['title'].iloc[:3]:
                print(f"   - {t}")
                
            # Embed
            print(" > Generating Embeddings...")
            embeddings = sent_model.encode(news_df['title'].tolist())
            mean_emb = np.mean(embeddings, axis=0)
            print(f" > Embedding Vector Shape: {mean_emb.shape} (Expected: (384,))")
            print(f" > Vector Mean Value: {mean_emb.mean():.6f}")
        else:
            print(" > WARNING: No News Found. Using Zero Vector.")
            mean_emb = np.zeros(384, dtype=np.float32)
    except Exception as e:
        print(f" > FAIL: {e}")
        return

    # 6. Feature Assembly
    print("\n[Step 5] Assembling Inputs...")
    feature_schema = get_feature_schema()
    missing_cols = [c for c in feature_schema if c not in df.columns]
    
    # Fill Zeros first
    zero_data = np.zeros((len(df), len(missing_cols)), dtype=np.float32)
    zero_df = pd.DataFrame(zero_data, columns=missing_cols, index=df.index)
    
    # Inject REAL embedding into last row
    if not isinstance(mean_emb, (int, float)):
        last_idx = df.index[-1]
        for i in range(384):
            col = f'emb_{i}'
            if col in zero_df.columns:
                zero_df.loc[last_idx, col] = mean_emb[i]
    
    df_merged = pd.concat([df, zero_df], axis=1)
    df_final = df_merged[feature_schema]
    print(f" > Input DataFrame Shape: {df_final.shape}")

    # 7. Inference
    print("\n[Step 6] Running Inference...")
    try:
        ds = CryptoDataset(dataframe=df_final, seq_len=96, news_emb_path=None, gas_path=None)
        loader, _, _ = ds.get_torch_loaders(batch_size=32, return_split=True)
        
        preds = []
        with torch.no_grad():
            for x_price, x_news, _ in loader:
                x_price, x_news = x_price.to(device), x_news.to(device)
                out = model(x_price, x_news)
                preds.extend(out.cpu().numpy().flatten())
        
        prediction = preds[-1]
        print(f" > Raw Prediction (Next 4h Return): {prediction:.6f}")
        
        # Thresholds
        pred_series = pd.Series(preds)
        rolling_mean = pred_series.rolling(96).mean().iloc[-1]
        rolling_std = pred_series.rolling(96).std().iloc[-1]
        
        print(f" > Stats (96 window): Mean={rolling_mean:.6f}, Std={rolling_std:.6f}")
        
    except Exception as e:
        print(f" > FAIL: {e}")
        return
        
    print("\n=== TEST RESULT ===")
    if prediction > (rolling_mean + 0.86*rolling_std):
        print("SIGNAL: LONG (TREND)")
    elif prediction < (rolling_mean - 0.86*rolling_std):
        print("SIGNAL: SHORT (TREND)")
    else:
        print("SIGNAL: GRID (RANGE)")
        
    print("System verified. Ready for launch.")

if __name__ == "__main__":
    test_pipeline()
