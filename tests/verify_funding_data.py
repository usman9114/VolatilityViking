
import sys
import os
import pandas as pd
# Add scripts to path to import fetch logic
sys.path.append(os.getcwd())
from scripts.train_ml_model import fetch_historical_data, fetch_funding_history
from src.ai.ml_predictor import MLPredictor

print("Testing Data Pipeline upgrade...")

# 1. Fetch small amount of data
symbol = "ETH/USDT"
print(f"Fetching 1 day OHLCV for {symbol}...")
df_ohlcv = fetch_historical_data(symbol, days=1)

print(f"Fetching 1 day Funding for {symbol}...")
df_funding = fetch_funding_history(symbol, days=1)

if df_funding.empty:
    print("❌ Failed to fetch funding data!")
    sys.exit(1)

# 2. Merge
print("Merging...")
df = pd.merge_asof(df_ohlcv, df_funding, on='timestamp', direction='backward')
df['fundingRate'] = df['fundingRate'].fillna(0)

print(f"Merged Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# 3. Feature Gen
print("Generating Features...")
predictor = MLPredictor()
features = predictor.prepare_features(df)

if 'funding_rate' in features.columns and 'funding_ma' in features.columns:
    print("✅ SUCCESS: 'funding_rate' and 'funding_ma' features generated!")
    print(features[['funding_rate', 'funding_ma']].tail())
else:
    print("❌ FAILURE: New features missing.")
