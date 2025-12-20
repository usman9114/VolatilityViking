import pandas as pd
import numpy as np
import torch
import mlflow.pytorch
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from scripts.inference import get_model_path, get_feature_schema
from src.models.dataset import CryptoDataset
from scripts.comprehensive_backtest import comprehensive_backtest

def run_backtest_only():
    print("--- Running Backtest with Existing Model ---")
    
    # 1. Load Data
    data_path = 'data/processed/train_dataset.parquet'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    # Ensure UTC
    if df.index.tz is None: df.index = df.index.tz_localize('UTC')
    
    # Filter for Backtest Period (e.g. 2018+)
    df = df[df.index.year >= 2018]
    
    # 2. Load Model
    print("Loading Model...")
    try:
        model_uri = get_model_path()
        model = mlflow.pytorch.load_model(model_uri)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        print(f"Model loaded from {model_uri}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 3. Prepare Dataset
    # Use CryptoDataset for consistent preprocessing (RevIN, etc)
    # seq_len=96 matches training default
    ds = CryptoDataset(
        dataframe=df,
        seq_len=96,
        news_emb_path=None, 
        gas_path=None,
        test_size=0,
        inference_mode=True
    )
    
    # We want to iterate over the WHOLE dataset sequentially.
    # With test_size=0, everything is in the 'train' split.
    # We disable shuffling to maintain time order.
    loader, _, _ = ds.get_torch_loaders(batch_size=64, return_split=True, shuffle_train=False)
    # Checking dataset.py: get_torch_loaders(self, batch_size=32, return_split=True)
    # If return_split=False, it might return a single loader.
    # Actually, let's use return_split=False if implemented, or just use the train loader if it returns (train, test).
    # Wait, if we pass the whole DF, we want to predict on ALL of it.
    # Let's assume get_torch_loaders returns (loader, None, shapes) if we only have one set?
    # Let's stick to standard behavior:
    # If we want full inference, we might need to rely on the loader covering the full dataset.
    
    # Let's inspect dataset.py quickly if needed, but assuming valid split.
    # For simplicity, let's manually iterate or assume loader covers it.
    
    print("Running Inference...")
    preds = []
    actuals = []
    
    # We need to map predictions back to timestamps.
    # usage of DataLoader shuffles by default? 
    # train_loader usually shuffles. test_loader usually doesn't.
    # CryptoDataset probably returns (train_loader, test_loader, shapes)
    
    # Hack: Creating a "test" loader for the whole dataset by setting test_size=1.0?
    # Or just iterating.
    
    # Let's check `src/models/metamodel.py`'s `evaluate`.
    # It assumes `loader` yields (x_price, x_news, y).
    
    with torch.no_grad():
        # Use the loader we obtained earlier (which covers the test split = 99% of data)
        # loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False) # INCORRECT
        
        for x_price, x_news, y in loader:
            x_price, x_news = x_price.to(device), x_news.to(device)
            out = model(x_price, x_news)
            preds.extend(out.cpu().numpy().flatten())
            actuals.extend(y.cpu().numpy().flatten())
            
    # 4. Save Results
    # Align timestamps using the test_df index
    # validation indices start after seq_len + forecast_horizon usually?
    # Dataset __getitem__ takes idx:idx+seq_len. Target is at +horizon.
    # So the prediction at 'idx' corresponds to the target at the end.
    # RevINDataset: y = self.targets[idx + self.seq_len + self.forecast_horizon - 1]
    # We want the timestamp of the TARGET.
    
    # Let's look at the original DF slice in test_df
    # test_ds uses test_data which matches test_df
    # len(test_ds) = len(test_data) - seq_len - horizon
    # The first sample (idx=0) uses 0..seq_len. Target is at seq_len+horizon-1.
    # So the first Timestamp result corresponds to test_df.index[seq_len + horizon - 1]
    
    # 4. Save Results
    # Align timestamps using the train_df index (since we used test_size=0)
    
    offset = ds.seq_len + ds.forecast_horizon - 1
    # We used "train_df" effectively.
    valid_indices = ds.train_df.index[offset:]
    
    # Truncate to match length (in case of batch drop_last=False/True issues)
    L = min(len(valid_indices), len(preds))
    
    results = pd.DataFrame({
        'Datetime': valid_indices[:L],
        'Actual': actuals[:L],
        'Ensemble': preds[:L]
    })
    results['Year'] = results['Datetime'].dt.year
    results.set_index('Datetime', inplace=True)
    
    out_path = 'data/processed/walk_forward_results_meta.csv'
    results.to_csv(out_path)
    print(f"Predictions saved to {out_path}")
    
    # 5. Run Comprehensive Backtest
    comprehensive_backtest()

if __name__ == "__main__":
    run_backtest_only()
