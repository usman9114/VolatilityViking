import pandas as pd
import numpy as np
from pathlib import Path
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Skipping embeddings.")

# Constants
DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
ETH_FILE = DATA_DIR / "ETH_USDT_4h.parquet"
GAS_FILE = DATA_DIR / "onchain_gas.parquet"
NEWS_FILE = DATA_DIR / "cryptopanic_historical.parquet"
OUTPUT_FILE = PROCESSED_DIR / "train_dataset.parquet"

def load_data():
    """Load raw parquet files."""
    print("Loading data...")
    eth = pd.read_parquet(ETH_FILE)
    gas = pd.read_parquet(GAS_FILE)
    news = pd.read_parquet(NEWS_FILE)
    return eth, gas, news

def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # Fill initial NaNs

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculate MACD."""
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def engineer_eth_features(df):
    """Add technical indicators to ETH data."""
    print("Engineering ETH features...")
    df = df.copy()
    
    # Ensure sorted index
    df = df.sort_index()
    
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # SMA
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'])
    
    # MACD
    df['macd'], df['macd_signal'] = calculate_macd(df['close'])
    
    return df

def engineer_gas_features(df):
    """Add gas features."""
    print("Engineering Gas features...")
    df = df.copy()
    df = df.sort_index()
    
    # Rolling stats
    df['gas_mean_24h'] = df['base_fee_gwei'].rolling(window=6).mean() # 6 * 4h = 24h
    df['gas_std_24h'] = df['base_fee_gwei'].rolling(window=6).std()
    
    return df

def process_news(df):
    """Process and align news data with embeddings."""
    print("Processing news data...")
    # Keep only relevant columns
    cols = ['headline', 'source', 'currencies']
    available_cols = [c for c in cols if c in df.columns]
    df = df[available_cols].copy()
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'datetime' in df.columns:
            df = df.set_index('datetime')
    
    # Ensure UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    # Resample to 4H to match ETH
    # We aggregate text: count, and list of headlines
    resampled = df.resample('4H').agg({
        'headline': lambda x: ' | '.join(x.dropna().astype(str)),
        'currencies': 'count' 
    }).rename(columns={'currencies': 'news_count'})
    
    if HAS_SENTENCE_TRANSFORMERS:
        print("Generating embeddings (this may take a while)...")
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available():
            device = 'mps'
            
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        # We need to encode the AGGREGATED headlines
        headlines = resampled['headline'].tolist()
        
        # Determine batch size based on available memory/device? 
        # 64 is safe-ish for CPU.
        embeddings = model.encode(headlines, show_progress_bar=True, batch_size=32)
        
        # Add embeddings as columns
        emb_df = pd.DataFrame(embeddings, index=resampled.index)
        emb_columns = [f'emb_{i}' for i in range(emb_df.shape[1])]
        emb_df.columns = emb_columns
        
        # Merge back
        resampled = resampled.join(emb_df)
        print(f"Generated {len(emb_columns)} embedding features.")
    
    return resampled

def create_target(df):
    """Create target variable for training."""
    print("Creating target variable...")
    # Target: 1 if next candle close is higher than current close (Binary Classification)
    # OR: Future return (Regression) - Let's do both or just return for now.
    # User said "save it for model training", usually implies a specific target.
    # I'll create 'target_return' and 'target_class'
    
    df['target_return'] = df['close'].shift(-1) / df['close'] - 1
    df['target_class'] = (df['target_return'] > 0).astype(int)
    
    # Drop last row as it has no target
    df = df.dropna(subset=['target_return'])
    return df

def main():
    # Ensure output dir exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    eth, gas, news = load_data()
    
    # Feature Engineering
    eth_processed = engineer_eth_features(eth)
    gas_processed = engineer_gas_features(gas)
    news_processed = process_news(news)
    
    # Alignment
    print("Aligning data...")
    # Left join on ETH index
    # Note: ETH is 4H. Gas might be. News is resampled to 4H.
    
    # Merge Gas
    # We use merge_asof or join. Since we resample, join is good.
    # We assume indexes are timezone aware and compatible.
    
    merged = eth_processed.join(gas_processed, how='left', rsuffix='_gas')
    merged = merged.join(news_processed, how='left', rsuffix='_news')
    
    # Fill NAs
    # Gas: Forward fill (last known gas price)
    merged['base_fee_gwei'] = merged['base_fee_gwei'].fillna(method='ffill')
    merged['gas_ratio'] = merged['gas_ratio'].fillna(method='ffill')
    merged['gas_mean_24h'] = merged['gas_mean_24h'].fillna(method='ffill')
    merged['gas_std_24h'] = merged['gas_std_24h'].fillna(0)
    
    # News: Fill news_count with 0, headlines with empty string
    merged['news_count'] = merged['news_count'].fillna(0)
    merged['headline'] = merged['headline'].fillna("")
    
    # Embeddings: Fill with 0
    emb_cols = [c for c in merged.columns if c.startswith('emb_')]
    if emb_cols:
        merged[emb_cols] = merged[emb_cols].fillna(0.0)
    
    # Create Target
    final_df = create_target(merged)
    
    # Drop defined NaNs (initial rolling windows)
    final_df = final_df.dropna()
    
    print(f"Saving processed data with shape {final_df.shape}...")
    final_df.to_parquet(OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    main()
