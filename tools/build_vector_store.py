import os
import pandas as pd
import sys
from pathlib import Path

# Fix Imports: Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.rag_memory import MarketMemory
from src.ai.features import calculate_alpha_features
from dotenv import load_dotenv

# Load env for OpenAI Key
load_dotenv("profiles/usman/.env")

DATA_DIR = "data/historical_1h"

def main():
    print("Initializing Vector Store Builder...")
    # Force rebuild?
    # shutil.rmtree("data/chroma_db", ignore_errors=True)
    
    sys.stdout.flush()
    
    memory = MarketMemory()
    
    # Get all CSVs
    # Target symbols - AVAX ONLY for now (User Request)
    targets = ["AVAXUSDT.csv"]
    
    # Get all CSVs but filter for targets
    all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    files = [f for f in all_files if f in targets]
    
    print(f"Found {len(files)} target datasets: {files}")
    sys.stdout.flush()
    
    for filename in files:
        # Parse symbol: BTCUSDT.csv -> BTC/USDT
        base = filename.replace('.csv', '')
        symbol = base.replace('USDT', '/USDT')
        filepath = os.path.join(DATA_DIR, filename)
        
        print(f"Loading {symbol} from {filepath}...")
        sys.stdout.flush()
        df = pd.read_csv(filepath)
        
        # --- FEATURE ENGINEERING ---
        print(f"  Calculating features (ADX, RSI, etc)...")
        if 'timestamp' in df.columns:
            df.sort_values('timestamp', inplace=True)
            
        df = calculate_alpha_features(df)
        df.dropna(inplace=True) # Drop warmup
        
        # --- NOISE REDUCTION / SIGNAL FILTERING ---
        # "Focus only on strong signals"
        # Criteria:
        # 1. Trend: ADX > 25
        # 2. Momentum Extremes: RSI > 65 or RSI < 35
        # 3. Volatility Expansion: BB Width > 0.15 (Coin specific, but ok for general)
        
        initial_len = len(df)
        print(f"  Filtering Noise (Initial: {initial_len} candles)...")
        
        mask_trend = df['adx'] > 25
        mask_rsi = (df['rsi'] > 65) | (df['rsi'] < 35)
        mask_vol = df['bb_width'] > 0.10 
        
        # Keep if ANY strong signal condition is met
        df_filtered = df[mask_trend | mask_rsi].copy()
        
        filtered_len = len(df_filtered)
        print(f"  Kept {filtered_len} Strong Signals ({filtered_len/initial_len:.1%} of data).")
        
        if filtered_len == 0:
            print("  ⚠️ No strong signals found? Skipping.")
            continue
        
        # Ingest (sync)
        memory.ingest_batch(df_filtered, symbol)
        print(f"Done with {symbol}.")
        sys.stdout.flush()
        
    print("Vector Store Build Complete!")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
