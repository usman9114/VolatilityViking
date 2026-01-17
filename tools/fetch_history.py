import os
import pandas as pd
import ccxt
import numpy as np
import time
import sys
from datetime import datetime, timedelta

# Configuration
SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 
    'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT', 
    'MATIC/USDT', 'LTC/USDT', 'UNI/USDT', 'ATOM/USDT', 'NEAR/USDT'
]
TIMEFRAME = '1h'
START_DATE = "2023-01-01 00:00:00"
OUTPUT_DIR = "data/historical_1h"

from src.ai.features import calculate_alpha_features

def fetch_ohlcv(exchange, symbol, start_ts, end_ts):
    """
    Fetch OHLCV data properly handling pagination.
    """
    all_candles = []
    current_ts = start_ts
    limit = 1000 
    
    print(f"Fetching {symbol} from {datetime.fromtimestamp(start_ts/1000)}...")
    sys.stdout.flush()
    
    while current_ts < end_ts:
        try:
            candles = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=current_ts, limit=limit)
            if not candles:
                break
            
            all_candles.extend(candles)
            current_ts = candles[-1][0] + 1  # Move to next ms
            
            # Show progress every 5000 candles
            if len(all_candles) % 5000 == 0:
                print(f"  Fetched {len(all_candles)} candles... Last: {datetime.fromtimestamp(candles[-1][0]/1000)}")
                sys.stdout.flush()
            
            # Rate limit handling
            time.sleep(0.2) 
            
            if len(candles) < limit:
                break 
                
        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")
            sys.stdout.flush()
            time.sleep(1)
            
    return all_candles

def main():
    print("Starting Ingestion...")
    sys.stdout.flush()
    
    exchange = ccxt.binance({'enableRateLimit': True}) 
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d %H:%M:%S")
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(time.time() * 1000)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    for symbol in SYMBOLS:
        print(f"Processing {symbol}...")
        sys.stdout.flush()
        
        # 1. Fetch
        raw_data = fetch_ohlcv(exchange, symbol, start_ts, end_ts)
        
        if not raw_data:
            print(f"No data for {symbol}, skipping...")
            sys.stdout.flush()
            continue
            
        # Convert to DF
        df = pd.DataFrame(raw_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 2. Process
        processed_df = calculate_alpha_features(df)
        
        # Add Answer Key (Labels) locally since shared lib is pure features
        processed_df['future_ret_4h'] = processed_df['close'].shift(-4) / processed_df['close'] - 1
        processed_df['future_ret_24h'] = processed_df['close'].shift(-24) / processed_df['close'] - 1
        processed_df.dropna(inplace=True)
        
        # 3. Save
        safe_sym = symbol.replace('/', '')
        out_path = os.path.join(OUTPUT_DIR, f"{safe_sym}.csv")
        processed_df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
