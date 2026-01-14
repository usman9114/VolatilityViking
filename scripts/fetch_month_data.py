
import ccxt
import pandas as pd
import time
from datetime import datetime
import os
import argparse

def fetch_month_data(symbol='ETH/USDT', days=30):
    exchange = ccxt.binance()
    all_ohlcv = []
    
    # ms
    now = exchange.milliseconds()
    since = now - (days * 24 * 60 * 60 * 1000)
    
    print(f"Fetching {symbol} 1m data for last {days} days...")
    
    while True:
        try:
            # Limit 1000 candles per call
            ohlcv = exchange.fetch_ohlcv(symbol, '1m', since=since, limit=1000)
            if not ohlcv:
                break
            
            all_ohlcv += ohlcv
            last_ts = ohlcv[-1][0]
            since = last_ts + 1
            
            # Progress param
            fetched_end = datetime.fromtimestamp(last_ts/1000)
            if len(all_ohlcv) % 10000 == 0:
                 print(f"Fetched up to {fetched_end}")
            
            if last_ts >= now:
                break
                
            time.sleep(0.05) 
        except Exception as e:
            print(e)
            time.sleep(1)
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Remove duplicates if any
    df.drop_duplicates(subset='timestamp', inplace=True)
    
    # Filter exact range
    start_ts = now - (days * 24 * 60 * 60 * 1000)
    df = df[df['timestamp'] >= start_ts]
    
    return df

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', help='Symbol to fetch (e.g. AVAX/USDT)', default=None)
    parser.add_argument('--days', type=int, default=30, help='Days of data')
    args = parser.parse_args()
    
    if args.symbol:
        symbols = [args.symbol]
    else:
        symbols = ['ETH/USDT', 'SOL/USDT', 'PEPE/USDT', 'BTC/USDT']
    
    for symbol in symbols:
        try:
            df = fetch_month_data(symbol, args.days)
            clean_sym = symbol.replace('/', '')
            filename = f'data/{clean_sym}_1m_30d.csv'
            df.to_csv(filename, index=False)
            print(f"Saved {len(df)} rows to {filename}")
        except Exception as e:
            print(f"Failed to fetch {symbol}: {e}")
