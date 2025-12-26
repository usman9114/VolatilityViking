
import ccxt
import pandas as pd
import time
from datetime import datetime

def fetch_data(symbol='ETH/USDT', timeframe='1m', since=None, limit=1000):
    exchange = ccxt.binance()
    all_ohlcv = []
    
    if since is None:
        # Fetch last 365 days
        since = exchange.milliseconds() - (365 * 24 * 60 * 60 * 1000)
    
    print(f"Fetching {symbol} {timeframe} since {datetime.fromtimestamp(since/1000)}")
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            
            all_ohlcv += ohlcv
            since = ohlcv[-1][0] + 1
            print(f"Fetched {len(ohlcv)} candles, Last: {datetime.fromtimestamp(ohlcv[-1][0]/1000)}")
            
            if len(ohlcv) < limit:
                break
            
            time.sleep(0.2)
        except Exception as e:
            print(e)
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df

if __name__ == "__main__":
    df = fetch_data()
    # Ensure data dir exists
    import os
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/ETHUSDT_1m.csv', index=False)
    print(f"Saved {len(df)} rows to data/ETHUSDT_1m.csv")
