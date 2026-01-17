import asyncio
import ccxt.async_support as ccxt_async
import os
from dotenv import load_dotenv

async def debug():
    # 1. Env Check
    load_dotenv('profiles/usman/.env')
    print("API_KEY:", os.getenv('BINANCE_API_KEY'))
    # Print first 5 chars of secret only
    secret = os.getenv('BINANCE_SECRET_KEY') or os.getenv('BINANCE_SECRET')
    print("SECRET:", secret[:5] if secret else "None")
    
    # 2. Ticker Check
    exchange = ccxt_async.binance({'options': {'defaultType': 'future'}})
    tickers = await exchange.fetch_tickers()
    print(f"Total Tickers: {len(tickers)}")
    
    keys = list(tickers.keys())
    print("Sample Keys:", keys[:5])
    
    # Check USDT filter
    usdt = [k for k in keys if k.endswith('/USDT')]
    print(f"Ends with /USDT: {len(usdt)}")
    
    usdt_colon = [k for k in keys if '/USDT:' in k]
    print(f"Contains /USDT:: {len(usdt_colon)}")
    
    await exchange.close()

if __name__ == "__main__":
    asyncio.run(debug())
