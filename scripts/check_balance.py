
import os
import ccxt
from dotenv import load_dotenv

def check_balance():
    load_dotenv()
    api_key = os.getenv('BINANCE_TESTNET_API_KEY')
    secret_key = os.getenv('BINANCE_TESTNET_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("Error: API keys not found in .env")
        return

    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': secret_key,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    exchange.set_sandbox_mode(True)
    
    try:
        balance = exchange.fetch_balance()
        usdt_total = balance['total'].get('USDT', 0)
        eth_total = balance['total'].get('ETH', 0)
        
        # Get ETH price to estimate total equity in USDT
        ticker = exchange.fetch_ticker('ETH/USDT')
        eth_price = ticker['last']
        
        total_equity = usdt_total + (eth_total * eth_price)
        
        print(f"USDT Balance: {usdt_total}")
        print(f"ETH Balance: {eth_total}")
        print(f"ETH Price: {eth_price}")
        print(f"Total Equity (est. USDT): {total_equity}")
        
    except Exception as e:
        print(f"Error fetching balance: {e}")

if __name__ == "__main__":
    check_balance()
