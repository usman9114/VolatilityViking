
import os
import json
import ccxt
from dotenv import load_dotenv

# Load Env
load_dotenv('.env_riz') 
# User was using .env_riz in previous conversation, let's try that or fallback to .env 
# The bot logs suggest it's using .env or .env_riz. khusha profile likely uses .env or specific key environment.
# Wait, previous conversation "Debugging Khusha Bot" used specific keys.
# Let's assume the bot is running correctly with the environment it has.
# I will use the args from verify_creds.py pattern but hardcode for speed.

def check_positions():
    print("Loading config...")
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    is_live = config.get('live_mode', False)
    print(f"Mode: {'LIVE' if is_live else 'TESTNET'}")
    
    if is_live:
        api_key = os.getenv('BINANCE_API_KEY')
        secret = os.getenv('BINANCE_SECRET_KEY') or os.getenv('BINANCE_SECRET')
    else:
        api_key = os.getenv('BINANCE_TESTNET_API_KEY')
        secret = os.getenv('BINANCE_TESTNET_SECRET_KEY')
        
    print(f"API Key (last 4): ...{api_key[-4:] if api_key else 'None'}")
    
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': secret,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'} 
    })
    
    if not is_live:
        exchange.set_sandbox_mode(True)
        
    print("Fetching positions for SOL/USDT and ETH/USDT...")
    try:
        positions = exchange.fetch_positions(['SOL/USDT', 'ETH/USDT'])
        print(f"Found {len(positions)} positions data points.")
        
        for p in positions:
            symbol = p['symbol']
            size = float(p.get('contracts', 0) or 0)
            entry = float(p.get('entryPrice', 0) or 0)
            unrealized = float(p.get('unrealizedPnl', 0) or 0)
            side = p.get('side', '?')
            print("-" * 40)
            print(f"Symbol: {symbol}")
            print(f"Size: {size}")
            print(f"Entry: {entry}")
            print(f"Side: {side}")
            print(f"PnL: {unrealized}")
            print(f"Raw: {json.dumps(p, indent=2)}")
            
        print("-" * 40)
        
        print("Fetching Open Orders...")
        orders = exchange.fetch_open_orders('SOL/USDT')
        print(f"Open Orders for SOL/USDT: {len(orders)}")
        for o in orders:
            print(f"ORDER: {o['side']} {o['amount']} @ {o['price']} ({o['type']})")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        exchange.close()

if __name__ == "__main__":
    check_positions()
