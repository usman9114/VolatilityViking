#!/usr/bin/env python3
import ccxt
import os
from dotenv import load_dotenv

load_dotenv('profiles/usman/.env')

api_key = os.getenv('BINANCE_API_KEY')
secret = os.getenv('BINANCE_SECRET_KEY') or os.getenv('BINANCE_SECRET')

# Check Spot
spot_exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': secret,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

# Check Futures  
futures_exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': secret,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

print("=" * 60)
print("BINANCE BALANCE CHECK")
print("=" * 60)

# Spot
balance_spot = spot_exchange.fetch_balance()
print(f"\n📊 SPOT WALLET:")
print(f"  USDT Total: ${balance_spot.get('USDT', {}).get('total', 0):.2f}")
print(f"  USDT Free:  ${balance_spot.get('USDT', {}).get('free', 0):.2f}")
print(f"  USDT Used:  ${balance_spot.get('USDT', {}).get('used', 0):.2f}")

# Futures
balance_futures = futures_exchange.fetch_balance()
print(f"\n📊 FUTURES WALLET:")
print(f"  USDT Total: ${balance_futures.get('USDT', {}).get('total', 0):.2f}")
print(f"  USDT Free:  ${balance_futures.get('USDT', {}).get('free', 0):.2f}")

info = balance_futures.get('info', {})
print(f"  Margin Balance: ${float(info.get('totalMarginBalance', 0)):.2f}")
print(f"  Wallet Balance: ${float(info.get('totalWalletBalance', 0)):.2f}")

print("\n" + "=" * 60)
