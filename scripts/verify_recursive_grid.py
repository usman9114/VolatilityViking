import sys
import os
import json
import pandas as pd
from unittest.mock import MagicMock
sys.path.append(os.getcwd())
try:
    from src.live.binance_bot import BinanceBot
except ImportError:
    print("Could not import BinanceBot. Make sure you are in the project root.")
    sys.exit(1)

def test_recursive_grid_logic():
    print("Testing Recursive Grid & EMA...")
    
    # 1. Setup Mock Bot
    bot = BinanceBot(config_path='config.json')
    bot.exchange = MagicMock()
    bot.symbol = 'ETH/USDT'
    
    # 2. Test EMA Calculation
    print("\n[Test] EMA Update")
    # Mock OHLCV: 1000 candles of rising price 3000 -> 4000
    prices = [3000 + i for i in range(1100)]
    ohlcv = [[0, p, p, p, p, 0] for p in prices] # timestamp, open, high, low, close, vol
    bot.exchange.fetch_ohlcv.return_value = ohlcv
    
    bot.update_ema()
    
    print(f"EMA Long: {bot.ema_long:.2f}")
    if bot.ema_long > 3000:
        print("✅ EMA Calculated")
    else:
        print("❌ EMA Calculation Failed")

    # 3. Test Neutral Price
    neutral = bot.get_neutral_price()
    print(f"Neutral Price (EMA): {neutral:.2f}")
    
    # 4. Test Recursive Grid Logic
    print("\n[Test] Recursive Grid Maintenance")
    # Mock Balance
    bot.exchange.fetch_balance.return_value = {
        'free': {'USDT': 1000, 'ETH': 0}, 
        'total': {'USDT': 1000, 'ETH': 0}
    }
    # Mock fetch_equity to return enough to trade
    bot.fetch_equity = MagicMock(return_value=(1000, 0, 1000))
    
    # Mock Open Orders (Empty initially)
    bot.exchange.fetch_open_orders.return_value = []
    
    # Run Maintenance
    bot.maintain_recursive_grid()
    
    # Debug: Check ideal grid calculation directly
    ideal_buys, ideal_sells = bot.calculate_ideal_grid(neutral)
    print(f"DEBUG: Ideal Buys: {len(ideal_buys)}")
    for b in ideal_buys: print(f"  {b}")
    
    # Check if orders were "placed"
    calls = bot.exchange.create_limit_buy_order.call_args_list
    print(f"Buy Orders Placed: {len(calls)}")
    
    if len(calls) > 0:
        print(f"First Order Price: {calls[0][0][2]}") # args[2] is price
        print("✅ Recursive Grid generated orders")
    else:
        print("❌ Recursive Grid failed to generate orders")

if __name__ == "__main__":
    test_recursive_grid_logic()
