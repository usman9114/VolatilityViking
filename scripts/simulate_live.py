
import os
import sys
import pandas as pd
import numpy as np
import warnings
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.getcwd())

# Import Bot Logic
from src.live.binance_bot import BinanceBot

def simulate():
    print("--- STARTING LIVE SIMULATION ---")
    
    # Init Bot (Testnet Mode safe)
    # We will hijack its 'get_signal' method slightly or just call it directly to inspect internals
    bot = BinanceBot(symbol='ETH/USDT', capital_limit=100)
    
    print("\n1. Running Full Inference Pipeline...")
    signal, price, pred = bot.get_signal()
    
    print("\n2. result Analysis...")
    print(f"Price: {price}")
    print(f"Prediction: {pred}")
    print(f"Signal: {signal}")
    
    # Assertions
    if pd.isna(pred):
        print("FAIL: Prediction is NaN!")
        sys.exit(1)
        
    if pd.isna(price):
        print("FAIL: Price is NaN!")
        sys.exit(1)
        
    print("\nPASS: Pipeline returned valid numbers.")
    print("--- SIMULATION COMPLETE ---")

if __name__ == "__main__":
    simulate()
