
import sys
import os
import logging
from dotenv import load_dotenv

# Load env immediately
load_dotenv('.env_riz') 
# Fallback to .env if needed, but priority to .env_riz as established
if not os.getenv('OPENAI_API_KEY'):
    load_dotenv('.env')

# Mock env vars for testing
os.environ['BINANCE_API_KEY'] = 'test'
os.environ['BINANCE_SECRET_KEY'] = 'test'

try:
    print("1. Testing Imports...")
    import ccxt.async_support
    import pandas
    import numpy
    import feedparser
    import openai
    import xgboost
    import sklearn
    print("   [OK] Imports successful")

    print("2. Testing AI Module Loading...")
    from src.ai.sentiment_engine import SentimentEngine
    from src.ai.ml_predictor import MLPredictor
    print("   [OK] AI Modules imported")

    print("3. Testing Bot Initialization...")
    from src.live.async_runner import AsyncMultiSymbolBot
    bot = AsyncMultiSymbolBot()
    print("   [OK] Bot initialized")
    
    print("\nSYSTEM VERIFICATION PASSED")

except Exception as e:
    print(f"\n[FAIL] System Verification Failed: {e}")
    sys.exit(1)
