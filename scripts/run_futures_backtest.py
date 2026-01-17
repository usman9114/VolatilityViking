
import sys
import os
import ccxt
import pandas as pd
import json
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.getcwd())

from src.backtest.robust_backtester import RobustBacktester

def fetch_recent_data(symbol='ETH/USDT', days=2, timeframe='1m'):
    print(f"Fetching last {days} days of {timeframe} data for {symbol}...")
    exchange = ccxt.binance()
    since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
    all_ohlcv = []
    
    while since < exchange.milliseconds():
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
        if not ohlcv: break
        since = ohlcv[-1][0] + 1
        all_ohlcv += ohlcv
        
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') # Backtester does this
    return df

def main():
    # 1. Fetch Data
    # Fetch 9 days to give buffer for indicators
    df = fetch_recent_data(days=9) 
    csv_path = 'data/temp_eth_recent.csv'
    os.makedirs('data', exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    # --- SCENARIO A: RAW MATH (Shorts Enabled) ---
    print("\n" + "="*60)
    print("SCENARIO A: RAW MATH MODE (No AI Guardrails)")
    print("="*60)
    overrides_raw = {
        "futures_settings": {
            "enabled": True,
            "leverage": 5,
            "shorting_enabled": True
        },
        "allow_shorts": True,
        "capital_limit": 1000,
        "aggressive_mode": False
    }
    bt = RobustBacktester(config_path='config.json', config_overrides=overrides_raw)
    bt.load_and_prep_data(csv_path, days=7)
    bt.run()
    
    # --- SCENARIO B: AI PROTECTED (Shorts Disabled) ---
    print("\n" + "="*60)
    print("SCENARIO B: AI PROTECTED MODE (Bullish Guardrails)")
    print("="*60)
    overrides_ai = {
        "futures_settings": {
            "enabled": True,
            "leverage": 5,
            "shorting_enabled": False # AI Block
        },
        "allow_shorts": False, 
        "capital_limit": 1000,
        "aggressive_mode": False
    }
    
    bt_ai = RobustBacktester(config_path='config.json', config_overrides=overrides_ai)
    bt_ai.load_and_prep_data(csv_path, days=7)
    bt_ai.run()

    # --- SCENARIO C: AGGRESSIVE GROWTH (AI Protected + Tighter Grid) ---
    print("\n" + "="*60)
    print("SCENARIO C: AGGRESSIVE GROWTH (Beat the Market?)")
    print("="*60)
    overrides_agg = {
        "futures_settings": {
            "enabled": True,
            "leverage": 10, # Higher Leverage
            "shorting_enabled": False 
        },
        "allow_shorts": False,
        "capital_limit": 1000,
        "aggressive_mode": True,
        "grid_settings": {
            "grid_span": 0.004, # Trace tighter (0.4%)
            "qty_step_multiplier": 1.1, # Add size faster
            "max_orders": 8
        },
        "wallet_exposure_limit": 2.5 # Use that leverage!
    }
    
    bt_agg = RobustBacktester(config_path='config.json', config_overrides=overrides_agg)
    bt_agg.load_and_prep_data(csv_path, days=7)
    bt_agg.run()

    print("\n[Analysis Complete]")
    
if __name__ == "__main__":
    main()
