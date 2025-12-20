import time
import schedule
import pandas as pd
from datetime import datetime
from src.data.data_fetcher import DataFetcher
from src.utils.db import log_prediction, init_db
import random # Placeholder for real inference if model not loaded

def job(symbol='ETH/USDT', timeframe='4h'):
    print(f"\n[Live] Running Job: {datetime.now()}")
    
    # 1. Fetch Latest Data (Just 1 candle or small batch)
    try:
        fetcher = DataFetcher(symbol=symbol, timeframe=timeframe)
        # Use robust fetcher (handles fallback) and get DF
        # fetch_price_data usually gets FULL history. For live, maybe overkill?
        # But it's robust. Let's use it but maybe we need a 'recent' mode?
        # fetch_price_data starts from 2017 by default.
        # Let's just use it but passing a recent start date to keep it light?
        start_date = (datetime.now() - pd.Timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
        df = fetcher.fetch_price_data(start_date_str=start_date)
        
        if df.empty:
            print("Failed to fetch data.")
            return

        current_price = df['close'].iloc[-1]
        
        # 2. Feature Engineering
        # (Simplified for demo - normally load full feature pipeline)
        # Here we just mock the prediction for the "Interface Demo" requested
        # To make it real, we would load the `metamodel` here.
        
        # MOCK PREDICTION FOR DEMO DASHBOARD
        # Since we haven't pickled the exact model state for inference in `metamodel.py`,
        # we will simulate a "Wait/Hold" or random signal for the Dashboard demo.
        
        predicted_return = random.uniform(-0.01, 0.01)
        signal = 0
        if predicted_return > 0.005: signal = 1
        elif predicted_return < -0.005: signal = 0 # Cash
        
        confidence = abs(predicted_return) * 100 # Mock confidence
        
        # 3. Log to DB
        log_prediction(
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            predicted_return=predicted_return,
            signal=signal,
            confidence=confidence,
            features_json="{}"
        )
        print(f"Logged: Price=${current_price}, Signal={signal}")
        
    except Exception as e:
        print(f"Job Error: {e}")

def start_trader(symbol, timeframe):
    init_db()
    print(f"Starting Live Trader Loop for {symbol}...")
    
    # Run once immediately
    job(symbol, timeframe)
    
    # Schedule
    if timeframe == '1m':
        schedule.every(1).minutes.do(job, symbol, timeframe)
    elif timeframe == '1h':
        schedule.every(1).hours.do(job, symbol, timeframe)
    elif timeframe == '4h':
        schedule.every(4).hours.do(job, symbol, timeframe)
        
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    start_trader('ETH/USDT', '4h')
