
import json
import sys
import os

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.garch_backtester import GarchBacktester

def main():
    print("Initializing 1-Year GARCH Backtest on 1m candles...")
    
    # Check data
    if not os.path.exists("data/ETHUSDT_1m.csv"):
        print("Error: data/ETHUSDT_1m.csv not found.")
        sys.exit(1)
        
    bt = GarchBacktester('config.json')
    bt.load_data("data/ETHUSDT_1m.csv")
    
    # SLICE LAST 3 DAYS (3 * 24 * 60 = 4320)
    bt.df = bt.df.iloc[-4320:]
    
    print(f"Data loaded: {len(bt.df)} candles")
    
    try:
        history = bt.run()
        final_equity = history[-1]['equity']
        profit = final_equity - bt.initial_balance
        profit_pct = (profit / bt.initial_balance) * 100
        
        print("\n" + "="*50)
        print(f"GARCH RESULT: ${final_equity:.2f} ({profit_pct:+.2f}%)")
        print("="*50)
        
    except Exception as e:
        print(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
