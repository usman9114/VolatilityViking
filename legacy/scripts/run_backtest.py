
import json
import sys
import os

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.event_backtester import EventBacktester

def main():
    print("Initializing 1-Year Backtest...")
    
    # Check data
    import os
    if not os.path.exists("data/ETHUSDT_1m.csv"):
        print("Error: data/ETHUSDT_1m.csv not found. Run scripts/fetch_data.py first.")
        sys.exit(1)
        
    bt = EventBacktester('config.json')
    bt.load_data("data/ETHUSDT_1m.csv")
    
    print(f"Data loaded: {len(bt.df)} candles ({bt.df['timestamp'].iloc[0]} to {bt.df['timestamp'].iloc[-1]})")
    
    try:
        history = bt.run()
        final_equity = history[-1]['equity']
        profit = final_equity - bt.initial_balance
        profit_pct = (profit / bt.initial_balance) * 100
        
        print("\n" + "="*50)
        print(f"RESULT: ${final_equity:.2f} ({profit_pct:+.2f}%)")
        print("="*50)
        
    except Exception as e:
        print(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
