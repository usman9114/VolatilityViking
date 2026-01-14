import sys
import os
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.backtest.garch_backtester import GarchBacktester

def run_test(symbol, data_path):
    print(f"\n--- Running Baseline Backtest for {symbol} ---")
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return
        
    try:
        bt = GarchBacktester(config_path='config.json')
        bt.load_data(data_path)
        history = bt.run()
        
        initial = bt.initial_balance
        final = history[-1]['equity'] if history else initial
        pnl = (final - initial) / initial * 100
        print(f"[{symbol}] Result: {pnl:.2f}%")
        return pnl
    except Exception as e:
        print(f"Error running {symbol}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure raw data files exist or assume they are in data/
    # The user has _1m.csv files.
    
    symbols = [
        ('ETH/USDT', 'data/ETHUSDT_1m.csv'),
        ('SOL/USDT', 'data/SOLUSDT_1m.csv'),
        ('LINK/USDT', 'data/LINKUSDT_1m.csv')
    ]
    
    results = {}
    for sym, path in symbols:
        pnl = run_test(sym, path)
        results[sym] = pnl
        
    print("\n=== SUMMARY ===")
    for sym, pnl in results.items():
        if pnl is not None:
            print(f"{sym}: {pnl:.2f}%")
        else:
            print(f"{sym}: Failed")
