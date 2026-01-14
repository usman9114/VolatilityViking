import sys
import os
import random
import concurrent.futures
import json
import numpy as np

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.backtest.garch_backtester import GarchBacktester

# Configuration
ITERATIONS = 50
DATA_FILE = 'data/ETHUSDT_1m.csv'
MAX_WORKERS = os.cpu_count() or 4

def generate_random_config():
    """Generates a random configuration variation."""
    return {
        'grid_settings': {
            'grid_span': round(random.uniform(0.003, 0.020), 4),       # 0.3% to 2.0%
            'max_orders': random.randint(3, 10),                       # 3 to 10 orders
            'drift_tolerance': round(random.uniform(0.001, 0.005), 4)  # 0.1% to 0.5%
        },
        'aggressive_mode': random.choice([True, False]),
        'core_wallet_exposure': round(random.uniform(0.3, 0.8), 2)
    }

def evaluate_config(config_overrides):
    """Runs backtest with specific overrides and returns score."""
    try:
        # Initialize backtester with overrides
        bt = GarchBacktester('config.json', config_overrides)
        
        # Load data (Assuming file exists)
        if not os.path.exists(DATA_FILE):
            return None
            
        bt.load_data(DATA_FILE)
        
        # Slice last 10,000 candles (~7 days) for speed
        if len(bt.df) > 10000:
            bt.df = bt.df.iloc[-10000:]
            
        history = bt.run()
        if not history: return None
        
        final_equity = history[-1]['equity']
        profit_pct = (final_equity - bt.initial_balance) / bt.initial_balance * 100
        
        # Calculate Max Drawdown
        peak = 0
        max_dd = 0
        for entry in history:
            val = entry['equity']
            if val > peak: peak = val
            dd = (peak - val) / peak
            if dd > max_dd: max_dd = dd
            
        # Score = Profit / MaxDD (Calmar-ish)
        # Avoid div by zero
        score = profit_pct / (max_dd * 100 + 0.1)
        
        return {
            'config': config_overrides,
            'profit': profit_pct,
            'max_dd': max_dd * 100,
            'score': score
        }
        
    except Exception as e:
        return {'error': str(e)}

def main():
    print(f"🚀 Starting Auto-Optimizer ({ITERATIONS} iterations)...")
    print(f"Target Data: {DATA_FILE}")
    print(f"Workers: {MAX_WORKERS}")
    
    tasks = []
    # Generate variations
    for _ in range(ITERATIONS):
        tasks.append(generate_random_config())
        
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(evaluate_config, cfg): cfg for cfg in tasks}
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            res = future.result()
            if res and 'error' not in res:
                results.append(res)
                print(f"[{completed}/{ITERATIONS}] Profit: {res['profit']:+.2f}% | DD: {res['max_dd']:.2f}% | Span: {res['config']['grid_settings']['grid_span']:.1%}")
            
    # Sort by Score from high to low
    results.sort(key=lambda x: x['profit'], reverse=True) # Sort by raw profit for now
    
    print("\n" + "="*60)
    print("🏆 TOP 3 CONFIGURATIONS")
    print("="*60)
    
    for i, res in enumerate(results[:3]):
        cfg = res['config']
        gs = cfg['grid_settings']
        print(f"Rank #{i+1}: Profit {res['profit']:+.2f}% | DD {res['max_dd']:.2f}%")
        print(f"   Grid Span: {gs['grid_span']:.1%} | Orders: {gs['max_orders']} | Drift: {gs['drift_tolerance']:.1%}")
        print(f"   Aggressive: {cfg['aggressive_mode']} | Exposure: {cfg['core_wallet_exposure']}")
        print("-" * 60)
        
    print("\nSuggested Action: Update config.json with Rank #1 settings if satisfied.")

if __name__ == "__main__":
    main()
