
import sys
import os
import json
import argparse

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.robust_backtester import RobustBacktester

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', help='Specific symbol to backtest (e.g., AVAX/USDT)')
    parser.add_argument('--days', type=int, default=30, help='Number of days of data to use')
    args = parser.parse_args()

    if args.symbol:
        # User passed "AVAX/USDT" or "AVAXUSDT"
        s = args.symbol.replace('/', '')
        symbols = [s]
    else:
        symbols = ['ETHUSDT', 'SOLUSDT', 'PEPEUSDT', 'BTCUSDT']
    
    print(f"{'Symbol':<10} | {'Return':<10} | {'Buy&Hold':<10} | {'MaxDD':<10} | {'Trades':<8} | {'Sharpe':<8}")
    print("-" * 75)
    
    for symbol in symbols:
        # Handle formatting
        symbol = symbol.replace('/', '')
        
        data_path = f'data/{symbol}_1m_30d.csv'
        if not os.path.exists(data_path):
             # Try fetching it if missing? For now just warn
             # Or try standard "AVAXUSDT" lookup
             pass
             
        if not os.path.exists(data_path):
            print(f"{symbol:<10} | Data Not Found ({data_path})")
            continue

        # Initialize Backtester
        # We can override config if needed, e.g. aggressive mode
        bt = RobustBacktester(config_path='config.json', config_overrides={
            'smart_grid': {'enabled': True, 'volatility_scale': 1.0},
            'allow_shorts': False,
            'core_wallet_exposure': 1.0, # Full Allocation for comparison
            'wallet_exposure_limit': 1.0
        })
        
        try:
            # Load Data
            bt.load_and_prep_data(data_path, days=args.days)
            # Run
            results = bt.run()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{symbol:<10} | Error: {e}")
            continue
            
        # Report Row
        ret = results.get('total_return_pct', 0)
        bh = 0 
        
        # Calculate BH manually here since we have df loaded in bt
        if not bt.df.empty:
            start = bt.df['close'].iloc[0]
            end = bt.df['close'].iloc[-1]
            bh = ((end - start) / start) * 100
            
        dd = results.get('max_drawdown_pct', 0)
        trades = results.get('total_trades', 0)
        sharpe = results.get('sharpe_ratio', 0)
        
        print(f"{symbol:<10} | {ret:>8.2f}% | {bh:>8.2f}% | {dd:>8.2f}% | {trades:>8} | {sharpe:>8.2f}")

if __name__ == "__main__":
    main()
