import sys
import os
import ccxt
import pandas as pd
import time
import concurrent.futures
from datetime import datetime
import json

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.backtest.garch_backtester import GarchBacktester

OS_CPU_COUNT = os.cpu_count() or 4
DAYS_TO_FETCH = 7  # Last week of data is plenty for scalping verification

def get_top_50_symbols():
    print("Fetching Top 50 USDT pairs by volume...")
    exchange = ccxt.binance()
    tickers = exchange.fetch_tickers()
    
    # Filter for USDT pairs, exclude stablecoins/leverage
    valid_tickers = []
    ignored = ['USDC/USDT', 'FDUSD/USDT', 'TUSD/USDT', 'DAI/USDT', 'EUR/USDT']
    
    for symbol, ticker in tickers.items():
        if '/USDT' in symbol and not any(x in symbol for x in ['UP/', 'DOWN/', 'BEAR/', 'BULL/']) and symbol not in ignored:
            valid_tickers.append({
                'symbol': symbol,
                'quoteVolume': ticker['quoteVolume'] if ticker['quoteVolume'] else 0
            })
            
    # Sort by volume
    valid_tickers.sort(key=lambda x: x['quoteVolume'], reverse=True)
    top_50 = [x['symbol'] for x in valid_tickers[:50]]
    print(f"Top 5 selected: {top_50[:5]}")
    return top_50

def fetch_and_save(symbol):
    safe_sym = symbol.replace('/', '')
    filepath = f"data/{safe_sym}_1m.csv"
    
    # Skip if file exists and is recent (optional, but let's force fresh for "current strategy")
    # Actually, let's fetch fresh.
    
    exchange = ccxt.binance()
    since = exchange.milliseconds() - (DAYS_TO_FETCH * 24 * 60 * 60 * 1000)
    all_ohlcv = []
    
    try:
        while True:
            ohlcv = exchange.fetch_ohlcv(symbol, '1m', since=since, limit=1000)
            if not ohlcv: break
            all_ohlcv += ohlcv
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000: break
            
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.to_csv(filepath, index=False)
        return (symbol, filepath, True)
    except Exception as e:
        return (symbol, str(e), False)

def run_backtest(args):
    symbol, filepath = args
    try:
        bt = GarchBacktester('config.json')
        bt.load_data(filepath)
        
        # Run
        history = bt.run()
        if not history: return None
        
        final_equity = history[-1]['equity']
        profit_pct = (final_equity - bt.initial_balance) / bt.initial_balance * 100
        
        # Calculate Max Drawdown
        peak = 0
        max_dd = 0
        for entry in history:
            if entry['equity'] > peak: peak = entry['equity']
            dd = (peak - entry['equity']) / peak
            if dd > max_dd: max_dd = dd
            
        return {
            'symbol': symbol,
            'profit_pct': profit_pct,
            'max_dd_pct': max_dd * 100,
            'trades': len(bt.trades),
            'final_equity': final_equity
        }
    except Exception as e:
        return {'symbol': symbol, 'error': str(e)}

def main():
    # 1. Get Symbols
    symbols = get_top_50_symbols()
    
    # 2. Fetch Data (Threaded - Network Bound)
    print(f"\nDownloading data for {len(symbols)} symbols ({DAYS_TO_FETCH} days)...")
    os.makedirs('data', exist_ok=True)
    
    successful_files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_and_save, symbols))
        
    for res in results:
        sym, path, success = res
        if success:
            successful_files.append((sym, path))
        else:
            print(f"Failed to fetch {sym}: {path}")

    # 3. Run Backtests (Process - CPU Bound)
    print(f"\nRunning Parallel Backtests on {len(successful_files)} symbols...")
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=OS_CPU_COUNT) as executor:
        for res in executor.map(run_backtest, successful_files):
            if res: results.append(res)

    # 4. Report
    print("\n" + "="*80)
    print(f"{'SYMBOL':<12} | {'PROFIT %':<10} | {'MAX DD %':<10} | {'TRADES':<8} | {'RESULT':<10}")
    print("-" * 80)
    
    results.sort(key=lambda x: x.get('profit_pct', -100), reverse=True)
    
    total_profit = 0
    winning = 0
    
    for r in results:
        if 'error' in r:
            print(f"{r['symbol']:<12} | ERROR: {r['error']}")
            continue
            
        profit = r['profit_pct']
        dd = r['max_dd_pct']
        total_profit += profit
        if profit > 0: winning += 1
        
        print(f"{r['symbol']:<12} | {profit:+6.2f}%    | {dd:6.2f}%    | {r['trades']:<8} | {'✅' if profit > 0 else '❌'}")

    print("="*80)
    avg_profit = total_profit / len(results) if results else 0
    print(f"AVERAGE PROFIT: {avg_profit:+.2f}%")
    print(f"WIN RATE: {winning}/{len(results)} ({winning/len(results)*100:.1f}%)")
    print("="*80)

if __name__ == "__main__":
    main()
