import sys
import os
import ccxt
import pandas as pd
import concurrent.futures
import json
import time
import logging
from src.backtest.garch_backtester import GarchBacktester

logger = logging.getLogger(__name__)

class MomentumAllocator:
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.last_run_time = 0

    def get_candidate_symbols(self):
        with open(self.config_path, 'r') as f:
            conf = json.load(f)
            
        core = conf.get('symbols', [])
        forager = conf.get('forager', {}).get('approved_symbols', [])
        return list(set(core + forager))

    def fetch_data(self, symbol, days=3):
        try:
            exchange = ccxt.binance()
            safe_sym = symbol.replace('/', '')
            filepath = f"data/{safe_sym}_3day.csv"
            
            # Skip if recently fetched (cache for 1 hr)
            if os.path.exists(filepath):
                if time.time() - os.path.getmtime(filepath) < 3600:
                    return filepath

            since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
            all_ohlcv = []
            
            while True:
                ohlcv = exchange.fetch_ohlcv(symbol, '1m', since=since, limit=1000)
                if not ohlcv: break
                all_ohlcv += ohlcv
                since = ohlcv[-1][0] + 1
                if len(ohlcv) < 1000: break
                time.sleep(0.1)
                
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            os.makedirs('data', exist_ok=True)
            df.to_csv(filepath, index=False)
            return filepath
        except Exception as e:
            logger.error(f"Alloc Fetch Error {symbol}: {e}")
            return None

    def analyze_symbol(self, symbol, days=3):
        filepath = self.fetch_data(symbol, days)
        if not filepath: return None
            
        try:
            bt = GarchBacktester(self.config_path)
            bt.load_data(filepath)
            history = bt.run()
            
            if not history: 
                return {'symbol': symbol, 'score': -999, 'profit': 0}
            
            final = history[-1]['equity']
            profit_pct = (final - bt.initial_balance) / bt.initial_balance * 100
            
            peak = 0
            max_dd = 0
            for x in history:
                val = x['equity']
                if val > peak: peak = val
                dd = (peak - val) / peak
                if dd > max_dd: max_dd = dd
                
            # Score Logic: Profit biased, penalized by DD
            score = profit_pct - (max_dd * 100 * 0.5) 
            
            return {
                'symbol': symbol,
                'profit': profit_pct,
                'max_dd': max_dd * 100,
                'score': score
            }
        except Exception as e:
            logger.error(f"Alloc Backtest Error {symbol}: {e}")
            return None

    def run(self):
        """Main execution method called by the bot"""
        logger.info("🏎️ Starting Momentum Allocator Run...")
        
        # Reload config to get latest settings
        with open(self.config_path, 'r') as f:
            conf = json.load(f)
            
        settings = conf.get('momentum_allocator', {})
        if not settings.get('enabled', False):
            logger.info("Momentum Allocator is Disabled.")
            return

        lookback_days = settings.get('lookback_days', 3)
        symbols = self.get_candidate_symbols()
        
        results = []
        # Use ThreadPool for fetching/backtesting combination (simplify concurrency for embed)
        # Using ProcessPool inside Thread might be tricky if not careful, sticking to simplified loop or threadpool
        # GarchBacktester is fast enough for 3 days x 10 coins
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
             futures = {executor.submit(self.analyze_symbol, sym, lookback_days): sym for sym in symbols}
             for future in concurrent.futures.as_completed(futures):
                 res = future.result()
                 if res: results.append(res)
                 
        results.sort(key=lambda x: x['score'], reverse=True)
        
        new_allocations = {}
        for i, r in enumerate(results):
            if i == 0:
                alloc = 0.40
                tier = "GOLD"
            elif i < 3:
                alloc = 0.25
                tier = "SLVR"
            else:
                alloc = 0.10
                tier = "POOR"
                
            if r['profit'] < -1.0 and i > 0:
                alloc = 0.05
                tier = "DEAD"
                
            new_allocations[r['symbol']] = alloc
            logger.info(f"Alloc: {r['symbol']} -> {alloc:.0%} ({tier}) [Profit:{r['profit']:.2f}%]")
            
        # Update Config
        conf['specific_exposures'] = new_allocations
        
        # Write back Atomically-ish
        tmp_path = self.config_path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(conf, f, indent=4)
        os.replace(tmp_path, self.config_path)
        
        self.last_run_time = time.time()
        logger.info("✅ Momentum Allocator Complete. Config Updated.")
