
import ccxt
import pandas as pd
import numpy as np
import os
import sys
import logging
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer

# Add project root to path
sys.path.append(os.getcwd())
from src.data.news_fetcher import NewsFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiFactorScanner:
    """
    Enhanced scanner that collects multiple factors for coin selection:
    1. Volatility (Passivbot style)
    2. Volume (24h quote volume)
    3. Price Momentum (24h change %)
    4. RSI (Relative Strength Index)
    5. Funding Rate (perpetual futures bias)
    6. Open Interest Change
    7. Bid-Ask Spread (liquidity)
    """
    
    def __init__(self, limit_top_volume=200):
        load_dotenv()
        self.limit_top_volume = limit_top_volume
        
        # Spot exchange for price data
        self.spot = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        # Futures exchange for funding/OI
        self.futures = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

    def get_top_volume_coins(self):
        """Fetch top coins by 24h volume."""
        try:
            logger.info("Fetching spot tickers...")
            tickers = self.spot.fetch_tickers()
            
            # Filter for USDT pairs, exclude leveraged tokens
            usdt_pairs = [
                symbol for symbol in tickers 
                if symbol.endswith('/USDT') 
                and 'UP/' not in symbol 
                and 'DOWN/' not in symbol
                and 'BEAR/' not in symbol
                and 'BULL/' not in symbol
            ]
            
            # Sort by Quote Volume
            sorted_pairs = sorted(
                usdt_pairs, 
                key=lambda x: tickers[x].get('quoteVolume', 0) or 0, 
                reverse=True
            )
            
            top_pairs = sorted_pairs[:self.limit_top_volume]
            logger.info(f"Found {len(top_pairs)} top volume coins")
            return top_pairs, tickers
            
        except Exception as e:
            logger.error(f"Error fetching top coins: {e}")
            return [], {}

    def calculate_volatility(self, symbol, limit=60):
        """Passivbot-style volatility: mean(log(high/low))"""
        try:
            ohlcv = self.spot.fetch_ohlcv(symbol, timeframe='1m', limit=limit)
            if not ohlcv or len(ohlcv) < 10:
                return 0.0
                
            log_ranges = []
            for candle in ohlcv:
                high, low = candle[2], candle[3]
                if low > 0:
                    log_ranges.append(np.log(high / low))
                    
            return np.mean(log_ranges) if log_ranges else 0.0
            
        except Exception as e:
            return 0.0

    def calculate_rsi(self, symbol, period=14, limit=100):
        """Calculate RSI (Relative Strength Index)"""
        try:
            ohlcv = self.spot.fetch_ohlcv(symbol, timeframe='1h', limit=limit)
            if not ohlcv or len(ohlcv) < period + 1:
                return 50.0  # Neutral
                
            closes = pd.Series([c[4] for c in ohlcv])
            delta = closes.diff()
            
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0
            
        except Exception as e:
            return 50.0

    def calculate_momentum(self, ticker):
        """Calculate price momentum from ticker data"""
        try:
            pct_change = ticker.get('percentage', 0) or 0
            return pct_change
        except:
            return 0.0

    def get_funding_rate(self, symbol):
        """Get perpetual funding rate (futures market sentiment)"""
        try:
            # Convert spot symbol to futures format
            futures_symbol = symbol.replace('/USDT', '/USDT:USDT')
            
            funding = self.futures.fetch_funding_rate(futures_symbol)
            rate = funding.get('fundingRate', 0) or 0
            return rate * 100  # Convert to percentage
            
        except Exception as e:
            return 0.0

    def get_open_interest_change(self, symbol):
        """Get open interest change (market participation)"""
        try:
            futures_symbol = symbol.replace('/USDT', '/USDT:USDT')
            
            # Fetch current OI
            oi = self.futures.fetch_open_interest(futures_symbol)
            current_oi = oi.get('openInterestValue', 0) or 0
            
            # We'd need historical OI for change, return current for now
            return current_oi
            
        except Exception as e:
            return 0.0

    def calculate_spread(self, symbol):
        """Calculate bid-ask spread (liquidity indicator)"""
        try:
            orderbook = self.spot.fetch_order_book(symbol, limit=5)
            
            if orderbook['bids'] and orderbook['asks']:
                best_bid = orderbook['bids'][0][0]
                best_ask = orderbook['asks'][0][0]
                mid = (best_bid + best_ask) / 2
                spread = (best_ask - best_bid) / mid * 100
                return spread
            return 1.0
            
        except Exception as e:
            return 1.0

    def scan(self, top_n=50):
        """Scan coins and collect all factors."""
        top_pairs, tickers = self.get_top_volume_coins()
        
        if not top_pairs:
            return []
        
        # Limit to top_n for API efficiency
        scan_pairs = top_pairs[:top_n]
        results = []
        
        logger.info(f"Scanning {len(scan_pairs)} coins with multi-factor analysis...")
        
        for i, symbol in enumerate(scan_pairs):
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i+1}/{len(scan_pairs)}")
            
            ticker = tickers.get(symbol, {})
            
            # Collect all factors
            data = {
                'symbol': symbol,
                'volume_24h': ticker.get('quoteVolume', 0) or 0,
                'volatility': self.calculate_volatility(symbol),
                'momentum_24h': self.calculate_momentum(ticker),
                'rsi': self.calculate_rsi(symbol),
                'funding_rate': self.get_funding_rate(symbol),
                'spread': self.calculate_spread(symbol),
                'last_price': ticker.get('last', 0) or 0
            }
            
            results.append(data)
        
        return results


class NewsAnalyzer:
    """Analyze news sentiment for coins."""
    
    def __init__(self):
        self.news_fetcher = NewsFetcher()
        logger.info("Loading Sentence Transformer...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Pre-compute anchor embeddings
        self.positive_anchor = self.model.encode(["Crypto price surge bull market positive growth breakthrough"])
        self.negative_anchor = self.model.encode(["Crypto crash dump market panic sell negative hack exploit"])
        logger.info("Model loaded.")

    def analyze_sentiment(self, symbol):
        """Fetch news and calculate sentiment score."""
        try:
            coin = symbol.split('/')[0]
            news_df = self.news_fetcher.fetch_news(coin, limit=20)
            
            if news_df.empty:
                return 0.0, 0
            
            headlines = news_df['title'].tolist()
            if not headlines:
                return 0.0, 0
                
            embeddings = self.model.encode(headlines)
            
            scores = []
            for emb in embeddings:
                pos_sim = np.dot(emb, self.positive_anchor[0])
                neg_sim = np.dot(emb, self.negative_anchor[0])
                scores.append(pos_sim - neg_sim)
                
            return np.mean(scores), len(headlines)
            
        except Exception as e:
            logger.error(f"Error analyzing news for {symbol}: {e}")
            return 0.0, 0


class PortfolioManager:
    """
    Enhanced Portfolio Manager with Multi-Factor Scoring.
    
    Factors:
    1. Volatility (higher = more grid opportunities)
    2. Volume (higher = better liquidity)
    3. Momentum (positive = bullish trend)
    4. RSI (30-70 range preferred, extremes = caution)
    5. News Sentiment (positive = bullish catalyst)
    6. Funding Rate (negative = longs paying shorts, potential squeeze)
    7. Spread (lower = better liquidity)
    """
    
    def __init__(self):
        self.scanner = MultiFactorScanner(limit_top_volume=200)
        self.analyzer = NewsAnalyzer()
        
        # Factor weights - OPTIMIZED FOR GRID TRADING
        # Volatility is king for grids - more swings = more fills
        self.weights = {
            'volatility': 0.40,    # Primary driver - grids need price swings
            'liquidity': 0.25,     # Must have volume + tight spread
            'funding': 0.15,       # Contrarian signal (negative = potential squeeze)
            'news': 0.10,          # Avoid bad news (hacks, delistings)
            'rsi_score': 0.05,     # Minor mean-reversion filter
            'momentum': 0.05       # Minor trend filter
        }
    
    def normalize(self, values):
        """Normalize values to 0-1 range."""
        arr = np.array(values)
        min_val, max_val = arr.min(), arr.max()
        if max_val == min_val:
            return np.ones_like(arr) * 0.5
        return (arr - min_val) / (max_val - min_val)
    
    def calculate_rsi_score(self, rsi):
        """Convert RSI to score (prefer 30-70 range)."""
        if rsi < 30:
            return 0.8  # Oversold = potential bounce
        elif rsi > 70:
            return 0.3  # Overbought = caution
        else:
            return 0.5  # Neutral
    
    def calculate_funding_score(self, funding):
        """Convert funding rate to score (contrarian)."""
        if funding < -0.01:
            return 0.8  # Negative funding = longs paying, potential squeeze
        elif funding > 0.05:
            return 0.3  # High positive = crowded longs
        else:
            return 0.5  # Neutral
    
    def run(self, top_volatile=20, top_final=10):
        print("\n" + "="*70)
        print("🚀 PORTFOLIO MANAGER AI - MULTI-FACTOR EDITION")
        print("="*70 + "\n")
        
        # 1. Multi-Factor Scan
        print("1. Scanning Market (Multi-Factor Analysis)...")
        candidates = self.scanner.scan(top_n=100)
        
        if not candidates:
            print("No candidates found!")
            return
        
        # 2. Filter by volatility (top N)
        candidates.sort(key=lambda x: x['volatility'], reverse=True)
        top_volatile_coins = candidates[:top_volatile]
        
        print(f"\nTop {len(top_volatile_coins)} Volatile Coins:")
        for c in top_volatile_coins[:5]:
            print(f"  - {c['symbol']}: vol={c['volatility']:.5f}, RSI={c['rsi']:.1f}")
        
        # 3. Analyze News Sentiment
        print(f"\n2. Analyzing News Sentiment for {len(top_volatile_coins)} candidates...")
        for c in top_volatile_coins:
            score, count = self.analyzer.analyze_sentiment(c['symbol'])
            c['news_score'] = score
            c['news_count'] = count
        
        # 4. Calculate Composite Scores
        print("\n3. Calculating Composite Scores...")
        
        # Normalize each factor
        volatilities = self.normalize([c['volatility'] for c in top_volatile_coins])
        momentums = self.normalize([c['momentum_24h'] for c in top_volatile_coins])
        news_scores = self.normalize([c['news_score'] for c in top_volatile_coins])
        
        # Calculate derived scores
        for i, c in enumerate(top_volatile_coins):
            c['volatility_norm'] = volatilities[i]
            c['momentum_norm'] = momentums[i]
            c['news_norm'] = news_scores[i]
            c['rsi_score'] = self.calculate_rsi_score(c['rsi'])
            c['funding_score'] = self.calculate_funding_score(c['funding_rate'])
            
            # Liquidity score (high volume, low spread)
            c['liquidity_score'] = 1.0 if c['spread'] < 0.1 else 0.5 if c['spread'] < 0.5 else 0.2
            
            # Composite score
            c['composite'] = (
                self.weights['volatility'] * c['volatility_norm'] +
                self.weights['momentum'] * c['momentum_norm'] +
                self.weights['news'] * c['news_norm'] +
                self.weights['rsi_score'] * c['rsi_score'] +
                self.weights['funding'] * c['funding_score'] +
                self.weights['liquidity'] * c['liquidity_score']
            )
        
        # 5. Final Ranking
        top_volatile_coins.sort(key=lambda x: x['composite'], reverse=True)
        final_picks = top_volatile_coins[:top_final]
        
        # 6. Display Results
        print("\n" + "="*70)
        print("🏆 FINAL RECOMMENDATIONS (MULTI-FACTOR RANKING)")
        print("="*70)
        
        print(f"\n{'RANK':<5} {'SYMBOL':<12} {'COMPOSITE':<10} {'VOL':<8} {'MOM%':<8} {'RSI':<6} {'NEWS':<8} {'FUNDING':<8} {'ACTION'}")
        print("-" * 85)
        
        for rank, c in enumerate(final_picks, 1):
            # Determine action
            if c['composite'] > 0.6 and c['momentum_24h'] > 0:
                action = "🟢 LONG"
            elif c['composite'] > 0.5:
                action = "🟡 WATCH"
            elif c['rsi'] > 70:
                action = "🔴 OVERBOUGHT"
            else:
                action = "⚪ NEUTRAL"
            
            print(f"{rank:<5} {c['symbol']:<12} {c['composite']:.4f}    "
                  f"{c['volatility']:.5f}  {c['momentum_24h']:>6.2f}%  "
                  f"{c['rsi']:>5.1f} {c['news_score']:>7.4f}  "
                  f"{c['funding_rate']:>7.4f}%  {action}")
        
        # 7. Summary
        print("\n" + "-"*70)
        print("FACTOR WEIGHTS:")
        for k, v in self.weights.items():
            print(f"  {k}: {v*100:.0f}%")
        
        # 8. Export recommended symbols
        recommended = [c['symbol'] for c in final_picks if c['composite'] > 0.5]
        print(f"\n📋 RECOMMENDED FOR FORAGER: {recommended[:5]}")
        
        return final_picks

    def calculate_optimal_positions(self, balance, min_order_size=10, 
                                     total_wel=1.0, initial_qty_pct=0.10):
        """
        Calculate optimal number of positions based on balance.
        
        Passivbot formula:
        initial_entry = balance × (total_WEL / n_positions) × entry_initial_qty_pct
        
        We need: initial_entry >= min_order_size
        Therefore: n_positions <= balance × total_WEL × initial_qty_pct / min_order_size
        """
        max_positions = int((balance * total_wel * initial_qty_pct) / min_order_size)
        
        # Apply sensible limits
        if balance < 100:
            return min(max_positions, 1)
        elif balance < 500:
            return min(max_positions, 2)
        elif balance < 2000:
            return min(max_positions, 4)
        elif balance < 10000:
            return min(max_positions, 8)
        else:
            return min(max_positions, 15)
    
    def update_config(self, config_path, recommended_symbols, n_positions=None):
        """
        Update config.json with recommended symbols from portfolio analysis.
        """
        import json
        
        try:
            # Read current config
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update forager approved_symbols
            if 'forager' not in config:
                config['forager'] = {}
            
            config['forager']['approved_symbols'] = recommended_symbols
            
            if n_positions:
                config['forager']['max_positions'] = n_positions
            
            # Write back
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"\n✅ Config updated at {config_path}")
            print(f"   Approved symbols: {recommended_symbols[:5]}...")
            if n_positions:
                print(f"   Max positions: {n_positions}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            return False
    
    def fetch_wallet_balance(self):
        """
        Fetch actual wallet balance from exchange.
        This is how Passivbot does it - reads real balance, not config.
        """
        try:
            # Try to use authenticated exchange
            api_key = os.getenv('BINANCE_API_KEY') or os.getenv('BINANCE_TESTNET_API_KEY')
            secret = os.getenv('BINANCE_SECRET_KEY') or os.getenv('BINANCE_SECRET') or os.getenv('BINANCE_TESTNET_SECRET_KEY')
            
            if not api_key or not secret:
                logger.warning("No API keys found, using config capital_limit")
                return None
            
            exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            
            # Check if testnet
            if 'TESTNET' in (os.getenv('BINANCE_TESTNET_API_KEY') or ''):
                exchange.set_sandbox_mode(True)
            
            balance = exchange.fetch_balance()
            
            # Calculate total in USDT
            usdt_balance = balance.get('USDT', {}).get('total', 0) or 0
            
            # Add value of other holdings
            total_usdt = usdt_balance
            for asset, amounts in balance.items():
                if asset not in ['USDT', 'info', 'timestamp', 'datetime', 'free', 'used', 'total']:
                    if isinstance(amounts, dict) and amounts.get('total', 0) > 0:
                        try:
                            ticker = self.scanner.spot.fetch_ticker(f"{asset}/USDT")
                            price = ticker.get('last', 0) or 0
                            total_usdt += amounts['total'] * price
                        except:
                            pass
            
            return total_usdt
            
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return None
    
    def run_auto_update(self, config_path='config.json'):
        """
        Run full analysis and auto-update config.
        Fetches actual wallet balance from exchange (Passivbot style).
        """
        print("\n" + "="*70)
        print("🤖 AUTO-UPDATE MODE")
        print("="*70)
        
        # Fetch ACTUAL balance from exchange
        balance = self.fetch_wallet_balance()
        
        if balance is None:
            # Fallback to config
            import json
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                balance = config.get('capital_limit', 1000)
                print(f"Using config capital_limit: ${balance:.2f}")
            except:
                balance = 1000
                print(f"Using default balance: ${balance}")
        else:
            print(f"💰 Actual wallet balance: ${balance:.2f}")
        
        # Calculate optimal positions
        n_positions = self.calculate_optimal_positions(balance)
        print(f"Optimal positions for balance: {n_positions}")
        
        # Run analysis
        results = self.run(top_volatile=30, top_final=n_positions * 2)
        
        # Get recommended symbols (those with composite > 0.4)
        recommended = [c['symbol'] for c in results if c['composite'] > 0.4]
        
        if not recommended:
            recommended = [c['symbol'] for c in results[:n_positions]]
        
        # Update config
        self.update_config(config_path, recommended, n_positions)
        
        return recommended


def run_scheduler(interval_hours=1, config_path='config.json'):
    """
    Run portfolio manager on a schedule.
    Updates forager every X hours with best candidates.
    """
    import time
    
    print(f"\n🕐 Starting Portfolio Scheduler (every {interval_hours}h)")
    print(f"   Config: {config_path}")
    print("-" * 50)
    
    pm = PortfolioManager()
    
    while True:
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running analysis...")
            pm.run_auto_update(config_path)
            
            next_run = datetime.now() + timedelta(hours=interval_hours)
            print(f"\n⏰ Next update at: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
            
            time.sleep(interval_hours * 3600)
            
        except KeyboardInterrupt:
            print("\n\n👋 Scheduler stopped.")
            break
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            time.sleep(60)  # Wait 1 min before retry


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Portfolio Manager - Multi-Factor Coin Selection')
    parser.add_argument('--auto', action='store_true', help='Run auto-update mode (updates config with best coins)')
    parser.add_argument('--schedule', type=float, help='Run on schedule (hours between updates)')
    parser.add_argument('--config', default='config.json', help='Config file path')
    
    args = parser.parse_args()
    
    if args.schedule:
        run_scheduler(interval_hours=args.schedule, config_path=args.config)
    elif args.auto:
        pm = PortfolioManager()
        pm.run_auto_update(config_path=args.config)
    else:
        pm = PortfolioManager()
        pm.run(top_volatile=20, top_final=10)
