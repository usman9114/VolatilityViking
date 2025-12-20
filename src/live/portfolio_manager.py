
import ccxt
import pandas as pd
import numpy as np
import os
import sys
import logging
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

class VolatilityScanner:
    def __init__(self, limit_top_volume=20):
        load_dotenv()
        self.limit_top_volume = limit_top_volume
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

    def get_top_volume_coins(self):
        """
        Fetch top coins by 24h volume to ensure liquidity.
        """
        try:
            logger.info("Fetching tickers...")
            tickers = self.exchange.fetch_tickers()
            
            # Filter for USDT pairs only
            usdt_pairs = [
                symbol for symbol in tickers 
                if symbol.endswith('/USDT') and 'UP/' not in symbol and 'DOWN/' not in symbol
            ]
            
            # Sort by Quote Volume
            sorted_pairs = sorted(
                usdt_pairs, 
                key=lambda x: tickers[x]['quoteVolume'] if tickers[x]['quoteVolume'] else 0, 
                reverse=True
            )
            
            top_pairs = sorted_pairs[:self.limit_top_volume]
            logger.info(f"Top {len(top_pairs)} coins by volume: {top_pairs}")
            return top_pairs
            
        except Exception as e:
            logger.error(f"Error fetching top coins: {e}")
            return []

    def calculate_volatility(self, symbol, limit=60):
        """
        Calculate Passivbot-style volatility: mean(log(high/low))
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='1m', limit=limit)
            if not ohlcv:
                return 0.0
                
            log_ranges = []
            for candle in ohlcv:
                high = candle[2]
                low = candle[3]
                if low > 0:
                    log_range = np.log(high / low)
                    log_ranges.append(log_range)
                    
            if not log_ranges:
                return 0.0
                
            volatility = np.mean(log_ranges)
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return 0.0

    def scan(self):
        """
        Scan top coins and rank by volatility.
        """
        top_coins = self.get_top_volume_coins()
        results = []
        
        logger.info(f"Calculating volatility for {len(top_coins)} coins...")
        for symbol in top_coins:
            vol = self.calculate_volatility(symbol)
            results.append({
                'symbol': symbol,
                'volatility': vol
            })
            
        # rank by volatility (highest first)
        ranked = sorted(results, key=lambda x: x['volatility'], reverse=True)
        return ranked

class NewsAnalyzer:
    def __init__(self):
        self.news_fetcher = NewsFetcher()
        logger.info("Loading Sentence Transformer...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded.")

    def analyze_sentiment(self, symbol):
        """
        Fetch news for symbol and calculate average sentiment score.
        """
        try:
            # Clean symbol for news fetcher (remove /USDT)
            coin = symbol.split('/')[0]
            
            news_df = self.news_fetcher.fetch_news(coin, limit=10)
            
            if news_df.empty:
                return 0.0, 0 # Score, Count
            
            headlines = news_df['title'].tolist()
            if not headlines:
                return 0.0, 0
                
            # Embeddings
            embeddings = self.model.encode(headlines)
            
            # Simple sentiment proxy:
            # In a real scenario, you'd train a classifier.
            # Here, we will assume similarity to "positive market sentiment" phrases 
            # or usage of a zero-shot approach if available.
            # But since we only have embeddings, we can't do direct "Positive/Negative" classification 
            # without a reference prototype or a fine-tuned head.
            
            # ALTERNATIVE: Use a simple keyword heuristic combined with embedding clustering?
            # Or simplified: projected similarity to "Crypto price bull run positive growth" vs "Crypto crash dump panic"
            
            positive_anchor = self.model.encode(["Crypto price nice surge bull market positive growth"])
            negative_anchor = self.model.encode(["Crypto crash dump market panic sell negative"])
            
            scores = []
            for emb in embeddings:
                pos_sim = np.dot(emb, positive_anchor[0])
                neg_sim = np.dot(emb, negative_anchor[0])
                # Score from -1 to 1 based on which anchor it's closer to
                score = pos_sim - neg_sim
                scores.append(score)
                
            avg_score = np.mean(scores)
            return avg_score, len(headlines)
            
        except Exception as e:
            logger.error(f"Error analyzing news for {symbol}: {e}")
            return 0.0, 0

class PortfolioManager:
    def __init__(self):
        self.scanner = VolatilityScanner(limit_top_volume=15) # Scan top 15 liquid pairs
        self.analyzer = NewsAnalyzer()
    
    def run(self):
        print("\n" + "="*50)
        print("🚀 PORTFOLIO MANAGER AI")
        print("="*50 + "\n")
        
        # 1. Scan for Volatility (Passivbot Logic)
        print("1. Scanning Market Volatility (Forager Mode)...")
        candidates = self.scanner.scan()
        
        # Take top 5 volatile coins
        top_volatile = candidates[:5]
        print(f"\nTop 5 Volatile Candidates:")
        for c in top_volatile:
            print(f"  - {c['symbol']}: {c['volatility']:.5f}")
            
        # 2. Analyze News Sentiment
        print("\n2. Analyzing News Sentiment (AI Powered)...")
        results = []
        for c in top_volatile:
            symbol = c['symbol']
            print(f"  > Analyzing {symbol}...")
            score, count = self.analyzer.analyze_sentiment(symbol)
            
            results.append({
                'symbol': symbol,
                'volatility': c['volatility'],
                'news_score': score,
                'news_count': count
            })
            
        # 3. Final Recommendation
        print("\n" + "="*50)
        print("🏆 FINAL RECOMMENDATIONS")
        print("="*50)
        
        # Rank by combined score (normalized vol + news score)
        # Assuming volatility is ~0.001 and news score is ~0.1
        # Let's simple rank by News Score for now, filtering only high volatile ones
        
        results.sort(key=lambda x: x['news_score'], reverse=True)
        
        print(f"{'SYMBOL':<10} | {'VOLATILITY':<12} | {'NEWS SCORE':<12} | {'ARTICLES':<8}")
        print("-" * 50)
        
        for r in results:
            # Color code
            score_display = f"{r['news_score']:.4f}"
            if r['news_score'] > 0.02:
                status = "BUY/LONG"
            elif r['news_score'] < -0.02:
                status = "SELL/SHORT"
            else:
                status = "NEUTRAL"
                
            print(f"{r['symbol']:<10} | {r['volatility']:.5f}      | {score_display:<12} | {r['news_count']:<8} -> {status}")
            
        print("\nDone.")

if __name__ == "__main__":
    pm = PortfolioManager()
    pm.run()
