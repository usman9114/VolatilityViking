import os
import feedparser
import re
import json
import logging
from datetime import datetime, timedelta
from openai import OpenAI
from src.ai.utils import call_openai_with_retry

# Setup logging
logger = logging.getLogger(__name__)

class SentimentEngine:
    """
    Layer 1: Sentiment Analysis Engine.
    Uses OpenAI GPT-4o-mini to analyze news/social sentiment.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = None
        self.prompts = self.load_prompts()
        
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info("SentimentEngine: OpenAI Client Initialized")
            except Exception as e:
                logger.error(f"SentimentEngine: Failed to init OpenAI: {e}")
        else:
            logger.warning("SentimentEngine: No OPENAI_API_KEY found. Sentiment will be neutral.")

    def reload(self):
        """Reload prompts or config if needed"""
        self.prompts = self.load_prompts()

    def load_prompts(self):
        try:
            with open('prompts.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load prompts.json: {e}. Using defaults.")
            return {}

    def fetch_news_rss(self, symbol):
        """
        Fetch news from RSS feeds (Cointelegraph, CryptoPanic, etc).
        Returns list of strings (headlines).
        """
        headlines = []
        try:
            # Map symbol to search terms
            base = symbol.split('/')[0].upper() # SOL
            
            # Name Mappings for better recall
            name_map = {
                'BTC': ['Bitcoin', 'BTC'],
                'ETH': ['Ethereum', 'Ether', 'ETH'],
                'SOL': ['Solana', 'SOL'],
                'XRP': ['Ripple', 'XRP'],
                'ADA': ['Cardano', 'ADA'],
                'DOGE': ['Dogecoin', 'DOGE'],
                'DOT': ['Polkadot', 'DOT'],
                'AVAX': ['Avalanche', 'AVAX'],
                'LTC': ['Litecoin', 'LTC'],
                'ATOM': ['Cosmos', 'ATOM'],
                'LINK': ['Chainlink', 'LINK']
            }
            
            keywords = name_map.get(base, [base])
            
            # RSS Sources (Main Feed)
            feeds = [
                "https://cointelegraph.com/rss", # Main Feed
                "https://cryptopanic.com/news/rss/" # General Aggregator
            ]
            
            for url in feeds:
                try:
                    feed = feedparser.parse(url)
                    for entry in feed.entries[:20]: # Check top 20 headlines
                        title = entry.title
                        # Cleaning
                        clean_title = re.sub('<[^<]+?>', '', title)
                        
                        # Local Filter: Check if any keyword matches
                        if any(k.lower() in clean_title.lower() for k in keywords):
                            headlines.append(clean_title)
                except Exception as e:
                    logger.warning(f"Failed to parse feed {url}: {e}")
                    
            return list(set(headlines)) # Dedup
            
        except Exception as e:
            logger.error(f"SentimentEngine: RSS Fetch Error: {e}")
            return []

    async def get_sentiment_score(self, symbol):
        """
        Get sentiment score for a symbol using OpenAI.
        """
        if not self.client:
            return 0.0, 0.0, "No API Key"
            
        # 1. Fetch Headlines
        headlines = self.fetch_news_rss(symbol)
        
        if not headlines:
            return 0.0, 0.0, "No headlines found"
            
        # 2. Construct Prompt
        cfg = self.prompts.get('sentiment_analysis', {})
        system_msg = cfg.get('system', "You are a crypto sentiment analyzer. Output only JSON.")
        user_msg = cfg.get('user_template', "").format(
            symbol=symbol,
            headlines=json.dumps(headlines)
        )
        
        # Fallback if config failed
        if not user_msg:
             user_msg = f"Analyze sentiment for {symbol} based on: {headlines}. Return JSON with score, confidence, reason."
        
        try:
            content = await call_openai_with_retry(
                self.client,
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ]
            )
            
            if not content: return 0.0, 0.0, "AI call failed"
            
            content = content.replace('```json', '').replace('```', '').strip()
            data = json.loads(content)
            
            score = float(data.get('score', 0.0))
            confidence = float(data.get('confidence', 0.0))
            reason = data.get('reason', 'Analysis complete')
            
            logger.info(f"[{symbol}] Sentiment: {score} (Conf: {confidence}) - {reason}")
            return score, confidence, reason
            
        except Exception as e:
            logger.error(f"SentimentEngine: AI Analysis Error: {e}")
            return 0.0, 0.0, f"Error: {str(e)}"
            
    async def analyze_trade_risk(self, symbol, action, ml_confidence, rsi, atr, adx, sent_score, sent_reason, technical_context=""):
        """
        Ask OpenAI to validate a proposed trade (Risk Manager Role).
        """
        if not self.client:
            return {"decision": "APPROVE", "reason": "No Valid AI Key"}
            
        cfg = self.prompts.get('trade_risk_validator', {})
        system_msg = cfg.get('system', "You are a Risk Manager. Output JSON.")
        
        # Format user message
        try:
            user_msg = cfg.get('user_template', "").format(
                symbol=symbol,
                action=action,
                ml_confidence=ml_confidence,
                rsi=rsi,
                atr=atr,
                adx=adx,
                sent_score=sent_score,
                sent_reason=sent_reason,
                technical_context=technical_context
            )
        except Exception as e:
            logger.error(f"Prompt formatting error: {e}")
            return {"decision": "APPROVE", "reason": "Prompt Error"}

        try:
            content = await call_openai_with_retry(
                self.client,
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ]
            )
            
            if not content: return {"decision": "APPROVE", "reason": "AI call failed"}
            
            content = content.replace('```json', '').replace('```', '').strip()
            data = json.loads(content)
            
            decision = data.get('decision', 'APPROVE').upper()
            reason = data.get('reason', 'AI Approved')
            
            logger.info(f"[{symbol}] Risk Manager: {decision} - {reason}")
            return data
            
        except Exception as e:
            logger.error(f"Risk Validator Error: {e}")
            return {"decision": "APPROVE", "reason": f"AI Error: {e}"}

# Self-test
if __name__ == "__main__":
    # Test run
    from dotenv import load_dotenv
    load_dotenv('.env_riz') # Load specific env
    
    logging.basicConfig(level=logging.INFO)
    engine = SentimentEngine()
    s, c, r = engine.get_sentiment_score("ETH/USDT")
    print(f"Result: Score={s}, Conf={c}, Reason={r}")
