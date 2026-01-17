
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import json
import sys
import os
import asyncio

# Correct path import
sys.path.append(os.getcwd())

from src.ai.sentiment_engine import SentimentEngine
from src.ai.global_sentinel import GlobalSentinel
from src.ai.ml_predictor import MLPredictor
from src.live.async_runner import AsyncMultiSymbolBot

class TestVolatilityVikingServices(unittest.TestCase):
    
    def setUp(self):
        # Suppress logging for clean test output
        import logging
        logging.disable(logging.CRITICAL)

    # --- 1. Sentiment Engine Tests ---
    @patch('src.ai.sentiment_engine.OpenAI')
    @patch('src.ai.sentiment_engine.feedparser.parse')
    def test_sentiment_analysis(self, mock_feed, mock_openai):
        """Test if Sentiment Engine parses OpenAI JSON correctly"""
        print("\n[TEST] Sentiment Engine Analysis...")
        
        # Mock Feed
        mock_feed.return_value.entries = [MagicMock(title="Bitcoin hits 100k"), MagicMock(title="ETH ETF Approved")]
        
        # Mock OpenAI Response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "score": 0.8,
            "confidence": 0.9,
            "reason": "Bullish news"
        })
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        engine = SentimentEngine()
        # Corrected Call: get_sentiment_score
        score, conf, reason = engine.get_sentiment_score("ETH/USDT")
        
        self.assertEqual(score, 0.8)
        self.assertEqual(conf, 0.9)
        print("✅ Sentiment Engine Logic Verified")

    # --- 2. Global Sentinel (Risk Officer) Tests ---
    @patch('src.ai.global_sentinel.OpenAI')
    def test_risk_officer_logic(self, mock_openai):
        """Test Risk Officer Leverage Decision"""
        print("\n[TEST] Global Sentinel (Risk Officer)...")
        
        # Mock Risk Officer Output
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "leverage": 2,
            "reason": "High Volatility detected"
        })
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        sentinel = GlobalSentinel()
        market_metrics = {"volatility": "HIGH (0.8%)", "trend": 50}
        decision = sentinel.run_risk_officer(market_metrics)
        
        self.assertEqual(decision['leverage'], 2)
        self.assertIn("High Volatility", decision['reason'])
        print("✅ Risk Officer (AI) Logic Verified")

    # --- 3. ML Predictor Tests ---
    @patch('xgboost.Booster')
    def test_ml_prediction(self, mock_booster):
        """Test ML Predictor interface"""
        print("\n[TEST] ML Predictor Interface...")
        
        # Mock Config
        dummy_config = {"ml_settings": {"enabled": True}, "symbols": ["ETH/USDT"]}
        predictor = MLPredictor(dummy_config)
        
        # Mock load model to avoid file I/O
        predictor.model = mock_booster
        predictor.model.predict.return_value = [0.05] # 5% pump predicted
        
        # Create dummy indicator features
        import pandas as pd
        dummy_df = pd.DataFrame({'close': [100]*200, 'low': [90]*200, 'high':[110]*200, 'volume': [1000]*200}) 
        dummy_df['rsi'] = 50
        
        # Corrected Call: only df
        pred, conf = predictor.predict_next_move(dummy_df)
        
        # Since we mocked predict returning 0.05 and class logic handles it
        # We just verify it runs without crashing
        self.assertIsInstance(pred, int)
        self.assertIsInstance(conf, float)
        print("✅ ML Predictor Interface Verified")

    # --- 4. Bot Risk & Order Validation Tests ---
    async def async_test_order_validation(self):
        """Async test for Bot Validation"""
        print("\n[TEST] Bot Order Validation (Async)...")
        
        bot = AsyncMultiSymbolBot()
        bot.exchange = AsyncMock() # Mock exchange
        
        # Test Kill Switch Blocking
        bot.kill_switch_active = True
        valid, reason = await bot.validate_order("ETH/USDT", 1.0, 'buy')
        self.assertFalse(valid)
        self.assertEqual(reason, "Kill-switch active")
        print("✅ Kill Switch Blocking Verified")
        
        # Test Min Cost Blocking
        bot.kill_switch_active = False
        bot.exchange.load_markets.return_value = {
            "ETH/USDT": {"limits": {"cost": {"min": 5.0}}}
        }
        bot.last_prices = {"ETH/USDT": {"price": 1000}} # 1000 price
        
        # Clean state for market load
        bot._markets = bot.exchange.load_markets.return_value

        # Small order check ($0.1 value)
        valid, reason = await bot.validate_order("ETH/USDT", 0.0001, 'buy') 
        # Cost = 0.0001 * 1000 = 0.1 < 5.0
        self.assertFalse(valid)
        self.assertIn("Below min cost", reason)
        print("✅ Min Cost Validation Verified")

    def test_bot_async_wrapper(self):
        """Wrapper to run async test"""
        asyncio.run(self.async_test_order_validation())

if __name__ == '__main__':
    unittest.main()
