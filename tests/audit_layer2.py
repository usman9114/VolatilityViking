
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os
import asyncio
import pandas as pd
import numpy as np

# Correct path import
sys.path.append(os.getcwd())

from src.live.async_runner import AsyncMultiSymbolBot

class AuditLayer2(unittest.TestCase):
    
    def setUp(self):
        # Suppress logging
        import logging
        logging.disable(logging.CRITICAL)
        
        # Mock Config
        self.mock_config = {
            "symbols": ["ETH/USDT"],
            "ema_settings": {"long_term_span": 20, "short_term_span": 5},
            "smart_grid": {"enabled": True, "atr_period": 14, "rsi_period": 14}
        }

    @patch('src.live.async_runner.ccxt_async.binance')
    def test_indicator_calculation(self, mock_ccxt):
        """Verify RSI, ATR, ADX calculation"""
        print("\n[AUDIT] Layer 2: Indicator Logic...")
        
        bot = AsyncMultiSymbolBot()
        bot.config = self.mock_config
        bot.exchange = AsyncMock()
        
        # Generate dummy OHLCV (100 candles)
        # Random walk
        closes = np.random.normal(100, 1, 100).cumsum() + 1000
        opens = closes - np.random.normal(0, 0.5, 100)
        highs = closes + np.random.normal(0.5, 0.1, 100)
        lows = opens - np.random.normal(0.5, 0.1, 100)
        vols = np.random.normal(1000, 100, 100)
        
        ohlcv = []
        for i in range(100):
            ohlcv.append([i*60000, opens[i], highs[i], lows[i], closes[i], vols[i]])
            
        bot.exchange.fetch_ohlcv.return_value = ohlcv
        
        # Run Update
        asyncio.run(bot.update_indicators(["ETH/USDT"]))
        
        # Verify State
        self.assertIn("ETH/USDT", bot.rsi)
        self.assertIn("ETH/USDT", bot.atr)
        self.assertIn("ETH/USDT", bot.adx)
        self.assertIn("ETH/USDT", bot.ema_long)
        
        rsi = bot.rsi["ETH/USDT"]
        atr = bot.atr["ETH/USDT"]
        
        # Sanity Checks
        self.assertTrue(0 <= rsi <= 100, f"RSI {rsi} out of bounds")
        self.assertTrue(atr > 0, "ATR should be positive")
        print(f"✅ Indicators Calculated: RSI={rsi:.2f}, ATR={atr:.2f}")

    @patch('src.live.async_runner.logger')
    @patch('src.live.async_runner.ccxt_async.binance')
    def test_ml_integration(self, mock_ccxt, mock_logger):
        """Verify ML Predictor is called and result stored"""
        print("\n[AUDIT] Layer 2: ML Integration...")
        
        bot = AsyncMultiSymbolBot()
        bot.config = self.mock_config
        bot.exchange = AsyncMock()
        
        # Mock ML Predictor
        bot.ml_predictor = MagicMock()
        bot.ml_predictor.predict_next_move.return_value = (1, 0.85) # Pump, 85%
        
        # Mock OHLCV (Increase to 100 to be safe)
        ohlcv = [[i*60000, 100, 101, 99, 100, 1000] for i in range(100)]
        bot.exchange.fetch_ohlcv.return_value = ohlcv
        
        # Run Update
        asyncio.run(bot.update_indicators(["ETH/USDT"]))
        
        # Check for swallowed errors
        if mock_logger.error.called:
             print(f"❌ Swallowed Error: {mock_logger.error.call_args}")
        
        # Verify ML Call
        bot.ml_predictor.predict_next_move.assert_called()
        
        # Verify Storage
        self.assertIn("ETH/USDT", bot.ml_predictions)
        pred = bot.ml_predictions["ETH/USDT"]
        self.assertEqual(pred['prediction'], 1)
        self.assertEqual(pred['confidence'], 0.85)
        print("✅ ML Predictor Called & Result Stored")

if __name__ == '__main__':
    unittest.main()
