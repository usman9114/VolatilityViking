
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os
import asyncio
import json

# Correct path import
sys.path.append(os.getcwd())

from src.live.async_runner import AsyncMultiSymbolBot

class AuditLayer1(unittest.TestCase):
    
    def setUp(self):
        # Suppress logging
        import logging
        logging.disable(logging.CRITICAL)
        
        # Mock Config
        self.mock_config = {
            "symbols": ["ETH/USDT"],
            "live_mode": False,
            "futures_settings": {
                "enabled": True,
                "leverage": 5,
                "margin_mode": "ISOLATED"
            }
        }

    @patch('src.live.async_runner.ccxt_async.binance')
    def test_exchange_init(self, mock_ccxt):
        """Verify Exchange Initialization and Key Loading"""
        print("\n[AUDIT] Layer 1: Exchange Connectivity...")
        
        # Mock env vars
        with patch.dict(os.environ, {
            'BINANCE_TESTNET_API_KEY': 'test_key', 
            'BINANCE_TESTNET_SECRET_KEY': 'test_secret'
        }):
            bot = AsyncMultiSymbolBot()
            bot.config = self.mock_config
            
            # Run Init
            # Fix: Make urls a real dict so we can verify assignment
            mock_ccxt.return_value.urls = {'api': {}}
            
            asyncio.run(bot.init_exchange())
            
            # Verify CCXT Init with Testnet Keys
            mock_ccxt.assert_called()
            call_args = mock_ccxt.call_args[0][0]
            self.assertEqual(call_args['apiKey'], 'test_key')
            self.assertEqual(call_args['secret'], 'test_secret')
            self.assertEqual(call_args['options']['defaultType'], 'future')
            
            # Verify URL Override for Testnet
            self.assertIn('fapiPublic', bot.exchange.urls['api'])
            print("✅ Exchange Init & Testnet Keys Verified")

    @patch('src.live.async_runner.ccxt_async.binance')
    def test_leverage_update(self, mock_ccxt):
        """Verify Leverage Setting"""
        print("\n[AUDIT] Layer 1: Leverage & Margin...")
        
        bot = AsyncMultiSymbolBot()
        bot.config = self.mock_config
        bot.exchange = AsyncMock()
        
        # Run Update
        asyncio.run(bot.update_leverage())
        
        # Verify Calls
        bot.exchange.set_leverage.assert_called_with(5, "ETH/USDT")
        bot.exchange.set_margin_mode.assert_called_with("ISOLATED", "ETH/USDT")
        print("✅ Leverage & Margin Mode Logic Verified")

    @patch('src.live.async_runner.ccxt_async.binance')
    def test_balance_parsing_futures(self, mock_ccxt):
        """Verify Futures Balance Parsing"""
        print("\n[AUDIT] Layer 1: Futures Balance Parsing...")
        
        bot = AsyncMultiSymbolBot()
        bot.config = self.mock_config
        bot.exchange = AsyncMock()
        
        # Mock Futures Balance (Recent ccxt structure)
        bot.exchange.fetch_balance.return_value = {
            "USDT": {"total": 1000.0, "free": 500.0},
            "total": {"USDT": 1000.0}
        }
        
        # Run fetch_all_positions (which triggers balance update)
        # Mock fetch_positions to be empty to isolate balance check
        bot.exchange.fetch_positions.return_value = []
        bot.last_prices = {"ETH/USDT": {"price": 2000}}
        
        asyncio.run(bot.fetch_all_positions(["ETH/USDT"]))
        
        self.assertEqual(bot.total_wallet_usdt, 1000.0)
        self.assertEqual(bot.free_wallet_usdt, 500.0)
        print("✅ Futures Balance Structure Verified")

    @patch('src.live.async_runner.ccxt_async.binance')
    def test_balance_parsing_spot(self, mock_ccxt):
        """Verify Spot Balance Parsing"""
        print("\n[AUDIT] Layer 1: Spot Balance Parsing...")
        
        bot = AsyncMultiSymbolBot()
        bot.config = self.mock_config.copy()
        bot.config["futures_settings"]["enabled"] = False
        bot.exchange = AsyncMock()
        
        # Mock Spot Balance
        bot.exchange.fetch_balance.return_value = {
            "USDT": {"total": 2000.0, "free": 2000.0},
            "ETH": {"total": 1.5, "free": 1.5} # Spot holdings
        }
        
        # Mock Ticker for value calc
        bot.last_prices = {"ETH/USDT": {"price": 3000.0}}
        
        asyncio.run(bot.fetch_all_positions(["ETH/USDT"]))
        
        # Check USDT Balance
        self.assertEqual(bot.total_wallet_usdt, 2000.0)
        
        # Check ETH Position (Spot Logic Fixed?)
        eth_pos = bot.positions["ETH/USDT"]
        self.assertEqual(eth_pos['amount'], 1.5)
        self.assertEqual(eth_pos['value_usdt'], 1.5 * 3000.0) # 4500
        print("✅ Spot Balance & Position Parsing Verified")

if __name__ == '__main__':
    unittest.main()
