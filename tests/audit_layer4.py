
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os
import asyncio

# Correct path import
sys.path.append(os.getcwd())

from src.live.async_runner import AsyncMultiSymbolBot

class AuditLayer4(unittest.TestCase):
    
    def setUp(self):
        import logging
        logging.disable(logging.CRITICAL)
        
        self.mock_config = {
            "symbols": ["ETH/USDT"],
            "orders": {"min_cost": 5.0, "min_amount": 0.001}
        }

    @patch('src.live.async_runner.logger')
    @patch('src.live.async_runner.AsyncMultiSymbolBot.check_wallet_exposure_limit')
    @patch('src.live.async_runner.ccxt_async.binance')
    def test_order_placement(self, mock_ccxt, mock_wel, mock_logger):
        """Verify Order Placement & Precision Format"""
        print("\n[AUDIT] Layer 4: Order Validation & Placement...")
        
        bot = AsyncMultiSymbolBot()
        bot.config = self.mock_config
        bot.exchange = AsyncMock()
        
        # Mock Precision Methods (Sync)
        bot.exchange.amount_to_precision = MagicMock(side_effect=lambda s, a: str(float(a)))
        bot.exchange.price_to_precision = MagicMock(side_effect=lambda s, p: str(float(p)))
        
        # Mock load_markets (Async)
        bot.exchange.load_markets.return_value = {
            "ETH/USDT": {
                "limits": {
                    "amount": {"min": 0.0001},
                    "cost": {"min": 5.0}
                }
            }
        }
        
        # Mock Balances (Success)
        bot.refresh_balance_before_order = AsyncMock(return_value=True)
        bot.last_prices = {"ETH/USDT": {"price": 2000.0}} # Required for validate_order cost check
        mock_wel.return_value = True
        
        # 1. Valid Order
        asyncio.run(bot.place_limit_order("ETH/USDT", "buy", 0.1, 2000.0))
        
        # Debug why it failed
        if not bot.exchange.create_limit_order.called:
            if mock_logger.debug.called:
                print(f"debug: {mock_logger.debug.call_args}")
            if mock_logger.warning.called:
                 print(f"warning: {mock_logger.warning.call_args}")
            if mock_logger.error.called:
                 print(f"error: {mock_logger.error.call_args}")
        
        bot.exchange.create_limit_order.assert_called_with(
            "ETH/USDT", "buy", 0.1, 2000.0
        )
        print("✅ Order Placed with Correct Precision")
        
        # 2. Invalid Order (Too Small)
        bot.exchange.create_limit_order.reset_mock()
        # Cost = 0.0001 * 2000 = 0.2 < 5.0 Min Cost
        asyncio.run(bot.place_limit_order("ETH/USDT", "buy", 0.0001, 2000.0))
        
        bot.exchange.create_limit_order.assert_not_called()
        print("✅ Min Cost Validation Blocked Small Order")

    def test_cancel_orders(self):
        """Verify Cancellation Logic"""
        print("\n[AUDIT] Layer 4: Order Cancellation...")
        
        bot = AsyncMultiSymbolBot()
        bot.exchange = AsyncMock()
        bot.open_orders = {
            "ETH/USDT": [{'id': '123'}, {'id': '456'}]
        }
        
        asyncio.run(bot.cancel_symbol_orders("ETH/USDT"))
        
        # Verify calls
        self.assertEqual(bot.exchange.cancel_order.call_count, 2)
        print("✅ All Open Orders Cancelled")

if __name__ == '__main__':
    unittest.main()
