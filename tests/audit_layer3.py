
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os
import asyncio

# Correct path import
sys.path.append(os.getcwd())

from src.live.async_runner import AsyncMultiSymbolBot

class AuditLayer3(unittest.TestCase):
    
    def setUp(self):
        import logging
        logging.disable(logging.CRITICAL)
        
        self.mock_config = {
            "symbols": ["ETH/USDT"],
            "max_drawdown_pct": 0.20,
            "total_wallet_exposure_limit": 1.0,
            "futures_settings": {"enabled": True},
            "grid_settings": {"max_orders": 5, "grid_span": 0.1, "grid_spacing_weight": 1.5}
        }

    # --- 1. Kill Switch Logic ---
    @patch('src.live.async_runner.AsyncMultiSymbolBot.cancel_all_orders', new_callable=AsyncMock)
    def test_kill_switch(self, mock_cancel):
        """Verify Kill Switch Logic"""
        print("\n[AUDIT] Layer 3: Kill Switch...")
        bot = AsyncMultiSymbolBot()
        bot.config = self.mock_config
        bot.max_drawdown_pct = 0.20
        bot.peak_equity = 1000.0
        
        # Current Equity = 900 (10% DD) -> Should FAIL (return False)
        bot.total_equity = 900.0
        triggered = bot.check_kill_switch()
        self.assertFalse(triggered)
        
        # Current Equity = 700 (30% DD) -> Should TRIGGER
        bot.total_equity = 700.0
        triggered = bot.check_kill_switch()
        self.assertTrue(triggered)
        self.assertTrue(bot.kill_switch_active)
        print("✅ Kill Switch Triggers Correctly")

    # --- 2. Wallet Exposure Logic ---
    def test_wallet_exposure(self):
        """Verify Wallet Exposure Limits"""
        print("\n[AUDIT] Layer 3: Wallet Exposure...")
        bot = AsyncMultiSymbolBot()
        bot.config = self.mock_config # Limit 1.0 (100%)
        bot.config['symbols'] = ["ETH/USDT"] # 1 symbol -> 100% allowed
        
        bot.total_equity = 1000.0
        bot.positions = {"ETH/USDT": {"value_usdt": 500.0, "wallet_exposure": 0.5}}
        
        # Try to add $650 (Total 1150 > 1100 allowed with buffer) -> Should Fail
        allowed = bot.check_wallet_exposure_limit("ETH/USDT", 650.0)
        self.assertFalse(allowed)
        
        # Try to add $400 (Total 900 < 1000) -> Should Pass
        allowed = bot.check_wallet_exposure_limit("ETH/USDT", 400.0)
        self.assertTrue(allowed)
        print("✅ Wallet Exposure Logic Verified")

    # --- 3. Grid Math Logic ---
    def test_grid_math(self):
        """Verify Geometric Grid & Martingale Math"""
        print("\n[AUDIT] Layer 3: Grid Math...")
        bot = AsyncMultiSymbolBot()
        
        # Geometric Grid
        center = 1000
        orders = 5
        prices = bot.calc_geometric_grid(center, orders, spacing_weight=1.5, grid_span=0.10, direction='down')
        
        self.assertEqual(len(prices), 5)
        self.assertTrue(prices[0] < center) # First buy below center
        self.assertTrue(prices[-1] < prices[0]) # Last buy lower
        
        # Verify Span
        max_dist = (center - prices[-1])/center
        self.assertAlmostEqual(max_dist, 0.10, delta=0.001)
        
        # Martingale
        qtys = bot.calc_martingale_qty(total_equity=1000, n_orders=5, multiplier=2.0, wel=0.1)
        # Alloc = $100
        # q0*(2^5 - 1)/(2-1) = 100 -> q0*31 = 100 -> q0 ~= 3.22
        expected_sum = 100.0
        self.assertAlmostEqual(sum(qtys), expected_sum, delta=0.01)
        print("✅ Grid & Martingale Math Verified")

if __name__ == '__main__':
    unittest.main()
