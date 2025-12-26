"""
Tests for Kill-Switch and Drawdown Protection
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.live.async_runner import AsyncMultiSymbolBot


class TestKillSwitch:
    """Test kill-switch and drawdown protection."""
    
    def test_kill_switch_not_triggered_at_low_drawdown(self, sample_config):
        """Kill-switch should NOT trigger below threshold."""
        bot = AsyncMultiSymbolBot.__new__(AsyncMultiSymbolBot)
        bot.config = sample_config
        bot.peak_equity = 1000.0
        bot.total_equity = 950.0  # 5% drawdown
        bot.max_drawdown_pct = 0.20
        bot.kill_switch_active = False
        
        result = bot.check_kill_switch()
        
        assert result is False
        assert bot.kill_switch_active is False
    
    def test_kill_switch_triggers_at_threshold(self, sample_config):
        """Kill-switch should trigger at threshold."""
        bot = AsyncMultiSymbolBot.__new__(AsyncMultiSymbolBot)
        bot.config = sample_config
        bot.peak_equity = 1000.0
        bot.total_equity = 790.0  # 21% drawdown
        bot.max_drawdown_pct = 0.20
        bot.kill_switch_active = False
        
        result = bot.check_kill_switch()
        
        assert result is True
        assert bot.kill_switch_active is True
    
    def test_kill_switch_exact_threshold(self, sample_config):
        """Kill-switch at exactly threshold."""
        bot = AsyncMultiSymbolBot.__new__(AsyncMultiSymbolBot)
        bot.config = sample_config
        bot.peak_equity = 1000.0
        bot.total_equity = 800.0  # Exactly 20%
        bot.max_drawdown_pct = 0.20
        bot.kill_switch_active = False
        
        # At exactly 20%, should NOT trigger (needs to exceed)
        result = bot.check_kill_switch()
        assert result is False
    
    def test_kill_switch_zero_peak_equity(self, sample_config):
        """Kill-switch should not trigger with zero peak."""
        bot = AsyncMultiSymbolBot.__new__(AsyncMultiSymbolBot)
        bot.config = sample_config
        bot.peak_equity = 0
        bot.total_equity = 100.0
        bot.max_drawdown_pct = 0.20
        bot.kill_switch_active = False
        
        result = bot.check_kill_switch()
        
        assert result is False


class TestWalletExposureLimit:
    """Test wallet exposure limit checking."""
    
    def test_order_allowed_within_limit(self, sample_config):
        """Order allowed when within WEL."""
        bot = AsyncMultiSymbolBot.__new__(AsyncMultiSymbolBot)
        bot.config = sample_config
        bot.total_equity = 5000.0
        bot.positions = {
            "ETH/USDT": {"value_usdt": 1000, "wallet_exposure": 0.20}
        }
        
        # Order for $500 on $5000 equity with 1.0 WEL and 1 position = OK
        result = bot.check_wallet_exposure_limit("ETH/USDT", 500)
        assert result is True
    
    def test_order_blocked_over_limit(self, sample_config):
        """Order blocked when exceeds WEL."""
        bot = AsyncMultiSymbolBot.__new__(AsyncMultiSymbolBot)
        sample_config['total_wallet_exposure_limit'] = 0.5
        bot.config = sample_config
        bot.total_equity = 1000.0
        bot.positions = {
            "ETH/USDT": {"value_usdt": 400, "wallet_exposure": 0.40}
        }
        
        def get_symbols():
            return ["ETH/USDT"]
        bot.get_active_symbols = get_symbols
        
        # Already at 40%, adding $200 would exceed 50% limit
        result = bot.check_wallet_exposure_limit("ETH/USDT", 200)
        assert result is False
