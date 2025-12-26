"""
Tests for Portfolio Manager
Tests multi-factor scoring and config updates.
"""
import pytest
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPositionCalculation:
    """Test optimal position calculation."""
    
    def test_small_balance_single_position(self):
        """Small balance should result in 1 position."""
        from src.live.portfolio_manager import PortfolioManager
        
        pm = PortfolioManager.__new__(PortfolioManager)
        
        n = pm.calculate_optimal_positions(50)  # $50
        assert n == 1
    
    def test_medium_balance_positions(self):
        """Medium balance should have 2-4 positions."""
        from src.live.portfolio_manager import PortfolioManager
        
        pm = PortfolioManager.__new__(PortfolioManager)
        
        n = pm.calculate_optimal_positions(300)  # $300
        assert 1 <= n <= 4
    
    def test_large_balance_positions(self):
        """Large balance should have more positions."""
        from src.live.portfolio_manager import PortfolioManager
        
        pm = PortfolioManager.__new__(PortfolioManager)
        
        n = pm.calculate_optimal_positions(5000)  # $5000
        assert n >= 4


class TestConfigUpdate:
    """Test config file updates."""
    
    def test_update_config_writes_symbols(self):
        """Config update should write new symbols."""
        from src.live.portfolio_manager import PortfolioManager
        
        # Create temp config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "symbols": ["ETH/USDT"],
                "forager": {
                    "enabled": False,
                    "approved_symbols": [],
                    "max_positions": 3
                }
            }
            json.dump(config, f)
            temp_path = f.name
        
        try:
            pm = PortfolioManager.__new__(PortfolioManager)
            pm.update_config(
                temp_path,
                recommended_symbols=["SOL/USDT", "BTC/USDT"],
                n_positions=2
            )
            
            # Read back
            with open(temp_path, 'r') as f:
                updated = json.load(f)
            
            assert "SOL/USDT" in updated["forager"]["approved_symbols"]
            assert "BTC/USDT" in updated["forager"]["approved_symbols"]
            assert updated["forager"]["max_positions"] == 2
        finally:
            os.unlink(temp_path)
