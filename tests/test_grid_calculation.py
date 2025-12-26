"""
Tests for Grid Calculation Logic
Tests geometric grid pricing and martingale sizing.
"""
import pytest
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.live.async_runner import AsyncMultiSymbolBot


class TestGeometricGrid:
    """Test geometric grid price calculation."""
    
    def test_grid_prices_decrease_from_center(self, sample_config):
        """Grid buy prices should decrease from center."""
        bot = AsyncMultiSymbolBot.__new__(AsyncMultiSymbolBot)
        bot.config = sample_config
        
        center = 1000.0
        prices = bot.calc_geometric_grid(center, n_orders=5, spacing_weight=1.2, grid_span=0.30)
        
        assert len(prices) == 5
        for i in range(len(prices) - 1):
            assert prices[i] > prices[i+1], "Prices should decrease"
        assert all(p < center for p in prices), "All prices below center"
    
    def test_grid_covers_span(self, sample_config):
        """Grid should cover approximately the specified span."""
        bot = AsyncMultiSymbolBot.__new__(AsyncMultiSymbolBot)
        bot.config = sample_config
        
        center = 1000.0
        grid_span = 0.30
        prices = bot.calc_geometric_grid(center, n_orders=10, spacing_weight=1.2, grid_span=grid_span)
        
        if prices:
            lowest = min(prices)
            actual_span = (center - lowest) / center
            assert math.isclose(actual_span, grid_span, rel_tol=0.05)
    
    def test_grid_spacing_increases(self, sample_config):
        """Spacing between orders should increase geometrically."""
        bot = AsyncMultiSymbolBot.__new__(AsyncMultiSymbolBot)
        bot.config = sample_config
        
        center = 1000.0
        prices = bot.calc_geometric_grid(center, n_orders=5, spacing_weight=1.5, grid_span=0.30)
        
        spacings = [prices[i] - prices[i+1] for i in range(len(prices)-1)]
        for i in range(len(spacings) - 1):
            assert spacings[i+1] > spacings[i], "Spacing should increase"
    
    def test_grid_empty_for_zero_orders(self, sample_config):
        """Empty grid for zero orders."""
        bot = AsyncMultiSymbolBot.__new__(AsyncMultiSymbolBot)
        bot.config = sample_config
        
        prices = bot.calc_geometric_grid(1000.0, n_orders=0)
        assert prices == []
    
    def test_uniform_spacing_when_weight_is_one(self, sample_config):
        """Uniform spacing when weight=1.0."""
        bot = AsyncMultiSymbolBot.__new__(AsyncMultiSymbolBot)
        bot.config = sample_config
        
        center = 1000.0
        prices = bot.calc_geometric_grid(center, n_orders=5, spacing_weight=1.0, grid_span=0.30)
        
        spacings = [prices[i] - prices[i+1] for i in range(len(prices)-1)]
        # All spacings should be equal
        for s in spacings:
            assert math.isclose(s, spacings[0], rel_tol=0.01)


class TestMartingaleSizing:
    """Test martingale quantity calculation."""
    
    def test_quantities_increase(self, sample_config):
        """Quantities should increase with multiplier > 1."""
        bot = AsyncMultiSymbolBot.__new__(AsyncMultiSymbolBot)
        bot.config = sample_config
        
        qtys = bot.calc_martingale_qty(1000.0, n_orders=5, multiplier=1.4, wel=1.0)
        
        assert len(qtys) == 5
        for i in range(len(qtys) - 1):
            assert qtys[i+1] > qtys[i], "Quantities should increase"
    
    def test_total_allocation_matches_wel(self, sample_config):
        """Total allocation should match equity * WEL."""
        bot = AsyncMultiSymbolBot.__new__(AsyncMultiSymbolBot)
        bot.config = sample_config
        
        equity = 1000.0
        wel = 0.5
        qtys = bot.calc_martingale_qty(equity, n_orders=5, multiplier=1.4, wel=wel)
        
        total = sum(qtys)
        expected = equity * wel
        assert math.isclose(total, expected, rel_tol=0.01)
    
    def test_uniform_when_multiplier_is_one(self, sample_config):
        """Uniform sizing when multiplier=1.0."""
        bot = AsyncMultiSymbolBot.__new__(AsyncMultiSymbolBot)
        bot.config = sample_config
        
        qtys = bot.calc_martingale_qty(1000.0, n_orders=5, multiplier=1.0, wel=1.0)
        
        assert len(qtys) == 5
        for q in qtys:
            assert math.isclose(q, qtys[0], rel_tol=0.01)
    
    def test_empty_for_zero_orders(self, sample_config):
        """Empty result for zero orders."""
        bot = AsyncMultiSymbolBot.__new__(AsyncMultiSymbolBot)
        bot.config = sample_config
        
        qtys = bot.calc_martingale_qty(1000.0, n_orders=0, multiplier=1.4)
        assert qtys == []
