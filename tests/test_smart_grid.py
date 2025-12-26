
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from src.live.async_runner import AsyncMultiSymbolBot

class TestSmartGrid:
    
    @pytest.fixture
    def bot(self):
        bot = AsyncMultiSymbolBot.__new__(AsyncMultiSymbolBot)
        bot.config = {
            'grid_settings': {
                'max_orders': 5,
                'grid_span': 0.05,
                'grid_spacing_weight': 1.0,
                'qty_step_multiplier': 1.0
            },
            'smart_grid': {
                'enabled': True,
                'volatility_scale': 1.0,
                'trend_bias_strength': 0.5
            },
            'wallet_exposure_limit': 1.0,
            'total_wallet_exposure_limit': 1.0
        }
        bot.atr = {}
        bot.rsi = {}
        bot.ema_long = {}
        bot.ema_short = {}
        bot.last_prices = {}
        bot.positions = {}
        bot.open_orders = {}
        
        # Mock exchange methods
        bot.exchange = MagicMock()
        bot.auto_unstuck = AsyncMock(return_value=False)
        bot.calc_geometric_grid = MagicMock(return_value=[100, 99, 98])
        bot.calc_martingale_qty = MagicMock(return_value=[1, 1, 1])
        bot.place_limit_order = AsyncMock(return_value={'id': '1'})
        bot.validate_order = AsyncMock(return_value=(True, None))
        bot.refresh_balance_before_order = AsyncMock()
        bot.get_active_symbols = MagicMock(return_value=['ETH/USDT'])
        bot.total_equity = 1000
        
        return bot

    @pytest.mark.asyncio
    async def test_atr_widens_grid(self, bot):
        # Setup: High ATR (5% of price)
        symbol = 'ETH/USDT'
        price = 1000.0
        atr = 50.0  # 5%
        
        bot.last_prices[symbol] = {'price': price}
        bot.atr[symbol] = atr
        bot.config['grid_settings']['grid_span'] = 0.05 # Base 5%
        
        # We need to inspect call to calc_geometric_grid to see if grid_span increased
        # Mock calc_geometric_grid to capture arguments
        bot.calc_geometric_grid = MagicMock(return_value=[]) 
        
        await bot.process_symbol(symbol)
        
        # Expected span calculation:
        # min_span = (atr / price) * 4 * vol_scale = 0.05 * 4 * 1.0 = 0.20 (20%)
        # Should be widened from 0.05 to 0.20
        
        args = bot.calc_geometric_grid.call_args
        assert args is not None
        # args[0] is tuple of positional args: (neutral_price, n_orders, spacing_weight, grid_span)
        # grid_span is the 4th argument (index 3)
        actual_span = args[0][3] 
        
        assert actual_span == pytest.approx(0.20)

    @pytest.mark.asyncio
    async def test_rsi_oversold_skews_neutral_up(self, bot):
        # Setup: RSI 20 (Oversold) -> Expect neutral price to move UP
        symbol = 'ETH/USDT'
        price = 1000.0
        bot.last_prices[symbol] = {'price': price}
        bot.atr[symbol] = 10.0 # Low ATR so span doesn't change
        bot.rsi[symbol] = 20.0
        bot.ema_long[symbol] = 1000.0
        
        await bot.process_symbol(symbol)
        
        args = bot.calc_geometric_grid.call_args
        actual_neutral = args[0][0]
        
        # Expected skew:
        # skew = (30 - 20) / 100 * 0.5 = 0.1 * 0.5 = 0.05 (5%)
        # neutral = 1000 * (1 + 0.05) = 1050
        
        assert actual_neutral == pytest.approx(1050.0)

    @pytest.mark.asyncio
    async def test_rsi_overbought_skews_neutral_down(self, bot):
        # Setup: RSI 80 (Overbought) -> Expect neutral price to move DOWN
        symbol = 'ETH/USDT'
        price = 1000.0
        bot.last_prices[symbol] = {'price': price}
        bot.atr[symbol] = 10.0
        bot.rsi[symbol] = 80.0
        bot.ema_long[symbol] = 1000.0
        
        await bot.process_symbol(symbol)
        
        args = bot.calc_geometric_grid.call_args
        actual_neutral = args[0][0]
        
        # Expected skew:
        # skew = (80 - 70) / 100 * 0.5 = 0.1 * 0.5 = 0.05
        # neutral = 1000 * (1 - 0.05) = 950
        
        assert actual_neutral == pytest.approx(950.0)
