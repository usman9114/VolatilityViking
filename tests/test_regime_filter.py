import pytest
from unittest.mock import MagicMock, AsyncMock
from src.live.async_runner import AsyncMultiSymbolBot

@pytest.fixture
def bot():
    b = AsyncMultiSymbolBot("config.json")
    b.exchange = MagicMock()
    b.exchange.create_limit_order = AsyncMock()
    b.place_limit_order = AsyncMock()
    b.calc_geometric_grid = MagicMock(return_value=[100, 99, 98])
    b.calc_martingale_qty = MagicMock(return_value=[1, 1, 1])
    b.validate_order = AsyncMock(return_value=(True, None))
    b.refresh_balance_before_order = AsyncMock(return_value=True)
    
    # Mock state
    b.last_prices = {'ETH/USDT': {'price': 100}}
    b.positions = {'ETH/USDT': {'amount': 10, 'free': 10}}
    b.open_orders = {'ETH/USDT': []}
    b.total_equity = 1000
    
    # Enable regime filter
    b.config['regime_filter'] = {
        'enabled': True,
        'adx_threshold': 25,
        'slope_threshold': 0.05,
        'top_trend_filter': True,
        'bottom_trend_filter': True
    }
    return b

@pytest.mark.asyncio
async def test_ranging_market(bot):
    # Ranging: Low ADX
    bot.adx = {'ETH/USDT': 15}
    bot.ema_slope = {'ETH/USDT': 0.0}
    
    await bot.process_symbol('ETH/USDT')
    
    # Both Buys and Sells should happen
    # We expect calls to place_limit_order for both 'buy' and 'sell' (since we have position)
    buy_calls = [c for c in bot.place_limit_order.call_args_list if c[0][1] == 'buy']
    sell_calls = [c for c in bot.place_limit_order.call_args_list if c[0][1] == 'sell']
    
    assert len(buy_calls) > 0, "Buys should be allowed in range"
    assert len(sell_calls) > 0, "Sells should be allowed in range"

@pytest.mark.asyncio
async def test_strong_uptrend(bot):
    # Strong Uptrend: High ADX, Positive Slope
    bot.adx = {'ETH/USDT': 30}
    bot.ema_slope = {'ETH/USDT': 0.1} # > 0.05
    
    await bot.process_symbol('ETH/USDT')
    
    buy_calls = [c for c in bot.place_limit_order.call_args_list if c[0][1] == 'buy']
    sell_calls = [c for c in bot.place_limit_order.call_args_list if c[0][1] == 'sell']
    
    assert len(buy_calls) > 0, "Buys should be allowed in uptrend"
    assert len(sell_calls) == 0, "Sells MUST be blocked in strong uptrend"

@pytest.mark.asyncio
async def test_strong_downtrend(bot):
    # Strong Downtrend: High ADX, Negative Slope
    bot.adx = {'ETH/USDT': 30}
    bot.ema_slope = {'ETH/USDT': -0.1} # < -0.05
    
    await bot.process_symbol('ETH/USDT')
    
    buy_calls = [c for c in bot.place_limit_order.call_args_list if c[0][1] == 'buy']
    sell_calls = [c for c in bot.place_limit_order.call_args_list if c[0][1] == 'sell']
    
    assert len(buy_calls) == 0, "Buys MUST be blocked in strong downtrend"
    assert len(sell_calls) > 0, "Sells should be allowed in downtrend"
