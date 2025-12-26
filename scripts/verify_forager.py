"""
Verification script for Forager feature.
Tests the volatility/volume calculation and symbol selection logic.
"""
import sys
import os
sys.path.append(os.getcwd())

from unittest.mock import MagicMock, patch

def test_forager():
    print("Testing Forager Feature...")
    
    # Import
    from src.live.binance_bot import BinanceBot
    
    # 1. Create Bot with Forager enabled config
    bot = BinanceBot(config_path='config.json')
    
    # Mock exchange
    bot.exchange = MagicMock()
    
    # 2. Test calc_volatility
    print("\n[Test] calc_volatility")
    # Mock OHLCV data: prices from 100 with varying ranges
    mock_ohlcv = []
    for i in range(60):
        close = 100 + i * 0.1
        high = close * 1.02   # 2% above
        low = close * 0.98    # 2% below
        mock_ohlcv.append([0, 0, high, low, close, 1000])
    
    bot.exchange.fetch_ohlcv.return_value = mock_ohlcv
    
    vol = bot.calc_volatility('ETH/USDT', 60)
    print(f"Volatility: {vol:.6f}")
    
    if vol > 0.03 and vol < 0.05:  # Expecting ~0.04 (4% range)
        print("✅ Volatility calculation works")
    else:
        print(f"❌ Unexpected volatility: {vol}")
    
    # 3. Test calc_volume
    print("\n[Test] calc_volume")
    volume = bot.calc_volume('ETH/USDT', 60)
    print(f"Avg Quote Volume: {volume:.0f}")
    
    if volume > 0:
        print("✅ Volume calculation works")
    else:
        print("❌ Volume calculation failed")
    
    # 4. Test select_forager_coins
    print("\n[Test] select_forager_coins")
    
    # Enable forager in config
    bot.config['forager'] = {
        'enabled': True,
        'approved_symbols': ['ETH/USDT', 'BTC/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT'],
        'max_positions': 3,
        'volume_drop_pct': 0.25,
        'volatility_lookback_minutes': 60,
        'volume_lookback_minutes': 60
    }
    
    # Mock different volatilities for each symbol
    volatility_map = {
        'ETH/USDT': 0.05,
        'BTC/USDT': 0.02,
        'SOL/USDT': 0.08,  # Highest
        'XRP/USDT': 0.03,
        'DOGE/USDT': 0.07  # Second highest
    }
    volume_map = {
        'ETH/USDT': 1000000,
        'BTC/USDT': 5000000,  # Highest volume
        'SOL/USDT': 500000,
        'XRP/USDT': 100000,   # Low volume - may be dropped
        'DOGE/USDT': 800000
    }
    
    def mock_volatility(symbol, lookback):
        return volatility_map.get(symbol, 0.01)
    
    def mock_volume(symbol, lookback):
        return volume_map.get(symbol, 0)
    
    bot.calc_volatility = mock_volatility
    bot.calc_volume = mock_volume
    
    selected = bot.select_forager_coins()
    print(f"Selected: {selected}")
    
    # Expect: SOL (highest vol) and DOGE (2nd highest) in top 3
    if 'SOL/USDT' in selected:
        print("✅ Highest volatility coin selected (SOL)")
    else:
        print("❌ SOL should be selected (highest volatility)")
    
    if len(selected) == 3:
        print(f"✅ Correct number of positions selected: {len(selected)}")
    else:
        print(f"❌ Expected 3 positions, got {len(selected)}")

if __name__ == "__main__":
    test_forager()
