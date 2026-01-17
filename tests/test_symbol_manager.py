"""
Tests for symbol_manager module.
"""
import pytest
import json
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

# Import after path setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.live.symbol_manager import (
    validate_symbol,
    can_remove_symbol,
    get_symbol_equity,
    add_symbol,
    remove_symbol,
    load_config,
    save_config
)


@pytest.fixture
def mock_exchange():
    """Create a mock exchange with common methods."""
    exchange = AsyncMock()
    exchange.id = "binance"
    exchange.load_markets = AsyncMock(return_value={
        "ETH/USDT": {"symbol": "ETH/USDT", "base": "ETH", "quote": "USDT"},
        "SOL/USDT": {"symbol": "SOL/USDT", "base": "SOL", "quote": "USDT"},
        "BTC/USDT": {"symbol": "BTC/USDT", "base": "BTC", "quote": "USDT"},
    })
    exchange.fetch_positions = AsyncMock(return_value=[])
    exchange.fetch_balance = AsyncMock(return_value={"ETH": {"total": 0}, "SOL": {"total": 0}})
    exchange.fetch_ticker = AsyncMock(return_value={"last": 100.0})
    exchange.fetch_open_orders = AsyncMock(return_value=[])
    return exchange


@pytest.fixture
def temp_config():
    """Create a temporary config file."""
    config = {
        "symbols": ["ETH/USDT"],
        "forager": {
            "approved_symbols": ["ETH/USDT", "SOL/USDT"]
        }
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        return f.name


class TestValidateSymbol:
    @pytest.mark.asyncio
    async def test_valid_symbol(self, mock_exchange):
        """Test that a valid symbol passes validation."""
        result = await validate_symbol("ETH/USDT", mock_exchange)
        assert result is True

    @pytest.mark.asyncio
    async def test_invalid_symbol_raises(self, mock_exchange):
        """Test that an invalid symbol raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            await validate_symbol("FAKE/USDT", mock_exchange)


class TestCanRemoveSymbol:
    @pytest.mark.asyncio
    async def test_can_remove_with_no_equity(self, mock_exchange):
        """Test removal allowed when no equity or orders."""
        can_remove, reason = await can_remove_symbol("ETH/USDT", mock_exchange)
        assert can_remove is True
        assert reason == "OK"

    @pytest.mark.asyncio
    async def test_cannot_remove_with_equity(self, mock_exchange):
        """Test removal blocked when symbol has equity."""
        mock_exchange.fetch_balance = AsyncMock(return_value={"ETH": {"total": 1.5}})
        mock_exchange.fetch_ticker = AsyncMock(return_value={"last": 3000.0})
        
        can_remove, reason = await can_remove_symbol("ETH/USDT", mock_exchange)
        assert can_remove is False
        assert "non-zero equity" in reason

    @pytest.mark.asyncio
    async def test_cannot_remove_with_open_orders(self, mock_exchange):
        """Test removal blocked when symbol has open orders."""
        mock_exchange.fetch_open_orders = AsyncMock(return_value=[{"id": "123", "symbol": "ETH/USDT"}])
        
        can_remove, reason = await can_remove_symbol("ETH/USDT", mock_exchange)
        assert can_remove is False
        assert "open order" in reason


class TestAddSymbol:
    @pytest.mark.asyncio
    async def test_add_to_forager(self, mock_exchange, temp_config):
        """Test adding a symbol to forager approved_symbols."""
        await add_symbol("BTC/USDT", temp_config, mock_exchange, to_core=False)
        
        config = load_config(temp_config)
        assert "BTC/USDT" in config["forager"]["approved_symbols"]

    @pytest.mark.asyncio
    async def test_add_to_core(self, mock_exchange, temp_config):
        """Test adding a symbol to core symbols."""
        await add_symbol("BTC/USDT", temp_config, mock_exchange, to_core=True)
        
        config = load_config(temp_config)
        assert "BTC/USDT" in config["symbols"]

    @pytest.mark.asyncio
    async def test_add_invalid_symbol_raises(self, mock_exchange, temp_config):
        """Test that adding an invalid symbol raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            await add_symbol("FAKE/USDT", temp_config, mock_exchange)


class TestRemoveSymbol:
    @pytest.mark.asyncio
    async def test_remove_symbol_success(self, mock_exchange, temp_config):
        """Test removing a symbol with no equity."""
        await remove_symbol("SOL/USDT", temp_config, mock_exchange)
        
        config = load_config(temp_config)
        assert "SOL/USDT" not in config["forager"]["approved_symbols"]

    @pytest.mark.asyncio
    async def test_remove_blocked_by_equity(self, mock_exchange, temp_config):
        """Test that removal is blocked when symbol has equity."""
        mock_exchange.fetch_balance = AsyncMock(return_value={"SOL": {"total": 10.0}})
        mock_exchange.fetch_ticker = AsyncMock(return_value={"last": 150.0})
        
        with pytest.raises(RuntimeError, match="Cannot remove"):
            await remove_symbol("SOL/USDT", temp_config, mock_exchange)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
