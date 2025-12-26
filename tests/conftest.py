"""
Pytest fixtures for VolatilityViking tests.
"""
import pytest
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "symbols": ["ETH/USDT", "BTC/USDT"],
        "capital_limit": 1000,
        "live_mode": False,
        "wallet_exposure_limit": 1.0,
        "total_wallet_exposure_limit": 1.5,
        "max_drawdown_pct": 0.20,
        "grid_settings": {
            "grid_span": 0.30,
            "grid_spacing_weight": 1.2,
            "qty_step_multiplier": 1.4,
            "max_orders": 5
        },
        "unstuck_settings": {
            "unstuck_threshold": 0.80,
            "unstuck_close_pct": 0.10,
            "unstuck_loss_allowance_pct": 0.02,
            "unstuck_price_distance_threshold": 0.20
        },
        "trailing_settings": {
            "close_trailing_threshold_pct": 0.02,
            "close_trailing_retracement_pct": 0.005
        },
        "forager": {
            "enabled": False,
            "approved_symbols": [],
            "max_positions": 3
        }
    }


@pytest.fixture
def sample_position():
    """Sample position data for testing."""
    return {
        "amount": 0.5,
        "free": 0.5,
        "value_usdt": 1750.0,
        "price": 3500.0,
        "wallet_exposure": 0.35
    }


@pytest.fixture
def sample_prices():
    """Sample price data for testing."""
    return {
        "ETH/USDT": {
            "price": 3500.0,
            "bid": 3499.0,
            "ask": 3501.0,
            "volume": 1000000000
        },
        "BTC/USDT": {
            "price": 97000.0,
            "bid": 96999.0,
            "ask": 97001.0,
            "volume": 5000000000
        }
    }


@pytest.fixture
def mock_exchange():
    """Mock exchange for testing without API calls."""
    class MockExchange:
        def __init__(self):
            self.orders = []
            self.balance = {
                "USDT": {"total": 5000, "free": 3000},
                "ETH": {"total": 0.5, "free": 0.5},
                "BTC": {"total": 0.01, "free": 0.01}
            }
        
        async def fetch_balance(self):
            return self.balance
        
        async def fetch_ticker(self, symbol):
            prices = {
                "ETH/USDT": {"last": 3500, "bid": 3499, "ask": 3501},
                "BTC/USDT": {"last": 97000, "bid": 96999, "ask": 97001}
            }
            return prices.get(symbol, {"last": 100, "bid": 99, "ask": 101})
        
        async def create_limit_order(self, symbol, side, amount, price):
            order = {
                "id": f"test-{len(self.orders)}",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": price,
                "status": "open"
            }
            self.orders.append(order)
            return order
        
        async def cancel_order(self, order_id, symbol):
            return {"id": order_id, "status": "cancelled"}
        
        async def fetch_open_orders(self, symbol):
            return [o for o in self.orders if o["symbol"] == symbol]
        
        async def load_markets(self):
            return {
                "ETH/USDT": {
                    "limits": {"amount": {"min": 0.001}, "cost": {"min": 10}},
                    "precision": {"amount": 4, "price": 2}
                },
                "BTC/USDT": {
                    "limits": {"amount": {"min": 0.00001}, "cost": {"min": 10}},
                    "precision": {"amount": 5, "price": 2}
                }
            }
        
        def amount_to_precision(self, symbol, amount):
            return round(amount, 4)
        
        def price_to_precision(self, symbol, price):
            return round(price, 2)
    
    return MockExchange()
