"""
Symbol Manager – Safe add/remove of trading symbols.

This module provides validation and safety checks for dynamically managing
the list of symbols the bot trades.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


async def validate_symbol(symbol: str, exchange) -> bool:
    """
    Validate that a symbol exists on the exchange.
    
    Args:
        symbol: Symbol in CCXT format, e.g. 'ETH/USDT'.
        exchange: Initialised CCXT exchange instance.
        
    Returns:
        True if valid.
        
    Raises:
        ValueError: If the symbol does not exist on the exchange.
    """
    try:
        markets = await exchange.load_markets()
        if symbol not in markets:
            raise ValueError(f"Symbol '{symbol}' not found on {exchange.id}. Available markets: {len(markets)}")
        logger.info(f"[SymbolManager] Validated symbol: {symbol}")
        return True
    except Exception as e:
        raise ValueError(f"Failed to validate symbol '{symbol}': {e}")


async def get_symbol_equity(symbol: str, exchange, is_futures: bool = False) -> float:
    """
    Get the current equity (position value) for a symbol.
    
    Returns:
        Float representing the position value (0 if no position).
    """
    try:
        if is_futures:
            positions = await exchange.fetch_positions([symbol])
            for pos in positions:
                if pos.get('symbol') == symbol:
                    contracts = abs(float(pos.get('contracts', 0) or 0))
                    mark_price = float(pos.get('markPrice', 0) or 0)
                    return contracts * mark_price
        else:
            # Spot: check base currency balance
            base = symbol.split('/')[0]
            balance = await exchange.fetch_balance()
            amount = float(balance.get(base, {}).get('total', 0) or 0)
            if amount > 0:
                ticker = await exchange.fetch_ticker(symbol)
                price = ticker.get('last', 0) or 0
                return amount * price
        return 0.0
    except Exception as e:
        logger.warning(f"[SymbolManager] Could not fetch equity for {symbol}: {e}")
        return 0.0


async def get_open_orders(symbol: str, exchange) -> list:
    """Return list of open orders for a symbol."""
    try:
        orders = await exchange.fetch_open_orders(symbol)
        return orders
    except Exception as e:
        logger.warning(f"[SymbolManager] Could not fetch open orders for {symbol}: {e}")
        return []


async def can_remove_symbol(symbol: str, exchange, is_futures: bool = False) -> tuple[bool, str]:
    """
    Check if a symbol can be safely removed.
    
    Returns:
        (can_remove: bool, reason: str)
    """
    # Check equity
    equity = await get_symbol_equity(symbol, exchange, is_futures)
    if equity > 0:
        return False, f"Symbol {symbol} has non-zero equity: ${equity:.2f}"
    
    # Check open orders
    orders = await get_open_orders(symbol, exchange)
    if orders:
        return False, f"Symbol {symbol} has {len(orders)} open order(s)"
    
    return True, "OK"


def load_config(config_path: str) -> dict:
    """Load config.json."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config_path: str, config: dict):
    """Save config.json atomically."""
    path = Path(config_path)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(config, f, indent=4)
    tmp.replace(path)
    logger.info(f"[SymbolManager] Config saved: {config_path}")


async def add_symbol(symbol: str, config_path: str, exchange, to_core: bool = False):
    """
    Add a symbol to the config after validation.
    
    Args:
        symbol: Symbol to add.
        config_path: Path to config.json.
        exchange: CCXT exchange instance.
        to_core: If True, add to 'symbols' (core). Otherwise add to 'forager.approved_symbols'.
    """
    # Validate
    await validate_symbol(symbol, exchange)
    
    config = load_config(config_path)
    
    if to_core:
        if symbol not in config.get('symbols', []):
            config.setdefault('symbols', []).append(symbol)
            logger.info(f"[SymbolManager] Added {symbol} to core symbols")
        else:
            logger.info(f"[SymbolManager] {symbol} already in core symbols")
    else:
        approved = config.get('forager', {}).get('approved_symbols', [])
        if symbol not in approved:
            config.setdefault('forager', {}).setdefault('approved_symbols', []).append(symbol)
            logger.info(f"[SymbolManager] Added {symbol} to forager approved_symbols")
        else:
            logger.info(f"[SymbolManager] {symbol} already in forager approved_symbols")
    
    save_config(config_path, config)


async def remove_symbol(symbol: str, config_path: str, exchange, is_futures: bool = False):
    """
    Remove a symbol from the config after safety checks.
    
    Args:
        symbol: Symbol to remove.
        config_path: Path to config.json.
        exchange: CCXT exchange instance.
        is_futures: Whether exchange is in futures mode.
        
    Raises:
        RuntimeError: If removal is blocked by safety checks.
    """
    can_remove, reason = await can_remove_symbol(symbol, exchange, is_futures)
    if not can_remove:
        raise RuntimeError(f"Cannot remove {symbol}: {reason}")
    
    config = load_config(config_path)
    
    # Remove from core symbols
    if symbol in config.get('symbols', []):
        config['symbols'].remove(symbol)
        logger.info(f"[SymbolManager] Removed {symbol} from core symbols")
    
    # Remove from forager approved_symbols
    approved = config.get('forager', {}).get('approved_symbols', [])
    if symbol in approved:
        approved.remove(symbol)
        logger.info(f"[SymbolManager] Removed {symbol} from forager approved_symbols")
    
    save_config(config_path, config)
