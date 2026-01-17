"""
CLI for managing trading symbols.

Usage:
    python -m src.cli.manage_symbols add ETH/USDT [--core]
    python -m src.cli.manage_symbols remove DASH/USDT
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.live.symbol_manager import add_symbol, remove_symbol, validate_symbol

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "config.json"


async def get_exchange():
    """Create exchange instance for validation."""
    import json
    import ccxt.async_support as ccxt
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    config_path = Path(__file__).resolve().parents[2] / DEFAULT_CONFIG
    with open(config_path) as f:
        config = json.load(f)
    
    is_futures = config.get('futures_settings', {}).get('enabled', False)
    
    exchange_class = ccxt.binance
    exchange = exchange_class({
        'apiKey': os.getenv('BINANCE_API_KEY', ''),
        'secret': os.getenv('BINANCE_SECRET', ''),
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future' if is_futures else 'spot'
        }
    })
    
    return exchange, is_futures, str(config_path)


async def cmd_add(args):
    """Add a symbol."""
    exchange, is_futures, config_path = await get_exchange()
    try:
        await add_symbol(args.symbol, config_path, exchange, to_core=args.core)
        print(f"✅ Successfully added {args.symbol}")
        return 0
    except Exception as e:
        print(f"❌ Failed to add {args.symbol}: {e}")
        return 1
    finally:
        await exchange.close()


async def cmd_remove(args):
    """Remove a symbol."""
    exchange, is_futures, config_path = await get_exchange()
    try:
        await remove_symbol(args.symbol, config_path, exchange, is_futures=is_futures)
        print(f"✅ Successfully removed {args.symbol}")
        return 0
    except Exception as e:
        print(f"❌ Failed to remove {args.symbol}: {e}")
        return 1
    finally:
        await exchange.close()


async def cmd_validate(args):
    """Validate a symbol exists on exchange."""
    exchange, _, _ = await get_exchange()
    try:
        await validate_symbol(args.symbol, exchange)
        print(f"✅ {args.symbol} exists on exchange")
        return 0
    except ValueError as e:
        print(f"❌ {e}")
        return 1
    finally:
        await exchange.close()


def main():
    parser = argparse.ArgumentParser(description="Manage trading symbols")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Add command
    add_parser = subparsers.add_parser('add', help='Add a symbol to config')
    add_parser.add_argument('symbol', help='Symbol in CCXT format, e.g. ETH/USDT')
    add_parser.add_argument('--core', action='store_true', help='Add to core symbols instead of forager')
    add_parser.set_defaults(func=cmd_add)
    
    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove a symbol from config')
    remove_parser.add_argument('symbol', help='Symbol in CCXT format')
    remove_parser.set_defaults(func=cmd_remove)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Check if symbol exists on exchange')
    validate_parser.add_argument('symbol', help='Symbol in CCXT format')
    validate_parser.set_defaults(func=cmd_validate)
    
    args = parser.parse_args()
    exit_code = asyncio.run(args.func(args))
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
