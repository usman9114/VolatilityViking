"""
CLI Script to manually trigger state sync from exchange history.
Usage: python -m src.cli.sync_state
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.live.state_manager import StateManager
from src.cli.manage_symbols import get_exchange

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("SyncState")

async def main():
    logger.info("Starting manual state sync (Forcing SPOT mode)...")
    
    # 1. Load Config & Env
    import json
    import os
    import ccxt.async_support as ccxt
    from dotenv import load_dotenv
    
    # Try loading from profiles/usman/.env first (user's profile), fallback to local .env
    if os.path.exists("profiles/usman/.env"):
        load_dotenv("profiles/usman/.env")
    else:
        load_dotenv()
    
    config_path = Path(__file__).resolve().parents[2] / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 2. Connect SPOT Exchange Direct
    exchange = ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET'),
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    try:
        await exchange.load_markets()
    except Exception as e:
        logger.error(f"Failed to connect to Spot API: {e}")
        await exchange.close()
        return

    core_syms = config.get('symbols', [])
    forager_syms = config.get('forager', {}).get('approved_symbols', [])
    all_syms = list(set(core_syms + forager_syms))
    
    logger.info(f"Syncing history for {len(all_syms)} symbols: {all_syms}")
    
    # 3. Sync
    sm = StateManager()
    await sm.sync_from_history(exchange, all_syms)
    
    # 4. Cleanup
    await exchange.close()
    logger.info("Sync complete. State saved to data/bot_state.json")

if __name__ == "__main__":
    asyncio.run(main())
