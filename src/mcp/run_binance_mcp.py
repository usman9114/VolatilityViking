import os
import sys
import runpy
import logging
from dotenv import load_dotenv

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BinanceMCP")

def main():
    # 1. Load Environment
    # Assuming run from project root, or adjusting path
    env_path = os.path.join(os.getcwd(), 'profiles/usman/.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
    else:
        logger.warning(f"No .env found at {env_path}")

    # 1.5 Map Keys for MCP Server (Expects BINANCE_API_SECRET)
    if not os.getenv('BINANCE_API_SECRET'):
        secret = os.getenv('BINANCE_SECRET_KEY') or os.getenv('BINANCE_SECRET')
        if secret:
            os.environ['BINANCE_API_SECRET'] = secret
            logger.info("Mapped BINANCE_SECRET_KEY/BINANCE_SECRET to BINANCE_API_SECRET")
        else:
            logger.error("Missing BINANCE_API_SECRET (and no alternatives found)")

    # 2. Handover to MCP Server (binance_futures_mcp)
    logger.info("Starting Binance Futures MCP Server...")
    
    try:
        # Try running as module first (standard for binance_futures_mcp)
        runpy.run_module('binance_mcp', run_name='__main__')
    except Exception as e:
        logger.error(f"Failed to run binance_mcp module: {e}")
        # Fallback to server.main if module run fails
        try:
             from binance_mcp.server import main
             main()
        except Exception as e2:
             logger.error(f"Failed fallback import: {e2}")
             sys.exit(1)

if __name__ == "__main__":
    main()
