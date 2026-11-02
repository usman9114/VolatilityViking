import argparse
import sys
import os
import pandas as pd
from dotenv import load_dotenv
import datetime

# load_dotenv() - Moved to main()

def main():
    parser = argparse.ArgumentParser(description="ETH-AI Meta Trader")
    parser.add_argument('--mode', type=str, choices=['live', 'dashboard'], help='Mode: live or dashboard')
    parser.add_argument('--symbol', type=str, default='ETH/USDT', help='Trading Pair')
    parser.add_argument('--timeframe', type=str, default='4h', help='Timeframe')
    parser.add_argument('--aggressive', action='store_true', help='Enable aggressive mode (5s checks, proactive grid). Default is conservative (60s checks, reactive grid).')
    parser.add_argument('--config', type=str, default='config.json', help='Path to strategy config file (required for live mode)')
    parser.add_argument('--async-mode', action='store_true', dest='async_mode', help='Use async runner (Passivbot style - faster)')
    parser.add_argument('--env-file', type=str, default='.env', help='Path to environment file (default: .env)')
    
    args = parser.parse_args()

    # Load environment variables
    load_dotenv(args.env_file, override=True)
    
    # Handle dashboard mode
    if hasattr(args, 'mode') and args.mode == 'dashboard':
        print("🚀 Launching Streamlit Dashboard...")
        import subprocess
        subprocess.run(['streamlit', 'run', 'dashboard.py'])
        return
    
    # Require mode for train/live
    if not hasattr(args, 'mode') or args.mode is None:
        parser.error("--mode is required (live or dashboard)")
    
    if args.mode == 'live':
        print("=" * 60)
        print("🚀 VOLATILITY VIKING - ASYNC LIVE TRADER")
        print("=" * 60)
        print(f"Config: {args.config}")
        print("Architecture: Passivbot-style async (all symbols parallel)")
        print("=" * 60)
        
        import asyncio
        from src.live.async_runner import main_async
        
        asyncio.run(main_async(args.config))

if __name__ == "__main__":
    main()
