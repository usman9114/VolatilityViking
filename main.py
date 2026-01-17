import argparse
import sys
import os
import pandas as pd
from dotenv import load_dotenv
def run_pipeline(config):
    symbol = config.get('symbol', 'ETH/USDT')
    timeframe = config.get('timeframe', '4h')
    
    print(f"--- Starting Pipeline for {symbol} ({timeframe}) [Meta Model] ---")
    
    # Use pre-processed data
    feat_path = "data/processed/train_dataset.parquet"
    
    # Load Price Data for Backtester
    df_price = pd.read_parquet(feat_path)
    
    # Train Model
    print("4. Training Meta Model (Price + News)...")
    from src.models.metamodel import MetamodelTrainer
    from src.backtest.backtester import Backtester
    
    print(f"   Using prepared data from {feat_path}")
    
    # Optimized Hyperparams are hardcoded in MetamodelTrainer, but we pass seq_len=96
    trainer = MetamodelTrainer(
        data_path=feat_path, 
        gas_path=None, 
        news_emb_path=None,
        seq_len=96 
    )
    
    results = trainer.walk_forward_train(start_year=2018) 
    
    # Report and Backtest using Smart Logic
    print("5. Generating Report...")
    bt = Backtester(results, df_price)
    metrics = bt.calculate_metrics()
    
    print("\n=== Training Complete ===")
    print(f"Symbol: {symbol}")
    print(f"Final Equity (from $10k): ${metrics['strategy_equity'].iloc[-1]:.2f}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="ETH-AI Meta Trader")
    parser.add_argument('--mode', type=str, choices=['train', 'live', 'dashboard'], help='Mode: train, live, or dashboard')
    parser.add_argument('--symbol', type=str, default='ETH/USDT', help='Trading Pair')
    parser.add_argument('--timeframe', type=str, default='4h', help='Timeframe')
    parser.add_argument('--aggressive', action='store_true', help='Enable aggressive mode (5s checks, proactive grid). Default is conservative (60s checks, reactive grid).')
    parser.add_argument('--config', type=str, default='config.json', help='Path to strategy config file (required for live mode)')
    parser.add_argument('--async-mode', action='store_true', dest='async_mode', help='Use async runner (Passivbot style - faster)')
    parser.add_argument('--env-file', type=str, default='.env', help='Path to environment file (default: .env)')
    parser.add_argument('--profile', type=str, help='Run as specific user profile (e.g. usman, rizwan)')
    
    args = parser.parse_args()

    # Handle Profile Logic (User Request)
    env_path = args.env_file
    if args.profile:
        # Check profiles/USER/.env
        profile_env = f"profiles/{args.profile}/.env"
        if os.path.exists(profile_env):
            env_path = profile_env
            print(f"👤 PROFILE ACTIVE: {args.profile.upper()} (Loaded {env_path})")
        else:
            parser.error(f"Profile '{args.profile}' not found at {profile_env}")
            
    # Load environment variables
    load_dotenv(env_path, override=True)
    
    # Handle dashboard mode
    if hasattr(args, 'mode') and args.mode == 'dashboard':
        print("🚀 Launching Streamlit Dashboard...")
        import subprocess
        subprocess.run(['streamlit', 'run', 'dashboard.py'])
        return
    
    # Require mode for train/live
    if not hasattr(args, 'mode') or args.mode is None:
        parser.error("--mode is required (train or live) unless using --dashboard")
    
    if args.mode == 'train':
        config = {
            'symbol': args.symbol,
            'timeframe': args.timeframe
        }
        run_pipeline(config)
    elif args.mode == 'live':
        print("=" * 60)
        print("🚀 VOLATILITY VIKING - ASYNC LIVE TRADER")
        print("=" * 60)
        print(f"Config: {args.config}")
        print("Architecture: Passivbot-style async (all symbols parallel)")
        print("=" * 60)
        
        import asyncio
        from src.live.async_runner import main_async
        
        profile_name = args.profile if args.profile else "default"
        asyncio.run(main_async(args.config, profile_name))

if __name__ == "__main__":
    main()
