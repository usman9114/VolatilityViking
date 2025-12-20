import argparse
import sys
import os
import pandas as pd
from dotenv import load_dotenv
import mlflow
import datetime

load_dotenv()

def run_pipeline(config):
    symbol = config.get('symbol', 'ETH/USDT')
    timeframe = config.get('timeframe', '4h')
    
    print(f"--- Starting Pipeline for {symbol} ({timeframe}) [Meta Model] ---")
    
    # MLflow Parent Run
    run_name = f"MetaModel_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.log_params(config) # Log CLI args
        
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
    parser.add_argument('--dashboard', action='store_true', help='Launch Streamlit dashboard (alternative to --mode dashboard)')
    
    args = parser.parse_args()
    
    # Handle dashboard mode
    if args.dashboard or (hasattr(args, 'mode') and args.mode == 'dashboard'):
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
        print(f"Starting Live Trader for {args.symbol} {args.timeframe}...")
        # from src.live.trader import start_trader # DEPRECATED: Uses old trader
        # start_trader(args.symbol, args.timeframe)
        
        # USE AUDITED BINANCE BOT
        from src.live.binance_bot import BinanceBot
        # Default amounts or add args to main.py? For now use defaults or hardcode safe test amount
        aggressive = getattr(args, 'aggressive', False)  # Get aggressive flag if exists
        bot = BinanceBot(symbol=args.symbol, capital_limit=1000, live_mode=True, aggressive_mode=aggressive) 
        # Note: 'live_mode=True' means MAINNET in binance_bot.py. 
        # We might want to pass this as an arg.
        bot.run()

if __name__ == "__main__":
    main()
