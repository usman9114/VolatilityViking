import argparse
import sys
import os
import pandas as pd
from dotenv import load_dotenv
def main():
    parser = argparse.ArgumentParser(description="VolatilityViking AI Trader")
    parser.add_argument('--mode', type=str, choices=['live', 'dashboard'], help='Mode: live or dashboard')
    parser.add_argument('--config', type=str, default='config.json', help='Path to strategy config file')
    parser.add_argument('--env-file', type=str, default='.env', help='Path to environment file (default: .env)')
    parser.add_argument('--profile', type=str, help='Run as specific user profile (e.g. usman)')
    
    args = parser.parse_args()

    # Handle Profile Logic (User Request)
    env_path = args.env_file
    config_path = args.config  # Default to --config argument
    
    if args.profile:
        # Check profiles/USER/.env and profiles/USER/config.json
        profile_dir = f"profiles/{args.profile}"
        profile_env = f"{profile_dir}/.env"
        profile_config = f"{profile_dir}/config.json"
        
        if os.path.exists(profile_env):
            env_path = profile_env
        else:
            parser.error(f"Profile '{args.profile}' not found at {profile_env}")
            
        # Use profile config if it exists
        if os.path.exists(profile_config):
            config_path = profile_config
            
        print(f"👤 PROFILE ACTIVE: {args.profile.upper()}")
        print(f"   Env: {env_path}")
        print(f"   Config: {config_path}")
            
    # Load environment variables
    load_dotenv(env_path, override=True)
    
    # Handle dashboard mode
    if hasattr(args, 'mode') and args.mode == 'dashboard':
        print("🚀 Launching Streamlit Dashboard...")
        import subprocess
        subprocess.run(['streamlit', 'run', 'dashboard.py'])
        return
    
    # Require mode for live
    if not hasattr(args, 'mode') or args.mode is None:
        parser.error("--mode is required (live or dashboard)")
    
    if args.mode == 'live':
        print("=" * 60)
        print("🚀 VOLATILITY VIKING - AGENTIC LIVE TRADER")
        print("=" * 60)
        print(f"Config: {config_path}")
        print("Architecture: Multi-Agent System (Scout, Risk, Supervisor, Architect)")
        print("=" * 60)
        
        import asyncio
        from src.live.async_runner import main_async, setup_logging
        from src.agents.meeting_room import start_scheduler
        
        # Setup logging for the profile
        profile_name = args.profile if args.profile else "default"
        setup_logging(profile_name)
        
        # Start the "Active Agency" AND the Bot in the same loop
        # This fixes the orphan task issue
        async def run_system():
            print("🤖 AGENT SYSTEM: Online (Background)")
            scheduler = asyncio.create_task(start_scheduler(gov_interval_hours=4))
            print("🚀 TRADING BOT: Online (Foreground)")
            bot = asyncio.create_task(main_async(config_path))
            
            # Wait for both (bot is main, scheduler is daemon-like)
            await asyncio.gather(scheduler, bot)

        try:
            asyncio.run(run_system())
        except KeyboardInterrupt:
            print("\nShutting down system...")

if __name__ == "__main__":
    main()
