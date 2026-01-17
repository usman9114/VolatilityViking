import asyncio
import time
import argparse
from datetime import datetime
from src.agents.governor import GovernorAgent, logger
from src.agents.sentinel import SentinelAgent
from src.agents.banker import BankerAgent

async def start_scheduler(gov_interval_hours=1, sentinel_interval_minutes=15):
    """
    Runs the Agency Scheduler.
    - Sentinel: Every 15 minutes (Optimized for API efficiency)
    - Banker: Runs with Sentinel (Liquidity Check)
    - Governor: Every 1 hours (Deep Strategy)
    """
    logger.info(f"🤖 AGENT SCHEDULER STARTED.")
    logger.info(f"   - Sentinel Interval: {sentinel_interval_minutes}m")
    logger.info(f"   - Governor Interval: {gov_interval_hours}h")
    
    governor = GovernorAgent()
    sentinel = SentinelAgent()
    banker = BankerAgent()
    
    last_gov_run = 0
    last_sentinel_run = 0
    
    while True:
        now = time.time()
        
        # --- SENTINEL & BANKER CHECK (High Freq) ---
        if now - last_sentinel_run > (sentinel_interval_minutes * 60):
            try:
                await asyncio.gather(
                    sentinel.run(),
                    banker.run()
                )
                last_sentinel_run = now
            except Exception as e:
                logger.error(f"Sentinel/Banker Error: {e}")
                
        # --- GOVERNOR CHECK (Low Freq) ---
        if now - last_gov_run > (gov_interval_hours * 3600):
            try:
                logger.info("🔔 MEETING CALL! Governor is entering the room...")
                await governor.run()
                logger.info("✅ Meeting Adjourned.")
                last_gov_run = now
            except Exception as e:
                logger.error(f"Governor Error: {e}")
        
        await asyncio.sleep(60)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv('profiles/usman/.env')

    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true', help='Run Governor once and exit')
    parser.add_argument('--sentinel-once', action='store_true', help='Run Sentinel once and exit')
    args = parser.parse_args()

    if args.once:
        agent = GovernorAgent()
        asyncio.run(agent.run())
    elif args.sentinel_once:
        agent = SentinelAgent()
        asyncio.run(agent.run())
    else:
        asyncio.run(start_scheduler())

