import os
import json
import logging
import hashlib
from src.agents.base_agent import BaseAgent, logger
from src.ai.sentiment_engine import SentimentEngine
from src.ai.utils import call_openai_with_retry
from src.ai.model_manager import ModelManager

class SentinelAgent(BaseAgent):
    """
    The Sentinel: High-Frequency News Monitor.
    Reflexively triggers critical halts (Red Phone) if Black Swan events are detected.
    
    OPTIMIZATION: Only calls OpenAI when headlines actually change.
    """
    def __init__(self):
        super().__init__()
        self.sentiment = SentimentEngine(self.config)
        self.model_manager = ModelManager()
        self.output_dir = "signals"
        os.makedirs(self.output_dir, exist_ok=True)
        self.lock_file = os.path.join(self.output_dir, "kill_switch.lock")
        self.last_headlines_hash = None  # Track headline changes

    def hash_headlines(self, headlines):
        """Generate hash of headline set to detect changes."""
        sorted_headlines = sorted(headlines)
        content = '|'.join(sorted_headlines)
        return hashlib.md5(content.encode()).hexdigest()

    async def run(self):
        logger.info(f"[{self.name}] Scanning for Black Swans...")
        
        # 1. Get Symbols
        symbols = self.config.get('symbols', [])
        if not symbols: return
        
        # 2. Batch Fetch News (ONE network pass)
        news_map = await self.sentiment.fetch_batch_news(symbols)
        
        # 3. Construct Aggregate Context
        # Flatten distinct headlines to avoid duplicates in prompt
        all_headlines = set()
        for sym, headlines in news_map.items():
            for h in headlines:
                all_headlines.add(h)
        
        if not all_headlines:
            logger.info(f"[{self.name}] No news found.")
            return

        # 4. INTELLIGENT TRIGGER: Check if headlines changed
        current_hash = self.hash_headlines(all_headlines)
        if current_hash == self.last_headlines_hash:
            logger.info(f"[{self.name}] Headlines unchanged, skipping OpenAI call (API optimization).")
            return
        
        self.last_headlines_hash = current_hash  # Update hash
        
        # 5. PRE-FILTER: Only call OpenAI if BLACK SWAN keywords detected
        # This saves 95%+ of API calls by filtering noise immediately
        black_swan_keywords = [
            'hack', 'hacked', 'breach', 'stolen', 'security',
            'sec', 'lawsuit', 'ban', 'banned', 'illegal',
            'depeg', 'depegged', 'usdt', 'usdc', 'stablecoin',
            'halt', 'halted', 'suspended', 'freeze',
            'insolvency', 'bankrupt', 'collapse', 'ftx',
            'war', 'attack', 'sanctions', 'emergency'
        ]
        
        # Check if ANY headline contains critical keywords
        headlines_text = ' '.join(all_headlines).lower()
        has_critical_keyword = any(kw in headlines_text for kw in black_swan_keywords)
        
        if not has_critical_keyword:
            logger.info(f"[{self.name}] No critical keywords detected. Skipping OpenAI (API optimization).")
            return
        
        logger.warning(f"[{self.name}] ⚠️  BLACK SWAN KEYWORDS DETECTED! Calling OpenAI for verification...")

        headlines_block = "\n".join(f"- {h}" for h in list(all_headlines)[:50]) # Limit to top 50 to fit context
        
        # 6. Consult AI (ONE API Call - only when necessary)
        prompts = self.load_prompts()
        cfg = prompts.get('sentinel', {})
        system_msg = cfg.get('system', "You are The Sentinel. Detect Threats.")
        user_template = cfg.get('user_template', "Analyze:\n{headlines_block}")
        
        user_msg = user_template.format(
            headlines_block=headlines_block,
            symbols=", ".join(symbols)
        )
        
        try:
            # Use 'scout' model preference (Speed)
            model_id = self.model_manager.get_model_for_role('scout')
            
            response = await call_openai_with_retry(
                self.sentiment.client, # Reuse client
                model=model_id,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ]
            )
            
            if not response: return

            data = json.loads(response.replace('```json', '').replace('```', '').strip())
            
            threat = data.get('threat_detected', False)
            reason = data.get('reason', 'No threat')
            
            if threat:
                logger.critical(f"🚨 SENTINEL DETECTED THREAT: {reason}")
                logger.critical(f"🚨 ENGAGING RED PHONE (Kill Switch) 🚨")
                self.engage_red_phone(reason)
            else:
                logger.info(f"[{self.name}] Sector Clear. ({reason})")
                
        except Exception as e:
            logger.error(f"Sentinel Error: {e}")

    def engage_red_phone(self, reason):
        """Create the lock file to halt trading."""
        with open(self.lock_file, 'w') as f:
            f.write(f"KILLED BY SENTINEL at {self.timestamp_str()}\nReason: {reason}")

    def timestamp_str(self):
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def load_prompts(self):
        try:
            with open('prompts.json', 'r') as f:
                return json.load(f)
        except:
            return {}

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    load_dotenv('profiles/usman/.env')
    sentinel = SentinelAgent()
    asyncio.run(sentinel.run())
