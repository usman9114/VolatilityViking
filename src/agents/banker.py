import os
import json
import ccxt
from src.agents.base_agent import BaseAgent, logger

class BankerAgent(BaseAgent):
    """
    The Banker: Liquidity & Margin Manager.
    - Monitors Futures Margin Level.
    - Auto-Topup from Spot if Risk is High.
    - Sweeps excess profits to Spot (optional).
    """
    def __init__(self):
        super().__init__()
        self.banker_config = self.config.get('banker', {})
        self.enabled = self.banker_config.get('enabled', False)
        
        # Connection (Private)
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret = os.getenv('BINANCE_SECRET_KEY') or os.getenv('BINANCE_SECRET')
        
        if self.api_key and self.secret:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'} 
            })
        else:
            self.exchange = None
            logger.warning("[Banker] No API Keys found. Banker disabled.")

    async def check_margin_health(self):
        """
        Check Margin Ratio (Maintenance Margin / Margin Balance).
        Returns risk_level (0.0 - 1.0).
        """
        try:
            balance = self.exchange.fetch_balance()
            info = balance.get('info', {})
            
            # Binance Futures 'info' structure verification needed usually, 
            # but fetch_balance parses 'total' and 'free' well.
            # We need Maintenance Margin vs Total Margin Balance.
            
            total_margin_balance = float(info.get('totalMarginBalance', 0))
            total_maint_margin = float(info.get('totalMaintMargin', 0))
            
            if total_margin_balance <= 0:
                return 0.0 # No funds, no risk? Or infinite risk?
                
            # Risk Ratio: Higher is bad. 1.0 = Liquidation.
            risk_ratio = total_maint_margin / total_margin_balance
            
            logger.info(f"[{self.name}] Margin Health: Balance=${total_margin_balance:.2f} | Maint=${total_maint_margin:.2f} | Risk={risk_ratio:.2%}")
            return risk_ratio, total_margin_balance
            
        except Exception as e:
            logger.error(f"[{self.name}] Failed to check margin: {e}")
            return 0.0, 0.0

    async def get_spot_balance(self):
        try:
            # We need a spot connection specifically or use method if supported
            # CCXT usually separates them. Safest to make a temp spot instance or change options?
            # Easiest: dedicated spot instance or unify.
            # Let's use a quick lightweight spot fetch
            spot_ex = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.secret,
                'enableRateLimit': True
            })
            try:
                bal = await spot_ex.fetch_balance()
                free_usdt = bal.get('USDT', {}).get('free', 0)
                return free_usdt
            finally:
                await spot_ex.close()
        except Exception as e:
            logger.error(f"[{self.name}] Spot balance check failed: {e}")
            return 0.0

    async def execute_transfer(self, amount, direction='spot_to_future'):
        try:
            if direction == 'spot_to_future':
                await self.exchange.transfer('USDT', amount, 'spot', 'future')
                logger.info(f"[{self.name}] 🚨 EMERGENCY TOPUP: Moved {amount} USDT to Futures.")
            elif direction == 'future_to_spot':
                await self.exchange.transfer('USDT', amount, 'future', 'spot')
                logger.info(f"[{self.name}] 💰 PROFIT SWEEP: Moved {amount} USDT to Spot.")
            return True
        except Exception as e:
            logger.error(f"[{self.name}] Transfer failed: {e}")
            return False

    async def run(self):
        if not self.enabled or not self.exchange:
            return

        logger.info(f"[{self.name}] Auditing Liquidity...")
        
        # Config
        safety_thresh = self.banker_config.get('margin_safety_threshold', 0.60) # Topup if risk > 60%
        topup_amount = self.banker_config.get('topup_amount', 50.0)
        min_spot = self.banker_config.get('min_spot_balance', 10.0)
        
        risk, margin_balance = await self.check_margin_health()
        
        # 1. EMERGENCY TOPUP
        if risk > safety_thresh:
            logger.warning(f"[{self.name}] RISK ALERT: Margin Risk {risk:.2%} > {safety_thresh:.2%}")
            
            spot_free = await self.get_spot_balance()
            available_to_send = spot_free - min_spot
            
            if available_to_send >= topup_amount:
                logger.info(f"[{self.name}] Initiating Topup of {topup_amount} USDT...")
                await self.execute_transfer(topup_amount, 'spot_to_future')
            elif available_to_send > 0:
                # Send whatever is left
                logger.info(f"[{self.name}] Initiating Topup of {available_to_send} USDT (Partial)...")
                await self.execute_transfer(available_to_send, 'spot_to_future')
            else:
                logger.critical(f"[{self.name}] ☠️ CRITICAL: Cannot Topup! Spot is empty.")
                
        # 2. PROFIT SWEEP (Optional)
        # Logic: If margin risk is super low (< 5%) AND balance > target, sweep?
        # Let's keep it simple for now: Only Topup. "Profit Taking" is complex logic.
        
        logger.info(f"[{self.name}] Audit Complete.")

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    load_dotenv('profiles/usman/.env')
    agent = BankerAgent()
    asyncio.run(agent.run())
