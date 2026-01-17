import json
import os
from openai import OpenAI
from src.agents.base_agent import BaseAgent, logger
from src.agents.scout import ScoutAgent
from src.agents.risk_agent import RiskAgent
from src.agents.architect import ArchitectAgent
from src.ai.model_manager import ModelManager
from src.ai.utils import call_openai_with_retry
from src.live.symbol_manager import validate_symbol, can_remove_symbol, add_symbol, remove_symbol, load_config, save_config

class GovernorAgent(BaseAgent):
    """
    The Governor (CEO).
    Replaces Supervisor. Uses best available reasoning model to control FULL Config.
    """
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv('OPENAI_API_KEY')
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            logger.warning("No OpenAI API Key found for Governor.")
            self.client = None
            
        self.scout = ScoutAgent()
        self.risk = RiskAgent()
        self.architect = ArchitectAgent()
        self.model_manager = ModelManager()

    def load_prompts(self):
        try:
            with open('prompts.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load prompts.json: {e}")
            return {}

    def deep_merge(self, source, destination):
        """
        Deep merge dictionary `source` into `destination`.
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
                self.deep_merge(value, destination[key])
            else:
                destination[key] = value
        return destination

    async def _auto_manage_symbols(self, scout_report: dict):
        """
        Auto-add/remove symbols based on Scout recommendations.
        Only runs if governor.auto_manage_symbols is enabled in config.
        """
        if not self.config.get('governor', {}).get('auto_manage_symbols', True):
            return
        
        # Get recommended symbols from scout report
        recommended = set()
        if isinstance(scout_report, dict):
            for entry in scout_report.get('top_coins', []):
                sym = entry.get('symbol')
                if sym:
                    recommended.add(sym)
        
        if not recommended:
            logger.info(f"[{self.name}] No symbol recommendations from Scout.")
            return
        
        # Get current approved symbols
        current_approved = set(self.config.get('forager', {}).get('approved_symbols', []))
        
        # Symbols to add (new recommendations not in approved)
        to_add = recommended - current_approved
        # Symbols to potentially remove (in approved but not recommended)
        to_check_remove = current_approved - recommended
        
        config_path = 'config.json'
        
        # Try to get exchange reference (we need it for validation)
        exchange = getattr(self, '_exchange', None)
        if not exchange:
            logger.warning(f"[{self.name}] No exchange reference for symbol management.")
            return
        
        is_futures = self.config.get('futures_settings', {}).get('enabled', False)
        
        added = []
        removed = []
        
        # Add new symbols
        for sym in to_add:
            try:
                await add_symbol(sym, config_path, exchange, to_core=False)
                added.append(sym)
            except Exception as e:
                logger.warning(f"[{self.name}] Could not add {sym}: {e}")
        
        # Remove deprecated symbols (only if zero equity)
        for sym in to_check_remove:
            try:
                await remove_symbol(sym, config_path, exchange, is_futures=is_futures)
                removed.append(sym)
            except RuntimeError as e:
                # Safety check failed (has equity or orders)
                logger.info(f"[{self.name}] Keeping {sym}: {e}")
            except Exception as e:
                logger.warning(f"[{self.name}] Could not remove {sym}: {e}")
        
        if added:
            logger.info(f"[{self.name}] Auto-added symbols: {added}")
        if removed:
            logger.info(f"[{self.name}] Auto-removed symbols: {removed}")
        
        # Reload config
        self.config = load_config(config_path)

    async def run(self):
        logger.info(f"[{self.name}] Governor is in session...")
        
        # 1. Select Best Brain
        model_id = self.model_manager.get_model_for_role('governor')
        logger.info(f"[{self.name}] Using Model: {model_id}")
        
        # 2. Gather Intelligence
        scout_report = await self.scout.run()
        risk_report = await self.risk.run()
        architect_report = await self.architect.run()
        
        # 3. Construct Prompt
        prompts = self.load_prompts()
        cfg = prompts.get('governor', {})
        system_msg = cfg.get('system', "You are the Governor.")
        
        user_template = cfg.get('user_template', "")
        
        current_config_dump = json.dumps(self.config, indent=2)
        
        try:
            user_msg = user_template.format(
                scout_report=json.dumps(scout_report),
                risk_report=json.dumps(risk_report),
                architect_report=json.dumps(architect_report),
                oracle_report="{}",  # Deprecated: Oracle removed
                current_config=current_config_dump
            )
        except Exception as e:
            logger.error(f"Prompt formatting error: {e}")
            return

        if not self.client:
            logger.error("Cannot govern without OpenAI.")
            return

        # 4. Consult LLM
        try:
            response = await call_openai_with_retry(
                self.client,
                model=model_id,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ]
            )
            
            if not response: return

            # Clean JSON
            txt = response.replace('```json', '').replace('```', '').strip()
            decision = json.loads(txt)
            
            rationale = decision.get('rationale', 'No rationale provided.')
            updates = decision.get('config_updates', {})
            
            logger.info(f"[{self.name}] Decision Rationale: {rationale}")
            
            if updates:
                logger.info(f"[{self.name}] Applying Config Updates: {json.dumps(updates, indent=2)}")
                
                # Apply updates via deep merge
                self.config = self.deep_merge(updates, self.config)
                self.save_config(self.config)
                logger.info(f"[{self.name}] Configuration Saved.")
            else:
                logger.info(f"[{self.name}] No config changes required.")
            
            # 5. Auto-manage symbols based on Scout recommendations
            await self._auto_manage_symbols(scout_report)
                
            return decision

        except Exception as e:
            logger.error(f"Governor Exception: {e}")
            if "not a chat model" in str(e) or "404" in str(e):
                 logger.warning(f"Model {model_id} failed. Falling back to gpt-4o...")
                 try:
                     response = await call_openai_with_retry(
                        self.client,
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg}
                        ]
                     )
                     if response:
                         txt = response.replace('```json', '').replace('```', '').strip()
                         decision = json.loads(txt)
                         logger.info(f"[{self.name}] Fallback Success. Rationale: {decision.get('rationale')}")
                         return decision
                 except Exception as e2:
                     logger.error(f"Fallback failed: {e2}")
            return None

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    load_dotenv('profiles/usman/.env')
    
    gov = GovernorAgent()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(gov.run())
