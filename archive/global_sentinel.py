
import logging
import json
import os
import time
import feedparser
from openai import OpenAI
from src.ai.utils import call_openai_with_retry
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class GlobalSentinel:
    """
    Layer 7: Autonomous AI Manager.
    - Macro Strategist: Sets DEFCON level based on global risks.
    - Narrative Hunter: Updates Forager list with trending coins.
    """
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.prompts = self.load_prompts()
        self.client = None
        
        load_dotenv('.env_riz')
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            try:
                self.client = OpenAI(api_key=api_key)
            except Exception as e:
                logger.error(f"GlobalSentinel: OpenAI Init Failed: {e}")
        
    def load_prompts(self):
        try:
            with open('prompts.json', 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def fetch_global_headlines(self):
        """Fetch broad market news (mocked RSS for now or use real generic feed)"""
        # For real macro, we'd want Yahoo Finance or CoinDesk Top Stories
        rss_url = "https://cointelegraph.com/rss"
        try:
            feed = feedparser.parse(rss_url)
            headlines = [entry.title for entry in feed.entries[:10]]
            return "\n".join(headlines)
        except Exception as e:
            logger.warning(f"Failed to fetch global news: {e}")
            return "Market is uncertain. Bitcoin checking key levels. Inflation data pending."

    def _validate_binance_symbols(self, symbols):
        """Check if symbols exist on Binance Futures. Returns valid symbols only."""
        import ccxt
        valid = []
        try:
            exchange = ccxt.binance({'enableRateLimit': True})
            exchange.load_markets()
            
            for symbol in symbols:
                if symbol in exchange.markets and exchange.markets[symbol].get('future'):
                    valid.append(symbol)
        except Exception as e:
            logger.error(f"Symbol validation failed: {e}")
            return symbols  # Fail-open
        
        return valid

    async def run_macro_strategist(self):
        """
        Consults the 'Macro Strategist' AI.
        Returns: Dict with defcon_level, suggested_leverage, etc.
        """
        if not self.client: return None
        
        prompt_cfg = self.prompts.get('macro_strategist', {})
        if not prompt_cfg: return None
        
        headlines = self.fetch_global_headlines()
        
        user_content = prompt_cfg.get('user_template', '').format(headlines=headlines)
        system_content = prompt_cfg.get('system', '')
        
        try:
            content = await call_openai_with_retry(
                self.client,
                model="gpt-5.1",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
            )
            if not content: return None
            
            # Clean JSON
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            logger.error(f"Macro Strategist failed: {e}")
            return None

    async def run_narrative_hunter(self):
        """
        Consults 'Narrative Hunter' AI.
        Returns: List of new coins to forage.
        """
        if not self.client: return None
        
        prompt_cfg = self.prompts.get('narrative_hunter', {})
        if not prompt_cfg: return None
        
        # Same headlines for now, or specific 'Trending' RSS
        headlines = self.fetch_global_headlines()
        
        user_content = prompt_cfg.get('user_template', '').format(headlines=headlines)
        system_content = prompt_cfg.get('system', '')
        
        try:
            content = await call_openai_with_retry(
                self.client,
                model="gpt-5.1",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
            )
            if not content: return None

            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            logger.error(f"Narrative Hunter failed: {e}")
            return None

    async def run_risk_officer(self, market_metrics):
        """
        Consults 'Risk Officer' AI for Leverage/Size.
        market_metrics: {
            'volatility': 'High (ATR=5%)',
            'trend_strength': 45,
            'ml_confidence': 88
        }
        """
        if not self.client: return None
        
        prompt_cfg = self.prompts.get('risk_officer', {})
        if not prompt_cfg: return None
        
        # We need the "Macro" context too?
        # Ideally Risk Officer sees Macro Sentiment too.
        # Let's run Macro Strategist FIRST in apply_ai_decisions and pass it here?
        # For now, just pass what we have.
        
        user_content = json.dumps(market_metrics, indent=2)
        system_content = prompt_cfg.get('role', '') + "\n" + "\n".join(prompt_cfg.get('rules', []))
        
        try:
            content = await call_openai_with_retry(
                self.client,
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
            )
            if not content: return None

            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            logger.error(f"Risk Officer failed: {e}")
            return None

    async def apply_ai_decisions(self, market_context=None):
        """
        The 'Board Meeting'. Runs AI managers and updates config.json.
        market_context: Optional dict with specific symbol metrics (e.g., ETH data).
        """
        logger.info("🤖 AI MANAGERS: Holding Strategy Meeting...")
        
        macro_decision = await self.run_macro_strategist()
        narrative_decision = await self.run_narrative_hunter()
        
        # Risk Officer needs context. If none provided, skip or use defaults.
        risk_decision = None
        if market_context:
            # We assume Macro Sentiment is part of context or we add it?
            if macro_decision:
                market_context['sentiment'] = f"DEFCON {macro_decision.get('defcon_level')} - {macro_decision.get('reason','')[:50]}..."
            
            risk_decision = await self.run_risk_officer(market_context)
        
        if not any([macro_decision, narrative_decision, risk_decision]):
            logger.info("AI Managers provided no output.")
            return 

        # Load Current Config
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        except:
            logger.error("Could not read config.json")
            return

        updated = False
        
        # 1. Apply Macro Strategy (sets Baseline)
        if macro_decision:
            defcon = macro_decision.get('defcon_level', 3)
            # We defer Leverage to Risk Officer, but keep Aggression logic
            reason = macro_decision.get('reason', '')
            logger.info(f"🌍 MACRO STRATEGIST: DEFCON {defcon} ({reason})")
            
            # Apply Aggression
            if defcon == 5:
                if not config.get('aggressive_mode'):
                    config['aggressive_mode'] = True
                    logger.info("   -> Enabled Aggressive Mode (Bull Run)")
                    updated = True

            
            # Liquidity Stance (Return for Async Runner)
            # Just log it here, return it to the caller
            liquidity_stance = macro_decision.get('liquidity_stance', 'BALANCED')
            logger.info(f"   -> Liquidity Stance: {liquidity_stance}")
                    
        # 2. Apply Risk Officer (Overrides Leverage)
        if risk_decision:
            leverage = risk_decision.get('leverage')
            reason = risk_decision.get('reason')
            if leverage:
                logger.info(f"👮 RISK OFFICER: Set Global Leverage to {leverage}x ({reason})")
                if 'futures_settings' in config:
                     if config['futures_settings']['leverage'] != leverage:
                        config['futures_settings']['leverage'] = leverage
                        updated = True
                        
        # 3. Apply Narrative Hunitng
        if narrative_decision:
            hot_coins = narrative_decision.get('suggested_coins', [])
            sectors = narrative_decision.get('top_sectors', [])
            
            if hot_coins:
                logger.info(f"🦅 NARRATIVE HUNTER: Hot Sectors {sectors} -> Found {hot_coins}")
                
                # Update Forager List
                if 'forager' in config:
                    current_list = set(config['forager'].get('approved_symbols', []))
                    
                    # Validate symbols exist on Binance
                    validated_coins = self._validate_binance_symbols(hot_coins)
                    if len(validated_coins) < len(hot_coins):
                        invalid = set(hot_coins) - set(validated_coins)
                        logger.warning(f"   ⚠️ Skipped invalid symbols: {invalid}")
                    
                    new_set = set(validated_coins)
                    merged = list(current_list.union(new_set))
                    
                    if len(merged) != len(current_list):
                        config['forager']['approved_symbols'] = merged
                        logger.info(f"   -> Added {new_set - current_list} to Forager")
                        updated = True

        # Save Config
        if updated:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info("✅ Config updated by AI Managers.")
        else:
            logger.info("AI Managers proposed no changes.")
            
        # Return decisions for Runner to act on (e.g. Liquidity Transfer)
        return {
            "liquidity_stance": macro_decision.get('liquidity_stance', 'BALANCED') if macro_decision else 'BALANCED'
        }
