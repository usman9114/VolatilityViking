
import json
import logging
from src.live.async_runner import AsyncMultiSymbolBot

# Mock args
class Args:
    config = 'profiles/khusha/config.json'
    profile = 'khusha'

bot = AsyncMultiSymbolBot(Args())
print(f"Liquidity Config: {bot.config.get('liquidity_manager')}")
