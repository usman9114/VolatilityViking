import logging
import json
import os
from abc import ABC, abstractmethod

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Abstract Base Class for all Agents.
    Enforces a standard structure for initialization and collecting reports.
    """
    def __init__(self, config_path='profiles/usman/config.json'):
        self.config_path = config_path
        self.config = self.load_config()
        self.name = self.__class__.__name__
        logger.info(f"[{self.name}] Initialized.")

    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Failed to load config: {e}")
            return {}

    def save_config(self, new_config):
        """Helper to save config updates."""
        try:
            # Backup first? Maybe later.
            with open(self.config_path, 'w') as f:
                json.dump(new_config, f, indent=4)
            self.config = new_config
            logger.info(f"[{self.name}] Config saved successfully.")
            return True
        except Exception as e:
            logger.error(f"[{self.name}] Failed to save config: {e}")
            return False

    @abstractmethod
    async def run(self):
        """
        Main execution method for the agent.
        Should return a dictionary (Report).
        """
        pass
