import os
import json
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages OpenAI Model Selection based on Config Preferences.
    """
    def __init__(self, config_path='profiles/usman/config.json'):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = None
        self.config_path = config_path
        self.available_models = []
        
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                logger.error(f"ModelManager Init Failed: {e}")
        
    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except:
            return {}

    def get_available_models(self):
        """Cache available models from API."""
        if self.available_models:
            return self.available_models
            
        if not self.client:
            return []
            
        try:
            models = self.client.models.list()
            self.available_models = [m.id for m in models.data]
            return self.available_models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def get_model_for_role(self, role="governor"):
        """
        Get best model for a specific role based on config preferences.
        """
        config = self.load_config()
        preferences = config.get('model_preferences', {})
        
        # Get preference list for role (default to Governor's list if 'default' used)
        role_prefs = preferences.get(role, ["gpt-4o", "gpt-4-turbo"])
        
        available = self.get_available_models()
        if not available:
            logger.warning("No available models found from API. Using preference #1 blindly.")
            return role_prefs[0]
            
        # Find first match
        for pref in role_prefs:
            # Exact match
            if pref in available:
                if 'codex' in pref or 'davinci' in pref: continue # Skip legacy
                return pref
            
            # Partial match (e.g. "o1" matches "o1-preview-2024")
            # Be careful with partials, but useful for versions.
            matches = [m for m in available if pref in m]
            if matches:
                # Filter legacy
                valid = [m for m in matches if 'codex' not in m and 'davinci' not in m]
                if valid:
                    return valid[0]
                    
        # Fallback
        logger.warning(f"No preferred model for {role} found in availability. Defaulting to gpt-4o.")
        return "gpt-4o"

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv('profiles/usman/.env')
    logging.basicConfig(level=logging.INFO)
    mm = ModelManager()
    print(f"Governor Model: {mm.get_model_for_role('governor')}")
    print(f"Scout Model: {mm.get_model_for_role('scout')}")
