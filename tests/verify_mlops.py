
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os
import asyncio
from datetime import datetime

# Add src to path
sys.path.append(os.getcwd())
from src.live.async_runner import AsyncMultiSymbolBot

class TestMLOpsPipeline(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = {
            "ml_settings": {
                "enabled": True,
                "auto_retrain_hours": 24
            },
            "api_key": "test",
            "secret_key": "test"
        }
        # Mock Exchange to avoid validation errors
        with patch('ccxt.async_support.binance') as mock_exch:
             # We must patch load_config BEFORE instantiation
            with patch.object(AsyncMultiSymbolBot, 'load_config') as mock_load:
                mock_load.return_value = self.config
                self.bot = AsyncMultiSymbolBot("dummy_config.json")
                self.bot.exchange = MagicMock()
                # Ensure config is set (in case init overrides it despite mock, though it shouldn't)
                self.bot.config = self.config
            
    async def test_sunday_tuning_trigger(self):
        """Test that Sunday triggers tune_model.py"""
        # Mock datetime to be Sunday
        # datetime.now().weekday() == 6
        
        # We need to mock datetime within async_runner module if possible, 
        # or mock the logic. Since we imported datetime in async_runner, 
        # we patch 'src.live.async_runner.datetime'
        
        with patch('src.live.async_runner.datetime') as mock_dt:
            mock_dt.now.return_value.weekday.return_value = 6 # Sunday
            
            # Reset timer to force trigger
            self.bot.last_ml_retrain = 0 
            
            with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_proc:
                mock_proc.return_value.pid = 1234
                mock_proc.return_value.communicate.return_value = (b"", b"")
                mock_proc.return_value.returncode = 0
                
                await self.bot.check_and_retrain_ml()
                
                # Check call args
                args = mock_proc.call_args[0]
                script_name = args[1]
                print(f"Sunday Script: {script_name}")
                self.assertIn("tune_model.py", script_name)
                
    async def test_monday_training_trigger(self):
        """Test that Monday triggers train_ml_model.py"""
        with patch('src.live.async_runner.datetime') as mock_dt:
            mock_dt.now.return_value.weekday.return_value = 0 # Monday
            
            self.bot.last_ml_retrain = 0 
            
            with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_proc:
                mock_proc.return_value.pid = 5678
                mock_proc.return_value.communicate.return_value = (b"", b"")
                mock_proc.return_value.returncode = 0
                
                await self.bot.check_and_retrain_ml()
                
                args = mock_proc.call_args[0]
                script_name = args[1]
                print(f"Monday Script: {script_name}")
                self.assertIn("train_ml_model.py", script_name)

if __name__ == "__main__":
    unittest.main()
