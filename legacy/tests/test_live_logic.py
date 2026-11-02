import unittest
import pandas as pd
import numpy as np

class TestBotLogic(unittest.TestCase):
    
    def decide_signal(self, current_prediction, rolling_mean, rolling_std, K=0.86):
        """Replicates the logic inside BinanceBot.get_signal"""
        threshold_up = rolling_mean + (K * rolling_std)
        threshold_down = rolling_mean - (K * rolling_std)
        
        signal = "NEUTRAL"
        if pd.isna(current_prediction) or pd.isna(threshold_up) or pd.isna(threshold_down):
            return "GRID (FALLBACK)", threshold_down, threshold_up
            
        if current_prediction > threshold_up:
            signal = "LONG"
        elif current_prediction < threshold_down:
            signal = "SHORT"
        else:
            signal = "GRID"
            
        return signal, threshold_down, threshold_up

    def test_grid_conditions(self):
        # Scenario: Prediction is within Bands -> GRID
        pred = 0.001
        mean = 0.000
        std = 0.005 # Bands: -0.0043 to +0.0043
        
        signal, low, high = self.decide_signal(pred, mean, std)
        self.assertEqual(signal, "GRID")

    def test_long_condition(self):
        # Scenario: Prediction > Upper Band -> LONG
        pred = 0.01 
        mean = 0.000
        std = 0.005 # Bands: -0.0043 to +0.0043 (Upper is 0.0043)
        
        signal, low, high = self.decide_signal(pred, mean, std)
        self.assertEqual(signal, "LONG")
        
    def test_short_condition(self):
        # Scenario: Prediction < Lower Band -> SHORT
        pred = -0.01 
        mean = 0.000
        std = 0.005 # Bands: -0.0043 to +0.0043 (Lower is -0.0043)
        
        signal, low, high = self.decide_signal(pred, mean, std)
        self.assertEqual(signal, "SHORT")
        
    def test_nan_fallback(self):
        # Scenario: Not enough history (NaN bands)
        pred = 0.005
        mean = np.nan
        std = np.nan
        
        signal, low, high = self.decide_signal(pred, mean, std)
        self.assertEqual(signal, "GRID (FALLBACK)")

if __name__ == '__main__':
    unittest.main()
