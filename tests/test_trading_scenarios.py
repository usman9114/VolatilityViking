
import unittest
import pandas as pd
import numpy as np

class TestTradingScenarios(unittest.TestCase):
    
    def decide_signal(self, current_prediction, rolling_mean, rolling_std, K=0.86):
        """Replicates the logic inside BinanceBot.get_signal"""
        threshold_up = rolling_mean + (K * rolling_std)
        threshold_down = rolling_mean - (K * rolling_std)
        
        signal = "NEUTRAL"
        if pd.isna(current_prediction) or pd.isna(threshold_up) or pd.isna(threshold_down):
            return "GRID (FALLBACK)"
            
        if current_prediction > threshold_up:
            signal = "LONG"
        elif current_prediction < threshold_down:
            signal = "SHORT"
        else:
            signal = "GRID"
            
        return signal

    # 1. Bull Trend
    def test_bull_trend(self):
        # Pred significantly above upper band
        signal = self.decide_signal(0.02, 0.00, 0.01)
        self.assertEqual(signal, "LONG")

    # 2. Bear Trend
    def test_bear_trend(self):
        # Pred significantly below lower band
        signal = self.decide_signal(-0.02, 0.00, 0.01)
        self.assertEqual(signal, "SHORT")

    # 3. Sideways (Chop)
    def test_sideways_chop(self):
        # Pred inside bands
        signal = self.decide_signal(0.005, 0.00, 0.01)
        self.assertEqual(signal, "GRID")

    # 4. Flash Crash
    def test_flash_crash(self):
        # Pred massive negative
        signal = self.decide_signal(-0.10, 0.00, 0.02)
        self.assertEqual(signal, "SHORT")

    # 5. Moon Mission
    def test_moon_mission(self):
        # Pred massive positive
        signal = self.decide_signal(0.10, 0.00, 0.02)
        self.assertEqual(signal, "LONG")

    # 6. Tight Squeeze (Low Volatility)
    def test_tight_squeeze(self):
        # Bands are very narrow, small movement triggers Trend
        signal = self.decide_signal(0.002, 0.00, 0.001) # Upper Band ~0.00086
        self.assertEqual(signal, "LONG")

    # 7. High Volatility
    def test_high_volatility(self):
        # Bands are very wide, large movement still Grid
        signal = self.decide_signal(0.02, 0.00, 0.05) # Upper Band ~0.043
        self.assertEqual(signal, "GRID")

    # 8. Recovery (Return to Mean)
    def test_recovery(self):
        # Pred is exactly mean
        signal = self.decide_signal(0.00, 0.00, 0.01)
        self.assertEqual(signal, "GRID")

    # 9. Data Outage (NaN)
    def test_data_outage(self):
        # Inputs are NaN
        signal = self.decide_signal(np.nan, 0.00, 0.01)
        self.assertEqual(signal, "GRID (FALLBACK)")

    # 10. Zero Variance (Edge Case)
    def test_zero_variance(self):
        # Std Dev is 0 (Flatline). Prediction slightly positive.
        # Thresholds = Mean. Pred > Mean -> LONG.
        signal = self.decide_signal(0.0001, 0.00, 0.00) 
        self.assertEqual(signal, "LONG")

if __name__ == '__main__':
    unittest.main()
