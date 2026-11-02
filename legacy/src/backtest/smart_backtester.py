import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SmartBacktester:
    def __init__(self, pred_path, price_path, initial_capital=10000):
        self.pred_path = pred_path
        self.price_path = price_path
        self.initial_capital = initial_capital
        
    def load_data(self):
        # Load Predictions (Regression: Log Returns)
        try:
            self.preds_df = pd.read_csv(self.pred_path, index_col=0, parse_dates=True)
            print(f"Loaded predictions from {self.pred_path}: {len(self.preds_df)} rows")
        except FileNotFoundError:
            print(f"Error: File not found {self.pred_path}")
            return False
            
        return True
        
    def run_dynamic_threshold(self, window=24, std_devs=1.0):
        """
        Signal = Long if Pred > MovingAverage(Pred) + k * Std(Pred)
        Signal = Short if Pred < MovingAverage(Pred) - k * Std(Pred)
        """
        df = self.preds_df.copy()
        df.sort_index(inplace=True)
        
        # Calculate Rolling Stats of PREDICTIONS
        df['pred_ma'] = df['Ensemble'].rolling(window=window).mean()
        df['pred_std'] = df['Ensemble'].rolling(window=window).std()
        
        # Dynamic Threshold
        df['upper_band'] = df['pred_ma'] + (std_devs * df['pred_std'])
        df['lower_band'] = df['pred_ma'] - (std_devs * df['pred_std'])
        
        # Generate Signals
        # 1 = Long, -1 = Short, 0 = Neutral
        df['position'] = 0
        
        # Signal Logic:
        # If prediction is abnormally high (above upper band) -> Long
        # If prediction is abnormally low (below lower band) -> Short
        
        # We also need to consider if prediction is POSITIVE or NEGATIVE?
        # Or does the band logic cover it? 
        # If trend is crashing, MA is negative. Prediction below lower band means "even more negative" -> Short.
        # If trend is mooning, MA is positive. Prediction above upper band means "even more positive" -> Long.
        
        df.loc[df['Ensemble'] > df['upper_band'], 'position'] = 1
        df.loc[df['Ensemble'] < df['lower_band'], 'position'] = -1
        
        # Strategy Return
        # We trade at Close, so we capture the return of the Next Candle?
        # In our dataset, 'Actual' is the log return of the target period (shifted).
        # So 'Actual' at index T is the return from T to T+1 (or T to T+horizon).
        # So Position at T should match Actual at T.
        
        df['strategy_return'] = df['position'] * df['Actual']
        
        # Transaction Costs (Simulated 0.1% per trade)
        # We need to detect position changes
        df['trade'] = df['position'].diff().fillna(0).abs()
        # df['cost'] = df['trade'] * 0.001 # 0.1% fee
        # Let's ignore fees for first pass optimization, but keep in mind
        
        # Equity Curve
        df['equity'] = self.initial_capital * (1 + df['strategy_return']).cumprod()
        
        # Metrics
        total_return = (df['equity'].iloc[-1] / self.initial_capital) - 1
        sharpe = df['strategy_return'].mean() / (df['strategy_return'].std() + 1e-9) * np.sqrt(365*6) # 4h candles
        
        # print(f"Dynamic Threshold (W={window}, K={std_devs}): Total Return = {total_return*100:.2f}%, Sharpe = {sharpe:.2f}")
        return df, total_return, sharpe
        
if __name__ == "__main__":
    # Use the best model predictions
    # We expect 'walk_forward_results_cnn.csv' to be the best regression one
    path = "data/processed/walk_forward_results_cnn.csv" 
    price_path = "data/processed/train_dataset.parquet"
    
    sb = SmartBacktester(path, price_path)
    if sb.load_data():
        print("--- Searching for Optimal Thresholds (Dynamic Bands) ---")
        best_ret = -999
        best_params = None
        
        # Grid Search
        # Window: 12 (2d), 24 (4d), 48 (8d), 96 (16d)
        # K: 0.5 to 3.0
        
        results = []
        
        for window in [6, 12, 24, 48, 96, 168]: 
            for k in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                df, ret, sharpe = sb.run_dynamic_threshold(window, k)
                results.append((window, k, ret, sharpe))
                
                if ret > best_ret:
                    best_ret = ret
                    best_params = (window, k)
                    
        print(f"\nBest Settings: Window={best_params[0]}, StdDevs={best_params[1]}")
        print(f"Best Return: {best_ret*100:.2f}%")
        
        # Run best again to show detail
        df_best, _, sharpe_best = sb.run_dynamic_threshold(best_params[0], best_params[1])
        print(f"Final Equity: ${df_best['equity'].iloc[-1]:.2f}")
        print(f"Sharpe: {sharpe_best:.2f}")
        
        # Save plot
        plt.figure(figsize=(12,6))
        plt.plot(df_best.index, df_best['equity'], label='Strategy')
        # Buy Hold
        buy_hold = sb.initial_capital * (1 + df_best['Actual']).cumprod()
        plt.plot(df_best.index, buy_hold, label='Buy & Hold', alpha=0.5)
        plt.title(f"Smart Strategy (W={best_params[0]}, K={best_params[1]})")
        plt.legend()
        plt.savefig('data/processed/smart_backtest_equity.png')
        print("Plot saved to data/processed/smart_backtest_equity.png")
