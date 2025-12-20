
import pandas as pd
import numpy as np

def calculate_stats():
    try:
        df = pd.read_csv('data/processed/walk_forward_results_meta.csv')
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
        
        # Assumption: 'Actual' is actual return, 'Ensemble' is predicted return
        # Strategy: Long if Ensemble > 0 (or some threshold, usually 0 or 0.005)
        # Let's assume Long-Only if Prediction > 0 for this calculation
        
        df['Signal'] = np.where(df['Ensemble'] > 0.00, 1, 0)
        df['Strategy_Return'] = df['Signal'] * df['Actual']
        
        # Cumulative Return
        df['Equity_Curve'] = (1 + df['Strategy_Return']).cumprod()
        total_return = (df['Equity_Curve'].iloc[-1] - 1) * 100
        
        # Win Rate
        wins = len(df[df['Strategy_Return'] > 0])
        total_trades = len(df[df['Signal'] == 1])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        # Drawdown
        df['Peak'] = df['Equity_Curve'].cummax()
        df['Drawdown'] = (df['Equity_Curve'] - df['Peak']) / df['Peak']
        max_drawdown = df['Drawdown'].min() * 100
        
        # Sharpe (Annualized) assuming 6 data points per day (4h candles)
        # 365 * 6 = 2190 periods per year
        mean_ret = df['Strategy_Return'].mean()
        std_ret = df['Strategy_Return'].std()
        sharpe = (mean_ret / std_ret) * np.sqrt(2190) if std_ret > 0 else 0
        
        print(f"Total Return: {total_return:.2f}%")
        print(f"Win Rate: {win_rate:.2f}% ({wins}/{total_trades})")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Data Points: {len(df)}")
        print(f"Ensemble Stats: Min={df['Ensemble'].min()}, Max={df['Ensemble'].max()}, Mean={df['Ensemble'].mean()}")
        print(f"Positive Predictions: {len(df[df['Ensemble'] > 0])}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    calculate_stats()
