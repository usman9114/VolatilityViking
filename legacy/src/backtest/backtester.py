import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import mlflow

class Backtester:
    def __init__(self, predictions_df, prices_df, initial_capital=10000, fee=0.001):
        self.predictions = predictions_df
        self.prices = prices_df
        self.initial_capital = initial_capital
        self.fee = fee
        
    def calculate_metrics(self):
        print("Aligning predictions with real prices...")
        
        if 'Datetime' in self.predictions.columns:
            pred_df = self.predictions.set_index('Datetime')
        else:
            pred_df = self.predictions.copy()
            
        pred_df.index = pd.to_datetime(pred_df.index)
        
        # Real Prices
        price_df = self.prices[['close']].copy()
        if price_df.index.tz is None:
            price_df.index = price_df.index.tz_localize('UTC')
            
        # Join
        # Include High/Low for Grid calculation
        df = price_df.join(pred_df, how='inner')
        # We need High/Low from prices_df if not present
        if 'high' not in df.columns:
             df = df.join(self.prices[['high', 'low']], rsuffix='_dup')
             
        df.dropna(inplace=True)
        
        df['real_log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # --- Smart Logic (Dynamic Thresholds) ---
        # Optimized Params: Window=96, Std=0.86
        WINDOW = 96
        K = 0.86
        
        # Calculate Pred Stats
        df['pred_ma'] = df['Ensemble'].rolling(window=WINDOW).mean()
        df['pred_std'] = df['Ensemble'].rolling(window=WINDOW).std()
        
        df['upper_band'] = df['pred_ma'] + (K * df['pred_std'])
        df['lower_band'] = df['pred_ma'] - (K * df['pred_std'])
        
        # Signal Generation
        df['position'] = 0
        df.loc[df['Ensemble'] > df['upper_band'], 'position'] = 1
        df.loc[df['Ensemble'] < df['lower_band'], 'position'] = -1
        
        # Trade Costs
        df['trade'] = df['position'].diff().abs()
        
        # Strategy Return
        # Position applies to CURRENT candle return?
        # If position[T] is decided at T using Pred[T] (Simulating valid_indices alignment),
        # And Pred[T] predicts Return[T+1].
        # Then position[T] should be applied to Return[T+1].
        # In this DF, real_log_return[T+1] is at index T+1.
        # So we align Position[T] with Return[T+1].
        # Effectively, StratRet[K] = Pos[K-1] * Ret[K].
        
        # 1. Trend Strategy Returns
        df['active_signal'] = df['position'].shift(1)
        df['trend_return'] = df['active_signal'] * df['real_log_return']
        
        # Trend Costs
        df['trend_cost'] = df['active_signal'].diff().abs() * self.fee
        df['trend_net'] = df['trend_return'] - df['trend_cost']
        
        # 2. Hybrid Grid Logic (Rolling Statistics)
        # Use Rolling Volatility to define "Opportunity Size".
        # If Volatility is High, we assume we can capture more range.
        # Volatility = Rolling StdDev of Returns (e.g. 7 days = 42 periods)
        
        ROLL_WINDOW = 42 # 7 Days (4h * 42 = 168h)
        df['rolling_vol'] = df['real_log_return'].rolling(window=ROLL_WINDOW).std()
        
        # Grid Yield Estimate:
        # We assume we capture a fraction of the Daily Volatility.
        # Annualized Vol = Std * sqrt(365*6). Daily Vol = Std * sqrt(6).
        # Yield = Daily_Vol * Efficiency_Factor
        
        daily_vol = df['rolling_vol'] * np.sqrt(6) 
        
        # Efficiency: How much of that daily wobble do we capture?
        # Conservative: 0.1 (10% of the daily move)
        GRID_EFFICIENCY = 0.5 
        
        # This yield is per day? No, this is yield per PERIOD (4h).
        # If Vol says "Price moves 2% per day", and we capture 10% of that -> 0.2% per day.
        # Per 4h candle: 0.2% / 6 = 0.03%.
        
        # Let's simplify:
        # Yield_per_4h = Rolling_Std_4h * Capture_Ratio
        # Capture_Ratio = 0.5 (We capture 50% of the 1-sigma move)
        
        df['vol_yield'] = df['rolling_vol'] * GRID_EFFICIENCY
        
        # Hybrid Return: If Signal!=0 -> Trend. If Signal==0 -> Vol Yield.
        df['hybrid_return'] = np.where(df['active_signal'] != 0, 
                                       df['trend_net'], 
                                       df['vol_yield'])
        
        # We set 'strategy_log_return' to Hybrid for final reporting
        df['strategy_log_return'] = df['hybrid_return']
        
        # Also keep Trend Only for comparison
        df['trend_only_return'] = df['trend_net']
        
        # Equity
        df['cumulative_market_return'] = df['real_log_return'].cumsum()
        df['cumulative_strategy_return'] = df['strategy_log_return'].cumsum()
        
        df['market_equity'] = self.initial_capital * np.exp(df['cumulative_market_return'])
        df['strategy_equity'] = self.initial_capital * np.exp(df['cumulative_strategy_return'])
        
        # Metrics
        final_equity = df['strategy_equity'].iloc[-1]
        market_final = df['market_equity'].iloc[-1]
        
        strat_ret = (final_equity - self.initial_capital) / self.initial_capital
        mkt_ret = (market_final - self.initial_capital) / self.initial_capital
        
        # Sharpe
        periods_per_year = 365 * 6
        if df['strategy_log_return'].std() > 0:
            sharpe = (df['strategy_log_return'].mean() / df['strategy_log_return'].std()) * np.sqrt(periods_per_year)
        else:
            sharpe = 0
            
        # Drawdown
        roll_max = df['strategy_equity'].cummax()
        drawdown = (df['strategy_equity'] - roll_max) / roll_max
        max_drawdown = drawdown.min()
        
        # Capture Ratios (Geometric)
        # Up Capture: CAGR of Strat during Up Periods / CAGR of Market during Up Periods
        # Simplified: Sum of returns during positive market days
        up_market = df[df['real_log_return'] > 0]
        down_market = df[df['real_log_return'] < 0]
        
        if len(up_market) > 0:
            up_capture = up_market['strategy_log_return'].sum() / up_market['real_log_return'].sum()
        else:
            up_capture = 0
            
        if len(down_market) > 0:
            down_capture = down_market['strategy_log_return'].sum() / down_market['real_log_return'].sum()
        else:
            down_capture = 0
            
        print(f"--- Smart Backtest Results ({df.index[0].date()} to {df.index[-1].date()}) ---")
        print(f"Initial Capital: ${self.initial_capital}")
        print(f"Final Equity:    ${final_equity:.2f}")
        print(f"Total Return:    {strat_ret*100:.2f}%")
        print(f"Buy & Hold:      {mkt_ret*100:.2f}%")
        print(f"Sharpe Ratio:    {sharpe:.2f}")
        print(f"Max Drawdown:    {max_drawdown*100:.2f}%")
        print(f"Upside Capture:  {up_capture*100:.1f}%")
        print(f"Downside Capture:{down_capture*100:.1f}% (Lower is better)")

        # MLflow Logging
        if mlflow.active_run():
            mlflow.log_metrics({
                "final_return": strat_ret,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_drawdown,
                "upside_capture": up_capture,
                "downside_capture": down_capture
            })

        # Plot
        os.makedirs('data/processed', exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['market_equity'], label='Buy & Hold (ETH)', color='gray', alpha=0.5)
        plt.plot(df.index, df['strategy_equity'], label='Hybrid Grid Strategy (Best)', color='green')
        
        # Add Trend Only for comparison
        trend_equity = self.initial_capital * np.exp(df['trend_only_return'].cumsum())
        plt.plot(df.index, trend_equity, label='Trend Only (No Grid)', color='blue', linestyle='--')
        
        plt.title(f"Best Strategy Performance (Hybrid Trend + Grid)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('data/processed/backtest_equity.png')
        print("Plot saved to data/processed/backtest_equity.png")

        if mlflow.active_run():
            mlflow.log_artifact('data/processed/backtest_equity.png')

        return df
