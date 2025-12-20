import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import os

def comprehensive_backtest():
    print("--- Comprehensive Backtest (Rolling Grid Logic + Detailed Stats) ---")
    
    # 1. Load Data
    pred_path = 'data/processed/walk_forward_results_meta.csv'
    price_path = 'data/processed/train_dataset.parquet'
    
    preds = pd.read_csv(pred_path)
    if 'Datetime' in preds.columns:
        preds['Datetime'] = pd.to_datetime(preds['Datetime'])
        # preds.set_index('Datetime', inplace=True) 
    
    prices = pd.read_parquet(price_path)
    if prices.index.tz is None: prices.index = prices.index.tz_localize('UTC')

    # Convert preds to right format for merging
    if 'Datetime' in preds.columns:
        preds.set_index('Datetime', inplace=True)
    if preds.index.tz is None:
        preds.index = preds.index.tz_localize('UTC')
        
    # Join
    df = prices[['close', 'high', 'low']].join(preds, how='inner')
    df.dropna(inplace=True)
    
    # 2. Logic (Rolling Grid)
    WINDOW = 96
    K = 0.86
    ROLL_WINDOW = 42 # 7 Days
    GRID_EFFICIENCY = 0.5 # 50% Capture of Volatility
    FEE = 0.001
    
    # Regimes
    df['pred_ma'] = df['Ensemble'].rolling(window=WINDOW).mean()
    df['pred_std'] = df['Ensemble'].rolling(window=WINDOW).std()
    df['upper'] = df['pred_ma'] + (K * df['pred_std'])
    df['lower'] = df['pred_ma'] - (K * df['pred_std'])
    
    # Signals
    df['signal'] = 0
    df.loc[df['Ensemble'] > df['upper'], 'signal'] = 1
    df.loc[df['Ensemble'] < df['lower'], 'signal'] = -1
    
    # Active Signal (Shifted for T+1 Return alignment)
    df['active_signal'] = df['signal'].shift(1)
    
    # Returns
    df['real_log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Trend Component
    df['trend_return'] = df['active_signal'] * df['real_log_return']
    df['trend_cost'] = df['active_signal'].diff().abs() * FEE
    df['trend_net'] = df['trend_return'] - df['trend_cost']
    
    # Grid Component
    df['rolling_vol'] = df['real_log_return'].rolling(window=ROLL_WINDOW).std()
    df['vol_yield'] = df['rolling_vol'] * GRID_EFFICIENCY
    
    # Hybrid Component
    df['hybrid_log_return'] = np.where(df['active_signal'] != 0, 
                                       df['trend_net'], 
                                       df['vol_yield'])
    
    # 3. Equity Curves (Compounding)
    initial = 10000
    df['cum_hybrid'] = df['hybrid_log_return'].cumsum()
    df['cum_market'] = df['real_log_return'].cumsum()
    
    df['equity_hybrid'] = initial * np.exp(df['cum_hybrid'])
    df['equity_market'] = initial * np.exp(df['cum_market'])
    
    # 4. Detailed Stats
    # Win Rate (Positive Periods)
    wins = (df['hybrid_log_return'] > 0).sum()
    total = len(df)
    win_rate = wins / total
    
    # Sortino Ratio (Downside deviation only)
    downside_returns = df.loc[df['hybrid_log_return'] < 0, 'hybrid_log_return']
    if len(downside_returns) > 0:
        sortino = (df['hybrid_log_return'].mean() / downside_returns.std()) * np.sqrt(365*6)
    else:
        sortino = 0
        
    # Max Drawdown
    roll_max = df['equity_hybrid'].cummax()
    drawdown = (df['equity_hybrid'] - roll_max) / roll_max
    max_dd = drawdown.min()
    
    print("\n--- Comprehensive Stats ---")
    print(f"Final Equity: ${df['equity_hybrid'].iloc[-1]:.2f}")
    print(f"Win Rate:     {win_rate*100:.1f}% (Periods with +Return)")
    print(f"Sortino:      {sortino:.2f}")
    print(f"Max Drawdown: {max_dd*100:.2f}%")
    
    # 5. MLflow Log & Plot
    mlflow.set_tracking_uri("file:///home/usman/eth-bot/mlruns")
    mlflow.set_experiment("ETH-Meta-Trader")
    
    with mlflow.start_run(run_name="Comprehensive_Realistic"):
        mlflow.log_metrics({
            "final_equity": df['equity_hybrid'].iloc[-1],
            "win_rate": win_rate,
            "sortino": sortino,
            "max_drawdown": max_dd
        })
        
        # LOG SCALE PLOT
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['equity_market'], label='Buy & Hold (ETH)', color='gray', alpha=0.5)
        plt.plot(df.index, df['equity_hybrid'], label='Hybrid Grid (Log Scale)', color='green')
        
        plt.yscale('log') # The Money Shot
        plt.title(f"Performance (Log Probability Scale)")
        plt.ylabel("Equity ($) - Log Scale")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        plt.tight_layout()
        
        plot_path = 'data/processed/comprehensive_plot.png'
        plt.savefig(plot_path)
        print(f"Log Scale Plot saved to {plot_path}")
        mlflow.log_artifact(plot_path)

if __name__ == "__main__":
    comprehensive_backtest()
