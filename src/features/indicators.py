import pandas as pd
import numpy as np
import ta

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()

    def add_technical_indicators(self):
        """Adds RSI, MACD, BB, ATR, etc. using 'ta' library."""
        
        # RSI
        self.df['RSI_14'] = ta.momentum.rsi(self.df['close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(self.df['close'])
        self.df['MACD_12_26_9'] = macd.macd()
        self.df['MACDs_12_26_9'] = macd.macd_signal()
        self.df['MACDh_12_26_9'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(self.df['close'], window=20, window_dev=2)
        self.df['BBL_20_2.0'] = bb.bollinger_lband()
        self.df['BBM_20_2.0'] = bb.bollinger_mavg()
        self.df['BBU_20_2.0'] = bb.bollinger_hband()
        self.df['BBB_20_2.0'] = bb.bollinger_wband()
        self.df['BBP_20_2.0'] = bb.bollinger_pband()
        
        # ATR (Volatility)
        self.df['ATR_14'] = ta.volatility.average_true_range(self.df['high'], self.df['low'], self.df['close'], window=14)
        
        # ADX (Trend Strength)
        adx = ta.trend.ADXIndicator(self.df['high'], self.df['low'], self.df['close'], window=14)
        self.df['ADX_14'] = adx.adx()
        self.df['DMP_14'] = adx.adx_pos()
        self.df['DMN_14'] = adx.adx_neg()

        # OBV (Volume)
        self.df['OBV'] = ta.volume.on_balance_volume(self.df['close'], self.df['volume'])

        # Moving Averages
        # SMA
        for length in [20, 50, 200]:
            self.df[f'SMA_{length}'] = ta.trend.sma_indicator(self.df['close'], window=length)
        
        # EMA
        for length in [12, 26, 50]:
            self.df[f'EMA_{length}'] = ta.trend.ema_indicator(self.df['close'], window=length)

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(self.df['high'], self.df['low'], self.df['close'], window=14, smooth_window=3)
        self.df['STOCHk_14_3_3'] = stoch.stoch()
        self.df['STOCHd_14_3_3'] = stoch.stoch_signal()

        # Parabolic SAR
        psar = ta.trend.PSARIndicator(self.df['high'], self.df['low'], self.df['close'])
        # ta lib returns separate indicators, we can combine or keep separate. 
        # For compatibility with old logic, let's keep it simple:
        self.df['PSARl'] = psar.psar_down()
        self.df['PSARs'] = psar.psar_up()
        self.df['PSAR'] = self.df['PSARl'].combine_first(self.df['PSARs'])
        
        # Simple Returns
        self.df['returns'] = self.df['close'].pct_change()
        # Log Returns (Manual calculation to be safe)
        self.df['log_returns'] = np.log(self.df['close'] / self.df['close'].shift(1))
        
        return self.df

    def merge_fear_greed(self, fear_greed_path):
        """Merges (forward fills) daily Fear & Greed index into 4h data."""
        try:
            fg_df = pd.read_parquet(fear_greed_path)
            
            # Ensure timestamps are timezone aware (UTC)
            if fg_df.index.tz is None:
                fg_df.index = fg_df.index.tz_localize('UTC')
            
            # Since FG is daily, we reindex it to match our 4h index using forward fill
            # First, sort both
            fg_df.sort_index(inplace=True)
            self.df.sort_index(inplace=True)
            
            merged = pd.merge_asof(
                self.df, 
                fg_df[['fear_greed_index']], 
                left_index=True, 
                right_index=True, 
                direction='backward' # use latest available value
            )
            
            self.df = merged
            return self.df
            
        except Exception as e:
            print(f"Error merging Fear & Greed data: {e}")
            return self.df

    def create_features(self, fear_greed_path=None):
        self.add_technical_indicators()
        if fear_greed_path:
            self.merge_fear_greed(fear_greed_path)
        
        # Cleanup
        self.df.dropna(inplace=True)
        return self.df

if __name__ == "__main__":
    # Load data
    try:
        df = pd.read_parquet('data/raw/ETH_USDT_4h.parquet')
        
        fe = FeatureEngineer(df)
        # Optional: check if fear_greed exists
        fg_path = 'data/raw/fear_greed.parquet'
        import os
        if not os.path.exists(fg_path): fg_path = None
        
        df_features = fe.create_features(fear_greed_path=fg_path)
        
        print(df_features.head())
        print(df_features.columns)
        
        # Save processed
        os.makedirs('data/processed', exist_ok=True)
        df_features.to_parquet('data/processed/eth_4h_features.parquet')
        print("Features saved to data/processed/eth_4h_features.parquet")
    except Exception as e:
        print(f"Main execution skipped/failed (likely no data): {e}")

def add_technical_indicators(df):
    """
    Standalone wrapper for compatibility.
    """
    fe = FeatureEngineer(df)
    return fe.create_features()
