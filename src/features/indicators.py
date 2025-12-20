import pandas as pd
import pandas_ta_classic as ta

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()

    def add_technical_indicators(self):
        """Adds RSI, MACD, BB, ATR, etc."""
        # RSI
        self.df['RSI_14'] = ta.rsi(self.df['close'], length=14)
        
        # MACD
        macd = ta.macd(self.df['close'])
        self.df = pd.concat([self.df, macd], axis=1)
        
        # Bollinger Bands
        bb = ta.bbands(self.df['close'])
        self.df = pd.concat([self.df, bb], axis=1)
        
        # ATR (Volatility)
        self.df['ATR_14'] = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=14)
        
        # ADX (Trend Strength)
        adx = ta.adx(self.df['high'], self.df['low'], self.df['close'])
        self.df = pd.concat([self.df, adx], axis=1)

        # OBV (Volume)
        self.df['OBV'] = ta.obv(self.df['close'], self.df['volume'])

        # Moving Averages
        # SMA
        for length in [20, 50, 200]:
            self.df[f'SMA_{length}'] = ta.sma(self.df['close'], length=length)
        
        # EMA
        for length in [12, 26, 50]:
            self.df[f'EMA_{length}'] = ta.ema(self.df['close'], length=length)

        # Stochastic Oscillator
        stoch = ta.stoch(self.df['high'], self.df['low'], self.df['close'])
        self.df = pd.concat([self.df, stoch], axis=1)

        # Parabolic SAR
        psar = ta.psar(self.df['high'], self.df['low'], self.df['close'])
        # PSAR returns separate columns for Long and Short. Combine them.
        # Column names are dynamic based on params, but usually like PSARl_0.02_0.2, PSARs_0.02_0.2
        # We can just sum them (NaN + Value = NaN? No, we want fillna)
        psar_l_col = [c for c in psar.columns if c.startswith('PSARl')][0]
        psar_s_col = [c for c in psar.columns if c.startswith('PSARs')][0]
        self.df['PSAR'] = psar[psar_l_col].combine_first(psar[psar_s_col])
        
        # Simple Returns
        self.df['returns'] = self.df['close'].pct_change()
        self.df['log_returns'] = ta.log_return(self.df['close'])
        
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
            
            # Merge asof or just reindex with ffill
            # We want the FG value known AT or BEFORE the candle time.
            # 'asof' merge is perfect for this.
            
            # fg_df = fg_df[['fear_greed_index']] # Keep numeric only for now
            
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
    df = pd.read_parquet('data/raw/ETH_USDT_4h.parquet')
    
    fe = FeatureEngineer(df)
    df_features = fe.create_features(fear_greed_path='data/raw/fear_greed.parquet')
    
    print(df_features.head())
    print(df_features.columns)
    
    # Save processed
    df_features.to_parquet('data/processed/eth_4h_features.parquet')
    print("Features saved to data/processed/eth_4h_features.parquet")

def add_technical_indicators(df):
    """
    Standalone wrapper for compatibility.
    """
    fe = FeatureEngineer(df)
    return fe.create_features()
