import pandas as pd
import numpy as np

# --- PANDAS INDICATOR IMPLEMENTATIONS ---

def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = calculate_ema(series, fast)
    exp2 = calculate_ema(series, slow)
    macd = exp1 - exp2
    signal_line = calculate_ema(macd, signal)
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return tr.rolling(window=period).mean()

def calculate_bollinger_width(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return (upper - lower) / sma

def calculate_obv(df):
    obv = np.where(df['close'] > df['close'].shift(1), df['volume'], 
                   np.where(df['close'] < df['close'].shift(1), -df['volume'], 0)).cumsum()
    return pd.Series(obv, index=df.index)

def calculate_adx(df, period=14):
    """
    Calculate Average Directional Index (ADX)
    """
    period = 14
    high = df['high']
    low = df['low']
    close = df['close']
    
    # TR
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    # Directional Movement
    up = high.diff()
    down = -low.diff()
    
    pos_dm = np.where((up > down) & (up > 0), up, 0.0)
    neg_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    pos_dm_s = pd.Series(pos_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    neg_dm_s = pd.Series(neg_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    
    # DI
    pos_di = 100 * (pos_dm_s / atr)
    neg_di = 100 * (neg_dm_s / atr)
    
    # DX
    dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx

def calculate_alpha_features(df):
    """
    Calculate Professional Grade 'Alpha' Technical Indicators using Pandas.
    """
    close = df['close']
    # Ensure Timestamp is handled if present, but we operate on columns
    
    # 1. Trend
    df['ema_20'] = calculate_ema(close, 20)
    df['ema_50'] = calculate_ema(close, 50)
    df['ema_200'] = calculate_ema(close, 200)
    
    df['dist_ema_20'] = (close - df['ema_20']) / df['ema_20']
    df['dist_ema_200'] = (close - df['ema_200']) / df['ema_200']
    
    df['adx'] = calculate_adx(df, 14) 
    df['trend_strength'] = abs(df['ema_50'].diff()) / df['ema_50']
    
    _, _, macdhist = calculate_macd(close)
    df['macd_hist'] = macdhist
    
    # 2. Momentum
    df['rsi'] = calculate_rsi(close, 14)
    df['roc'] = close.pct_change(periods=12)
    
    # 3. Volatility
    df['atr'] = calculate_atr(df, 14)
    df['atr_pct'] = df['atr'] / close 
    df['bb_width'] = calculate_bollinger_width(close)
    
    # 4. Volume
    df['obv'] = calculate_obv(df)
    avg_vol = df['volume'].rolling(20).mean()
    df['obv_slope'] = df['obv'].diff(3) / avg_vol

    # 5. Cyclical
    # Check if we have timestamp column
    if 'timestamp' in df.columns:
        dates = pd.to_datetime(df['timestamp'], unit='ms')
        df['day_of_week'] = dates.dt.dayofweek
        df['hour_of_day'] = dates.dt.hour
    
    return df
