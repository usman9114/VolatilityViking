
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ta
import joblib
import os

logger = logging.getLogger(__name__)

class MLPredictor:
    """
    Layer 2: ML Prediction Engine (FreqAI-Lite).
    Predicts short-term price moves using XGBoost.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.model = None
        self.model_path = "models/xgboost_price_predictor.json"
        
        # Load existing model if available
        if os.path.exists(self.model_path):
            try:
                self.model = xgb.XGBClassifier()
                self.model.load_model(self.model_path)
                logger.info(f"Loaded ML Model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
    
    def prepare_features(self, df):
        """
        Generate technical features like FreqAI.
        Input: DataFrame with ohlcv.
        Returns: Feature DataFrame.
        """
        df = df.copy()
        
        # 1. Momentum
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_slope'] = df['rsi'].diff(3)
        
        # 2. Volatility
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        df['bb_width'] = ta.volatility.bollinger_wband(df['close'])
        
        # 3. Trend
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
        df['ema_short'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_long'] = ta.trend.ema_indicator(df['close'], window=26)
        df['trend_dist'] = (df['ema_short'] - df['ema_long']) / df['ema_long']
        
        # 4. Volume
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']
        
        # 5. Market Data (Futures) - NEW
        if 'fundingRate' in df.columns:
            df['funding_rate'] = df['fundingRate']
            df['funding_ma'] = df['fundingRate'].rolling(10).mean()
            # Spread or other metrics could go here
        else:
            df['funding_rate'] = 0.0
            df['funding_ma'] = 0.0
        
        features = ['rsi', 'rsi_slope', 'atr', 'bb_width', 'adx', 'trend_dist', 'vol_ratio', 'funding_rate', 'funding_ma']
        return df[features].fillna(0)

    def train_model(self, df):
        """
        Train XGBoost on recent data.
        Target: Does price go UP or DOWN by > 0.5% in next 5 candles?
        """
        try:
            # 1. Create Target
            lookahead = 5
            threshold = 0.005 # 0.5% move
            
            df['future_close'] = df['close'].shift(-lookahead)
            df['change'] = (df['future_close'] - df['close']) / df['close']
            
            # Classes: 1 (Pump), -1 (Dump), 0 (Chop)
            df['target'] = 0
            df.loc[df['change'] > threshold, 'target'] = 1
            df.loc[df['change'] < -threshold, 'target'] = 2 # XGB prefers benign integers (0,1,2)
            
            # Map: 0=Chop, 1=Pump, 2=Dump
            
            # Drop NaN from shift
            df = df.dropna()
            
            # Assign Sample Weights based on Volatility (Profit-Weighted Loss)
            # abs(change) * 100 ensures a 1% move has weight 1.0, 5% move has weight 5.0
            # We clip minimum weight to 0.1 so 'Chop' isn't ignored completely.
            weights = df['change'].abs() * 100
            weights = weights.clip(lower=0.1)
            
            X = self.prepare_features(df)
            y = df['target']
            
            # Split (Stratify not used due to TimeSeriesSplit nature)
            # We must split weights too!
            split_idx = int(len(df) * 0.8)
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            w_train = weights.iloc[:split_idx]
            
            # Load optimized params if available
            params_path = "models/best_hyperparameters.json"
            model_params = {
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 5,
                'use_label_encoder': False,
                'eval_metric': 'mlogloss'
            }
            
            if os.path.exists(params_path):
                try:
                    with open(params_path, 'r') as f:
                        best_params = json.load(f)
                        model_params.update(best_params)
                        logger.info("Loaded Optimized Hyperparameters!")
                except Exception as e:
                    logger.warning(f"Failed to load optimized params: {e}")
                    
            # Train
            self.model = xgb.XGBClassifier(**model_params)
            # Pass weights to emphasize profitable trades
            self.model.fit(X_train, y_train, sample_weight=w_train)
            
            # Eval
            preds = self.model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            logger.info(f"ML Training Complete. Test Accuracy: {acc:.2%}")
            
            # Save
            os.makedirs('models', exist_ok=True)
            self.model.save_model(self.model_path)
            
            return True, acc
            
        except Exception as e:
            logger.error(f"ML Training Error: {e}")
            return False, 0.0

    def predict_next_move(self, df):
        """
        Predict next 5 candles.
        Returns: 
            prediction (int): 0 (Chop), 1 (Pump), 2 (Dump)
            confidence (float): Probability of that class
        """
        try:
            if not self.model:
                return 0, 0.0
            
            # Feature Gen on latest data
            features = self.prepare_features(df)
            last_row = features.iloc[[-1]] # 2D array
            
            # Predict
            pred_class = self.model.predict(last_row)[0]
            probs = self.model.predict_proba(last_row)[0]
            confidence = probs[pred_class]
            
            return int(pred_class), float(confidence)
            
        except Exception as e:
            logger.error(f"Prediction Error: {e}")
            return 0, 0.0
