
import os
import sys
import logging
import json
import optuna
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.ai.ml_predictor import MLPredictor
from scripts.train_ml_model import fetch_historical_data, fetch_funding_history

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def objective(trial, X_train, y_train, w_train, X_test, y_test):
    """
    Optuna Objective Function.
    Optimizes XGBoost Hyperparameters.
    """
    param = {
        'verbosity': 0,
        'objective': 'multi:softmax',
        'num_class': 3,
        'booster': 'gbtree',
        'eval_metric': 'mlogloss',
        # Search Space
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }

    model = xgb.XGBClassifier(**param, use_label_encoder=False)
    
    # Train with sample weights (Profit-Aware)
    model.fit(X_train, y_train, sample_weight=w_train)
    
    # Predict
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    
    return accuracy

def main():
    logger.info("Starting Hyperparameter Tuning (Optuna)...")
    
    # 1. Fetch Data (Just 1 symbol for speed, or aggregate?)
    # For tuning, we want a representative sample. Let's use ETH/USDT (High volume leader).
    symbol = "ETH/USDT"
    logger.info(f"Fetching 90 days of data for {symbol} to tune hyperparameters...")
    
    df = fetch_historical_data(symbol, days=90)
    funding_df = fetch_funding_history(symbol, days=90)
    
    if df.empty or funding_df.empty:
        logger.error("Failed to fetch tuning data.")
        return

    # Merge
    df = pd.merge_asof(df, funding_df, on='timestamp', direction='backward')
    df['fundingRate'] = df['fundingRate'].fillna(0)
    
    # 2. Prepare Feature/Target (using Logic from MLPredictor but manual split for Optuna)
    predictor = MLPredictor()
    
    # Target
    lookahead = 5
    threshold = 0.005 # 0.5%
    df['future_close'] = df['close'].shift(-lookahead)
    df['change'] = (df['future_close'] - df['close']) / df['close']
    df['target'] = 0
    df.loc[df['change'] > threshold, 'target'] = 1
    df.loc[df['change'] < -threshold, 'target'] = 2
    df = df.dropna()
    
    # Weights
    weights = df['change'].abs() * 100
    weights = weights.clip(lower=0.1)
    
    X = predictor.prepare_features(df)
    y = df['target']
    
    # Split
    split_idx = int(len(df) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    w_train = weights.iloc[:split_idx]
    
    # 3. Run Optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, w_train, X_test, y_test), n_trials=50)
    
    logger.info("="*40)
    logger.info("✅ Tuning Complete!")
    logger.info(f"🏆 Best Trial Accuracy: {study.best_value:.2%}")
    logger.info("🌟 Best Params:")
    logger.info(json.dumps(study.best_params, indent=4))
    
    # Save best params to a file for the main trainer to use
    with open('models/best_hyperparameters.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)
    logger.info("Saved best params to models/best_hyperparameters.json")

if __name__ == "__main__":
    main()
