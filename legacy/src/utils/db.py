import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_PATH = 'data/predictions.db'

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create Logs Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            symbol TEXT,
            timeframe TEXT,
            current_price REAL,
            predicted_return REAL,
            signal INTEGER,
            confidence REAL,
            features_json TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

def log_prediction(timestamp, symbol, timeframe, current_price, predicted_return, signal, confidence, features_json=""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO prediction_logs (timestamp, symbol, timeframe, current_price, predicted_return, signal, confidence, features_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, symbol, timeframe, current_price, predicted_return, signal, confidence, features_json))
    
    conn.commit()
    conn.close()

def get_logs(symbol=None, limit=100):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM prediction_logs"
    params = []
    
    if symbol:
        query += " WHERE symbol = ?"
        params.append(symbol)
        
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df
