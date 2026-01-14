import streamlit as st
import ccxt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
import os
import sys

# Load environment variables
# Priority: 1. ENV_FILE var, 2. .env_riz, 3. .env
custom_env = os.getenv('ENV_FILE')
if custom_env and os.path.exists(custom_env):
    load_dotenv(custom_env, override=True)
    print(f"Loaded config from {custom_env}")
elif os.path.exists('.env_riz'):
    load_dotenv('.env_riz', override=True)
else:
    load_dotenv(override=True)

# Page config
st.set_page_config(
    page_title="Volatility Viking Profit Tracker",
    page_icon="💰",
    layout="wide"
)

# Initialize Binance connection
@st.cache_resource
def get_exchange():
    api_key = os.getenv('BINANCE_API_KEY')
    secret = os.getenv('BINANCE_SECRET_KEY') or os.getenv('BINANCE_SECRET')
    
    if not api_key:
        return None
        
    return ccxt.binance({
        'apiKey': api_key,
        'secret': secret,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })

exchange = get_exchange()

# Load Config for Symbols
def load_symbols():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            return config.get('symbols', ['ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'LINK/USDT']) + config.get('forager', {}).get('approved_symbols', [])
    except:
        return ['ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'LINK/USDT']

SYMBOLS = list(set(load_symbols()))

# Fetch filled orders for ALL symbols
@st.cache_data(ttl=60)
def fetch_all_trades(symbols):
    all_trades = []
    
    # Calculate start time (e.g., 30 days ago to keep it fast, or since bot start)
    since = exchange.milliseconds() - (30 * 24 * 60 * 60 * 1000) 
    
    for symbol in symbols:
        try:
            trades = exchange.fetch_my_trades(symbol, since=since, limit=1000)
            for t in trades:
                all_trades.append({
                    'Symbol': symbol,
                    'Time': pd.to_datetime(t['timestamp'], unit='ms'),
                    'Side': t['side'].upper(),
                    'Price': float(t['price']),
                    'Amount': float(t['amount']),
                    'Cost': float(t['cost']), # Total USDT value
                    'Fee': float(t['fee']['cost']) if t.get('fee') else 0.0,
                    'FeeCurrency': t['fee']['currency'] if t.get('fee') else ''
                })
        except Exception as e:
            pass
            
    if not all_trades:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_trades)
    df = df.sort_values('Time', ascending=False)
    return df

# Strict Realized Profit Calculation
def calculate_realized_pnl(df):
    """
    Calculates Realized Profit using FIFO matching.
    Profit = (Sell Price - Buy Price) * Matched Amount
    """
    realized_pnl = 0.0
    symbol_pnl = {}
    
    for symbol in df['Symbol'].unique():
        sym_df = df[df['Symbol'] == symbol].sort_values('Time') # Time asc for FIFO
        
        inventory = [] # List of (Price, Amount) tuples
        pnl = 0.0
        fees = 0.0
        
        for _, row in sym_df.iterrows():
            if row['Side'] == 'BUY':
                inventory.append({'price': row['Price'], 'amount': row['Amount']})
                fees += row['Fee'] if row['FeeCurrency'] == 'USDT' else 0 # Simplified fee tracking
            elif row['Side'] == 'SELL':
                sell_amt = row['Amount']
                sell_price = row['Price']
                fees += row['Fee'] if row['FeeCurrency'] == 'USDT' else 0
                
                # Match against inventory (FIFO)
                while sell_amt > 0 and inventory:
                    buy_item = inventory[0]
                    
                    matched_amt = min(sell_amt, buy_item['amount'])
                    
                    # Calculate PnL on this chunk
                    trade_profit = (sell_price - buy_item['price']) * matched_amt
                    pnl += trade_profit
                    
                    # Update counters
                    sell_amt -= matched_amt
                    buy_item['amount'] -= matched_amt
                    
                    if buy_item['amount'] <= 1e-8: # floating point epsilon
                        inventory.pop(0)
                        
        symbol_pnl[symbol] = pnl - fees
        realized_pnl += (pnl - fees)
        
    return realized_pnl, symbol_pnl

# Dashboard Layout
st.title("💰 Realized Profit Tracker")

if not exchange:
    st.error("API Keys missing! strict checking .env_riz or .env")
    st.stop()

trades_df = fetch_all_trades(SYMBOLS)

if not trades_df.empty:
    # 1. Calculate Realized PnL
    total_realized, sym_pnl = calculate_realized_pnl(trades_df)
    
    # 2. Display Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("💵 Total Realized Profit", f"${total_realized:.2f}", help="Net profit from closed trades (sells - cost of buys)")
    
    # 3. Per Symbol Breakdown
    st.subheader("Profit by Symbol")
    pnl_data = [{"Symbol": s, "Realized Profit": p} for s, p in sym_pnl.items()]
    pnl_df = pd.DataFrame(pnl_data).sort_values("Realized Profit", ascending=False)
    
    st.dataframe(
        pnl_df.style.format({"Realized Profit": "${:.2f}"})
              .background_gradient(cmap='RdYlGn', subset=['Realized Profit']),
        use_container_width=True
    )
    
    # 4. Recent Trades Log
    st.subheader("Recent Trades")
    st.dataframe(
        trades_df[['Time', 'Symbol', 'Side', 'Price', 'Amount', 'Cost']].head(50)
        .style.format({'Price': '${:.4f}', 'Cost': '${:.2f}', 'Amount': '{:.4f}'}),
        use_container_width=True
    )

else:
    st.info("No trades found in the last 30 days.")
    
# Auto-refresh
if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()
