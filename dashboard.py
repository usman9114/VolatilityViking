import streamlit as st
import ccxt
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="ETH Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize Binance connection
@st.cache_resource
def get_exchange():
    return ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET_KEY') or os.getenv('BINANCE_SECRET'),
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })

exchange = get_exchange()

# Fetch current balance
@st.cache_data(ttl=10)  # Cache for 10 seconds
def fetch_balance():
    balance = exchange.fetch_balance()
    usdt = balance['total'].get('USDT', 0)
    eth = balance['total'].get('ETH', 0)
    
    # Get current ETH price
    ticker = exchange.fetch_ticker('ETH/USDT')
    eth_price = ticker['last']
    
    total_equity = usdt + (eth * eth_price)
    
    return {
        'usdt': usdt,
        'eth': eth,
        'eth_price': eth_price,
        'total_equity': total_equity
    }

# Fetch filled orders
# Fetch filled orders
@st.cache_data(ttl=60)  # Cache for 60 seconds
def fetch_filled_orders(symbol='ETH/USDT'):
    try:
        # Fetch trades from the last year (ensure we get history)
        # 1 year ago = 365 * 24 * 60 * 60 * 1000 ms
        since = exchange.milliseconds() - (365 * 24 * 60 * 60 * 1000)
        
        # Limit 1000 is generous for this bot's frequency
        trades = exchange.fetch_my_trades(symbol, since=since, limit=1000)
        
        if trades:
            df = pd.DataFrame([{
                'Time': pd.to_datetime(t['timestamp'], unit='ms'),
                'Side': t['side'].upper(),
                'Price': float(t['price']),
                'Amount': float(t['amount']),
                'Total': float(t['cost']), # cost = price * amount
                'Fee': float(t['fee']['cost']) if t.get('fee') else 0.0
            } for t in trades])
            
            df = df.sort_values('Time', ascending=False)
            return df
            
    except Exception as e:
        st.error(f"Error fetching orders from Binance: {e}")
        
    return pd.DataFrame()

# Calculate P&L
def calculate_pnl(df, current_balance):
    if df.empty:
        # No filled orders yet - show current balance as baseline
        return 0, 0, current_balance['total_equity']
    
    # Calculate total bought and sold
    buys = df[df['Side'] == 'BUY']
    sells = df[df['Side'] == 'SELL']
    
    total_bought_usdt = buys['Total'].sum()
    total_sold_usdt = sells['Total'].sum()
    
    total_bought_eth = buys['Amount'].sum()
    total_sold_eth = sells['Amount'].sum()
    
    # Current holdings
    current_eth = current_balance['eth']
    
    # Avoid division by zero
    if total_bought_eth > 0:
        avg_buy_price = total_bought_usdt / total_bought_eth
    else:
        avg_buy_price = 0
        
    # Calculate realized P&L (from sells)
    realized_pnl = total_sold_usdt - (total_sold_eth * avg_buy_price)
    
    # Calculate unrealized P&L (current holdings)
    # Note: using current_balance['eth'] for unrealized P&L
    unrealized_pnl = (current_eth * current_balance['eth_price']) - (current_eth * avg_buy_price)
    
    # Total P&L
    total_pnl = realized_pnl + unrealized_pnl
    
    # Capital at risk = total bought (what you've invested)
    capital_at_risk = total_bought_usdt
    
    # ROI based on current equity vs what you started with
    # If you bought $100 of ETH and now have $110 total = +10% ROI
    pnl_pct = (total_pnl / capital_at_risk * 100) if capital_at_risk > 0 else 0
    
    return total_pnl, pnl_pct, capital_at_risk

# Dashboard
st.title("📈 ETH Trading Dashboard")
st.markdown("---")

# Fetch data
try:
    balance = fetch_balance()
    orders_df = fetch_filled_orders()
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Total Equity", f"${balance['total_equity']:.2f}")
    
    with col2:
        st.metric("💵 USDT Balance", f"${balance['usdt']:.2f}")
    
    with col3:
        st.metric("🪙 ETH Balance", f"{balance['eth']:.4f} ETH")
    
    with col4:
        st.metric("📊 ETH Price", f"${balance['eth_price']:.2f}")
    
    st.markdown("---")
    
    # P&L Section
    if not orders_df.empty:
        total_pnl, pnl_pct, initial_capital = calculate_pnl(orders_df, balance)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "💼 Capital at Risk",
                f"${initial_capital:.2f}"
            )
        
        with col2:
            pnl_color = "normal" if total_pnl >= 0 else "inverse"
            st.metric(
                "💹 Total P&L",
                f"${total_pnl:.2f}",
                delta=f"{pnl_pct:.2f}%",
                delta_color=pnl_color
            )
        
        with col3:
            roi = (balance['total_equity'] / initial_capital - 1) * 100 if initial_capital > 0 else 0
            st.metric(
                "📈 ROI",
                f"{roi:.2f}%",
                delta_color="normal" if roi >= 0 else "inverse"
            )
        
        st.markdown("---")
        
        # Cumulative P&L Graph
        st.subheader("📊 Cumulative Profit/Loss")
        
        # Calculate cumulative P&L over time
        df_sorted = orders_df.sort_values('Time')
        df_sorted['Cumulative_USDT'] = 0.0
        
        cumulative = 0
        for idx, row in df_sorted.iterrows():
            if row['Side'] == 'BUY':
                cumulative -= row['Total']  # Spent USDT
            else:
                cumulative += row['Total']  # Received USDT
            df_sorted.at[idx, 'Cumulative_USDT'] = cumulative
        
        # Create plotly chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_sorted['Time'],
            y=df_sorted['Cumulative_USDT'],
            mode='lines+markers',
            name='Cumulative P&L',
            line=dict(color='green' if cumulative >= 0 else 'red', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,255,0,0.1)' if cumulative >= 0 else 'rgba(255,0,0,0.1)'
        ))
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Cumulative P&L (USDT)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Filled Orders Table
        st.subheader("📋 Filled Transactions")
        
        # Format the dataframe for display
        display_df = orders_df.copy()
        display_df['Time'] = display_df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
        display_df['Amount'] = display_df['Amount'].apply(lambda x: f"{x:.4f} ETH")
        display_df['Total'] = display_df['Total'].apply(lambda x: f"${x:.2f}")
        display_df['Fee'] = display_df['Fee'].apply(lambda x: f"${x:.4f}")
        
        # Color code by side
        def highlight_side(row):
            if row['Side'] == 'BUY':
                return ['background-color: rgba(0, 255, 0, 0.1)'] * len(row)
            else:
                return ['background-color: rgba(255, 0, 0, 0.1)'] * len(row)
        
        st.dataframe(
            display_df.style.apply(highlight_side, axis=1),
            use_container_width=True,
            height=400
        )
        
    else:
        st.info("📊 **No filled orders yet!**")
        st.markdown("""
        Your bot has placed orders, but none have been filled yet.
        
        **What this means:**
        - Orders are waiting on Binance
        - They will fill when price reaches your order levels
        - Once filled, they'll appear here with P&L tracking
        
        **Current Status:**
        - Total Equity: ${:.2f}
        - This is your starting capital
        """.format(balance['total_equity']))
    
    # Auto-refresh
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Last updated
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
except Exception as e:
    st.error(f"Error connecting to Binance: {e}")
    st.info("Make sure your API keys are set in the .env file")
