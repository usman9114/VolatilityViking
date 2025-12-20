import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()

    # Introduction
    intro_md = """# 🎓 AI Trading Masterclass: Inside the Black Box
    
**Objective**: Demystify how our `ETH-AI` bot predicts the future. 
We will peel back the layers of the Neural Network and inspect the raw data it sees.

**What we will explore:**
1.  **The Inputs (411 Features)**: What exactly is the bot looking at?
2.  **On-Chain Truths**: Does High Gas really mean a Top?
3.  **The Signal**: Visualizing the "Buy" moments.
4.  **Strategy Performance**: Inspecting the Equity Curve.
"""

    # Imports
    import_code = """import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Set Plotly theme to dark
import plotly.io as pio
pio.templates.default = "plotly_dark"
"""

    # Load Data
    load_md = """## 1. Loading the Brain's Memory 🧠
We load the engineered dataset (`eth_4h_features.parquet`). This contains the **411 features** for every 4-hour candle.
"""
    load_code = """df = pd.read_parquet('../data/processed/eth_4h_features.parquet')
print(f"Dataset Shape: {df.shape}")
print(f"Total Features: {len(df.columns)}")
display(df.tail(3))
"""

    # Feature Inspection
    feat_md = """## 2. The 411 Inputs: A Breakdown
The model doesn't just see Price. It sees a massive vector. Let's break it down.
"""
    feat_code = """# 1. Technicals (Price Action)
tech_cols = ['open', 'close', 'RSI_14', 'MACD_12_26_9', 'ATR_14', 'fear_greed_index']
print("Technical Indicators (Sample):", tech_cols)

# 2. On-Chain (Gas) -> Note: These were merged into the dataset earlier
# Let's verify if gas columns exist or if we need to visualize from raw
if 'base_fee_gwei' in df.columns:
    print("On-Chain Data Found!")
else:
    print("Loading On-Chain Raw Data for visualization...")
    gas_df = pd.read_parquet('../data/raw/onchain_gas.parquet')
    # Join for viz
    df = df.join(gas_df, how='left')

# 3. News Embeddings (The Hidden 384)
emb_cols = [c for c in df.columns if str(c).startswith('emb_')]
print(f"Number of 'Hidden' News Embedding Features: {len(emb_cols)}")
"""

    # Visualizing On-Chain
    gas_md = """## 3. On-Chain Alpha: Gas Fees vs Price ⛽
**Hypothesis**: High Gas Fees (Network Congestion) mark "Euphoria" tops. Low Gas Fees mark "Capitulation" bottoms.
"""
    gas_code = """# Filter last 2 years for clearer view
viz_df = df.loc['2023':].copy()

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.03, subplot_titles=('ETH Price', 'Ethereum Base Fee (Gwei)'),
                    row_heights=[0.7, 0.3])

# Price
fig.add_trace(go.Candlestick(x=viz_df.index,
                             open=viz_df['open'], high=viz_df['high'],
                             low=viz_df['low'], close=viz_df['close'],
                             name='ETH'), row=1, col=1)

# Gas
fig.add_trace(go.Scatter(x=viz_df.index, y=viz_df['base_fee_gwei'], 
                         line=dict(color='orange', width=2), name='Base Fee'), row=2, col=1)

fig.update_layout(title='Does Gas Predict Tops?', height=800, xaxis_rangeslider_visible=False)
fig.show()
"""

    # Model Predictions
    pred_md = """## 4. The Prediction: Accuracy Check 🎯
Let's look at the Backtest Results.
"""
    pred_code = """res_df = pd.read_csv('../data/processed/walk_forward_results_with_news.csv', index_col='Datetime', parse_dates=True)

# Plot last 3 months
subset = res_df.iloc[-500:]

fig = go.Figure()

# Actual Return (Cumulative for visualization sake, or just the raw signal)
# 'Ensemble' column is the Predicted Log Return for the NEXT period.
# 'Actual' column is the Real Log Return that happened.

fig.add_trace(go.Scatter(x=subset.index, y=subset['Actual'],
                         mode='lines', name='Actual Return (4h)', line=dict(color='gray', width=1, dash='dot')))

fig.add_trace(go.Scatter(x=subset.index, y=subset['Ensemble'],
                         mode='lines', name='AI Predicted Return', line=dict(color='#00ff00', width=2)))

fig.update_layout(title='AI Prediction vs Reality (Last 500 Candles)', 
                  yaxis_title='Log Return', height=600)
fig.show()
"""

    # Equity Curve
    equity_md = """## 5. The Result: Equity Curve 🚀
How much money did this strategy actually make?
"""
    equity_code = """# We reload the simulated trades log if available, or just recalculate
# For now, let's load the saved equity curve image or just run the backtest logic briefly
sys_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
if sys_path not in sys.path:
    sys.path.append(sys_path)
    
from src.backtest.backtester import Backtester

predictions = pd.read_csv('../data/processed/walk_forward_results_with_news.csv')
prices = pd.read_parquet('../data/raw/eth_4h.parquet')

bt = Backtester(predictions, prices)
res = bt.calculate_metrics()

fig = go.Figure()
fig.add_trace(go.Scatter(x=res.index, y=res['strategy_equity'], 
                         mode='lines', name='AI Strategy', line=dict(color='#00ff00', width=3)))
fig.add_trace(go.Scatter(x=res.index, y=res['market_equity'], 
                         mode='lines', name='Buy & Hold', line=dict(color='gray', width=2)))

fig.update_layout(title='Strategy Performance (Log Scale)', yaxis_type="log", height=600)
fig.show()
"""

    # Add cells
    nb['cells'] = [
        nbf.v4.new_markdown_cell(intro_md),
        nbf.v4.new_code_cell(import_code),
        nbf.v4.new_markdown_cell(load_md),
        nbf.v4.new_code_cell(load_code),
        nbf.v4.new_markdown_cell(feat_md),
        nbf.v4.new_code_cell(feat_code),
        nbf.v4.new_markdown_cell(gas_md),
        nbf.v4.new_code_cell(gas_code),
        nbf.v4.new_markdown_cell(pred_md),
        nbf.v4.new_code_cell(pred_code),
        nbf.v4.new_markdown_cell(equity_md),
        nbf.v4.new_code_cell(equity_code)
    ]

    # Save
    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/Masterclass_Analysis.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("Notebook created at notebooks/Masterclass_Analysis.ipynb")

if __name__ == "__main__":
    create_notebook()
