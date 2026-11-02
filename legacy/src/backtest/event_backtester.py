import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

class EventBacktester:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.initial_balance = self.config.get('capital_limit', 1000)
        self.usdt_balance = self.initial_balance
        self.eth_balance = 0.0
        self.equity_history = []
        self.orders = []
        self.trades = []
        
        # Grid settings
        self.grid_settings = self.config.get('grid_settings', {})
        self.fee_rate = 0.001 # 0.1% fee
        
    def load_data(self, csv_path):
        """ Load 1m data for tick simulation """
        self.df = pd.read_csv(csv_path)
        if 'timestamp' not in self.df.columns:
            # Assuming standard OHLCV without header or different names
            self.df.rename(columns={'0': 'timestamp', '1': 'open', '2': 'high', '3': 'low', '4': 'close'}, inplace=True)
            
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
        self.df.sort_values('timestamp', inplace=True)
        
        # Calculate Indicators for Smart Grid
        # RSI
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        self.df['atr'] = true_range.rolling(window=14).mean()
        
        self.df.fillna(0, inplace=True)
        
    def run(self):
        """ Run the simulation """
        print(f"Starting Backtest with ${self.initial_balance}...")
        
        # Simplified simulation: Iterating every minute
        # For a real "tick" backtest, we'd need trade data, but 1m OHLCV is decent for grid if we assume High/Low hits.
        
        position = 0.0
        avg_entry = 0.0
        
        for i, row in self.df.iterrows():
            price = row['close']
            high = row['high']
            low = row['low']
            
            # 1. Check Fills (Limit Orders)
            # For each open order, check if High/Low touched it
            filled_indices = []
            for idx, order in enumerate(self.orders):
                if order['side'] == 'buy':
                    if low <= order['price']:
                         # FILL BUY
                         cost = order['qty'] * order['price']
                         if self.usdt_balance >= cost:
                             self.usdt_balance -= cost
                             self.eth_balance += order['qty'] * (1 - self.fee_rate)
                             self.trades.append({'time': row['timestamp'], 'side': 'buy', 'price': order['price'], 'qty': order['qty']})
                             filled_indices.append(idx)
                elif order['side'] == 'sell':
                    if high >= order['price']:
                        # FILL SELL
                        revenue = order['qty'] * order['price']
                        if self.eth_balance >= order['qty']:
                            self.eth_balance -= order['qty']
                            self.usdt_balance += revenue * (1 - self.fee_rate)
                            self.trades.append({'time': row['timestamp'], 'side': 'sell', 'price': order['price'], 'qty': order['qty']})
                            filled_indices.append(idx)
                            
            # Remove filled orders (in reverse to avoid index shift)
            for idx in sorted(filled_indices, reverse=True):
                del self.orders[idx]
                
            # 2. Strategy Logic (Recursive Grid)
            # Calculate Equity
            equity = self.usdt_balance + (self.eth_balance * price)
            self.equity_history.append({'time': row['timestamp'], 'equity': equity, 'price': price})
            
            # Rebalance Grid (Simplified: Every 60 mins or on large div)
            if i % 60 == 0: 
                # Pass current row data for indicators
                self.rebalance_grid(price, equity, row)
                
        # End
        final_equity = self.equity_history[-1]['equity']
        print(f"Final Equity: ${final_equity:.2f} ({(final_equity - self.initial_balance)/self.initial_balance*100:.2f}%)")
        return self.equity_history

    def rebalance_grid(self, center_price, total_equity, row_data=None):
        """ Simplified version of BinanceBot.calculate_ideal_grid + diff """
        # Clear existing orders (Simplification: In backtest we can cancel all and replace)
        self.orders = []
        
        n_orders = self.grid_settings.get('max_orders', 10)
        span = self.grid_settings.get('grid_span', 0.30)
        weight = self.grid_settings.get('grid_spacing_weight', 1.2)
        qty_mult = self.grid_settings.get('qty_step_multiplier', 1.4)
        we_limit = self.config.get('wallet_exposure_limit', 1.0)
        
        # --- SMART GRID LOGIC ---
        smart_conf = self.config.get('smart_grid', {})
        if smart_conf.get('enabled', False) and row_data is not None:
            # 1. Volatility
            atr = row_data.get('atr', 0)
            if atr > 0:
                vol_scale = smart_conf.get('volatility_scale', 1.0)
                min_span = (atr / center_price) * 4 * vol_scale
                if min_span > span:
                    span = min_span
            
            # 2. Trend Bias
            rsi = row_data.get('rsi', 50)
            bias = smart_conf.get('trend_bias_strength', 0.5)
            if rsi < 30:
                skew = (30 - rsi) / 100 * bias
                center_price *= (1 + skew)
            elif rsi > 70:
                skew = (rsi - 70) / 100 * bias
                center_price *= (1 - skew)
        # ------------------------
        
        # Buys
        buy_prices = self._calc_geometric_prices(center_price, n_orders, weight, span, 'down')
        buy_alloc = total_equity * we_limit
        buy_qtys_usdt = self._calc_martingale_values(buy_alloc, n_orders, qty_mult)
        
        for p, q_usdt in zip(buy_prices, buy_qtys_usdt):
            qty = q_usdt / p
            if self.usdt_balance >= (qty * p):
                self.orders.append({'side': 'buy', 'price': p, 'qty': qty})
                
        # Sells
        if self.eth_balance > 0.001:
            sell_prices = self._calc_geometric_prices(center_price, n_orders, weight, span, 'up')
            # Distribute inventory
            sell_qtys_values = self._calc_martingale_values(self.eth_balance, n_orders, qty_mult) # This treats eth_balance as the total to distribute
            # Wait, _calc_martingale_values returns fractions of the total.
            # So if we pass eth_balance, we get ETH amounts.
            
            for p, q_eth in zip(sell_prices, sell_qtys_values):
                self.orders.append({'side': 'sell', 'price': p, 'qty': q_eth})

    def _calc_geometric_prices(self, center, n, weight, span, direction):
        if n < 1: return []
        if weight == 1.0:
            d0 = span / n
        else:
            d0 = span * (weight - 1) / (pow(weight, n) - 1)
        
        prices = []
        cum_dist = 0
        for i in range(n):
            dist = d0 * pow(weight, i)
            cum_dist += dist
            if direction == 'down':
                prices.append(center * (1 - cum_dist))
            else:
                prices.append(center * (1 + cum_dist))
        return prices

    def _calc_martingale_values(self, total, n, mult):
        if n < 1: return []
        if mult == 1.0:
            q0 = total / n
        else:
            q0 = total * (mult - 1) / (pow(mult, n) - 1)
        
        qtys = []
        for i in range(n):
            q = q0 * pow(mult, i)
            qtys.append(q)
        return qtys
