
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from src.live.recursive_grid import RecursiveGrid

class RobustBacktester:
    def __init__(self, config_path='config.json', config_overrides=None):
        # Load Config
        with open(config_path, 'r') as f:
            self.base_config = json.load(f)
            
        if config_overrides:
            self.base_config.update(config_overrides)
            if 'grid_settings' in config_overrides:
                if 'grid_settings' not in self.base_config: self.base_config['grid_settings'] = {}
                self.base_config['grid_settings'].update(config_overrides['grid_settings'])
        
        # State
        self.initial_capital = self.base_config.get('capital_limit', 1000)
        self.usdt_balance = self.initial_capital
        self.eth_balance = 0.0
        self.orders = [] 
        self.trades = [] 
        self.equity_history = []
        
        # Settings
        self.fee_rate = 0.001
        self.strategy_interval = '5min' 
        self.allow_shorts = self.base_config.get('allow_shorts', False)
        if self.allow_shorts:
             print(">> SHORTING ENABLED (Futures Simulation) <<") 
        
    def load_and_prep_data(self, csv_path, days=None):
        """
        Loads 1m data, resamples to 5m for indicators, merges back.
        """
        print(f"Loading data from {csv_path}...")
        df_1m = pd.read_csv(csv_path)
        
        # Standardize columns
        if 'timestamp' not in df_1m.columns:
            # Check if columns match expected layout
            if len(df_1m.columns) >= 5:
                df_1m.rename(columns={df_1m.columns[0]: 'timestamp', df_1m.columns[1]: 'open', df_1m.columns[2]: 'high', df_1m.columns[3]: 'low', df_1m.columns[4]: 'close'}, inplace=True)
            if len(df_1m.columns) >= 6:
                df_1m.rename(columns={df_1m.columns[5]: 'volume'}, inplace=True)
            
        df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], unit='ms') if df_1m['timestamp'].dtype != 'datetime64[ns]' else pd.to_datetime(df_1m['timestamp'])
        df_1m.sort_values('timestamp', inplace=True)
        df_1m.set_index('timestamp', inplace=True)
        
        # Filter by days if requested
        if days:
            end_ts = df_1m.index[-1]
            start_ts = end_ts - pd.Timedelta(days=days)
            df_1m = df_1m[df_1m.index >= start_ts]
            print(f"Filtered data to last {days} days: {len(df_1m)} rows ({df_1m.index[0]} to {df_1m.index[-1]})")
        
        # Resample to Strategy Interval for Indicators
        print(f"Resampling to {self.strategy_interval} for indicator calculation...")
        df_strat = df_1m.resample(self.strategy_interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # --- Calculate Indicators on Strategy Data ---
        df_strat = self._calculate_indicators(df_strat)
        
        # --- Merge back to 1m ---
        # Shift 1 so that T's close is available at T+1
        df_strat_shifted = df_strat.shift(1)
        
        # Reindex to 1m and ffill
        df_indicators = df_strat_shifted.reindex(df_1m.index, method='ffill')
        
        # Combine
        self.df = df_1m.join(df_indicators[['garch_vol', 'rsi', 'adx', 'ema_slope', 'ema_l', 'ema_s']], rsuffix='_strat')
        self.df.dropna(inplace=True) 
        print(f"Data prepared: {len(self.df)} 1m rows.")
        
    def _calculate_indicators(self, df):
        # Log Returns
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        
        # GARCH(1,1)
        omega = 1e-8; alpha = 0.05; beta = 0.90
        returns = df['log_ret'].values
        sigma2 = np.zeros(len(returns))
        if len(returns) > 0:
            sigma2[0] = np.var(returns)
        for t in range(1, len(returns)):
            sigma2[t] = omega + alpha * (returns[t-1]**2) + beta * sigma2[t-1]
        df['garch_vol'] = np.sqrt(sigma2)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ADX
        up = df['high'].diff()
        down = -df['low'].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        
        # True Range
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        adx_window = 14
        tr_s = tr.rolling(window=adx_window).mean()
        p_dm_s = pd.Series(plus_dm, index=df.index).rolling(window=adx_window).mean()
        m_dm_s = pd.Series(minus_dm, index=df.index).rolling(window=adx_window).mean()
        
        p_di = 100 * (p_dm_s / tr_s)
        m_di = 100 * (m_dm_s / tr_s)
        dx = 100 * np.abs(p_di - m_di) / (p_di + m_di)
        df['adx'] = dx.rolling(window=adx_window).mean()
        
        # EMA Slope
        # Need EMA Long first (default 1000 in live, but 5m data? )
        # Live 1000m EMA on 1m data = 1000 mins.
        # Backtest 5m data -> EMA span should be 200 (200 * 5 = 1000 mins).
        ema_span = self.base_config.get('ema_settings', {}).get('long_term_span', 1000)
        # Adjust span for timeframe difference (1m vs 5m)
        # EMA Slope
        ema_span = self.base_config.get('ema_settings', {}).get('long_term_span', 1000)
        if self.strategy_interval == '5min':
            ema_span = max(1, int(ema_span / 5))
            
        # Debug
        # print(f"DEBUG: EMA Span: {ema_span}, DF Shape: {df.shape}")
        
        try:
            df['ema_l'] = df['close'].ewm(span=ema_span, adjust=False).mean()
            # EMA Short (100 in live -> 20 in 5m)
            ema_s_span = max(1, int(100 / 5)) if self.strategy_interval == '5min' else 100
            df['ema_s'] = df['close'].ewm(span=ema_s_span, adjust=False).mean()
            
            # Verify ema_l type
            if df['ema_l'] is None:
                 print("ERROR: ema_l is None!")
            
            # Use safe calculation
            ema_series = df['ema_l']
            shifted = ema_series.shift(1)
            df['ema_slope'] = (ema_series - shifted) / shifted * 100
        except Exception as e:
            print(f"DEBUG Error in Slope Calc: {e}")
            df['ema_slope'] = 0.0
            
        return df

    def run(self):
        print(f"Starting Robust Backtest on {len(self.df)} periods...")
        
        last_rebalance_ts = None
        
        # Trailing State
        # { 'active': False, 'high': 0.0, 'trailing_stop_price': 0.0 }
        self.trailing_state = {'active': False, 'high': 0.0}
        
        for ts, row in self.df.iterrows():
            curr_price = row['close']
            curr_high = row['high']
            curr_low = row['low']
            
            # --- 1. Check Fills (Buys only usually) ---
            self._check_fills(curr_low, curr_high, ts)
            
            # --- 2. Check Exits (Trailing TP - Long & Short) ---
            self._check_trailing_stop(curr_price, curr_high, curr_low, ts)
            
            # --- 3. Update Equity ---
            # Equity = USDT + (AssetQty * Price)
            # If Short (Negative Qty):
            # Equity = USDT (which holds the short proceeds) + (NegativeQty * Price)
            # Example: Sold 1 ETH @ 1000. USDT=2000 (1000 start + 1000 proceeds). ETH=-1. Price=1000.
            # Equity = 2000 + (-1 * 1000) = 1000. Correct.
            # Price drops to 900.
            # Equity = 2000 + (-1 * 900) = 1100. Profit 100. Correct.
            equity = self.usdt_balance + (self.eth_balance * curr_price)
            self.equity_history.append({'time': ts, 'equity': equity, 'price': curr_price})
            
            # --- 4. Strategy Rebalance (Buys) ---
            if ts.minute % 5 == 0:
                if last_rebalance_ts != ts:
                    # Sync with Live Bot: Anchor is EMA Long (default)
                    anchor_price = row.get('ema_l', curr_price)
                    
                    self._rebalance_grid(row, equity, anchor_price, ts)
                    last_rebalance_ts = ts
                    
        return self._generate_report()

    def _check_fills(self, low, high, timestamp):
        filled_indices = []
        for i, order in enumerate(self.orders):
            if order['side'] == 'buy':
                if low <= order['price']:
                    # Fill
                    cost = order['qty'] * order['price']
                    if self.usdt_balance >= cost * 0.999: 
                        self.usdt_balance -= cost
                        # Update Avg Entry
                        old_qty = self.eth_balance
                        new_qty = order['qty'] * (1 - self.fee_rate)
                        total_qty = old_qty + new_qty
                        
                        # Avg Entry = (OldCost + NewCost) / TotalQty? 
                        # Tracking cost basis properly
                        # Avg Entry = (OldQty * OldAvg + NewQty * NewPrice) / TotalQty
                        if old_qty + new_qty > 0:
                            prev_entry = getattr(self, 'avg_entry', 0.0)
                            # If previous balance was zero or near zero, use new price
                            if old_qty < 0.0001:
                                self.avg_entry = order['price'] # First buy
                            else:
                                total_val = (prev_entry * old_qty) + cost 
                                self.avg_entry = total_val / total_qty
                        else:
                            self.avg_entry = 0.0
                        
                        self.eth_balance = total_qty
                        
                        self.trades.append({
                            'time': timestamp, 'side': 'buy', 
                            'price': order['price'], 'qty': order['qty'],
                            'cost': cost
                        })
                        filled_indices.append(i)
            
            elif order['side'] == 'sell' and self.allow_shorts:
                 if high >= order['price']:
                    # Fill Short
                    cost = order['qty'] * order['price']
                    revenue = cost # We get USDT
                    
                    # Margin Check (Simple: Can't have position > balance * leverage?)
                    # Let's assume 1x leverage cap for simplicity, or just allow it if equity > 0
                    current_equity = self.usdt_balance + (self.eth_balance * order['price'])
                    if current_equity > 0:
                        self.usdt_balance += revenue * (1 - self.fee_rate)
                        
                        # Update Avg Entry Short
                        # If we are already short (eth_balance < 0)
                        old_qty_abs = abs(self.eth_balance) if self.eth_balance < 0 else 0
                        new_qty = order['qty']
                        total_qty = old_qty_abs + new_qty
                        
                        prev_entry = getattr(self, 'avg_entry_short', 0.0)
                        total_revenue_sq = (prev_entry * old_qty_abs) + revenue
                        self.avg_entry_short = total_revenue_sq / total_qty
                        
                        self.eth_balance -= new_qty # Become more negative
                        
                        self.trades.append({
                            'time': timestamp, 'side': 'sell_short', 
                            'price': order['price'], 'qty': order['qty'],
                            'revenue': revenue
                        })
                        filled_indices.append(i)
                        
        for i in sorted(filled_indices, reverse=True):
            del self.orders[i]

    def _check_trailing_stop(self, curr_price, high, low, timestamp):
        if self.eth_balance < 0.0001:
            self.trailing_state = {'active': False, 'high': 0.0}
            self.avg_entry = 0.0
            return

        trailing_conf = self.base_config.get('trailing_settings', {})
        trail_act = trailing_conf.get('close_trailing_threshold_pct', 0.03) # Increased to 3%
        trail_cb = trailing_conf.get('close_trailing_retracement_pct', 0.01) # Increased to 1%
        
        # Enable check
        if trail_act <= 0: return
        
        # PnL Check
        avg_entry = getattr(self, 'avg_entry', 0.0)
        if avg_entry <= 0: return # Should not happen if balance > 0
        
        pnl_pct = (curr_price - avg_entry) / avg_entry
        
        # 1. Activation
        if not self.trailing_state['active']:
            if pnl_pct > trail_act:
                self.trailing_state['active'] = True
                self.trailing_state['high'] = curr_price
                # print(f"DEBUG: Trailing Activated at {curr_price} (Entry: {avg_entry})")
        
        # 2. Update / Trigger
        if self.trailing_state['active']:
            if curr_price > self.trailing_state['high']:
                self.trailing_state['high'] = curr_price
            
            stop_price = self.trailing_state['high'] * (1 - trail_cb)
            
            # Use 'low' to simulate stop hit? 
            # Or just close ('market sell') if current close < stop?
            # Being conservative: if 'low' hit stop, we exit at stop_price 
            
            if low <= stop_price:
                # Sell All
                revenue = self.eth_balance * stop_price
                self.usdt_balance += revenue * (1 - self.fee_rate)
                
                self.trades.append({
                    'time': timestamp, 'side': 'sell',
                    'price': stop_price, 'qty': self.eth_balance,
                    'revenue': revenue
                })
                
                self.orders = [] # Clear buys? Passivbot keeps buys usually, but let's clear to reset grid
                
                # RESET STATE
                self.eth_balance = 0.0 # Floating point dust?
                self.avg_entry = 0.0
                self.trailing_state = {'active': False, 'high': 0.0}
                
        # --- SHORT TRAILING STOP ---
        if self.allow_shorts and self.eth_balance < -0.0001:
             avg_entry = getattr(self, 'avg_entry_short', 0.0)
             if avg_entry <= 0: return # Should not happen

             # PnL for Short: (Entry - Price) / Entry
             # If Price < Entry, we make money.
             pnl_pct = (avg_entry - curr_price) / avg_entry
             
             # State for Short
             if not hasattr(self, 'trailing_state_short'):
                 self.trailing_state_short = {'active': False, 'low': float('inf')}
                 
             # 1. Activation
             if not self.trailing_state_short['active']:
                 if pnl_pct > trail_act:
                     self.trailing_state_short['active'] = True
                     self.trailing_state_short['low'] = curr_price
                     
             # 2. Update / Trigger
             if self.trailing_state_short['active']:
                 if curr_price < self.trailing_state_short['low']:
                     self.trailing_state_short['low'] = curr_price
                 
                 # Trailing Buy Price (Callback upwards)
                 buy_price = self.trailing_state_short['low'] * (1 + trail_cb)
                 
                 if high >= buy_price:
                     # COVER (Buy back)
                     qty_to_cover = abs(self.eth_balance)
                     cost = qty_to_cover * buy_price
                     
                     self.usdt_balance -= cost * (1 + self.fee_rate) # Pay cost + fee
                     self.eth_balance = 0.0
                     self.avg_entry_short = 0.0
                     self.trailing_state_short = {'active': False, 'low': float('inf')}
                     
                     self.trades.append({
                        'time': timestamp, 'side': 'cover_short',
                        'price': buy_price, 'qty': qty_to_cover,
                        'cost': cost
                     })
                     self.orders = []
                
    def _rebalance_grid(self, row, total_equity, center_price, ts):
        # Clear existing BUYS only? 
        # Typically we cancel old grid buys and replace
        self.orders = []
        
        grid_conf = self.base_config.get('grid_settings', {})
        n_orders = grid_conf.get('max_orders', 10)
        span = grid_conf.get('grid_span', 0.15) 
        weight = grid_conf.get('grid_spacing_weight', 1.2)
        qty_mult = grid_conf.get('qty_step_multiplier', 1.4)
        
        # Wallet Exposure: Live bot prioritizes 'core_wallet_exposure' (0.35) for main symbols.
        # Fallback to wallet_exposure_limit if not set.
        we_limit = self.base_config.get('core_wallet_exposure', self.base_config.get('wallet_exposure_limit', 1.0))
        
        # --- SMART GRID LOGIC ---
        smart_conf = self.base_config.get('smart_grid', {})
        if smart_conf.get('enabled', False):
            # Indicators are in row
            garch_vol = row.get('garch_vol', 0)
            adx = row.get('adx', 0)
            rsi = row.get('rsi', 50)
            
            if garch_vol > 0:
                # --- RSI PANIC SELL ---
                rsi_sell_thresh = smart_conf.get('rsi_sell_threshold', 90)
                if rsi > rsi_sell_thresh:
                    # Cancel all orders (clear grid)
                    self.orders = []
                    
                    # Sell Held Assets
                    if self.eth_balance > 0:
                        revenue = self.eth_balance * row['close'] * 0.995 # Slippage?
                        self.usdt_balance += revenue * (1 - self.fee_rate)
                        
                        self.trades.append({
                            'time': row.name, 'side': 'panic_sell',
                            'price': row['close'], 'qty': self.eth_balance,
                            'revenue': revenue
                        })
                        
                        print(f"[{row.name}] *** RSI PANIC SELL *** RSI={rsi:.1f} Price={row['close']:.2f} Qty={self.eth_balance:.4f}")
                        
                        self.eth_balance = 0.0
                        self.avg_entry = 0.0
                        self.trailing_state = {'active': False, 'high': 0.0}
                        return
                
                vol_scale = smart_conf.get('volatility_scale', 1.0)
                
                # Trend Scale
                trend_scale = 1.0
                if adx > 40:
                    trend_scale = 1.0 + min((adx - 40) / 40, 0.5)
                vol_scale *= trend_scale
                
                # Safety Factor
                safety_factor = 4.0
                if adx < 30: safety_factor = 2.5
                elif adx < 60: safety_factor = 2.5 + 1.5 * ((adx - 30) / 30)
                
                daily_vol = garch_vol * 17.0
                min_span = daily_vol * safety_factor * vol_scale
                
                if not self.base_config.get('aggressive_mode', False):
                    if min_span > span:
                        span = min_span
                        
        # Grid Init
        allow_shorts = True 
            
        # Trend Bias
        bias = smart_conf.get('trend_bias_strength', 0.5)
        if rsi < 30:
            skew = (30 - rsi) / 100 * bias
            center_price *= (1 + skew)
        elif rsi > 70:
            skew = (rsi - 70) / 100 * bias
            center_price *= (1 - skew)

            center_price_adaptive = center_price 
            
            # --- ADAPTIVE ANCHORING ---
            # If ADX > 25, use EMA Short
            if row.get('adx', 0) > 25:
                # center_price was based on EMA Long? 
                # Actually _rebalance_grid receives 'center_price' which is just 'curr_price' passed in??
                # Wait, rebalance_grid(row, equity, curr_price)...
                # Ah, we need to pass the EMA L/S or calculate it here.
                # Actually, the caller passed 'curr_price' as center_price? 
                # Let's check call site: self._rebalance_grid(row, equity, curr_price)
                # NO. The bot usually grids around EMA. 
                # Standard bot logic: neutral_price = ema_l.
                
                # Backtester Logic Flaw Check:
                # Caller: _rebalance_grid(row, equity, curr_price)
                # This means it was gridding around CURRENT PRICE, not EMA?
                # If so, that explains why it was "buying the dip" too fast?
                # Or maybe it was intended to act as 'neutral'?
                
                # Let's fix this to match LIVE bot.
                # Live bot: neutral_price = ema_l
                
                ema_l = row.get('ema_l', center_price)
                ema_s = row.get('ema_s', center_price)
                
                center_price_adaptive = ema_l # Default to EMA Long
                
                if row.get('adx', 0) > 25:
                     center_price_adaptive = ema_s
                     # Debug
                     # print(f"DEBUG: Adaptive Trigger! ADX={row.get('adx',0):.1f} Price={center_price:.2f} -> {center_price_adaptive:.2f}")
            
            # Use the adaptive price
            center_price = center_price_adaptive
        regime_conf = self.base_config.get('regime_filter', {})
        allow_buys = True
        if regime_conf.get('enabled', False):
            slope = row.get('ema_slope', 0)
            adx_thresh = regime_conf.get('adx_threshold', 25)
            slope_thresh = regime_conf.get('slope_threshold', 0.05)
            
            # Strong DOWNTREND -> Block Buys
            if adx > adx_thresh and slope < -slope_thresh:
                if regime_conf.get('bottom_trend_filter', True):
                    allow_buys = False
                    
            # Strong UPTREND -> Block Shorts
            if adx > adx_thresh and slope > slope_thresh:
                 if regime_conf.get('top_trend_filter', True):
                     allow_shorts = False

        if not allow_buys and not (self.allow_shorts and allow_shorts):
            self.orders = []
            return

        # --- EXECUTE ---
        grid_params = {
            'max_orders': n_orders,
            'grid_span': span,
            'wallet_exposure_limit': we_limit,
            'grid_spacing_weight': weight,
            'qty_step_multiplier': qty_mult
        }
        
        # Buys
        # RecursiveGrid calc uses Total Equity
        ideal_buys = RecursiveGrid.calc_recursive_grid(center_price, total_equity, grid_params, side='long')
        
        # Filter: only buys below current price (margin usually handled by calc_recursive_grid logic but purely)
        # We append them
        if allow_buys:
             for order in ideal_buys:
                 # Filter: Only buy below current price (Standard Grid)
                 if order['price'] < row['close']:
                     self.orders.append({'side': 'buy', 'price': order['price'], 'qty': order['qty_base']})
            
        # Short Grid (If enabled) used inverted prices (above)
        if allow_shorts:
             ideal_shorts = RecursiveGrid.calc_recursive_grid(center_price, total_equity, grid_params, side='short')
             if ideal_shorts:
                 if self.allow_shorts:
                     for order in ideal_shorts:
                         self.orders.append({'side': 'sell', 'price': order['price'], 'qty': order['qty_base']})
                 else:
                     # AI BLOCKED SHORT
                     if row.get('rsi', 50) > 80: # Just log the egregious ones to save space
                         print(f"[{ts}] 🛡️ AI GUARDRAIL: Blocked {len(ideal_shorts)} Short Orders (RSI={row.get('rsi',0):.1f})")

    def _generate_report(self):
        if not self.equity_history:
            return {}
            
        df = pd.DataFrame(self.equity_history)
        df.set_index('time', inplace=True)
        
        initial = float(self.initial_capital)
        final = float(df['equity'].iloc[-1])
        profit = final - initial
        ret_pct = (profit / initial) * 100
        
        # Max DD
        roll_max = df['equity'].cummax()
        dd = (df['equity'] - roll_max) / roll_max
        max_dd = dd.min() * 100
        
        # Daily Return Stats
        daily_resample = df['equity'].resample('D').last().pct_change().dropna()
        sharpe = 0
        if daily_resample.std() > 0:
            sharpe = (daily_resample.mean() / daily_resample.std()) * np.sqrt(365)
            
        # Win Rate
        win_rate = 0
        if self.trades:
            # Approximate win rate from trades? Grid bots buy/sell continuously. 
            # Realized PnL is hard to track without FIFO queue.
            # We'll just report Total Trades.
            pass
            
        # Fees Paid
        total_fees = 0.0
        for t in self.trades:
            if 'cost' in t: total_fees += t['cost'] * self.fee_rate
            if 'revenue' in t: total_fees += t['revenue'] * self.fee_rate # Fee on revenue for sell
            
        # Buy & Hold
        market_ret = 0
        if not df.empty:
            start_price = df['price'].iloc[0]
            end_price = df['price'].iloc[-1]
            market_ret = ((end_price - start_price) / start_price) * 100
            
        print("\n=== Robust Backtest Results ===")
        print(f"Period: {df.index[0]} to {df.index[-1]}")
        print(f"Strategy Interval: {self.strategy_interval}")
        print(f"Final Equity: ${final:.2f}")
        print(f"Total Return: {ret_pct:.2f}%")
        print(f"Buy & Hold:   {market_ret:.2f}%")
        print(f"Max Drawdown: {max_dd:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Total Trades: {len(self.trades)}")
        print(f"Total Fees Paid: ${total_fees:.2f}")
        
        return {
            "total_return_pct": ret_pct,
            "final_equity": final,
            "max_drawdown_pct": max_dd,
            "sharpe_ratio": sharpe,
            "total_trades": len(self.trades),
            "total_fees": total_fees
        }
