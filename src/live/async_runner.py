"""
Async Multi-Symbol Bot Runner - Passivbot Style
Single event loop, all symbols processed in parallel.
"""
import asyncio
import ccxt.async_support as ccxt_async
import json
import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from src.live.state_manager import StateManager


# Setup logging with profile support
def setup_logging(profile_name="default"):
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"logs/{profile_name}_{timestamp}.log"
    
    logging.getLogger().handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_filename}")
    return logger

logger = logging.getLogger(__name__)


class AsyncMultiSymbolBot:
    """
    Passivbot-style async bot that handles ALL symbols in a single event loop.
    Much faster than thread-per-symbol approach.
    """
    
    def __init__(self, config_path='config.json'):
        # load_dotenv() - Handled in main.py
        self.config_path = config_path
        self.config = self.load_config()
        
        # Exchange connections
        self.exchange = None
        
        # State per symbol
        self.positions = {}
        self.open_orders = {}
        self.last_prices = {}
        self.ema_long = {}
        self.ema_short = {}
        self.atr = {}
        self.rsi = {}
        self.adx = {}
        self.ema_slope = {}
        self.garch_vol = {}
        
        # State Manager (Cost Basis Tracking)
        self.state_manager = StateManager()
        
        # Trailing Stop State
        self.trailing_high_water_mark = {}  # Highest price since activation
        
        # Wallet tracking (Passivbot style)
        self.total_equity = 0
        self.total_wallet_usdt = 0
        self.free_wallet_usdt = 0
        
        # Kill-Switch & Drawdown Protection
        self.peak_equity = 0
        self.max_drawdown_pct = self.config.get('max_drawdown_pct', 0.20)  # 20% default
        self.kill_switch_active = False
        
        # Control
        self.stop_signal = False
        self.last_config_reload = 0
        
    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Config load error: {e}")
            return {}
    
    async def init_exchange(self):
        """Initialize async exchange connection."""
        live_mode = self.config.get('live_mode', False)
        
        if live_mode:
            api_key = os.getenv('BINANCE_API_KEY')
            secret = os.getenv('BINANCE_SECRET_KEY') or os.getenv('BINANCE_SECRET')
        else:
            api_key = os.getenv('BINANCE_TESTNET_API_KEY')
            secret = os.getenv('BINANCE_TESTNET_SECRET_KEY')
        
        
        futures_enabled = self.config.get('futures_settings', {}).get('enabled', False)
        default_type = 'future' if futures_enabled else 'spot'
        logger.info(f"Exchange type: {default_type} (futures_enabled={futures_enabled})")
        
        self.exchange = ccxt_async.binance({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {'defaultType': default_type}
        })
        
        if not live_mode:
            self.exchange.set_sandbox_mode(True)
            logger.info("Running in TESTNET mode")
        else:
            logger.warning("Running in LIVE mode!")
    
    async def close_exchange(self):
        if self.exchange:
            await self.exchange.close()
    
    def get_active_symbols(self):
        """Get all active symbols from config."""
        core = self.config.get('symbols', [])
        
        if self.config.get('forager', {}).get('enabled', False):
            forager_syms = self.config.get('forager', {}).get('approved_symbols', [])
            max_pos = self.config.get('forager', {}).get('max_positions', 3)
            extra = [s for s in forager_syms if s not in core][:max_pos]
            return core + extra
        
        return core if core else ['ETH/USDT']
    

    
    async def fetch_all_data(self, symbols):
        """
        Fetch data for ALL symbols in parallel using asyncio.gather.
        This is the key performance optimization - Passivbot style.
        """
        async def fetch_symbol_data(symbol):
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                return symbol, {
                    'price': ticker['last'],
                    'bid': ticker['bid'],
                    'ask': ticker['ask'],
                    'volume': ticker['quoteVolume']
                }
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                return symbol, None
        
        # Fetch ALL symbols in parallel
        results = await asyncio.gather(*[fetch_symbol_data(s) for s in symbols])
        
        for symbol, data in results:
            if data:
                self.last_prices[symbol] = data
        
        return len([r for r in results if r[1] is not None])
    
    async def fetch_all_positions(self, symbols):
        """
        Fetch positions and wallet balance for all symbols.
        CRITICAL: This provides fresh data before ANY order placement.
        """
        try:
            # Check if Futures or Spot
            is_future = self.exchange.options['defaultType'] == 'future'
            
            if is_future:
                # FUTURES: Fetch Positions explicitly
                positions = await self.exchange.fetch_positions(symbols)
                # Map to local dict
                pos_map = {p['symbol']: p for p in positions}
                
                # Fetch Balance (Margin)
                balance = await self.exchange.fetch_balance()
                self.total_wallet_usdt = balance.get('USDT', {}).get('total', 0)
                self.free_wallet_usdt = balance.get('USDT', {}).get('free', 0)
                
                # Calculate Equity (Wallet + Unrealized PnL)
                # Note: fetch_balance 'total' in futures usually includes unrealized PnL on Binance? 
                # Actually usually 'total' is margin. simple logic:
                self.total_equity = self.total_wallet_usdt # Approximate
                
                for symbol in symbols:
                    p = pos_map.get(symbol, {})
                    amount = float(p.get('contracts', 0) or 0)
                    side = p.get('side', 'long') # 'long' or 'short'
                    if side == 'short': amount = -amount
                    
                    price = self.last_prices.get(symbol, {}).get('price', 0)
                    value_usdt = abs(amount * price)
                    
                    self.positions[symbol] = {
                        'amount': amount,
                        'free': amount, # futures don't lock amount in orders usually, but let's assume all free for now
                        'value_usdt': value_usdt,
                        'price': price,
                        'unrealizedPnL': float(p.get('unrealizedPnl', 0) or 0),
                        'leverage': p.get('leverage', 1)
                    }
                    
                    # Add PnL to Equity for accurate checking
                    # (Assuming fetch_balance total is Wallet Balance)
                    # self.total_equity += p.get('unrealizedPnl', 0) - handled by exchange usually?
            else:
                # SPOT: Use Balance
                balance = await self.exchange.fetch_balance()
                
                # Total wallet in USDT (quote currency)
                self.total_wallet_usdt = balance.get('USDT', {}).get('total', 0) or 0
                self.free_wallet_usdt = balance.get('USDT', {}).get('free', 0) or 0
                
                # Calculate value of all holdings
                self.total_equity = self.total_wallet_usdt
                
                for symbol in symbols:
                    base = symbol.split('/')[0]
                    amount = balance.get(base, {}).get('total', 0) or 0
                    free = balance.get(base, {}).get('free', 0) or 0
                    
                    # Get current price for value calculation
                    price = self.last_prices.get(symbol, {}).get('price', 0)
                    value_usdt = amount * price if price else 0
                    
                    self.positions[symbol] = {
                        'amount': amount,
                        'free': free,
                        'value_usdt': value_usdt,
                        'price': price
                    }
                    
                    self.total_equity += value_usdt
            
            # Calculate wallet exposure per symbol (Shared logic)
            for symbol in symbols:
                pos = self.positions.get(symbol, {})
                if self.total_equity > 0:
                    pos['wallet_exposure'] = pos.get('value_usdt', 0) / self.total_equity
                else:
                    pos['wallet_exposure'] = 0
            
            return True
            
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return False
    
    async def refresh_balance_before_order(self):
        """
        ALWAYS call this before placing an order.
        Ensures we have up-to-date balance to prevent over-exposure.
        """
        try:
            balance = await self.exchange.fetch_balance()
            self.total_wallet_usdt = balance.get('USDT', {}).get('total', 0) or 0
            self.free_wallet_usdt = balance.get('USDT', {}).get('free', 0) or 0
            
            logger.debug(f"[Balance Refresh] USDT: {self.total_wallet_usdt:.2f} (free: {self.free_wallet_usdt:.2f})")
            return True
        except Exception as e:
            logger.error(f"Balance refresh failed: {e}")
            return False
    
    def check_wallet_exposure_limit(self, symbol, order_value_usdt):
        """
        Check if placing this order would exceed wallet exposure limits.
        Passivbot-style risk management.
        """
        total_wel = self.config.get('total_wallet_exposure_limit', 1.0)
        n_positions = len(self.get_active_symbols())
        per_symbol_wel = total_wel / max(n_positions, 1)
        
        current_exposure = self.positions.get(symbol, {}).get('wallet_exposure', 0)
        new_exposure = (self.positions.get(symbol, {}).get('value_usdt', 0) + order_value_usdt) / max(self.total_equity, 1)
        
        if new_exposure > per_symbol_wel * 1.1:  # 10% buffer
            logger.warning(f"[{symbol}] Order blocked: exposure {new_exposure:.2%} > limit {per_symbol_wel:.2%}")
            return False
        
        return True
    
    def check_kill_switch(self):
        """
        Global Kill-Switch - Passivbot Safety Feature
        Triggers when drawdown exceeds max_drawdown_pct
        OR when 'Red Phone' lock file is present.
        """
        # 1. Red Phone (Sentinel/Manual)
        if os.path.exists("signals/kill_switch.lock"):
            logger.critical("=" * 60)
            logger.critical(f"🚨 RED PHONE ACTIVATED: signals/kill_switch.lock found! 🚨")
            logger.critical("HALTING TRADING IMMEDIATELY.")
            logger.critical("=" * 60)
            self.kill_switch_active = True
            return True

        # 2. Drawdown
        if self.peak_equity <= 0:
            return False
        
        current_drawdown = (self.peak_equity - self.total_equity) / self.peak_equity
        
        if current_drawdown > self.max_drawdown_pct:
            logger.critical("=" * 60)
            logger.critical(f"🚨 KILL SWITCH ACTIVATED 🚨")
            logger.critical(f"Drawdown: {current_drawdown:.1%} > Max: {self.max_drawdown_pct:.1%}")
            logger.critical(f"Peak: ${self.peak_equity:.2f} → Current: ${self.total_equity:.2f}")
            logger.critical("=" * 60)
            self.kill_switch_active = True
            return True
        
        return False
    
    async def cancel_all_orders(self):
        """Cancel all open orders - used on kill-switch activation."""
        try:
            symbols = self.get_active_symbols()
            for symbol in symbols:
                orders = self.open_orders.get(symbol, [])
                for order in orders:
                    try:
                        await self.exchange.cancel_order(order['id'], symbol)
                        logger.info(f"Cancelled order {order['id']} on {symbol}")
                    except Exception as e:
                        logger.error(f"Failed to cancel {order['id']}: {e}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            return False

    async def cancel_order(self, symbol, order_id):
        """Cancel a single order by ID."""
        try:
            await self.exchange.cancel_order(order_id, symbol)
            # logger.info(f"Cancelled {order_id} on {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel {order_id} ({symbol}): {e}")
            return False
    
    async def validate_order(self, symbol, amount, side='buy'):
        """
        Validate order before placement.
        Checks: minimum size, wallet exposure, kill-switch.
        """
        # Kill-switch check
        if self.kill_switch_active:
            logger.warning(f"[{symbol}] Order blocked: Kill-switch active")
            return False, "Kill-switch active"
        
        # Get market info
        try:
            if not hasattr(self, '_markets') or not self._markets:
                self._markets = await self.exchange.load_markets()
            
            market = self._markets.get(symbol, {})
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
            min_cost = market.get('limits', {}).get('cost', {}).get('min', 0)
            
            price = self.last_prices.get(symbol, {}).get('price', 0)
            order_cost = amount * price
            
            if amount < min_amount:
                return False, f"Below min amount: {min_amount}"
            
            if order_cost < min_cost:
                return False, f"Below min cost: ${min_cost}"
            
            # WE check for buys
            if side == 'buy':
                if not self.check_wallet_exposure_limit(symbol, order_cost):
                    return False, "Exceeds wallet exposure limit"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False, str(e)
    
    async def fetch_all_orders(self, symbols):
        """Fetch open orders for all symbols in parallel."""
        async def fetch_orders(symbol):
            try:
                orders = await self.exchange.fetch_open_orders(symbol)
                return symbol, orders
            except Exception as e:
                return symbol, []
        
        results = await asyncio.gather(*[fetch_orders(s) for s in symbols])
        
        for symbol, orders in results:
            self.open_orders[symbol] = orders
    
    async def update_indicators(self, symbols):
        """Update EMAs, ATR, and RSI for all symbols in parallel."""
        async def fetch_ohlcv(symbol):
            try:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, '1m', limit=1100)
                return symbol, ohlcv
            except Exception as e:
                return symbol, None
        
        results = await asyncio.gather(*[fetch_ohlcv(s) for s in symbols])
        
        long_span = self.config.get('ema_settings', {}).get('long_term_span', 1000)
        short_span = self.config.get('ema_settings', {}).get('short_term_span', 100)
        smart_conf = self.config.get('smart_grid', {})
        atr_period = smart_conf.get('atr_period', 14)
        rsi_period = smart_conf.get('rsi_period', 14)
        
        for symbol, ohlcv in results:
            if ohlcv and len(ohlcv) >= long_span:
                try:
                    df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # EMAs
                    df['ema_l'] = df['close'].ewm(span=long_span, adjust=False).mean()
                    df['ema_s'] = df['close'].ewm(span=short_span, adjust=False).mean()
                    self.ema_long[symbol] = df['ema_l'].iloc[-1]
                    self.ema_short[symbol] = df['ema_s'].iloc[-1]
                    
                    # --- NEW: GARCH(1,1) Recursive Calc ---
                    # Calculate log returns
                    df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
                    returns = df['log_ret'].values
                    
                    # GARCH Params (Standard/Robust)
                    omega = 1e-8
                    alpha = 0.05
                    beta = 0.90
                    
                    n_garch = len(returns)
                    sigma2 = np.zeros(n_garch)
                    sigma2[0] = np.var(returns) if len(returns) > 0 else 0
                    
                    # Loop last portion (full history recursive)
                    for t in range(1, n_garch):
                        sigma2[t] = omega + alpha * (returns[t-1]**2) + beta * sigma2[t-1]
                        
                    current_garch_vol = np.sqrt(sigma2[-1])
                    self.garch_vol[symbol] = current_garch_vol
                    # -------------------------------------
                    
                    # ATR
                    df['tr1'] = df['high'] - df['low']
                    df['tr2'] = abs(df['high'] - df['close'].shift())
                    df['tr3'] = abs(df['low'] - df['close'].shift())
                    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
                    df['atr'] = df['tr'].rolling(window=atr_period).mean()
                    self.atr[symbol] = df['atr'].iloc[-1]
                    
                    # RSI
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
                    rs = gain / loss
                    df['rsi'] = 100 - (100 / (1 + rs))
                    self.rsi[symbol] = df['rsi'].iloc[-1]
                    
                    # --- NEW: ADX (Wilder's Smoothing) ---
                    # 1. DM
                    up = df['high'].diff()
                    down = -df['low'].diff()
                    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
                    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
                    
                    # 2. Smooth
                    # For performance in live loop, simple rolling sum is decent proxy for Wilder's
                    adx_window = 14
                    tr_s = df['tr'].rolling(window=adx_window).mean()
                    p_dm_s = pd.Series(plus_dm).rolling(window=adx_window).mean()
                    m_dm_s = pd.Series(minus_dm).rolling(window=adx_window).mean()
                    
                    # 3. DI
                    p_di = 100 * (p_dm_s / tr_s)
                    m_di = 100 * (m_dm_s / tr_s)
                    
                    # 4. DX
                    dx = 100 * abs(p_di - m_di) / (p_di + m_di)
                    df['adx'] = dx.rolling(window=adx_window).mean()
                    self.adx[symbol] = df['adx'].iloc[-1]
                    
                    # --- NEW: EMA Slope ---
                    # % change over last 5 mins
                    ema_l_series = df['ema_l']
                    slope = (ema_l_series.iloc[-1] - ema_l_series.iloc[-5]) / ema_l_series.iloc[-5] * 100
                    self.ema_slope[symbol] = slope
                    
                except Exception as e:
                    logger.error(f"Indicator calc error for {symbol}: {e}")
    
    # ==================== GRID CALCULATION ====================
    
    def calc_geometric_grid(self, center_price, n_orders, spacing_weight=1.2, grid_span=0.30, side='long'):
        """Calculate geometric grid prices (Passivbot style). Supports Long/Short."""
        try:
            if n_orders < 1: return []
            
            if spacing_weight == 1.0:
                d0 = grid_span / n_orders
            else:
                d0 = grid_span * (spacing_weight - 1) / (pow(spacing_weight, n_orders) - 1)
            
            prices = []
            cum_dist = 0
            for i in range(n_orders):
                dist = d0 * pow(spacing_weight, i)
                cum_dist += dist
                
                if side == 'short':
                    prices.append(center_price * (1 + cum_dist))
                else:
                    prices.append(center_price * (1 - cum_dist))
            
            return prices
        except Exception as e:
            logger.error(f"Grid calc error: {e}")
            return []
    
    def calc_martingale_qty(self, total_equity, n_orders, multiplier=1.4, wel=1.0):
        """Calculate martingale sizing for grid orders."""
        try:
            total_allocation = total_equity * wel
            
            if n_orders < 1: return []
            
            if multiplier == 1.0:
                q0 = total_allocation / n_orders
            else:
                q0 = total_allocation * (multiplier - 1) / (pow(multiplier, n_orders) - 1)
            
            return [q0 * pow(multiplier, i) for i in range(n_orders)]
        except Exception as e:
            logger.error(f"Martingale calc error: {e}")
            return []
    
    # ==================== ORDER MANAGEMENT ====================
    
    async def place_limit_order(self, symbol, side, amount, price):
        """Place a limit order with validation."""
        try:
            # Validate first
            valid, reason = await self.validate_order(symbol, amount, side)
            if not valid:
                # Only log as debug to reduce spam (these are expected with small balance)
                logger.debug(f"[{symbol}] Skipped: {reason}")
                return None
            
            # Refresh balance
            await self.refresh_balance_before_order()
            
            # Format precision
            market = self._markets.get(symbol, {})
            amount = float(self.exchange.amount_to_precision(symbol, amount))
            price = float(self.exchange.price_to_precision(symbol, price))
            
            # --- SAFETY CHECK: Prevent Selling at Loss (Spot) ---
            is_futures = self.config.get('futures_settings', {}).get('enabled', False)
            safety_conf = self.config.get('safety_settings', {})
            
            if side == 'sell' and not is_futures and safety_conf.get('prevent_loss_sells', False):
                 avg_entry = self.state_manager.get_avg_entry(symbol)
                 if avg_entry > 0:
                     min_profit = safety_conf.get('min_profit_pct', 0.002)
                     min_price = avg_entry * (1 + min_profit)
                     
                     # Allow Panic Sells / Stop Losses (identified by aggressive pricing or huge distance?)
                     # For now, strict check on Grid orders.
                     if price < min_price:
                         logger.warning(f"[{symbol}] 🛡️ SAFETY: Blocked SELL {price} < Min {min_price:.2f} (Avg: {avg_entry:.2f})")
                         return None
            
            # Place order
            order = await self.exchange.create_limit_order(symbol, side, amount, price)
            logger.info(f"[{symbol}] {side.upper()} {amount} @ {price}")
            return order
            
        except Exception as e:
            error_msg = str(e)
            # Only log balance errors once (they spam a lot)
            if "insufficient balance" in error_msg.lower():
                if not hasattr(self, '_balance_warned') or symbol not in self._balance_warned:
                    logger.warning(f"[{symbol}] Insufficient balance - reduce positions or add funds")
                    if not hasattr(self, '_balance_warned'):
                        self._balance_warned = set()
                    self._balance_warned.add(symbol)
            else:
                logger.error(f"[{symbol}] Order error: {e}")
            return None
    
    async def cancel_symbol_orders(self, symbol):
        """Cancel all open orders for a symbol."""
        try:
            orders = self.open_orders.get(symbol, [])
            for order in orders:
                await self.exchange.cancel_order(order['id'], symbol)
            return len(orders)
        except Exception as e:
            logger.error(f"[{symbol}] Cancel error: {e}")
            return 0
    
    # ==================== AUTO-UNSTUCK ====================
    
    async def auto_unstuck(self, symbol):
        """
        Auto-Unstuck for a symbol - Passivbot Risk Management.
        Closes partial position when underwater and over-exposed.
        """
        try:
            pos = self.positions.get(symbol, {})
            amount = pos.get('amount', 0)
            value = pos.get('value_usdt', 0)
            we = pos.get('wallet_exposure', 0)
            
            if amount <= 0:
                return False
            
            # Config
            unstuck_threshold = self.config.get('unstuck_settings', {}).get('unstuck_threshold', 0.80)
            unstuck_close_pct = self.config.get('unstuck_settings', {}).get('unstuck_close_pct', 0.10)
            unstuck_distance = self.config.get('unstuck_settings', {}).get('unstuck_price_distance_threshold', 0.20)
            
            # Check if over-exposed
            n_positions = len(self.get_active_symbols())
            per_symbol_wel = self.config.get('total_wallet_exposure_limit', 1.0) / max(n_positions, 1)
            
            if we < unstuck_threshold * per_symbol_wel:
                return False  # Not over-exposed
            
            # Check if underwater
            current_price = self.last_prices.get(symbol, {}).get('price', 0)
            avg_entry = pos.get('avg_entry', current_price)  # Would need to track this
            
            if avg_entry > 0:
                distance = (avg_entry - current_price) / avg_entry
                if distance < unstuck_distance:
                    return False  # Not underwater enough
            
            # Execute unstuck
            unstuck_qty = amount * unstuck_close_pct
            
            if unstuck_qty > pos.get('free', 0):
                unstuck_qty = pos.get('free', 0) * 0.99
            
            if unstuck_qty <= 0:
                return False
            
            logger.warning(f"[{symbol}] AUTO-UNSTUCK: Closing {unstuck_qty:.6f} ({unstuck_close_pct:.0%} of position)")
            
            order = await self.exchange.create_market_order(symbol, 'sell', unstuck_qty)
            
            if order:
                logger.info(f"[{symbol}] Unstuck complete: {order.get('id')}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"[{symbol}] Unstuck error: {e}")
            return False
    
    # ==================== MAIN TRADING LOGIC ====================
    
    async def process_symbol(self, symbol):
        """
        Process trading logic for a single symbol.
        Supports LONG and SHORT modes.
        """
        try:
            price_data = self.last_prices.get(symbol, {})
            position = self.positions.get(symbol, {})
            orders = self.open_orders.get(symbol, [])
            
            if not price_data:
                return
            
            current_price = price_data['price']
            ema_l = self.ema_long.get(symbol, current_price)
            ema_s = self.ema_short.get(symbol, current_price)
            
            # --- DETERMINE GRID MODE ---
            # Default to 'long'.
            # Future: Supervisor Agent updates this in config per symbol.
            grid_mode = self.config.get('grid_mode', 'long').lower()
            if hasattr(self, 'symbol_modes') and symbol in self.symbol_modes:
                grid_mode = self.symbol_modes.get(symbol)
                
            is_long = (grid_mode == 'long')
            
            # Neutral price default is long-term EMA
            neutral_price = ema_l
            
            # --- ADAPTIVE ANCHORING ---
            adx_val = self.adx.get(symbol, 0)
            if adx_val > 25:
                neutral_price = ema_s
            
            # Get config
            grid_settings = self.config.get('grid_settings', {})
            n_orders = grid_settings.get('max_orders', 5)
            grid_span = grid_settings.get('grid_span', 0.30)
            spacing_weight = grid_settings.get('grid_spacing_weight', 1.2)
            qty_multiplier = grid_settings.get('qty_step_multiplier', 1.4)
            
            # --- SMART GRID MANAGER ---
            smart_conf = self.config.get('smart_grid', {})
            if smart_conf.get('enabled', False):
                # --- RSI PANIC SELL/COVER ---
                rsi = self.rsi.get(symbol, 50)
                
                # LONG Panic
                if is_long:
                    rsi_sell_thresh = smart_conf.get('rsi_sell_threshold', 90)
                    if rsi > rsi_sell_thresh:
                        if int(time.time()) % 60 < 5:
                            logger.critical(f"[{symbol}] RSI {rsi:.1f} > {rsi_sell_thresh} -> PANIC SELL!")
                        await self.cancel_symbol_orders(symbol)
                        free_amount = position.get('free', 0)
                        if free_amount > 0:
                            # Sell 2% below price (taker)
                            await self.place_limit_order(symbol, 'sell', free_amount, current_price * 0.98)
                        return
                
                # SHORT Panic
                else:
                    rsi_buy_thresh = smart_conf.get('rsi_buy_threshold', 10)
                    if rsi < rsi_buy_thresh:
                        if int(time.time()) % 60 < 5:
                            logger.critical(f"[{symbol}] RSI {rsi:.1f} < {rsi_buy_thresh} -> PANIC BUY (Short Cover)!")
                        await self.cancel_symbol_orders(symbol)
                        # We need to BUY to cover. Amount is negative for shorts in our logic?
                        # Check fetch_all_positions logic: 
                        # If futures: amount = -amount for shorts? No, I implemented `if side == 'short': amount = -amount`
                        # So 'amount' is negative. 'free' is negative.
                        # We need to buy abs(amount).
                        pos_amt = position.get('amount', 0)
                        if pos_amt < 0:
                            buy_qty = abs(pos_amt)
                            await self.place_limit_order(symbol, 'buy', buy_qty, current_price * 1.02)
                        return

                # 1. Volatility Adaptation (GARCH) + Trend Scaling (ADX)
                garch = self.garch_vol.get(symbol, 0)
                if garch > 0:
                    vol_scale = smart_conf.get('volatility_scale', 1.0)
                    
                    trend_scale = 1.0
                    if adx_val > 40:
                        trend_scale = 1.0 + min((adx_val - 40) / 40, 0.5)
                    
                    vol_scale *= trend_scale
                    
                    safety_factor = 4.0
                    if adx_val < 30: safety_factor = 2.5
                    elif adx_val < 60: safety_factor = 2.5 + 1.5 * ((adx_val - 30) / 30)
                        
                    daily_vol = garch * 38
                    min_span = daily_vol * safety_factor * vol_scale
                    
                    aggressive = self.config.get('aggressive_mode', False)
                    if not aggressive:
                        # Bidirectional GARCH Adjustment (Widen OR Tighten)
                        # We blend the dynamic span with the config span for stability
                        # 0.7 dynamic + 0.3 static default
                        grid_span = (min_span * 0.7) + (grid_span * 0.3)
                        
                        # Hard clamps
                        grid_span = max(0.005, min(grid_span, 0.10))               
                
                # 2. Trend Bias (RSI)
                # Skew neutral price based on RSI
                bias_strength = smart_conf.get('trend_bias_strength', 0.5)
                if rsi < 30: # Oversold
                    skew = (30 - rsi) / 100 * bias_strength
                    # If Long: buying cheaper is good, but skewing grid UP catches it earlier
                    # If Short: oversold means we should be covering, or selling higher
                    neutral_price *= (1 + skew)
                elif rsi > 70: # Overbought
                    skew = (rsi - 70) / 100 * bias_strength
                    neutral_price *= (1 - skew)

            # --- REGIME FILTER ---
            regime_conf = self.config.get('regime_filter', {})
            allow_entries = True
            allow_closes = True
            
            if regime_conf.get('enabled', False):
                slope = self.ema_slope.get(symbol, 0)
                adx_thresh = regime_conf.get('adx_threshold', 25)
                slope_thresh = regime_conf.get('slope_threshold', 0.05)
                
                # Strong UPTREND
                if adx_val > adx_thresh and slope > slope_thresh:
                    if regime_conf.get('top_trend_filter', True):
                        if is_long:
                             # Don't sell into uptrend (let it run)
                             allow_closes = False
                        else:
                             # Don't short into uptrend
                             allow_entries = False
                            
                # Strong DOWNTREND
                elif adx_val > adx_thresh and slope < -slope_thresh:
                    if regime_conf.get('bottom_trend_filter', True):
                        if is_long:
                             # Don't buy (catch knife)
                             allow_entries = False
                        else:
                             # Don't cover (let it drop)
                             allow_closes = False
            
            # --- CHECK AUTO-UNSTUCK ---
            await self.auto_unstuck(symbol)
            
            # --- CALCULATE IDEAL GRID ---
            # Long: Prices < Neutral
            # Short: Prices > Neutral
            grid_prices = self.calc_geometric_grid(neutral_price, n_orders, spacing_weight, grid_span, side=grid_mode)
            
            # Dynamic Capital Allocation
            core_symbols = self.config.get('symbols', [])
            base_wel = self.config.get('core_wallet_exposure', 0.35) if symbol in core_symbols \
                    else self.config.get('forager_wallet_exposure', 0.15)
            
            # --- VOLATILITY-BASED SIZING (ATR/GARCH) ---
            target_wel = base_wel
            
            if self.config.get('smart_grid', {}).get('rapid_volatility_sizing', False):
                # Penalty for High Volatility (Safety First)
                garch_current = self.garch_vol.get(symbol, 0)
                vol_penalty = 1.0
                
                if garch_current > 0.003:  # High vol (>0.3% per min)
                    vol_penalty = 0.70  # Reduce size by 30%
                elif garch_current > 0.002:  # Medium vol
                    vol_penalty = 0.85  # Reduce size by 15%
                
                if vol_penalty < 1.0:
                    target_wel = base_wel * vol_penalty
                    if int(time.time()) % 300 < 5:
                        logger.info(f"[{symbol}] Volatility Sizing: Penalty={vol_penalty:.2f} → WEL={target_wel:.1%}")
            
            start_qty_multiplier = qty_multiplier
            buy_qtys = self.calc_martingale_qty(self.total_equity, n_orders, start_qty_multiplier, target_wel)
            
            # --- GRID MAINTENANCE (ENTRIES) ---
            # Long: Entries are Buys
            # Short: Entries are Sells
            
            entry_side = 'buy' if is_long else 'sell'
            
            existing_entry_orders = [o for o in orders if o['side'] == entry_side]
            existing_entry_prices = set(o['price'] for o in existing_entry_orders)
            
            drift_tol = grid_settings.get('drift_tolerance', 0.005)
            
            orders_to_keep = []
            matched_grid_indices = set()
            
            if allow_entries:
                for order in existing_entry_orders:
                    is_valid = False
                    for i, target_price in enumerate(grid_prices):
                        if i in matched_grid_indices: continue
                        
                        if abs(order['price'] - target_price) / target_price < drift_tol:
                            is_valid = True
                            matched_grid_indices.add(i)
                            orders_to_keep.append(order['id'])
                            break
                    
                    if not is_valid:
                        # Cancel extra entries
                        await self.cancel_order(symbol, order['id'])
                
                # Place missing entries
                for i, (price, qty_usdt) in enumerate(zip(grid_prices, buy_qtys)):
                    if i in matched_grid_indices: continue
                    
                    # Logic Check:
                    # Long: Price must be < Current * 0.995
                    # Short: Price must be > Current * 1.005
                    threshold = 0.995 if is_long else 1.005
                    should_place = (price <= current_price * threshold) if is_long else (price >= current_price * threshold)
                    
                    if not should_place: continue
                    
                    qty_base = qty_usdt / price
                    await self.place_limit_order(symbol, entry_side, qty_base, price)

            # --- CLOSE ORDERS (EXITS) ---
            # Long: Sells above Neutral
            # Short: Buys below Neutral
            
            pos_amt = position.get('amount', 0)
            has_position = (pos_amt > 0) if is_long else (pos_amt < 0)
            
            if has_position and allow_closes:
                # Calculate exit prices (Mirror the grid)
                # Long: Neutral + (Neutral - BuyPrice)
                # Short: Neutral - (SellPrice - Neutral) => 2*Neutral - SellPrice
                # Formula is same: 2*Neutral - EntryPrice
                exit_prices = [2 * neutral_price - p for p in grid_prices[:3]]
                
                close_side = 'sell' if is_long else 'buy'
                existing_close_orders = [o for o in orders if o['side'] == close_side]
                
                matched_close_indices = set()
                for order in existing_close_orders:
                    is_valid = False
                    for i, target_price in enumerate(exit_prices):
                         if i in matched_close_indices: continue
                         if abs(order['price'] - target_price) / target_price < drift_tol:
                             is_valid = True
                             matched_close_indices.add(i)
                             break
                    if not is_valid:
                        await self.cancel_order(symbol, order['id'])
                
                # --- TRAILING STOP LOGIC ---
                await self.process_trailing_stop(symbol, current_price, position, is_long, neutral_price)
                
                # Place missing closest orders (if not trailing)
                # We assume process_trailing_stop handles pure exits if trailing activated
                
                abs_free = abs(position.get('free', 0))
                if abs_free > 0:
                     # Simple logic: Distribute free amount to first few exit levels
                     exit_qty_chunk = abs_free / max(len(exit_prices), 1)
                     
                     for i, price in enumerate(exit_prices):
                         if i in matched_close_indices: continue
                         
                         # Check distance
                         # Long: Exit > Current
                         # Short: Exit < Current
                         safe_to_place = (price > current_price * 1.005) if is_long else (price < current_price * 0.995)
                         if not safe_to_place: continue
                         
                         await self.place_limit_order(symbol, close_side, exit_qty_chunk, price)

            # Log summary
            if int(time.time()) % 60 < 5:
                we = position.get('wallet_exposure', 0)
                garch = self.garch_vol.get(symbol, 0) * 100
                logger.info(f"[{symbol}] {grid_mode.upper()} | Price: {current_price:.2f} | WE: {we:.1%} | GARCH: {garch:.2f}%")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()

    async def process_trailing_stop(self, symbol, current_price, position, is_long, neutral_price):
        """Separate Trailing Stop Logic for clarity."""
        trailing_conf = self.config.get('trailing_settings', {})
        activation_pct = trailing_conf.get('close_trailing_threshold_pct', 0.02)
        callback_pct = trailing_conf.get('close_trailing_retracement_pct', 0.005)
        
        # Estimate Entry logic
        avg_entry = position.get('avgEntryPrice', neutral_price) # Fallback to Neutral
        if avg_entry <= 0: avg_entry = neutral_price
        
        # PnL Calculation
        if is_long:
            pnl_pct = (current_price - avg_entry) / avg_entry
        else:
            pnl_pct = (avg_entry - current_price) / avg_entry
            
        if pnl_pct < activation_pct:
            return # Not triggered
            
        # Update Water Mark
        last_wm = self.trailing_high_water_mark.get(symbol, 0)
        
        # Init Water Mark if 0
        if last_wm == 0:
            last_wm = current_price
            
        new_record = False
        if is_long:
            if current_price > last_wm:
                self.trailing_high_water_mark[symbol] = current_price
                new_record = True
        else:
            if current_price < last_wm or last_wm == 0:
                self.trailing_high_water_mark[symbol] = current_price
                new_record = True
                
        if new_record and int(time.time()) % 60 < 5:
             logger.info(f"[{symbol}] Trailing New Record: {current_price:.4f} (PnL: {pnl_pct:.2%})")
            
        # Check Stop
        wm = self.trailing_high_water_mark.get(symbol, current_price)
        stop_triggered = False
        
        if is_long:
            stop_price = wm * (1 - callback_pct)
            if current_price < stop_price: stop_triggered = True
        else:
            stop_price = wm * (1 + callback_pct)
            if current_price > stop_price: stop_triggered = True
            
        if stop_triggered:
            logger.info(f"[{symbol}] 🛑 TRAILING STOP HIT! Price: {current_price:.4f} / Stop: {stop_price:.4f}")
            await self.cancel_symbol_orders(symbol)
            
            # Close position
            abs_amount = abs(position.get('amount', 0))
            if abs_amount > 0:
                close_side = 'sell' if is_long else 'buy'
                # Market close (Limit with tolerance)
                close_price = current_price * 0.99 if is_long else current_price * 1.01
                await self.place_limit_order(symbol, close_side, abs_amount, close_price)
                
                # Reset
                self.trailing_high_water_mark[symbol] = 0

    
    async def hot_reload_config(self):
        """Hot-reload config if changed."""
        try:
            new_config = self.load_config()
            
            # Change Detection: Only update if changed
            if new_config == self.config:
                return False

            old_syms = set(self.get_active_symbols())
            
            # DEEP UPDATE: Update member variables that are cached
            self.config = new_config
            self.wallet_exposure_limit = self.config.get('wallet_exposure_limit', 1.0)
            self.max_drawdown_pct = self.config.get('max_drawdown_pct', 0.15)
            
            # Update specific subsections if needed
            if 'futures_settings' in self.config:
                 self.leverage = self.config['futures_settings'].get('leverage', 1)
            
            # Hot-Reload State (Cost Basis)
            if hasattr(self, 'state_manager'):
                self.state_manager.reload()

            # Log changes
            logger.info(f"[Hot-Reload] Config & State updated. WEL: {self.wallet_exposure_limit}, Leverage: {self.leverage}")

            new_syms = set(self.get_active_symbols())
            if old_syms != new_syms:
                added = new_syms - old_syms
                removed = old_syms - new_syms
                if added:
                    logger.info(f"[Hot-Reload] Added: {added}")
                if removed:
                    logger.info(f"[Hot-Reload] Removed: {removed}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Config reload error: {e}")
            return False
    
    async def run(self):
        """
        Main async execution loop - Passivbot style.
        All symbols processed in parallel each iteration.
        """
        logger.info("Starting Async Multi-Symbol Bot...")
        
        await self.init_exchange()
        
        iteration = 0
        last_ema_update = 0
        last_status_log = 0
        
        try:
            while not self.stop_signal:
                now = time.time()
                iteration += 1
                
                # Hot-reload config every 10 seconds (Responsive to Governor/User)
                if now - self.last_config_reload > 10:
                    await self.hot_reload_config()
                    self.last_config_reload = now
                
                # Get active symbols
                symbols = self.get_active_symbols()
                
                if not symbols:
                    logger.warning("No symbols configured")
                    await asyncio.sleep(10)
                    continue
                
                # ===== FETCH ALL DATA IN PARALLEL (FRESH DATA) =====
                t0 = time.time()
                
                # Parallel data fetch - ensures fresh data before any orders
                await asyncio.gather(
                    self.fetch_all_data(symbols),
                    self.fetch_all_positions(symbols),
                    self.fetch_all_orders(symbols)
                )
                
                # Update indicators less frequently (every minute)
                if now - last_ema_update > 60:
                    await self.update_indicators(symbols)
                    last_ema_update = now
                
                fetch_time = time.time() - t0
                
                # ===== UPDATE PEAK EQUITY & CHECK KILL-SWITCH =====
                if self.total_equity > self.peak_equity:
                    self.peak_equity = self.total_equity
                
                if self.check_kill_switch():
                    logger.critical("Cancelling all orders...")
                    await self.cancel_all_orders()
                    self.stop_signal = True
                    break
                
                # ===== LOG STATUS (every minute) =====
                if now - last_status_log > 60:
                    drawdown = 0
                    if self.peak_equity > 0:
                        drawdown = (self.peak_equity - self.total_equity) / self.peak_equity
                    
                    logger.info("=" * 60)
                    logger.info(f"💰 Total Equity: ${self.total_equity:.2f} | Peak: ${self.peak_equity:.2f}")
                    logger.info(f"📉 Drawdown: {drawdown:.1%} / {self.max_drawdown_pct:.1%} max")
                    logger.info(f"📊 Active Symbols: {len(symbols)}")
                    for sym in symbols[:5]:  # Show top 5
                        pos = self.positions.get(sym, {})
                        price = self.last_prices.get(sym, {}).get('price', 0)
                        exposure = pos.get('wallet_exposure', 0)
                        logger.info(f"   {sym}: ${pos.get('value_usdt', 0):.2f} ({exposure:.1%} WE) @ {price:.4f}")
                    if len(symbols) > 5:
                        logger.info(f"   ... and {len(symbols) - 5} more")
                    logger.info("=" * 60)
                    last_status_log = now
                
                # ===== PROCESS ALL SYMBOLS IN PARALLEL =====
                t1 = time.time()
                await asyncio.gather(*[self.process_symbol(s) for s in symbols])
                process_time = time.time() - t1
                
                # Performance logging
                if iteration % 20 == 0:
                    logger.info(f"[Perf] Fetch: {fetch_time:.2f}s | Process: {process_time:.2f}s | Iter: {iteration}")
                
                # Sleep
                await asyncio.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested...")
        finally:
            await self.close_exchange()
            logger.info("Bot stopped.")


async def run_portfolio_manager_async(config_path, interval_hours=1, stop_event=None):
    """Run portfolio manager in async context."""
    from src.live.portfolio_manager import PortfolioManager
    
    pm = PortfolioManager()
    
    # Skip initial run - assume config is up to date
    logger.info("[Portfolio Manager] Will run first update in 1 hour (config assumed up-to-date)")
    
    try:
        # Periodic updates
        while stop_event is None or not stop_event.is_set():
            await asyncio.sleep(interval_hours * 3600)
            logger.info("[Portfolio Manager] Running scheduled update...")
            pm.run_auto_update(config_path=config_path)
    except asyncio.CancelledError:
        logger.info("[Portfolio Manager] Stopped.")


async def main_async(config_path='config.json'):
    """
    Main entry point - runs bot only.
    Portfolio manager disabled (config assumed up-to-date).
    """
    bot = AsyncMultiSymbolBot(config_path)
    
    try:
        await bot.run()
    except asyncio.CancelledError:
        logger.info("Shutting down gracefully...")
    finally:
        await bot.close_exchange()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--profile', default='default')
    args = parser.parse_args()
    
    # Setup logging with profile name
    logger = setup_logging(args.profile)
    
    # Load profile .env
    env_path = f"profiles/{args.profile}/.env"
    if os.path.exists(env_path):
        logger.info(f"Loading env from {env_path}")
        load_dotenv(env_path, override=True)
    
    asyncio.run(main_async(args.config))
