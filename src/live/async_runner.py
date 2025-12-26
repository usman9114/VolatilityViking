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

# Setup logging to both console and file
os.makedirs('logs', exist_ok=True)
log_filename = f"logs/async_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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


class AsyncMultiSymbolBot:
    """
    Passivbot-style async bot that handles ALL symbols in a single event loop.
    Much faster than thread-per-symbol approach.
    """
    
    def __init__(self, config_path='config.json'):
        load_dotenv()
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
        
        self.exchange = ccxt_async.binance({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
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
            balance = await self.exchange.fetch_balance()
            
            # Total wallet in USDT (quote currency)
            self.total_wallet_usdt = balance.get('USDT', {}).get('total', 0) or 0
            self.free_wallet_usdt = balance.get('USDT', {}).get('free', 0) or 0
            
            # Calculate value of all holdings
            total_equity = self.total_wallet_usdt
            
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
                
                total_equity += value_usdt
            
            self.total_equity = total_equity
            
            # Calculate wallet exposure per symbol
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
        Triggers when drawdown exceeds max_drawdown_pct.
        """
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
    
    def calc_geometric_grid(self, center_price, n_orders, spacing_weight=1.2, grid_span=0.30):
        """Calculate geometric grid prices (Passivbot style)."""
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
        Maintains grid around EMA (neutral price).
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
            
            # Neutral price is long-term EMA
            neutral_price = ema_l
            
            # Get config
            grid_settings = self.config.get('grid_settings', {})
            n_orders = grid_settings.get('max_orders', 5)
            grid_span = grid_settings.get('grid_span', 0.30)
            spacing_weight = grid_settings.get('grid_spacing_weight', 1.2)
            qty_multiplier = grid_settings.get('qty_step_multiplier', 1.4)
            
            # --- SMART GRID MANAGER ---
            smart_conf = self.config.get('smart_grid', {})
            if smart_conf.get('enabled', False):
                # 1. Volatility Adaptation (GARCH) + Trend Scaling (ADX)
                garch = self.garch_vol.get(symbol, 0)
                adx = self.adx.get(symbol, 0)
                
                if garch > 0:
                    vol_scale = smart_conf.get('volatility_scale', 1.0)
                    
                    # --- NEW: Trend Adaptive Spacing ---
                    # Revised for Scalping:
                    # Ignore trends until they are EXTREME (ADX > 40)
                    # Max expansion capped at 1.5x (was 2.0x) to allow closer orders
                    trend_scale = 1.0
                    if adx > 40:
                        trend_scale = 1.0 + min((adx - 40) / 40, 0.5) # Max 1.5x at ADX 80+
                    
                    vol_scale *= trend_scale
                    # -----------------------------------

                    # DYNAMIC SAFETY FACTOR
                    # Low ADX (Chop) -> Covet 2.5 std devs (Tight)
                    # High ADX (Trend) -> Cover 4.0 std devs (Safe)
                    safety_factor = 4.0
                    if adx < 30:
                        safety_factor = 2.5
                    elif adx < 60:
                        # Linear scale 2.5 -> 4.0
                        safety_factor = 2.5 + 1.5 * ((adx - 30) / 30)
                        
                    daily_vol = garch * 38
                    min_span = daily_vol * safety_factor * vol_scale
                    
                    # --- OVERRIDE LOGIC ---
                    # If aggressive_mode is ON, we trust the manual grid_span even if GARCH says it's unsafe.
                    # We only widen if aggressive_mode is OFF.
                    aggressive = self.config.get('aggressive_mode', False)
                    
                    if not aggressive:
                         if min_span > grid_span:
                             logger.debug(f"[{symbol}] Widening span: {grid_span:.2%} -> {min_span:.2%} (GARCH x{safety_factor:.1f})")
                             grid_span = min_span
                    else:
                        # In aggressive mode, we might still want to KNOW what GARCH thinks, but we don't force it.
                        # Maybe Log it?
                        if int(time.time()) % 60 < 5:
                             pass # logger.debug(f"[{symbol}] Aggressive Mode: Ignoring GARCH floor ({min_span:.2%}). Using {grid_span:.2%}")
                
                # 2. Trend Bias (RSI)
                rsi = self.rsi.get(symbol, 50)
                bias_strength = smart_conf.get('trend_bias_strength', 0.5)
                
                if rsi < 30: # Oversold -> Shift Neutral UP (more buys below)
                    skew = (30 - rsi) / 100 * bias_strength
                    neutral_price *= (1 + skew)
                    if int(time.time()) % 60 < 5:
                        logger.info(f"[{symbol}] RSI {rsi:.1f} Oversold -> Skewing grid UP")
                elif rsi > 70: # Overbought -> Shift Neutral DOWN (more sells above)
                    skew = (rsi - 70) / 100 * bias_strength
                    neutral_price *= (1 - skew)
                    if int(time.time()) % 60 < 5:
                        logger.info(f"[{symbol}] RSI {rsi:.1f} Overbought -> Skewing grid DOWN")

            # --- REGIME FILTER (NEW) ---
            regime_conf = self.config.get('regime_filter', {})
            allow_buys = True
            allow_sells = True
            
            if regime_conf.get('enabled', False):
                adx = self.adx.get(symbol, 0)
                slope = self.ema_slope.get(symbol, 0)
                adx_thresh = regime_conf.get('adx_threshold', 25)
                slope_thresh = regime_conf.get('slope_threshold', 0.05)
                
                # Strong UPTREND (ADX > 25, Positive Slope) -> Block Shorts?
                if adx > adx_thresh and slope > slope_thresh:
                    if regime_conf.get('top_trend_filter', True):
                        allow_sells = False
                        if int(time.time()) % 60 < 5:
                            logger.info(f"[{symbol}] STRONG UPTREND (ADX:{adx:.1f} Slope:{slope:.2f}) -> Blocking Sells")
                            
                # Strong DOWNTREND (ADX > 25, Negative Slope) -> Block Longs?
                elif adx > adx_thresh and slope < -slope_thresh:
                    if regime_conf.get('bottom_trend_filter', True):
                        allow_buys = False
                        if int(time.time()) % 60 < 5:
                            logger.info(f"[{symbol}] STRONG DOWNTREND (ADX:{adx:.1f} Slope:{slope:.2f}) -> Blocking Buys")
            # ---------------------------
            
            # --- CHECK AUTO-UNSTUCK ---
            await self.auto_unstuck(symbol)
            
            # --- CALCULATE IDEAL GRID ---
            buy_prices = self.calc_geometric_grid(neutral_price, n_orders, spacing_weight, grid_span)
            
            # Dynamic Capital Allocation
            # Core Symbols (Hardcoded in config) -> High Tier
            # Forager Symbols (Auto-discovered) -> Low Tier
            core_symbols = self.config.get('symbols', [])
            if symbol in core_symbols:
                target_wel = self.config.get('core_wallet_exposure', 0.35)
            else:
                target_wel = self.config.get('forager_wallet_exposure', 0.15)
                
            buy_qtys = self.calc_martingale_qty(
                self.total_equity, 
                n_orders,
                qty_multiplier,
                target_wel
            )
            
            # --- GRID MAINTENANCE ---
            existing_buy_orders = [o for o in orders if o['side'] == 'buy']
            existing_buy_prices = set(o['price'] for o in existing_buy_orders)
            
            # Place missing buy orders
            if allow_buys:
                for i, (price, qty) in enumerate(zip(buy_prices, buy_qtys)):
                    # Skip if order already exists near this price
                    if any(abs(p - price) / price < 0.005 for p in existing_buy_prices):
                        continue
                    
                    # Only place orders below current price
                    if price >= current_price * 0.995:
                        continue
                    
                    # Convert qty from USDT to base asset
                    qty_base = qty / price
                    
            # Get detailed grid settings
            drift_tol = grid_settings.get('drift_tolerance', 0.005)

            # --- 1. IDENTIFY ORDERS TO KEEP VS CANCEL ---
            orders_to_keep = []
            matched_grid_indices = set() # Track which grid levels are satisfied
            
            # Buys
            if allow_buys:
                existing_buy_orders = [o for o in orders if o['side'] == 'buy']
                for order in existing_buy_orders:
                    # Check if this order matches ANY target grid price
                    is_valid = False
                    for i, target_price in enumerate(buy_prices):
                        if i in matched_grid_indices: continue # This level already claimed
                        
                        if abs(order['price'] - target_price) / target_price < drift_tol:
                            is_valid = True
                            matched_grid_indices.add(i)
                            orders_to_keep.append(order['id'])
                            break
                    
                    if not is_valid:
                        # This order is "orphaned" (too far from any target) -> Cancel it
                        await self.cancel_order(symbol, order['id'])

            # --- 2. PLACE MISSING ORDERS ---
            if allow_buys:
                for i, (price, qty) in enumerate(zip(buy_prices, buy_qtys)):
                    if i in matched_grid_indices:
                        continue # Already have a valid order for this level
                    
                    # Place new order
                    qty_base = qty / price
                    await self.place_limit_order(symbol, 'buy', qty_base, price)
            
            # --- CLOSE ORDERS (Sells above neutral) ---
            pos_amount = position.get('amount', 0)
            if pos_amount > 0 and allow_sells:
                # Calculate sell prices (mirror of buys, above neutral)
                sell_prices = [neutral_price * (1 + (neutral_price - p) / neutral_price) for p in buy_prices[:3]]
                
                existing_sell_orders = [o for o in orders if o['side'] == 'sell']
                
                # --- 1. CLEANUP OLD SELLS ---
                matched_sell_indices = set()
                for order in existing_sell_orders:
                    is_valid = False
                    for i, target_price in enumerate(sell_prices):
                         if i in matched_sell_indices: continue
                         
                         if abs(order['price'] - target_price) / target_price < drift_tol:
                             is_valid = True
                             matched_sell_indices.add(i)
                             break
                    
                    if not is_valid:
                        await self.cancel_order(symbol, order['id'])

                # --- 2. PLACE MISSING SELLS ---
                free_amount = position.get('free', 0)
                sell_qty_chunk = free_amount / max(len(sell_prices), 1)
                
                for i, price in enumerate(sell_prices):
                    if i in matched_sell_indices:
                        continue
                        
                    if price <= current_price * 1.005:
                        continue
                    
                    if sell_qty_chunk > 0:
                        await self.place_limit_order(symbol, 'sell', sell_qty_chunk, price)
            
            # Log summary occasionally
            if int(time.time()) % 60 < 5:
                we = position.get('wallet_exposure', 0)
                adx = self.adx.get(symbol, 0)
                slope = self.ema_slope.get(symbol, 0)
                garch = self.garch_vol.get(symbol, 0) * 100 # Show as %
                logger.info(f"[{symbol}] Price: {current_price:.2f} | ADX: {adx:.1f} | Slope: {slope:.3f}% | GARCH: {garch:.2f}% | WE: {we:.1%}")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    async def hot_reload_config(self):
        """Hot-reload config if changed."""
        try:
            new_config = self.load_config()
            
            old_syms = set(self.get_active_symbols())
            self.config = new_config
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
                
                # Hot-reload config every 5 minutes
                if now - self.last_config_reload > 300:
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
    args = parser.parse_args()
    
    asyncio.run(main_async(args.config))
