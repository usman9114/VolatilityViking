import os
import time
import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import sys
import logging
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.getcwd())
from scripts.inference import inference, fetch_live_data, get_feature_schema, get_model_path
from src.data.news_fetcher import NewsFetcher
from sentence_transformers import SentenceTransformer

import torch
import mlflow.pytorch
from src.models.dataset import CryptoDataset

# ANSI Color Codes for Terminal
class Colors:
    GREEN = '\033[92m'  # For buys (ETH increased)
    RED = '\033[91m'    # For sells (ETH decreased)
    YELLOW = '\033[93m' # For warnings
    BLUE = '\033[94m'   # For info
    RESET = '\033[0m'   # Reset to default

# Setup Logging
os.makedirs('logs', exist_ok=True)
log_filename = f"logs/bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging to capture everything
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='a'),
        logging.StreamHandler()
    ],
    force=True  # Force reconfiguration
)

# Set logger
logger = logging.getLogger(__name__)

# Also redirect print() to logger
class PrintLogger:
    def __init__(self, logger):
        self.logger = logger
    
    def write(self, message):
        if message.strip():  # Avoid logging empty lines
            self.logger.info(message.strip())
    
    def flush(self):
        pass

# Redirect stdout to logger
import sys
sys.stdout = PrintLogger(logger)

logger.info(f"="*60)
logger.info(f"Bot started - Log file: {log_filename}")
logger.info(f"="*60)

class BinanceBot:
    def __init__(self, config_path='config.json', symbol_override=None):
        load_dotenv()
        self.config = self.load_config(config_path)
        
        # Use override if provided, else fall back to config
        if symbol_override:
            self.symbol = symbol_override
        else:
            # Support both 'symbol' (string) and 'symbols' (list, take first)
            symbols = self.config.get('symbols', [])
            if symbols:
                self.symbol = symbols[0]
            else:
                self.symbol = self.config.get('symbol', 'ETH/USDT')
        
        try:
            self.base_asset = self.symbol.split('/')[0]
            self.quote_asset = self.symbol.split('/')[1]
        except:
            self.base_asset = "ETH"
            self.quote_asset = "USDT"
            
        self.capital_limit = self.config.get('capital_limit', 1000)
        self.live_mode = self.config.get('live_mode', False)
        self.aggressive_mode = self.config.get('aggressive_mode', False)
        
        # Log mode
        if self.aggressive_mode:
            print(f"\n{Colors.YELLOW}⚡ AGGRESSIVE MODE ENABLED{Colors.RESET}")
            print(f"{Colors.YELLOW}   - 5-second checks{Colors.RESET}")
            print(f"{Colors.YELLOW}   - Proactive grid maintenance{Colors.RESET}")
            print(f"{Colors.YELLOW}   - Continuous optimization{Colors.RESET}\n")
        else:
            print(f"\n{Colors.GREEN}🛡️  CONSERVATIVE MODE (Default){Colors.RESET}")
            print(f"{Colors.GREEN}   - 60-second checks{Colors.RESET}")
            print(f"{Colors.GREEN}   - Reactive grid management{Colors.RESET}")
            print(f"{Colors.GREEN}   - Proven approach{Colors.RESET}\n")
        
        # Wallet Exposure Limits (Passivbot Risk Management)
        self.wallet_exposure_limit = self.config.get('wallet_exposure_limit', 1.0)
        self.total_wallet_exposure_limit = self.config.get('total_wallet_exposure_limit', 1.5)
        self.current_wallet_exposure = 0.0  # Tracked dynamically
        self.peak_balance = 0.0  # Track historical peak for unstucking
        
        # Grid Settings
        self.grid_settings = self.config.get('grid_settings', {})
        self.grid_span = self.grid_settings.get('grid_span', 0.30) # 30% range
        self.qty_step_multiplier = self.grid_settings.get('qty_step_multiplier', 1.4)
        self.max_orders = self.grid_settings.get('max_orders', 10)
        
        # Auto-Unstuck Parameters
        unstuck = self.config.get('unstuck_settings', {})
        self.unstuck_threshold = unstuck.get('unstuck_threshold', 0.80)
        self.unstuck_close_pct = unstuck.get('unstuck_close_pct', 0.10)
        self.unstuck_loss_allowance_pct = unstuck.get('unstuck_loss_allowance_pct', 0.02)
        self.unstuck_price_distance_threshold = unstuck.get('unstuck_price_distance_threshold', 0.20)
        
        # Dynamic Grid Spacing Parameters (Legacy support until full Geometric takeover)
        self.grid_spacing_base = 0.01  # 1% base spacing
        self.grid_spacing_we_weight = 0.5  # Widen by 50% per WE unit
        self.grid_spacing_volatility_weight = 2.0  # Widen by 200% per volatility unit
        self.volatility_window_minutes = 60  # 1-hour volatility window
        self.volatility_ema = 0.02  # Initialize with 2% volatility estimate
        
        # Trailing Close Parameters (Passivbot)
        trailing = self.config.get('trailing_settings', {})
        self.close_trailing_threshold_pct = trailing.get('close_trailing_threshold_pct', 0.02)
        self.close_trailing_retracement_pct = trailing.get('close_trailing_retracement_pct', 0.005)
        self.close_trailing_qty_pct = trailing.get('close_trailing_qty_pct', 0.20)
        self.highest_price_since_entry = None  # Track for trailing
        
        # Close Grid Markup Parameters (Passivbot)
        self.close_grid_markup_start = 0.005  # Start at 0.5% profit
        self.close_grid_markup_end = 0.015  # End at 1.5% profit
        self.close_grid_qty_pct = 0.20  # 20% per TP order
        self.close_grid_lines = 5  # Number of TP orders
        
        # Trailing Entry Parameters (Passivbot)
        self.entry_trailing_threshold_pct = trailing.get('entry_trailing_threshold_pct', 0.01)
        self.entry_trailing_retracement_pct = trailing.get('entry_trailing_retracement_pct', 0.003)
        self.lowest_price_since_signal = None  # Track for trailing entries
        
        # Select Keys based on Mode
        if self.live_mode:
            self.api_key = os.getenv('BINANCE_API_KEY')
            self.secret_key = os.getenv('BINANCE_SECRET_KEY') or os.getenv('BINANCE_SECRET')
            self.log_file = 'data/logs/live_mainnet_trades.csv'
            print("\n" + "!"*40)
            print("WARNING: RUNNING IN LIVE MAINNET MODE (REAL MONEY)")
            print("!"*40 + "\n")
        else:
            self.api_key = os.getenv('BINANCE_TESTNET_API_KEY')
            self.secret_key = os.getenv('BINANCE_TESTNET_SECRET_KEY')
            self.log_file = 'data/logs/live_trades.csv'
            print("Running in TESTNET Mode.")
            
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Init Log Header
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("timestamp,signal,action,price,quantity,order_id,type,status,info,usdt_bal,eth_bal,total_equity_usdt,wallet_exposure\n")
        
        # SAFETY: Global Kill-Switch
        self.high_water_mark_equity = 0.0
        self.MAX_DRAWDOWN_PCT = 0.20  # 20% Max Drawdown allowed
        
        # SAFETY: Auto-Unstuck Tracker
        self.position_age_start = None
        self.unstuck_threshold_hours = 24
    
    def load_config(self, path):
        try:
            with open(path, 'r') as f:
                content = f.read()
                
            # Strip comments (// or #)
            import re
            # Remove // and # comments, but keep lines intact
            content = re.sub(r'#.*', '', content)
            content = re.sub(r'//.*', '', content)
            
            return json.loads(content)
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
            return {}

    def hot_reload_config(self):
        """
        Hot-reload config from file (Passivbot style).
        Updates forager symbols without restart.
        Called periodically from main loop.
        """
        try:
            new_config = self.load_config(self.config_path)
            
            # Track what changed
            old_forager = self.config.get('forager', {})
            new_forager = new_config.get('forager', {})
            
            old_symbols = set(old_forager.get('approved_symbols', []))
            new_symbols = set(new_forager.get('approved_symbols', []))
            
            # Check if forager symbols changed
            if old_symbols != new_symbols:
                added = new_symbols - old_symbols
                removed = old_symbols - new_symbols
                
                if added:
                    print(f"[Hot-Reload] New symbols added: {added}")
                if removed:
                    print(f"[Hot-Reload] Symbols removed: {removed}")
                
                # Update config
                self.config = new_config
                print(f"[Hot-Reload] Config reloaded. Active forager symbols: {len(new_symbols)}")
                return True
            
            # Check if other important settings changed
            if new_forager.get('max_positions') != old_forager.get('max_positions'):
                self.config = new_config
                print(f"[Hot-Reload] max_positions changed to {new_forager.get('max_positions')}")
                return True
            
            return False
            
        except Exception as e:
            print(f"[Hot-Reload] Error: {e}")
            return False

    def calc_geometric_grid(self, center_price, n_orders, spacing_weight=1.2, grid_span=0.30):
        """
        Calculate geometric grid prices.
        Spacing increases geometrically to cover grid_span with n_orders.
        
        r = grid_span (e.g. 0.3 for 30%)
        d0 = initial_spacing (small)
        alpha = spacing_weight (e.g. 1.2)
        
        We want sum(d0 * alpha^i for i in 0..n-1) = r
        d0 * (alpha^n - 1) / (alpha - 1) = r
        d0 = r * (alpha - 1) / (alpha^n - 1)
        """
        try:
            if n_orders < 1: return []
            
            # Solve for initial spacing d0
            if spacing_weight == 1.0:
                d0 = grid_span / n_orders
            else:
                d0 = grid_span * (spacing_weight - 1) / (pow(spacing_weight, n_orders) - 1)
            
            prices = []
            cum_dist = 0
            for i in range(n_orders):
                dist = d0 * pow(spacing_weight, i)
                cum_dist += dist
                
                # Buy side
                prices.append(center_price * (1 - cum_dist))
            
            return prices
        except Exception as e:
            print(f"Geometric Grid Error: {e}")
            return []

    def calc_martingale_qty(self, total_equity, n_orders, multiplier=1.4):
        """
        Calculate quantities for martingale sizing.
        Total allocation = Wallet Exposure Limit * Total Equity
        
        q0 * (mult^n - 1) / (mult - 1) = Total_Allocation
        q0 = Total_Allocation * (mult - 1) / (mult^n - 1)
        """
        try:
            total_allocation = total_equity * self.wallet_exposure_limit
            
            if n_orders < 1: return []
            
            if multiplier == 1.0:
                q0 = total_allocation / n_orders
            else:
                q0 = total_allocation * (multiplier - 1) / (pow(multiplier, n_orders) - 1)
            
            qtys = []
            for i in range(n_orders):
                q = q0 * pow(multiplier, i)
                qtys.append(q)
                
            return qtys
        except Exception as e:
             print(f"Martingale Qty Error: {e}")
             return []

        
        if not self.api_key or not self.secret_key:
            missing_key = []
            if self.live_mode:
                if not self.api_key: missing_key.append('BINANCE_API_KEY')
                if not self.secret_key: missing_key.append('BINANCE_SECRET_KEY (or BINANCE_SECRET)')
            else:
                if not self.api_key: missing_key.append('BINANCE_TESTNET_API_KEY')
                if not self.secret_key: missing_key.append('BINANCE_TESTNET_SECRET_KEY')
                
            print(f"ERROR: Missing Keys in .env: {', '.join(missing_key)}")
            sys.exit(1)
            
        print(f"Connecting to Binance {'MAINNET' if self.live_mode else 'TESTNET'}...")
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.secret_key,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'} 
        })
        
        # Enable Sandbox ONLY if NOT in live mode
        if not self.live_mode:
            self.exchange.set_sandbox_mode(True) 
        
        # Init News & Embeddings
        print("Loading News Fetcher & Sentence Transformer...")
        self.news_fetcher = NewsFetcher()
        self.sent_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Sentence Transformer Ready.")
        
        # Verify Connection & Init Model
        self.check_connection()
        self.init_model()

    def check_connection(self):
        """Verify API connection and display account info"""
        try:
            # Display log file locations
            print("\n" + "="*70)
            print("📁 LOG FILES:")
            print("="*70)
            print(f"Bot Activity Log: {os.path.abspath(log_filename)}")
            print(f"Trade History CSV: {os.path.abspath(self.log_file)}")
            print("="*70 + "\n")
            
            balance = self.exchange.fetch_balance()
            total_usdt, total_eth, total_equity = self.fetch_equity()
            
            print(f"Connected! Wallet Balance: {total_usdt:.2f} USDT, {total_eth:.4f} ETH")
            print(f"Total Account Equity: ${total_equity:.2f}")
            
            # Check if capital is sufficient
            if not self.live_mode:  # Testnet
                 if total_usdt < self.capital_limit and total_equity < self.capital_limit:
                     print(f"WARNING: Account Equity ({total_equity:.2f}) is less than Capital Limit ({self.capital_limit})")
                     
        except Exception as e:
            print(f"Connection Failed: {e}")
            sys.exit(1)

    def fetch_equity(self):
        try:
            balance = self.exchange.fetch_balance()
            usdt_free = balance.get('USDT', {}).get('free', 0)
            usdt_used = balance.get('USDT', {}).get('used', 0)
            total_usdt = usdt_free + usdt_used
            
            eth_free = balance.get('ETH', {}).get('free', 0)
            eth_used = balance.get('ETH', {}).get('used', 0)
            total_eth = eth_free + eth_used
            
            ticker = self.exchange.fetch_ticker(self.symbol)
            price = ticker['last']
            
            total_equity = total_usdt + (total_eth * price)
        
            # SAFETY: Update High Water Mark & Check Drawdown
            if total_equity > self.high_water_mark_equity:
                self.high_water_mark_equity = total_equity
            
            drawdown = (self.high_water_mark_equity - total_equity) / self.high_water_mark_equity if self.high_water_mark_equity > 0 else 0
            
            if drawdown > self.MAX_DRAWDOWN_PCT:
                msg = f"CRITICAL: Max Drawdown Hit! ({drawdown:.2%} > {self.MAX_DRAWDOWN_PCT:.2%}). Stopping Bot."
                logging.critical(msg)
                print(f"\n{'!'*50}\n{msg}\n{'!'*50}\n")
                try:
                    print("Attempting to cancel all orders...")
                    self.exchange.cancel_all_orders(self.symbol)
                except:
                    pass
                import sys
                sys.exit(1)

            return total_usdt, total_eth, total_equity
        except Exception as e:
            print(f"Error fetching equity: {e}")
            return 0.0, 0.0, 0.0

    def calculate_wallet_exposure(self):
        """
        Calculate Wallet Exposure (WE) - Passivbot Risk Management
        WE = (position_value) / (unleveraged_balance)
        
        Returns:
            float: Current wallet exposure (0.0 = no position, 1.0 = 100% of balance)
        """
        try:
            total_usdt, total_eth, total_equity = self.fetch_equity()
            
            if total_equity == 0:
                return 0.0
            
            # Calculate ETH position value
            current_price = self.get_market_price()
            eth_value = total_eth * current_price
            
            # WE = Position Value / Unleveraged Balance
            wallet_exposure = eth_value / total_equity
            
            # Update tracking
            self.current_wallet_exposure = wallet_exposure
            
            # Update peak balance for unstucking
            if total_equity > self.peak_balance:
                self.peak_balance = total_equity
            
            return wallet_exposure
            
        except Exception as e:
            print(f"WE Calculation Error: {e}")
            return 0.0

    def auto_unstuck(self):
        """
        Auto-Unstuck Mechanism - Passivbot Risk Management
        
        Prevents permanent stuck positions by:
        1. Detecting when position is underwater and WE is high
        2. Realizing small controlled losses (10% of position)
        3. Freeing margin for better re-entry prices
        
        Returns:
            bool: True if unstuck action was taken, False otherwise
        """
        try:
            # 1. Calculate current state
            total_usdt, total_eth, total_equity = self.fetch_equity()
            we = self.calculate_wallet_exposure()
            current_price = self.get_market_price()
            
            # Skip if no position
            if total_eth < 0.001:
                return False
            
            # 2. Check if position is "stuck"
            # Stuck = WE > threshold AND price moved significantly from entry
            if we < self.unstuck_threshold:
                return False  # Not stuck enough
            
            # 3. Calculate average entry price (estimate from current holdings)
            # Since we don't track avg entry explicitly, use equity/eth ratio
            eth_value = total_eth * current_price
            
            # Estimate if we're underwater (simplified check)
            # If we're at high WE but price hasn't moved much, we're likely stuck
            price_distance = abs(current_price - (total_equity / total_eth)) / current_price
            
            if price_distance < self.unstuck_price_distance_threshold:
                return False  # Not far enough from entry
            
            # 4. Check loss allowance
            # Don't unstuck if we've already taken too many losses
            loss_allowance = self.peak_balance * (1 - self.unstuck_loss_allowance_pct * self.total_wallet_exposure_limit)
            
            if total_equity < loss_allowance:
                print(f"[Auto-Unstuck] Loss allowance exceeded. Peak: ${self.peak_balance:.2f}, Current: ${total_equity:.2f}, Allowance: ${loss_allowance:.2f}")
                return False
            
            # 5. Check FREE ETH balance (not locked in orders)
            balance = self.exchange.fetch_balance()
            free_eth = balance['free'].get('ETH', 0)
            
            if free_eth < 0.001:
                print(f"[Auto-Unstuck] No free ETH available (all locked in orders). Cancelling grid first...")
                # Cancel all open orders to free up ETH
                try:
                    open_orders = self.exchange.fetch_open_orders(self.symbol)
                    for order in open_orders:
                        self.exchange.cancel_order(order['id'], self.symbol)
                        print(f"[Auto-Unstuck] Cancelled order {order['id']}")
                    
                    # Re-fetch balance after cancelling
                    balance = self.exchange.fetch_balance()
                    free_eth = balance['free'].get('ETH', 0)
                except Exception as e:
                    print(f"[Auto-Unstuck] Error cancelling orders: {e}")
                    return False
            
            # 6. Execute unstuck: Close 10% of TOTAL position (use free ETH)
            unstuck_qty = total_eth * self.unstuck_close_pct
            
            # But only sell what's available (free)
            if unstuck_qty > free_eth:
                unstuck_qty = free_eth * 0.99  # Use 99% of free to be safe
            
            unstuck_qty = float(self.exchange.amount_to_precision(self.symbol, unstuck_qty))
            
            # Check minimum order size
            trade_val = unstuck_qty * current_price
            if trade_val < 6.0:
                print(f"[Auto-Unstuck] Trade value ${trade_val:.2f} too small. Skipping.")
                return False
            
            print(f"\n{'='*50}")
            print(f"[AUTO-UNSTUCK TRIGGERED]")
            print(f"WE: {we:.2f} ({we/self.wallet_exposure_limit:.0%} of limit)")
            print(f"Price Distance: {price_distance:.1%}")
            print(f"Closing {unstuck_qty:.4f} ETH")
            print(f"{'='*50}\n")
            
            # Calculate EMA bands for exit price
            try:
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, '1m', limit=1000)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Calculate EMAs
                ema_short = df['close'].ewm(span=100).mean().iloc[-1]
                ema_long = df['close'].ewm(span=1000).mean().iloc[-1]
                
                # Upper band (for selling)
                upper_ema = max(ema_short, ema_long)
                
                # Place limit order at upper EMA + 1% (unstuck_ema_dist)
                unstuck_price = upper_ema * 1.01
                unstuck_price = float(self.exchange.price_to_precision(self.symbol, unstuck_price))
                
                print(f"[Auto-Unstuck] EMA-Based Exit:")
                print(f"  Upper EMA: ${upper_ema:.2f}")
                print(f"  Unstuck Price: ${unstuck_price:.2f} (+1% above EMA)")
                
                # Place limit order (not market)
                order = self.exchange.create_limit_sell_order(self.symbol, unstuck_qty, unstuck_price)
                
            except Exception as ema_error:
                print(f"[Auto-Unstuck] EMA calculation failed: {ema_error}")
                print(f"[Auto-Unstuck] Falling back to market order")
                # Fallback to market order
                order = self.exchange.create_market_order(self.symbol, 'sell', unstuck_qty)
                unstuck_price = current_price
            
            # Log the unstuck action
            if 'fills' in order:
                fill_price = sum(f['price'] * f['amount'] for f in order['fills']) / sum(f['amount'] for f in order['fills'])
            else:
                fill_price = order['price'] if order['price'] else current_price
            
            self.log_trade("AUTO_UNSTUCK", "MARKET_SELL", fill_price, unstuck_qty, order['id'], "MARKET", "FILLED", f"WE: {we:.2f}, Distance: {price_distance:.1%}")
            
            print(f"[Auto-Unstuck] Successfully closed {unstuck_qty:.4f} ETH. New WE will be lower.")
            return True
            
        except Exception as e:
            print(f"[Auto-Unstuck] Error: {e}")
            return False

    def update_volatility_ema(self):
        """
        Update volatility EMA based on recent price action.
        Volatility = log(high/low) for recent candles
        
        This is used to widen grid spacing in choppy markets.
        """
        try:
            # Fetch recent 1m candles for volatility calculation
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe='1m', limit=self.volatility_window_minutes)
            
            if not ohlcv or len(ohlcv) < 10:
                return  # Not enough data
            
            # Calculate log range for each candle: ln(high/low)
            log_ranges = []
            for candle in ohlcv:
                high, low = candle[2], candle[3]
                if low > 0:
                    log_range = np.log(high / low)
                    log_ranges.append(log_range)
            
            # Calculate mean log range (volatility)
            current_volatility = np.mean(log_ranges)
            
            # Update EMA: new_ema = old_ema * 0.9 + new_val * 0.1
            alpha = 0.1
            self.volatility_ema = self.volatility_ema * (1 - alpha) + current_volatility * alpha
            
        except Exception as e:
            print(f"Volatility Update Error: {e}")
    
    def calculate_grid_spacing(self):
        """
        Calculate dynamic grid spacing based on:
        1. Wallet Exposure (widen as position grows)
        2. Market Volatility (widen in choppy markets)
        
        Returns:
            float: Grid spacing percentage (e.g., 0.02 = 2%)
        """
        try:
            # Get current WE
            we = self.current_wallet_exposure
            
            # Base spacing
            spacing = self.grid_spacing_base
            
            # Adjust for Wallet Exposure
            # As WE increases, widen spacing to slow down entries
            we_multiplier = 1 + (we * self.grid_spacing_we_weight)
            
            # Adjust for Volatility
            # In high volatility, widen spacing to avoid getting filled too quickly
            vol_multiplier = 1 + (self.volatility_ema * self.grid_spacing_volatility_weight)
            
            # Final spacing
            dynamic_spacing = spacing * we_multiplier * vol_multiplier
            
            # Cap at reasonable limits (0.5% to 10%)
            dynamic_spacing = max(0.005, min(0.10, dynamic_spacing))
            
            return dynamic_spacing
            
        except Exception as e:
            print(f"Grid Spacing Calculation Error: {e}")
            return self.grid_spacing_base  # Fallback to base spacing

    def check_trailing_close(self, current_price):
        """
        Trailing Close Orders - Passivbot Strategy
        
        Waits for price to move favorably (threshold), then closes on retracement.
        This locks in better profits than fixed TP levels.
        
        Returns:
            bool: True if trailing close was executed
        """
        try:
            # Get current position
            total_usdt, total_eth, total_equity = self.fetch_equity()
            
            if total_eth < 0.001:
                self.highest_price_since_entry = None
                return False  # No position
            
            # Initialize highest price tracker
            if self.highest_price_since_entry is None:
                self.highest_price_since_entry = current_price
                return False
            
            # Update highest price
            if current_price > self.highest_price_since_entry:
                self.highest_price_since_entry = current_price
            
            # Estimate average entry price (simplified)
            avg_entry_price = total_equity / total_eth
            
            # Check if threshold met (price moved favorably)
            threshold_price = avg_entry_price * (1 + self.close_trailing_threshold_pct)
            
            if self.highest_price_since_entry < threshold_price:
                return False  # Haven't reached threshold yet
            
            # Check if retraced
            retracement_price = self.highest_price_since_entry * (1 - self.close_trailing_retracement_pct)
            
            if current_price > retracement_price:
                return False  # No retracement yet
            
            # TRIGGER: Threshold met + Retraced
            close_qty = total_eth * self.close_trailing_qty_pct
            
            # Check free balance
            balance = self.exchange.fetch_balance()
            free_eth = balance['free'].get('ETH', 0)
            
            if close_qty > free_eth:
                # Cancel grid orders to free ETH
                try:
                    open_orders = self.exchange.fetch_open_orders(self.symbol)
                    for order in open_orders:
                        self.exchange.cancel_order(order['id'], self.symbol)
                    balance = self.exchange.fetch_balance()
                    free_eth = balance['free'].get('ETH', 0)
                except:
                    pass
            
            close_qty = min(close_qty, free_eth * 0.99)
            close_qty = float(self.exchange.amount_to_precision(self.symbol, close_qty))
            
            if close_qty * current_price < 6.0:
                return False  # Too small
            
            print(f"\n{'='*50}")
            print(f"[TRAILING CLOSE TRIGGERED]")
            print(f"Avg Entry: ${avg_entry_price:.2f}")
            print(f"Highest: ${self.highest_price_since_entry:.2f} (+{((self.highest_price_since_entry/avg_entry_price-1)*100):.1f}%)")
            print(f"Current: ${current_price:.2f} (retraced {((1-current_price/self.highest_price_since_entry)*100):.1f}%)")
            print(f"Closing {close_qty:.4f} ETH @ ${current_price:.2f}")
            print(f"{'='*50}\n")
            
            # Execute close
            order = self.exchange.create_market_order(self.symbol, 'sell', close_qty)
            
            if 'fills' in order:
                fill_price = sum(f['price'] * f['amount'] for f in order['fills']) / sum(f['amount'] for f in order['fills'])
            else:
                fill_price = order['price'] if order['price'] else current_price
            
            profit_pct = ((fill_price / avg_entry_price) - 1) * 100
            self.log_trade("TRAILING_CLOSE", "MARKET_SELL", fill_price, close_qty, order['id'], "MARKET", "FILLED", f"Profit: {profit_pct:.2f}%")
            
            # Reset tracker
            self.highest_price_since_entry = None
            
            return True
            
        except Exception as e:
            print(f"[Trailing Close] Error: {e}")
            return False

    def deploy_close_grid(self):
        """
        Close Grid with Markup - Passivbot Strategy
        
        Places multiple take-profit orders at different markup levels.
        Complements trailing closes with fixed profit targets.
        
        Example:
        - Entry: $2,950
        - Markup Start: 0.5% → $2,965
        - Markup End: 1.5% → $2,994
        - 5 TP orders: $2,965, $2,972, $2,979, $2,987, $2,994
        - Each: 20% of position
        """
        try:
            # Get current position
            total_usdt, total_eth, total_equity = self.fetch_equity()
            
            if total_eth < 0.001:
                return  # No position
            
            # Estimate average entry price
            current_price = self.get_market_price()
            avg_entry_price = total_equity / total_eth
            
            # Calculate markup prices
            markup_start_price = avg_entry_price * (1 + self.close_grid_markup_start)
            markup_end_price = avg_entry_price * (1 + self.close_grid_markup_end)
            
            # Only deploy if we're in profit territory
            if current_price < markup_start_price:
                return  # Not profitable enough yet
            
            # Create price levels
            prices = pd.interval_range(
                start=markup_start_price,
                end=markup_end_price,
                periods=self.close_grid_lines
            ).mid
            
            # Calculate quantity per order
            qty_per_order = total_eth * self.close_grid_qty_pct
            qty_per_order = float(self.exchange.amount_to_precision(self.symbol, qty_per_order))
            
            # Check if orders already exist
            try:
                open_orders = self.exchange.fetch_open_orders(self.symbol)
                # Cancel existing TP orders to redeploy
                for order in open_orders:
                    if order['side'] == 'sell' and order['type'] == 'limit':
                        self.exchange.cancel_order(order['id'], self.symbol)
            except:
                pass
            
            print(f"\n[Close Grid Markup]")
            print(f"Avg Entry: ${avg_entry_price:.2f}")
            print(f"TP Range: ${markup_start_price:.2f} (+{self.close_grid_markup_start*100:.1f}%) to ${markup_end_price:.2f} (+{self.close_grid_markup_end*100:.1f}%)")
            print(f"Deploying {self.close_grid_lines} TP orders, {qty_per_order:.4f} ETH each:")
            
            # Place TP orders
            for idx, price in enumerate(prices):
                try:
                    price = float(self.exchange.price_to_precision(self.symbol, price))
                    
                    # Skip if too small
                    if qty_per_order * price < 5.0:
                        continue
                    
                    order = self.exchange.create_limit_sell_order(self.symbol, qty_per_order, price)
                    profit_pct = ((price / avg_entry_price) - 1) * 100
                    print(f"  TP{idx+1}: {qty_per_order:.4f} ETH @ ${price:.2f} (+{profit_pct:.2f}%)")
                    self.log_trade("CLOSE_GRID", "SELL_LIMIT", price, qty_per_order, order['id'], "LIMIT", "OPEN", f"TP {profit_pct:.2f}%")
                    
                except Exception as e:
                    print(f"  TP{idx+1} Failed: {e}")
            
            print(f"Close Grid Deployed.\n")
            
        except Exception as e:
            print(f"[Close Grid] Error: {e}")

    def auto_unstuck(self):
        """
        PASSIVBOT-STYLE AUTO-UNSTUCK
        Checks if position has been held too long and tries to exit at Break-Even or small loss
        to recycle capital.
        """
        try:
            total_usdt, total_eth, total_equity = self.fetch_equity()
            
            # 1. Track Position Age
            if total_eth > 0.001:
                if self.position_age_start is None:
                    self.position_age_start = time.time()
                    print(f"[Unstuck] New Position Detected. Timer Started.")
            else:
                if self.position_age_start is not None:
                    self.position_age_start = None
                    print(f"[Unstuck] Position Closed. Timer Reset.")
                return # No position to unstuck

            # 2. Check Duration
            if self.position_age_start is None: return
            
            elapsed_hours = (time.time() - self.position_age_start) / 3600
            if elapsed_hours < self.unstuck_threshold_hours:
                return # Not stuck yet
                
            # 3. Analyze "Stuck" Position
            current_price = self.get_market_price()
            avg_entry = total_equity / total_eth if total_eth > 0 else 0
            
            if avg_entry == 0: return

            pnl_pct = (current_price / avg_entry) - 1
            
            print(f"\n[Unstuck Check] Position held for {elapsed_hours:.1f} hours.")
            print(f"  Avg Entry: ${avg_entry:.2f} | Current: ${current_price:.2f} | PnL: {pnl_pct*100:.2f}%")
            
            # 4. Action Logic
            # Condition A: We are in profit (even small), just close it to free liquidity
            if pnl_pct > 0.002: # 0.2% profit
                print("  Condition met: Minimal Profit. Closing to recycle.")
                self.close_position_market()
                self.position_age_start = None
                
            # Condition B: We are at small loss but stuck long time (Recycle)
            elif pnl_pct > -0.02 and elapsed_hours > 48:
                print("  Condition met: Small Loss (-2%) after 48h. Closing to recycle.")
                self.close_position_market()
                self.position_age_start = None
                
            # Condition C: Deep loss? Maybe Hedge or just wait (User preference)
            # For now, we only handle small loss recycling.
            
        except Exception as e:
            print(f"[Unstuck] Error: {e}")

    def close_position_market(self):
        balance = self.exchange.fetch_balance()
        eth_free = balance['free'].get('ETH', 0)
        if eth_free > 0.001:
            print(f"Executing Market Sell for {eth_free:.4f} ETH...")
            self.exchange.create_market_order(self.symbol, 'sell', eth_free)

    def log_trade(self, signal, action, price, quantity, order_id, order_type, status, info=""):
        timestamp = pd.Timestamp.now()
        
        # Fetch Snapshot
        usdt_bal, eth_bal, equity = self.fetch_equity()
        
        # Calculate Wallet Exposure
        we = self.calculate_wallet_exposure()
        
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp},{signal},{action},{price},{quantity},{order_id},{order_type},{status},{info},{usdt_bal:.2f},{eth_bal:.4f},{equity:.2f},{we:.3f}\n")
        print(f"logged: {action} {quantity} @ {price} | Eq: ${equity:.2f} | WE: {we:.2f} ({we/self.wallet_exposure_limit:.0%})")

    def init_model(self):
        # Set device first (needed even if model fails to load)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            model_uri = get_model_path()
            # Handle CPU-only environment
            map_location = None if torch.cuda.is_available() else torch.device('cpu')
            self.model = mlflow.pytorch.load_model(model_uri, map_location=map_location)
            
            self.model.to(self.device)
            self.model.eval()
            print("Brain Loaded (Model Ready).")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None  # Continue without AI model, use grid trading
            print("Continuing with grid trading strategy (no AI predictions)")

    def get_signal(self):
        print("Thinking...")
        # 1. Fetch Price (STRICT MODE)
        try:
             df = fetch_live_data(self.symbol, limit=200, strict=True)
        except Exception as e:
             print(f"CRITICAL: Price Fetch Failed: {e}")
             return None, None, None

        current_price = df['close'].iloc[-1]
        
        # 2. Fetch & Embed News (REAL TIME)
        print("Fetching News...")
        news_df = self.news_fetcher.fetch_news('ETH', limit=50) # Last 50 items
        
        if not news_df.empty:
            # Get news from last 4 hours
            cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=4)
            if news_df.index.tz is None: news_df.index = news_df.index.tz_localize('UTC') # Ensure TZ aware
            
            recent_news = news_df[news_df.index > cutoff]
            
            if not recent_news.empty:
                headlines = recent_news['title'].tolist()
                print(f"Found {len(headlines)} recent headlines. Embedding...")
                embeddings = self.sent_model.encode(headlines)
                mean_emb = np.mean(embeddings, axis=0) # (384,)
            else:
                # FALLBACK: Use the LAST available news item
                print("No news in last 4h. Using LAST AVAILABLE news persistence.")
                last_item = news_df.iloc[-1]
                print(f"Fallback News: {last_item['title']} ({last_item.name})")
                embeddings = self.sent_model.encode([last_item['title']])
                mean_emb = np.mean(embeddings, axis=0)
        else:
             print("STRICT MODE: News Fetch Failed or Empty. Aborting.")
             return None, None, None

        # 4. Integrate into DF
        # We need to add 'emb_0'...'emb_383' cols.
        feature_schema = get_feature_schema()
        
        # First, ensure basic alignment
        missing_cols = [c for c in feature_schema if c not in df.columns]
        
        # Create zero matrix
        zero_data = np.zeros((len(df), len(missing_cols)), dtype=np.float32)
        zero_df = pd.DataFrame(zero_data, columns=missing_cols, index=df.index)
        
        # FILL THE LAST ROW with our Real Embedding
        # Identify embedding columns
        emb_cols = [c for c in missing_cols if c.startswith('emb_')]
        # This assumes emb_0 to emb_383 are in order in missing_cols
        # We must be careful mapping mean_emb indices to column names.
        
        # Let's map explicitly
        if not isinstance(mean_emb, (int, float)): # If it's a vector
            last_idx = df.index[-1]
            # Update the zero_df last row
            for i in range(384):
                 col_name = f'emb_{i}'
                 if col_name in zero_df.columns:
                     zero_df.loc[last_idx, col_name] = mean_emb[i]
        
        # Merge
        df = pd.concat([df, zero_df], axis=1)
        df = df[feature_schema]
        
        # 5. Predict
        # Check if model is available
        if self.model is None:
            # No AI model - default to GRID strategy
            print("No AI model available - using GRID strategy")
            signal = "GRID"
            return signal, current_price, 0.0  # prediction = 0.0 (neutral)
        
        # FIX: Set test_size=0 to use ALL fetched data for inference (no train/test split)
        # Use inference_mode=True to prevent dropping last row
        ds = CryptoDataset(dataframe=df, seq_len=96, news_emb_path=None, gas_path=None, test_size=0, inference_mode=True)
        # Disable shuffling to preserve order
        loader, _, _ = ds.get_torch_loaders(batch_size=32, return_split=True, shuffle_train=False)
        
        preds = []
        with torch.no_grad():
            for x_price, x_news, _ in loader:
                x_price, x_news = x_price.to(self.device), x_news.to(self.device)
                out = self.model(x_price, x_news)
                preds.extend(out.cpu().numpy().flatten())
                
        current_prediction = preds[-1]
        
        # 4. Thresholds
        pred_series = pd.Series(preds)
        WINDOW = 96
        K = 0.86
        rolling_mean = pred_series.rolling(window=WINDOW).mean().iloc[-1]
        rolling_std = pred_series.rolling(window=WINDOW).std().iloc[-1]
        
        thresh_up = rolling_mean + (K * rolling_std)
        thresh_down = rolling_mean - (K * rolling_std)
        
        print(f"Price: {current_price} | Pred: {current_prediction:.5f} | Bands: {thresh_down:.5f}/{thresh_up:.5f}")
        
        # GRID PRIORITY: Tighten thresholds by 10% to favor GRID mode
        GRID_BIAS = 0.10
        thresh_up_biased = thresh_up * (1 - GRID_BIAS)
        thresh_down_biased = thresh_down * (1 + GRID_BIAS)
        
        signal = "NEUTRAL"
        if current_prediction > thresh_up_biased: 
            signal = "LONG"
        elif current_prediction < thresh_down_biased: 
            signal = "SHORT"
        else: 
            signal = "GRID"  # Favored zone is now wider
        
        print(f"[Signal Logic] Biased Bands: {thresh_down_biased:.5f}/{thresh_up_biased:.5f} → {signal}")
        
        return signal, current_price, current_prediction

    def calculate_ideal_grid(self, center_price):
        """
        Calculate the ideal grid state based on Geometric Progression & Martingale.
        
        Returns:
            (ideal_buys, ideal_sells): Lists of dicts with 'price' and 'quantity'
        """
        # Fetch current side
        balance = self.exchange.fetch_balance()
        quote_free = balance['free'].get(self.quote_asset, 0)
        base_free = balance['free'].get(self.base_asset, 0)
        total_quote, total_base, total_equity = self.fetch_equity()
        
        # Grid Params
        n_orders = self.max_orders
        spacing_weight = self.grid_settings.get('grid_spacing_weight', 1.2)
        grid_span = self.grid_span
        
        ideal_buys = []
        ideal_sells = []
        
        # --- BUYS (Geometric Down) ---
        buy_prices = self.calc_geometric_grid(center_price, n_orders, spacing_weight, grid_span)
        
        # Calculate Martingale Distribution (in Quote Currency)
        buy_quote_values = self.calc_martingale_qty(total_equity, n_orders, self.qty_step_multiplier)
        
        if quote_free > 10: # Min buffer
             remaining_quote = quote_free
             for p, val_quote in zip(buy_prices, buy_quote_values):
                 price = float(self.exchange.price_to_precision(self.symbol, p))
                 
                 # CONVERT VALUE TO BASE QTY
                 raw_qty = val_quote / price
                 qty = float(self.exchange.amount_to_precision(self.symbol, raw_qty))
                 
                 cost = qty * price
                 
                 # Limits check (Min notional usually 5-10 USDT)
                 if remaining_quote >= cost and cost >= 5.0 and qty >= 0.00001:
                     ideal_buys.append({'price': price, 'quantity': qty, 'side': 'buy'})
                     remaining_quote -= cost

        # --- SELLS (Geometric Up) ---
        sell_prices = []
        d0 = (grid_span * (spacing_weight - 1) / (pow(spacing_weight, n_orders) - 1)) if spacing_weight != 1 else grid_span/n_orders
        cum_dist = 0
        for i in range(n_orders):
            dist = d0 * pow(spacing_weight, i)
            cum_dist += dist
            sell_prices.append(center_price * (1 + cum_dist))
            
        # For sells, distribute the CURRENT AVAILABLE BASE ASSET using the same martingale curve
        if base_free > 0.0001:
             # Sell distribution logic
             spacing_mult = self.qty_step_multiplier
             if spacing_mult == 1:
                 q0 = base_free / n_orders
             else:
                 q0 = base_free * (spacing_mult - 1) / (pow(spacing_mult, n_orders) - 1)
             
             total_base_allocated = 0
             for i, p in enumerate(sell_prices):
                 q_raw = q0 * pow(spacing_mult, i)
                 
                 price = float(self.exchange.price_to_precision(self.symbol, p))
                 qty = float(self.exchange.amount_to_precision(self.symbol, q_raw))
                 value = qty * price
                 
                 if (total_base_allocated + qty) <= base_free and value >= 5.0 and qty >= 0.00001:
                    ideal_sells.append({'price': price, 'quantity': qty, 'side': 'sell'})
                    total_base_allocated += qty
        
        return ideal_buys, ideal_sells

    def compare_grids(self, ideal_orders, actual_orders):
        """
        Compare ideal grid vs actual open orders.
        
        Returns:
            (orders_to_cancel, orders_to_place)
        """
        # Tolerance for price matching (0.1%)
        PRICE_TOLERANCE = 0.001
        # Tolerance for quantity matching (1%)
        QTY_TOLERANCE = 0.01
        
        orders_to_cancel = []
        orders_to_place = []
        
        # Convert actual orders to comparable format
        actual_by_side = {'buy': [], 'sell': []}
        for order in actual_orders:
            side = order['side']
            actual_by_side[side].append({
                'price': float(order['price']),
                'quantity': float(order['amount']),
                'id': order['id'],
                'side': side
            })
        
        # Check each ideal order
        for ideal in ideal_orders:
            side = ideal['side']
            ideal_price = ideal['price']
            ideal_qty = ideal['quantity']
            
            # Find matching actual order
            matched = False
            for actual in actual_by_side[side]:
                price_diff = abs(actual['price'] - ideal_price) / ideal_price
                qty_diff = abs(actual['quantity'] - ideal_qty) / ideal_qty if ideal_qty > 0 else 0
                
                if price_diff <= PRICE_TOLERANCE and qty_diff <= QTY_TOLERANCE:
                    # Order matches ideal, keep it
                    matched = True
                    actual_by_side[side].remove(actual)
                    break
            
            if not matched:
                # Ideal order doesn't exist, need to place it
                orders_to_place.append(ideal)
        
        # Any remaining actual orders are not in ideal grid, cancel them
        for side in ['buy', 'sell']:
            for actual in actual_by_side[side]:
                orders_to_cancel.append(actual)
        
        return orders_to_cancel, orders_to_place

    def maintain_ideal_grid(self, current_price):
        """
        Proactively maintain ideal grid (Passivbot approach).
        Called every 5 seconds.
        """
        try:
            # Anti-flapping: Don't update if we just updated recently
            if hasattr(self, 'last_grid_update_time'):
                time_since_update = time.time() - self.last_grid_update_time
                if time_since_update < 3:
                    return  # Skip if updated within last 3 seconds
            
            # Calculate ideal grid
            ideal_buys, ideal_sells = self.calculate_ideal_grid(current_price)
            ideal_orders = ideal_buys + ideal_sells
            
            # Fetch actual orders
            actual_orders = self.exchange.fetch_open_orders(self.symbol)
            
            # Compare
            orders_to_cancel, orders_to_place = self.compare_grids(ideal_orders, actual_orders)
            
            # If no changes needed, we're done
            if not orders_to_cancel and not orders_to_place:
                return
            
            # Log optimization action
            print(f"\n{Colors.YELLOW}[Grid Optimization]{Colors.RESET}")
            print(f"  To Cancel: {len(orders_to_cancel)}")
            print(f"  To Place: {len(orders_to_place)}")
            
            # Cancel deviating orders
            for order in orders_to_cancel:
                try:
                    self.exchange.cancel_order(order['id'], self.symbol)
                    print(f"  {Colors.RED}✗ Cancelled{Colors.RESET}: {order['side'].upper()} {order['quantity']:.4f} @ ${order['price']:.2f}")
                except Exception as e:
                    print(f"  Failed to cancel {order['id']}: {e}")
            
            # Place missing orders
            for order in orders_to_place:
                try:
                    if order['side'] == 'buy':
                        result = self.exchange.create_limit_buy_order(self.symbol, order['quantity'], order['price'])
                        print(f"  {Colors.GREEN}✓ Placed{Colors.RESET}: BUY {order['quantity']:.4f} @ ${order['price']:.2f}")
                    else:
                        result = self.exchange.create_limit_sell_order(self.symbol, order['quantity'], order['price'])
                        print(f"  {Colors.GREEN}✓ Placed{Colors.RESET}: SELL {order['quantity']:.4f} @ ${order['price']:.2f}")
                    
                    self.log_trade("GRID_OPT", f"{order['side'].upper()}_LIMIT", order['price'], order['quantity'], result['id'], "LIMIT", "OPEN", "Ideal Grid Maintenance")
                except Exception as e:
                    print(f"  Failed to place {order['side']} order: {e}")
            
            # Update timestamp
            self.last_grid_update_time = time.time()
            
        except Exception as e:
            print(f"[Grid Optimization] Error: {e}")

    def execute_grid(self, center_price):
        print(f"\n{'='*70}")
        print(f"EXECUTING GEOMETRIC GRID CENTERED AT ${center_price:.2f}")
        print(f"{'='*70}")
        
        # 1. Fetch State
        total_usdt, total_eth, total_equity = self.fetch_equity()
        open_orders = self.exchange.fetch_open_orders(self.symbol)
        print(f"[Geometric Grid] Found {len(open_orders)} existing orders.")

        # 2. Generate IDEAL Orders (Using new Geometric Logic)
        ideal_buys, ideal_sells = self.calculate_ideal_grid(center_price)
        
        # Transform to flat list of dicts consistent with previous structure
        ideal_orders = []
        for b in ideal_buys:
             ideal_orders.append({'side': 'buy', 'qty': b['quantity'], 'price': b['price'], 'type': 'limit'})
        for s in ideal_sells:
             ideal_orders.append({'side': 'sell', 'qty': s['quantity'], 'price': s['price'], 'type': 'limit'})
             
        print(f"[Geometric Grid] Generated {len(ideal_orders)} ideal orders.")

        # 4. Diff & Rebalance
        to_create = []
        to_cancel = []
        
        # Match Open Orders with Ideal Orders
        # We try to find an "Approximate Match" for each open order in the ideal list
        # If found, we remove it from ideal list (it's satisfied). 
        # If not found, mark for cancellation.
        # Remaining ideal orders are marked for creation.
        
        # Copy ideal list to consume
        unmet_ideal = ideal_orders.copy()
        
        print(f"[Smart Grid] Reconciling {len(open_orders)} Open vs {len(ideal_orders)} Ideal...")
        
        for order in open_orders:
            match_found = False
            for i, ideal in enumerate(unmet_ideal):
                # Match logic: Same Side, Same Type, Price +/- 0.5%, Qty +/- 5%
                price_match = abs(order['price'] - ideal['price']) / ideal['price'] < 0.005
                qty_match = abs(order['amount'] - ideal['qty']) / ideal['qty'] < 0.05
                
                if order['side'] == ideal['side'] and price_match and qty_match:
                    match_found = True
                    # Remove from unmet, as this order satisfies it
                    unmet_ideal.pop(i) 
                    # print(f"  ✓ Order matches ideal: {order['side']} @ {order['price']}")
                    break
            
            if not match_found:
                to_cancel.append(order)
                
        to_create = unmet_ideal
        
        # 5. Execute Diff
        print(f"[Smart Grid] Action Plan: Cancel {len(to_cancel)}, Create {len(to_create)}")
        
        # A. Cancel first (to free balance)
        for order in to_cancel:
            try:
                print(f"  ✗ Cancelling: {order['side']} {order['amount']} @ {order['price']}")
                self.exchange.cancel_order(order['id'], self.symbol)
            except Exception as e:
                print(f"    Error canceling {order['id']}: {e}")
                
        # B. Create new
        for order in to_create:
            try:
                if order['side'] == 'buy':
                    self.exchange.create_limit_buy_order(self.symbol, order['qty'], order['price'])
                else:
                    self.exchange.create_limit_sell_order(self.symbol, order['qty'], order['price'])
                print(f"  ✓ Created: {order['side']} {order['qty']} @ {order['price']}")
            except Exception as e:
                 print(f"    Error creating {order['side']}: {e}")

        print(f"Smart Grid Rebalancing Complete.\n")

    def execute_trend(self, direction):
        # logging.info(f"Verifying TREND Position: {direction}") 
        # Don't print every minute unless actionable, or use debug level.
        # For now, let's just print less alarming text.
        print(f"Checking {direction} alignment...")
        
        price = self.get_market_price()
        side = 'buy' if direction == "LONG" else 'sell'
        
        # USE TOTAL ACCOUNT EQUITY (not just capital_limit)
        if side == 'buy':
            # REBALANCING LOGIC for LONG
            # 1. Get Total Equity
            total_usdt, total_eth, total_equity = self.fetch_equity()
            
            # 2. Calc Target Position Value (95% of TOTAL EQUITY)
            target_value = total_equity * 0.95
            print(f"[Trend] Total Equity: ${total_equity:.2f} | Target: ${target_value:.2f}")
            
            # 3. Calc Current Base Asset Value
            balance = self.exchange.fetch_balance()
            base_bal = balance['total'].get(self.base_asset, 0)
            current_val = base_bal * price
            
            # 3. Calc Deficit
            deficit_quote = target_value - current_val
            
            if deficit_quote < 10.0: # Minimum interaction threshold ($10)
                print(f"Already positioned (Deficit ${deficit_quote:.2f} < $10). Holding.")
                return
            
            # 4. Calc Qty to Buy
            qty = deficit_quote / price
            
            # 5. Check actual Quote validity and use available balance
            quote_bal = balance['free'].get(self.quote_asset, 0)
            if quote_bal < (qty * price):
                # Use whatever Quote is available (99% for fees)
                print(f"[Balance Safety] Calculated qty needs ${qty * price:.2f}, but only ${quote_bal:.2f} available")
                print(f"[Balance Safety] Using available balance instead")
                max_buy_val = quote_bal * 0.99  # Safety for fees
                qty = max_buy_val / price
                
                # If still too small, skip
                if qty * price < 6.0:
                    print(f"[Balance Safety] Available balance ${qty * price:.2f} too small. Skipping.")
                    return
            
        elif side == 'sell':
            # For SHORT, sell Base to reach target
            # Use available (free) Base balance
            balance = self.exchange.fetch_balance() # Ensure balance is fresh for this branch
            base_bal = balance['free'].get(self.base_asset, 0)
            
            if base_bal < 0.001:
                print(f"[Balance Safety] No free {self.base_asset} available to sell.")
                return
            
            # Calculate qty needed, but cap at available
            # The 'deficit_quote' variable is only defined in the 'buy' branch.
            # For a 'sell' in trend, we're likely reducing Base exposure.
            # A simpler approach for 'sell' trend is to sell a fixed percentage of available Base,
            # or to sell down to a target Quote percentage.
            # For now, let's assume 'qty_needed' would be derived from a target Base balance.
            # If we're just selling available Base, then 'qty' is simply 'base_bal'.
            qty = base_bal * 0.99 # Sell most of available Base
            
            print(f"[Balance Safety] Free ETH: {eth_bal:.4f}, Using: {qty:.4f}")
            
            # Check if trade is big enoughn
        
        # Precision
        qty = float(self.exchange.amount_to_precision(self.symbol, qty))

        # Final Min Value Check (Binance Min ~$5-10)
        trade_val = qty * price
        if trade_val < 6.0:
             print(f"Trade value ${trade_val:.2f} too small. Skipping.")
             return

        try:
             order = self.exchange.create_market_order(self.symbol, side, qty)
             # ... logging ...
             if 'fills' in order:
                fill_price = sum(f['price'] * f['amount'] for f in order['fills']) / sum(f['amount'] for f in order['fills'])
             else:
                fill_price = order['price'] if order['price'] else price
             
             slippage = (fill_price - price) / price
             self.log_trade("TREND", f"MARKET_{side.upper()}", fill_price, qty, order['id'], "MARKET", "FILLED", f"Slippage: {slippage*100:.2f}%")
             print(f"Market {side} Filled at {fill_price}")
        except Exception as e:
             print(f"Order Failed: {e}")

    def get_market_price(self):
        ticker = self.exchange.fetch_ticker(self.symbol)
        return ticker['last']


    def update_ema(self):
        """
        Update Long and Short EMAs for Trend Following.
        """
        try:
            # 1. Fetch OHLCV (enough for long EMA)
            long_span = self.config.get('ema_settings', {}).get('long_term_span', 1000)
            short_span = self.config.get('ema_settings', {}).get('short_term_span', 100)
            
            limit = long_span + 100
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe='1m', limit=limit)
             
            if not ohlcv: return
             
            closes = [x[4] for x in ohlcv]
            df = pd.Series(closes)
             
            self.ema_long = df.ewm(span=long_span).mean().iloc[-1]
            self.ema_short = df.ewm(span=short_span).mean().iloc[-1]
            
            # log occasionaly
            if int(time.time()) % 300 < 10:
                print(f"[EMA] Long ({long_span}): {self.ema_long:.2f} | Short ({short_span}): {self.ema_short:.2f}")
                
        except Exception as e:
            print(f"EMA Error: {e}")

    def get_neutral_price(self):
        """
        Get the Neutral Price for Grid Centering.
        If EMA is available, use EMA (Trend Following).
        Else, use Current Market Price.
        """
        if hasattr(self, 'ema_long') and self.ema_long > 0:
            # Passivbot often uses the Long EMA as the "Center" of the grid
            return self.ema_long
        
        return self.get_market_price()

    # ==================== FORAGER ====================
    def calc_volatility(self, symbol, lookback_minutes=60):
        """
        Calculate volatility as mean((high - low) / close) for recent 1m candles.
        This is Passivbot's 'log-range' volatility measure.
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='1m', limit=lookback_minutes)
            if not ohlcv or len(ohlcv) < 10:
                return 0.0
            
            ranges = []
            for candle in ohlcv:
                high, low, close = candle[2], candle[3], candle[4]
                if close > 0:
                    ranges.append((high - low) / close)
            
            return sum(ranges) / len(ranges) if ranges else 0.0
        except Exception as e:
            print(f"[Forager] Volatility calc failed for {symbol}: {e}")
            return 0.0

    def calc_volume(self, symbol, lookback_minutes=60):
        """
        Calculate average quote volume over recent 1m candles.
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='1m', limit=lookback_minutes)
            if not ohlcv or len(ohlcv) < 10:
                return 0.0
            
            # Volume is typically in base, multiply by close for quote volume
            volumes = [candle[5] * candle[4] for candle in ohlcv]
            return sum(volumes) / len(volumes) if volumes else 0.0
        except Exception as e:
            print(f"[Forager] Volume calc failed for {symbol}: {e}")
            return 0.0

    def select_forager_coins(self):
        """
        Select symbols to trade.
        - 'symbols' list = ALWAYS active (core coins)
        - Forager ON = Adds best N from 'approved_symbols' to core list
        """
        # Core symbols - always trade these
        core_symbols = self.config.get('symbols', [])
        
        forager_config = self.config.get('forager', {})
        
        if not forager_config.get('enabled', False):
            # Forager disabled - just trade core symbols
            if core_symbols:
                return core_symbols
            else:
                return [self.symbol]  # Fallback
        
        # Forager enabled - add best coins to core
        approved = forager_config.get('approved_symbols', [])
        
        # Remove core symbols from approved pool (don't double-count)
        approved = [s for s in approved if s not in core_symbols]
        
        if not approved:
            print("[Forager] No additional approved symbols to scan.")
            return core_symbols if core_symbols else [self.symbol]
        
        max_positions = forager_config.get('max_positions', 3)
        volume_drop_pct = forager_config.get('volume_drop_pct', 0.25)
        vol_lookback = forager_config.get('volatility_lookback_minutes', 60)
        volume_lookback = forager_config.get('volume_lookback_minutes', 60)
        
        print(f"[Forager] Core symbols: {core_symbols}")
        print(f"[Forager] Scanning {len(approved)} additional symbols...")
        
        # Calculate metrics for approved symbols
        coin_metrics = []
        for symbol in approved:
            try:
                volatility = self.calc_volatility(symbol, vol_lookback)
                volume = self.calc_volume(symbol, volume_lookback)
                coin_metrics.append({
                    'symbol': symbol,
                    'volatility': volatility,
                    'volume': volume
                })
            except Exception as e:
                print(f"[Forager] Skipping {symbol}: {e}")
        
        if not coin_metrics:
            return core_symbols if core_symbols else [self.symbol]
        
        # Step 1: Sort by volume and drop lowest X%
        coin_metrics.sort(key=lambda x: x['volume'], reverse=True)
        keep_count = int(len(coin_metrics) * (1 - volume_drop_pct))
        coin_metrics = coin_metrics[:max(keep_count, 1)]
        
        # Step 2: Sort remaining by volatility (highest first)
        coin_metrics.sort(key=lambda x: x['volatility'], reverse=True)
        
        # Step 3: Take top N
        forager_picks = [c['symbol'] for c in coin_metrics[:max_positions]]
        
        print(f"[Forager] Adding {len(forager_picks)} symbols: {forager_picks}")
        for c in coin_metrics[:max_positions]:
            print(f"  - {c['symbol']}: vol={c['volatility']:.6f}, volume={c['volume']:.0f}")
        
        # Combine: Core + Forager picks
        all_symbols = list(core_symbols) + forager_picks
        print(f"[Forager] Total active symbols: {all_symbols}")
        
        return all_symbols

    def maintain_recursive_grid(self):
        """
        Passivbot "Recursive" Grid Maintenance.
        Constantly recalculates the Ideal Grid based on:
        1. Current Neutral Price (EMA)
        2. Current Equity
        
        And then aligns open orders to this Ideal Grid.
        """
        try:
            # Anti-flapping
            if hasattr(self, 'last_grid_update_time'):
                if time.time() - self.last_grid_update_time < 5: return

            # 1. Get Ideal Grid based on Neutral Price
            neutral_price = self.get_neutral_price()
            ideal_buys, ideal_sells = self.calculate_ideal_grid(neutral_price)
            
            # Flatten
            ideal_orders = []
            for b in ideal_buys: ideal_orders.append({'side': 'buy', 'qty': b['quantity'], 'price': b['price']})
            for s in ideal_sells: ideal_orders.append({'side': 'sell', 'qty': s['quantity'], 'price': s['price']})
            
            # 2. Get Actual Orders
            actual_orders = self.exchange.fetch_open_orders(self.symbol)
            
            # 3. Compare & Rebalance
            orders_to_cancel, orders_to_place = self.compare_grids(ideal_orders, actual_orders)
            
            if not orders_to_cancel and not orders_to_place: return

            print(f"\n[Recursive Grid] Rebalancing around ${neutral_price:.2f}...")
            
            # Cancel
            for order in orders_to_cancel:
                try:
                    self.exchange.cancel_order(order['id'], self.symbol)
                except: pass
            
            # Place
            for order in orders_to_place:
                try:
                    if order['side'] == 'buy':
                        self.exchange.create_limit_buy_order(self.symbol, order['qty'], order['price'])
                    else:
                        self.exchange.create_limit_sell_order(self.symbol, order['qty'], order['price'])
                except: pass
                
            self.last_grid_update_time = time.time()
            
        except Exception as e:
            print(f"Recursive Grid Error: {e}")

    def run(self):
        print("Bot Started. Press Ctrl+C to stop.")
        
        # State Tracking
        last_prediction_time = None
        last_config_reload = time.time()
        current_signal = "NEUTRAL"
        
        # Initialize balance tracking
        self.fetch_equity()
        
        PREDICTION_INTERVAL = 4 * 3600 # 4 Hours
        CONFIG_RELOAD_INTERVAL = 5 * 60  # 5 Minutes - check for config changes
        
        while True:
            try:
                now = time.time()
                time_since_last = 0 
                
                # --- 0. HOT-RELOAD CONFIG (Every 5 minutes) ---
                if now - last_config_reload >= CONFIG_RELOAD_INTERVAL:
                    self.hot_reload_config()
                    last_config_reload = now
                
                # --- 0.5 UPDATE EMA (Every minute) ---
                if int(now) % 60 < 10:
                    self.update_ema()

                # --- 1. PREDICTION LOGIC (Every 4 Hours) ---
                if last_prediction_time is None or (now - last_prediction_time >= PREDICTION_INTERVAL):
                    print("\n[AI] Time for new prediction...")
                    signal, price, pred = self.get_signal()
                    
                    if signal is None:
                        time.sleep(60) 
                        continue
                        
                    current_signal = signal
                    last_prediction_time = now
                    print(f"[AI] New Signal: {signal} | Next prediction in 4 hours")
                    
                # --- 2. AUTO-UNSTUCK CHECK ---
                check_interval = 5 if self.aggressive_mode else 60
                if last_prediction_time and int(now) % 60 < check_interval:
                    self.auto_unstuck()
                
                # --- 3. TRAILING CLOSE CHECK ---
                if last_prediction_time and int(now) % 60 < check_interval:
                    price = self.get_market_price()
                    self.check_trailing_close(price)
                
                # --- 4. CLOSE GRID DEPLOYMENT ---
                if int(time.time()) % 300 < check_interval:
                    self.deploy_close_grid()
                
                # --- 5. RECURSIVE GRID MAINTENANCE ---
                # Replaces the old "Grid Management" block
                if current_signal == "GRID" or self.aggressive_mode:
                    # In Passivbot, grid is always maintained relative to Neutral Price
                    self.maintain_recursive_grid()
                
                # Sleep based on mode
                sleep_interval = 5 if self.aggressive_mode else 60
                time.sleep(sleep_interval)
                
            except KeyboardInterrupt:
                print("Stopping...")
                break
            except Exception as e:
                print(f"Error in Loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Binance Trading Bot")
    # Args are now optional/overrides since we have config.json
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    
    args = parser.parse_args()
    
    bot = BinanceBot(config_path=args.config)
    bot.run()
