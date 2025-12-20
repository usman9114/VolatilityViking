import os
import time
import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import sys
import logging
from datetime import datetime

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
    def __init__(self, symbol='ETH/USDT', capital_limit=1000, live_mode=False, aggressive_mode=False):
        load_dotenv()
        self.live_mode = live_mode
        self.aggressive_mode = aggressive_mode  # Feature flag for Passivbot-style aggressive grid management
        self.symbol = symbol
        self.capital_limit = capital_limit
        
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
        self.wallet_exposure_limit = 1.0  # Max 100% of balance per position
        self.total_wallet_exposure_limit = 1.5  # Max 150% total (allows leverage)
        self.current_wallet_exposure = 0.0  # Tracked dynamically
        self.peak_balance = 0.0  # Track historical peak for unstucking
        
        # Auto-Unstuck Parameters
        self.unstuck_threshold = 0.80  # Trigger when WE > 80% of limit
        self.unstuck_close_pct = 0.10  # Close 10% of position each time
        self.unstuck_loss_allowance_pct = 0.02  # Allow 2% loss below peak
        self.unstuck_price_distance_threshold = 0.20  # Trigger when price is 20% away from avg entry
        
        # Dynamic Grid Spacing Parameters
        self.grid_spacing_base = 0.01  # 1% base spacing
        self.grid_spacing_we_weight = 0.5  # Widen by 50% per WE unit
        self.grid_spacing_volatility_weight = 2.0  # Widen by 200% per volatility unit
        self.volatility_window_minutes = 60  # 1-hour volatility window
        self.volatility_ema = 0.02  # Initialize with 2% volatility estimate
        
        # Trailing Close Parameters (Passivbot)
        self.close_trailing_threshold_pct = 0.02  # Wait for 2% profit
        self.close_trailing_retracement_pct = 0.005  # Close on 0.5% pullback
        self.close_trailing_qty_pct = 0.20  # Close 20% per trigger
        self.highest_price_since_entry = None  # Track for trailing
        
        # Close Grid Markup Parameters (Passivbot)
        self.close_grid_markup_start = 0.005  # Start at 0.5% profit
        self.close_grid_markup_end = 0.015  # End at 1.5% profit
        self.close_grid_qty_pct = 0.20  # 20% per TP order
        self.close_grid_lines = 5  # Number of TP orders
        
        # Trailing Entry Parameters (Passivbot)
        self.entry_trailing_threshold_pct = 0.01  # Wait for 1% move
        self.entry_trailing_retracement_pct = 0.003  # Enter on 0.3% pullback
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
        Calculate the ideal grid state based on current conditions.
        
        Returns:
            (ideal_buys, ideal_sells): Lists of dicts with 'price' and 'quantity'
        """
        # Fetch current balance
        balance = self.exchange.fetch_balance()
        usdt_free = balance['free'].get('USDT', 0)
        eth_free = balance['free'].get('ETH', 0)
        
        # Calculate dynamic spacing
        self.update_volatility_ema()
        spacing_pct = self.calculate_grid_spacing()
        
        up_range = center_price * (1 + spacing_pct)
        down_range = center_price * (1 - spacing_pct)
        
        ideal_buys = []
        ideal_sells = []
        
        # Calculate ideal buy orders
        max_buy_lines = 5
        buy_prices = pd.interval_range(start=down_range, end=center_price, periods=max_buy_lines).mid
        
        if usdt_free > 20:
            usdt_for_buys = usdt_free * 0.90
            qty_per_buy = (usdt_for_buys / max_buy_lines) / center_price
            qty_per_buy = float(self.exchange.amount_to_precision(self.symbol, qty_per_buy))
            
            remaining_usdt = usdt_free
            for p in buy_prices:
                price = float(self.exchange.price_to_precision(self.symbol, p))
                cost = qty_per_buy * price
                
                # Ensure qty meets minimum precision (0.0001)
                if remaining_usdt >= cost and cost >= 6.0 and qty_per_buy >= 0.0001:
                    ideal_buys.append({'price': price, 'quantity': qty_per_buy, 'side': 'buy'})
                    remaining_usdt -= cost
        
        # Calculate ideal sell orders
        max_sell_lines = 5
        sell_prices = pd.interval_range(start=center_price, end=up_range, periods=max_sell_lines).mid
        
        if eth_free > 0.001:
            base_qty = eth_free * 0.05
            multipliers = [1, 2, 4, 8, 5]
            
            total_eth_allocated = 0
            for p, mult in zip(sell_prices, multipliers):
                price = float(self.exchange.price_to_precision(self.symbol, p))
                sell_qty = base_qty * mult
                sell_qty = float(self.exchange.amount_to_precision(self.symbol, sell_qty))
                value = sell_qty * price
                
                if (total_eth_allocated + sell_qty) <= eth_free and value >= 5.0 and sell_qty >= 0.0001:
                    ideal_sells.append({'price': price, 'quantity': sell_qty, 'side': 'sell'})
                    total_eth_allocated += sell_qty
        
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
        print(f"EXECUTING GRID CENTERED AT ${center_price:.2f}")
        print(f"{'='*70}")
        
        # 1. Cancel Open Orders
        open_orders = self.exchange.fetch_open_orders(self.symbol)
        if open_orders:
            print(f"[Grid] Cancelling {len(open_orders)} open orders...")
            self.exchange.cancel_all_orders(self.symbol)
            self.log_trade("GRID", "CANCEL_ALL", center_price, 0, "N/A", "CANCEL", "OK", "Resetting Grid")
            
        # 2. Fetch Fresh Balance (CRITICAL: Must reflect latest Binance state)
        total_usdt, total_eth, total_equity = self.fetch_equity()
        balance = self.exchange.fetch_balance()
        usdt_free = balance['free'].get('USDT', 0)
        eth_free = balance['free'].get('ETH', 0)
        
        print(f"\n[Balance Check - Pre-Grid]")
        print(f"  Total Equity: ${total_equity:.2f}")
        print(f"  USDT: ${total_usdt:.2f} (Free: ${usdt_free:.2f})")
        print(f"  ETH: {total_eth:.4f} (Free: {eth_free:.4f})")
        print(f"  ETH Value: ${total_eth * center_price:.2f}")
        
        # DYNAMIC GRID SPACING (Passivbot Enhancement)
        self.update_volatility_ema()
        spacing_pct = self.calculate_grid_spacing()
        
        up_range = center_price * (1 + spacing_pct)
        down_range = center_price * (1 - spacing_pct)
        
        print(f"\n[Grid Parameters]")
        print(f"  Spacing: {spacing_pct:.2%} (Base: {self.grid_spacing_base:.2%}, Vol: {self.volatility_ema:.4f}, WE: {self.current_wallet_exposure:.2f})")
        print(f"  Price Range: ${down_range:.2f} - ${up_range:.2f}")
        
        # 3. DYNAMIC BUY GRID (Based on Available USDT)
        max_buy_lines = 5  # Target
        buy_prices = pd.interval_range(start=down_range, end=center_price, periods=max_buy_lines).mid
        
        if usdt_free > 20:  # Minimum threshold
            # Calculate how much USDT per buy order (use 90% of free USDT)
            usdt_for_buys = usdt_free * 0.90
            qty_per_buy = (usdt_for_buys / max_buy_lines) / center_price
            qty_per_buy = float(self.exchange.amount_to_precision(self.symbol, qty_per_buy))
            
            print(f"\n[Buy Grid] Deploying {max_buy_lines} orders")
            print(f"  USDT Available: ${usdt_free:.2f}")
            print(f"  USDT Per Order: ${usdt_for_buys / max_buy_lines:.2f}")
            print(f"  Qty Per Order: {qty_per_buy:.4f} ETH")
            
            buy_orders_placed = 0
            remaining_usdt = usdt_free
            
            for idx, p in enumerate(buy_prices):
                try:
                    price = float(self.exchange.price_to_precision(self.symbol, p))
                    cost = qty_per_buy * price
                    
                    # Validate balance before each order
                    if remaining_usdt >= cost and cost >= 6.0:
                        order = self.exchange.create_limit_buy_order(self.symbol, qty_per_buy, price)
                        self.log_trade("GRID", "BUY_LIMIT", price, qty_per_buy, order['id'], "LIMIT", "OPEN")
                        remaining_usdt -= cost
                        buy_orders_placed += 1
                        print(f"  ✓ Buy #{idx+1}: {qty_per_buy:.4f} ETH @ ${price:.2f} (Cost: ${cost:.2f})")
                    else:
                        if cost < 6.0:
                            print(f"  ✗ Buy #{idx+1}: Skipped (value ${cost:.2f} too small)")
                        else:
                            print(f"  ✗ Buy #{idx+1}: Skipped (insufficient USDT: ${remaining_usdt:.2f} < ${cost:.2f})")
                        
                except Exception as e: 
                    print(f"  ✗ Buy #{idx+1}: Failed - {e}")
            
            print(f"  Summary: {buy_orders_placed}/{max_buy_lines} buy orders placed")
        else:
            print(f"\n[Buy Grid] SKIPPED - Insufficient USDT (${usdt_free:.2f} < $20)")

        # 4. DYNAMIC SELL GRID (Based on Available ETH)
        max_sell_lines = 5  # Target
        sell_prices = pd.interval_range(start=center_price, end=up_range, periods=max_sell_lines).mid
        
        if eth_free > 0.001:  # Minimum threshold
            # MARTINGALE GRID SIZING (Passivbot Strategy)
            base_qty = eth_free * 0.05  # 5% base quantity
            multipliers = [1, 2, 4, 8, 5]  # 5%, 10%, 20%, 40%, 25% (total: 100%)
            
            print(f"\n[Sell Grid] Deploying {max_sell_lines} orders (Martingale)")
            print(f"  ETH Available: {eth_free:.4f}")
            print(f"  Base Qty: {base_qty:.4f} (5% of free ETH)")
            
            sell_orders_placed = 0
            total_eth_allocated = 0
            
            for idx, (p, mult) in enumerate(zip(sell_prices, multipliers)):
                try:
                    price = float(self.exchange.price_to_precision(self.symbol, p))
                    sell_qty = base_qty * mult
                    sell_qty = float(self.exchange.amount_to_precision(self.symbol, sell_qty))
                    value = sell_qty * price
                    
                    # Validate balance before each order
                    # Check if we have enough ETH remaining (accounting for already allocated)
                    if (total_eth_allocated + sell_qty) <= eth_free and value >= 5.0:
                        order = self.exchange.create_limit_sell_order(self.symbol, sell_qty, price)
                        self.log_trade("GRID", "SELL_LIMIT", price, sell_qty, order['id'], "LIMIT", "OPEN")
                        total_eth_allocated += sell_qty
                        sell_orders_placed += 1
                        pct_of_free = (sell_qty / eth_free) * 100
                        print(f"  ✓ Sell #{idx+1}: {sell_qty:.4f} ETH @ ${price:.2f} ({pct_of_free:.0f}% of free, ${value:.2f})")
                    else:
                        if value < 5.0:
                            print(f"  ✗ Sell #{idx+1}: Skipped (value ${value:.2f} too small)")
                        else:
                            remaining_eth = eth_free - total_eth_allocated
                            print(f"  ✗ Sell #{idx+1}: Skipped (insufficient ETH: {remaining_eth:.4f} < {sell_qty:.4f})")
                        
                except Exception as e: 
                    print(f"  ✗ Sell #{idx+1}: Failed - {e}")
            
            print(f"  Summary: {sell_orders_placed}/{max_sell_lines} sell orders placed")
            print(f"  Total ETH Allocated: {total_eth_allocated:.4f}/{eth_free:.4f} ({(total_eth_allocated/eth_free)*100:.0f}%)")
        else:
            print(f"\n[Sell Grid] SKIPPED - Insufficient ETH ({eth_free:.4f} < 0.001)")
            
        # 5. Post-Grid Balance Check
        post_usdt, post_eth, post_equity = self.fetch_equity()
        post_balance = self.exchange.fetch_balance()
        post_usdt_free = post_balance['free'].get('USDT', 0)
        post_eth_free = post_balance['free'].get('ETH', 0)
        
        print(f"\n[Balance Check - Post-Grid]")
        print(f"  USDT Free: ${usdt_free:.2f} → ${post_usdt_free:.2f} (Δ: ${post_usdt_free - usdt_free:.2f})")
        print(f"  ETH Free: {eth_free:.4f} → {post_eth_free:.4f} (Δ: {post_eth_free - eth_free:.4f})")
        print(f"{'='*70}\n")
        
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
            
            # 3. Calc Current ETH Value
            balance = self.exchange.fetch_balance()
            eth_bal = balance['total'].get('ETH', 0)
            current_val = eth_bal * price
            
            # 3. Calc Deficit
            deficit_usdt = target_value - current_val
            
            if deficit_usdt < 10.0: # Minimum interaction threshold ($10)
                print(f"Already positioned (Deficit ${deficit_usdt:.2f} < $10). Holding.")
                return
            
            # 4. Calc Qty to Buy
            qty = deficit_usdt / price
            
            # 5. Check actual USDT validity and use available balance
            usdt_bal = balance['free'].get('USDT', 0)
            if usdt_bal < (qty * price):
                # Use whatever USDT is available (99% for fees)
                print(f"[Balance Safety] Calculated qty needs ${qty * price:.2f}, but only ${usdt_bal:.2f} available")
                print(f"[Balance Safety] Using available balance instead")
                max_buy_val = usdt_bal * 0.99  # Safety for fees
                qty = max_buy_val / price
                
                # If still too small, skip
                if qty * price < 6.0:
                    print(f"[Balance Safety] Available balance ${qty * price:.2f} too small. Skipping.")
                    return
            
        elif side == 'sell':
            # For SHORT, sell ETH to reach target
            # Use available (free) ETH balance
            eth_bal = balance['free'].get('ETH', 0)
            
            if eth_bal < 0.001:
                print("[Balance Safety] No free ETH available to sell.")
                return
            
            # Calculate qty needed, but cap at available
            qty_needed = deficit_usdt / price if 'deficit_usdt' in locals() else eth_bal
            qty = min(qty_needed, eth_bal * 0.99)  # Use available, with safety margin
            
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


    def run(self):
        print("Bot Started. Press Ctrl+C to stop.")
        
        # State Tracking
        last_prediction_time = None
        current_signal = "NEUTRAL"
        current_grid_center = None
        
        # Initialize balance tracking for colored logging
        total_usdt, total_eth, total_equity = self.fetch_equity()
        self.last_eth_balance = total_eth
        self.last_usdt_balance = total_usdt
        
        PREDICTION_INTERVAL = 4 * 3600 # 4 Hours
        
        while True:
            try:
                now = time.time()
                time_since_last = 0  # Initialize to prevent reference errors
                
                # --- 1. PREDICTION LOGIC (Every 4 Hours) ---
                if last_prediction_time is None or (now - last_prediction_time >= PREDICTION_INTERVAL):
                    print("\n[AI] Time for new prediction...")
                    signal, price, pred = self.get_signal()
                    
                    if signal is None:
                        print(">> ABORT: Missing Data. Skipping this cycle & Waiting 60s.")
                        time.sleep(60) 
                        continue
                        
                    current_signal = signal
                    last_prediction_time = now
                    print(f"[AI] New Signal: {signal} | Next prediction in 4 hours")
                    
                    # Force Grid Reset on new signal
                    if signal == "GRID":
                        current_grid_center = None 
                else:
                    # Just fetch current price for maintenance
                    price = self.get_market_price()
                    time_since_last = now - last_prediction_time
                    time_until_next = PREDICTION_INTERVAL - time_since_last
                    hours_until = time_until_next / 3600
                    # Print this occasionally (every 10 minutes)
                    if int(time_since_last) % 600 < 60:
                        print(f"[Timer] Next AI prediction in {hours_until:.1f} hours")
                    
                # --- 2. AUTO-UNSTUCK CHECK ---
                # Check if position needs unstucking before executing strategy
                check_interval = 5 if self.aggressive_mode else 60
                if last_prediction_time and int(time_since_last) % 60 < check_interval:
                    self.auto_unstuck()
                
                # --- 3. TRAILING CLOSE CHECK ---
                # Check for profitable trailing close opportunities
                if last_prediction_time and int(time_since_last) % 60 < check_interval:
                    self.check_trailing_close(price)
                
                # --- 4. CLOSE GRID DEPLOYMENT (Every 5 Minutes) ---
                # Deploy TP grid if position is profitable
                if int(time.time()) % 300 < check_interval:
                    self.deploy_close_grid()
                
                # --- 5. GRID MANAGEMENT ---
                if current_signal == "GRID":
                    if current_grid_center is None:
                        # Deploy Initial Grid
                        print(f"\n[Grid] Deploying initial grid...")
                        self.execute_grid(price)
                        current_grid_center = price
                    else:
                        if self.aggressive_mode:
                            # AGGRESSIVE MODE: Proactive grid maintenance (Passivbot)
                            self.maintain_ideal_grid(price)
                            
                            # Update grid center if price deviates significantly
                            deviation = abs(price - current_grid_center) / current_grid_center
                            if deviation > 0.02:  # 2% deviation
                                print(f"\n[Grid] Price deviated {deviation*100:.1f}% from center. Recentering...")
                                self.execute_grid(price)
                                current_grid_center = price
                        else:
                            # CONSERVATIVE MODE: Reactive grid replenishment (Original)
                            try:
                                open_orders = self.exchange.fetch_open_orders(self.symbol)
                                
                                # If we have fewer than 4 orders, grid needs replenishment
                                if len(open_orders) < 4:
                                    print(f"\n{'='*70}")
                                    print(f"[Grid Replenishment] {len(open_orders)} orders remaining (< 4)")
                                    print(f"{'='*70}")
                                    
                                    # Show what orders are still open
                                    if open_orders:
                                        print(f"[Remaining Orders]")
                                        for order in open_orders:
                                            side = order['side'].upper()
                                            print(f"  {side}: {order['amount']:.4f} ETH @ ${order['price']:.2f}")
                                    
                                    # Fetch current balance to show what changed
                                    current_usdt, current_eth, current_equity = self.fetch_equity()
                                    
                                    # Determine if we bought or sold by comparing ETH balance
                                    if not hasattr(self, 'last_eth_balance'):
                                        self.last_eth_balance = current_eth
                                        self.last_usdt_balance = current_usdt
                                    
                                    eth_change = current_eth - self.last_eth_balance
                                    usdt_change = current_usdt - self.last_usdt_balance
                                    
                                    # Color code based on what happened
                                    if eth_change > 0.0001:  # Bought ETH
                                        color = Colors.GREEN
                                        action = "BOUGHT ETH"
                                    elif eth_change < -0.0001:  # Sold ETH
                                        color = Colors.RED
                                        action = "SOLD ETH"
                                    else:
                                        color = Colors.RESET
                                        action = "NO CHANGE"
                                    
                                    print(f"\n{color}[Balance Changes - {action}]{Colors.RESET}")
                                    print(f"{color}  USDT: ${self.last_usdt_balance:.2f} → ${current_usdt:.2f} (Δ: ${usdt_change:+.2f}){Colors.RESET}")
                                    print(f"{color}  ETH: {self.last_eth_balance:.4f} → {current_eth:.4f} (Δ: {eth_change:+.4f}){Colors.RESET}")
                                    print(f"  Total Equity: ${current_equity:.2f}")
                                    
                                    # Update tracking
                                    self.last_eth_balance = current_eth
                                    self.last_usdt_balance = current_usdt
                                    
                                    print(f"\n[Action] Redeploying grid with updated balances...")
                                    self.execute_grid(price)
                                    current_grid_center = price
                                else:
                                    # Check Deviation
                                    deviation = abs(price - current_grid_center) / current_grid_center
                                    if deviation > 0.02:  # 2% Deviation Threshold
                                        print(f"Price deviation {deviation*100:.2f}% > 2%. Recentering Grid...")
                                        self.execute_grid(price)
                                        current_grid_center = price
                                    else:
                                        print("Grid is within range. holding.")
                            except Exception as e:
                                print(f"[Grid] Error checking orders: {e}")
                                print("Grid is within range. holding.")
                            
                else:
                    # Trend Logic (Maintain Position)
                    # Only execute every 60 seconds to avoid excessive trades
                    if last_prediction_time and int(time_since_last) % 60 < check_interval:
                        self.execute_trend(current_signal)
                    
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
    parser.add_argument('--symbol', type=str, default='ETH/USDT', help='Trading Pair')
    parser.add_argument('--amount', type=float, default=1000, help='Capital Limit in USDT')
    parser.add_argument('--live', action='store_true', help='ENABLE REAL MONEY TRADING (Mainnet). Default is False (Testnet).')
    parser.add_argument('--aggressive', action='store_true', help='Enable aggressive mode (5s checks, proactive grid). Default is conservative (60s checks, reactive grid).')
    
    args = parser.parse_args()
    
    bot = BinanceBot(
        symbol=args.symbol, 
        capital_limit=args.amount,
        live_mode=args.live,
        aggressive_mode=args.aggressive
    )
    bot.run()
