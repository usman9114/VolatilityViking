
import os
import sys
import json
import argparse
import ccxt
from dotenv import load_dotenv

def verify_credentials():
    parser = argparse.ArgumentParser(description="Verify Binance API Credentials")
    parser.add_argument('--env-file', default='.env', help='Path to .env file')
    parser.add_argument('--config', default='config.json', help='Path to config.json')
    args = parser.parse_args()

    # 1. Load Environment Variables
    if not os.path.exists(args.env_file):
        print(f"❌ Error: Env file '{args.env_file}' not found.")
        sys.exit(1)
    
    print(f"📖 Loading environment from: {args.env_file}")
    load_dotenv(args.env_file, override=True)

    # 2. Load Config for Mode (Live vs Testnet)
    is_live = False
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
                is_live = config.get('live_mode', False)
                print(f"⚙️  Config loaded: {'LIVE' if is_live else 'TESTNET'} mode detected.")
        except Exception as e:
            print(f"⚠️  Warning: Could not load {args.config} ({e}). Defaulting to TESTNET.")
    else:
         print(f"⚠️  Warning: {args.config} not found. Defaulting to TESTNET.")

    # 3. Select Keys
    if is_live:
        api_key = os.getenv('BINANCE_API_KEY')
        secret = os.getenv('BINANCE_SECRET_KEY') or os.getenv('BINANCE_SECRET')
        base_url = None # Default
    else:
        api_key = os.getenv('BINANCE_TESTNET_API_KEY')
        secret = os.getenv('BINANCE_TESTNET_SECRET_KEY')
        base_url = 'https://testnet.binancefuture.com'

    if not api_key or not secret:
        print("❌ Error: Missing API Key or Secret for the selected mode.")
        print(f"   Checked vars: {'BINANCE_API_KEY' if is_live else 'BINANCE_TESTNET_API_KEY'}")
        sys.exit(1)
        
    print(f"🔑 Keys found. Testing connection...")

    # 4. Connect with CCXT
    try:
        # Default options to future since that's what the bot mostly uses
        exchange_class = getattr(ccxt, 'binance')
        exchange = exchange_class({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'} 
        })
        
        if not is_live:
            exchange.set_sandbox_mode(True)
            # Some ccxt versions need manual URL override for futures testnet if sandbox mode doesn't hit the right one
            # But set_sandbox_mode(True) is usually enough for binance
        
        # 5. Active Check
        print("📡 Sending request to Binance (fetch_balance)...")
        balance = exchange.fetch_balance()
        
        usdt_bal = balance.get('USDT', {}).get('total', 0)
        print("\n✅ SUCCESS: Connection Established!")
        print(f"   Account Mode: {'LIVE' if is_live else 'TESTNET'}")
        print(f"   USDT Balance: {usdt_bal}")
        
    except ccxt.AuthenticationError:
        print("\n❌ FAILURE: Authentication Error. Check your API Key and Secret.")
    except ccxt.NetworkError:
        print("\n❌ FAILURE: Network Error. Check your internet connection or proxy.")
    except Exception as e:
        print(f"\n❌ FAILURE: An unexpected error occurred: {e}")
    finally:
        pass

if __name__ == "__main__":
    verify_credentials()
