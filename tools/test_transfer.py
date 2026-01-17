import ccxt
import os
import sys
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv('profiles/usman/.env')

def transfer_spot_to_future(amount=10):
    """
    Transfers USDT from Spot Wallet to USDT-M Futures Wallet.
    """
    api_key = os.getenv('BINANCE_API_KEY')
    secret = os.getenv('BINANCE_SECRET_KEY') or os.getenv('BINANCE_SECRET')

    if not api_key or not secret:
        print("Error: BINANCE_API_KEY or BINANCE_SECRET_KEY not found in env.")
        return

    print("Initializing Binance Connection...")
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': secret,
        'enableRateLimit': True
    })

    try:
        # 1. Check Spot Balance
        print("Fetching Spot Balance...")
        balance = exchange.fetch_balance()
        
        # 'free' contains available funds
        usdt_free = balance.get('USDT', {}).get('free', 0)
        print(f"Current Spot USDT Free: {usdt_free:.2f} USDT")

        if usdt_free < amount:
            print(f"❌ Insufficient funds. You need at least {amount} USDT in Spot.")
            return

        # 2. Execute Transfer
        # type: 'future' means USDT-M Futures
        print(f"🚀 Transferring {amount} USDT from Spot to Futures...")
        
        # ccxt unification: transfer(code, amount, fromAccount, toAccount)
        # For Binance: fromAccount='spot', toAccount='future'
        transfer_result = exchange.transfer('USDT', amount, 'spot', 'future')
        
        print("✅ Transfer Submitted!")
        print(f"Transaction ID: {transfer_result.get('id')}")
        print(f"Info: {transfer_result}")

    except Exception as e:
        print(f"❌ Transfer Failed: {e}")
        # Debugging info
        if "API key" in str(e):
            print("Tip: Check if your API Key has 'Enable Futures' AND 'Enable Spot & Margin Trading' permissions.")
            print("Tip: Also check 'Enable Universal Transfer' permit if applicable.")

if __name__ == "__main__":
    # Check if amount arg provided
    amount = 10
    if len(sys.argv) > 1:
        try:
            amount = float(sys.argv[1])
        except:
            pass
            
    transfer_spot_to_future(amount)
