
import os
import sys
import argparse
from dotenv import load_dotenv

def check():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-file', default='.env')
    args = parser.parse_args()

    print(f"Loading from: {args.env_file}")
    load_dotenv(args.env_file, override=True)

    key = os.getenv('BINANCE_API_KEY')
    secret = os.getenv('BINANCE_SECRET_KEY') or os.getenv('BINANCE_SECRET')

    print(f"BINANCE_API_KEY: {'Found' if key else 'MISSING'}")
    if key:
        print(f"  Value: {key[:4]}...{key[-4:]} (Length: {len(key)})")

    print(f"BINANCE_SECRET:  {'Found' if secret else 'MISSING'}")
    if secret:
        print(f"  Value: {secret[:4]}...{secret[-4:]} (Length: {len(secret)})")

if __name__ == "__main__":
    check()
