#!/usr/bin/env python3
"""Quick test to verify Vector Store is working"""
import sys
import os
sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv

# Load env
load_dotenv('profiles/usman/.env')

from src.ai.rag_memory import MarketMemory

print("=" * 60)
print("VECTOR STORE TEST")
print("=" * 60)

try:
    print("\n1. Initializing MarketMemory...")
    memory = MarketMemory()
    print(f"   ✅ Connected! Items in DB: {memory.collection.count()}")
    
    print("\n2. Testing query...")
    test_query = "[REGIME: Neutral] [VOLATILITY: Squeeze] [MOMENTUM: Overbought] Details: RSI is 72.5. Price is 3.5% from 200 EMA."
    results = memory.query_similar_patterns(test_query, n_results=3)
    
    print(f"   ✅ Query returned {len(results)} results")
    
    if results:
        print("\n3. Sample result:")
        r = results[0]
        meta = r.get('metadata', {})
        print(f"   - Symbol: {meta.get('symbol', 'N/A')}")
        print(f"   - 4h Return: {meta.get('ret_4h', 0)*100:.2f}%")
        print(f"   - 24h Return: {meta.get('ret_24h', 0)*100:.2f}%")
        print(f"   - Distance: {r.get('distance', 0):.4f}")
    
    print("\n" + "=" * 60)
    print("✅ VECTOR STORE IS WORKING!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
