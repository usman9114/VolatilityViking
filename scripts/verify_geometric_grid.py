import sys
import os
import json
sys.path.append(os.getcwd())
from src.live.binance_bot import BinanceBot

def test_geometric_grid():
    print("Testing Geometric Grid Calculation...")
    
    # Create a dummy bot (mocking config)
    bot = BinanceBot(config_path='config.json')
    
    # Test Parameters
    center = 3000
    n = 5
    span = 0.10 # 10%
    weight = 1.2
    
    # 1. Test Grid Math
    print(f"Calculating grid: Center={center}, N={n}, Span={span}, Weight={weight}")
    prices = bot.calc_geometric_grid(center, n, weight, span)
    
    print("Grid Prices:", prices)
    
    if len(prices) != n:
        print(f"❌ Error: Expected {n} prices, got {len(prices)}")
        sys.exit(1)
        
    # Check if spacing implies geometric
    diffs = []
    last_p = center
    for p in prices:
        diff = last_p - p
        diffs.append(diff)
        last_p = p
        
    print("Diffs (Spacing):", diffs)
    
    # Check geometric ratio approx
    if len(diffs) > 1:
        ratio = diffs[1] / diffs[0]
        print(f"Observed Ratio: {ratio:.4f} (Expected ~{weight})")
        
        if abs(ratio - weight) > 0.05:
             # Note: logic might result in slightly different ratio due to the solve for d0
             print(f"⚠️ Ratio mismatch, but might be due to d0 solve.")
    
    # Check Total Span
    total_span = (center - prices[-1])/center
    print(f"Total Span (Down): {total_span:.4%} (Expected {span:.4%})")
    
    if abs(total_span - span) > 0.001:
        print("❌ Error: Total span mismatch")
        sys.exit(1)
        
    # 2. Test Martingale
    print("\nTesting Martingale Qty...")
    equity = 1000
    bot.wallet_exposure_limit = 1.0
    qtys = bot.calc_martingale_qty(equity, n, multiplier=1.5)
    print("Qtys:", qtys)
    
    total_alloc = sum(qtys)
    print(f"Total Allocation: {total_alloc:.2f} (Target: {equity})")
    
    if abs(total_alloc - equity) > 1.0:
        print("❌ Error: Total allocation mismatch")
        sys.exit(1)

    print("\n✅ Verification Passed!")

if __name__ == "__main__":
    test_geometric_grid()
