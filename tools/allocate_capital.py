import sys
import os
import logging

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.live.momentum_allocator import MomentumAllocator

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    allocator = MomentumAllocator('config.json')
    allocator.run()

if __name__ == "__main__":
    main()
