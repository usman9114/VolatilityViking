"""Quick test for news fetcher per-symbol filtering."""
import sys
import os
sys.path.append(os.getcwd())

from src.data.news_fetcher import NewsFetcher

def test_news():
    fetcher = NewsFetcher()
    
    print("=" * 50)
    print("Testing News Fetcher Per-Symbol Filtering")
    print("=" * 50)
    
    coins = ['ETH', 'BTC', 'SOL']
    
    for coin in coins:
        print(f"\n--- News for {coin} ---")
        df = fetcher.fetch_news(coin, limit=20)
        
        if df.empty:
            print(f"  No news found for {coin}")
        else:
            print(f"  Found {len(df)} articles:")
            for title in df['title'].head(3).tolist():
                print(f"    - {title[:70]}...")
                
    print("\n✅ Test complete - check if news differs per coin")

if __name__ == "__main__":
    test_news()
