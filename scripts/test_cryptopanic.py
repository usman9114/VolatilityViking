import os
import sys
import requests
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.getcwd())

load_dotenv()

def test_cryptopanic():
    api_key = os.getenv('CRYPTOPANIC_API_KEY')
    print(f"API Key Present: {bool(api_key)}")
    
    if not api_key:
        print("Error: CRYPTOPANIC_API_KEY is missing in env.")
        return

    urls = [
        "https://cryptopanic.com/api/v1/posts/",
        "https://cryptopanic.com/api/v1/posts",
        "https://public-api.cryptopanic.com/api/v1/posts/"
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    params = {
        "auth_token": api_key,
        "kind": "news",
        "filter": "important", 
        "currencies": "ETH",
        "limit": 5
    }

    for u in urls:
        print(f"\nRequesting: {u}...")
        try:
            response = requests.get(u, params=params, headers=headers, timeout=10)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'results' in data:
                        print("SUCCESS! Data received.")
                        for post in data['results'][:3]:
                            print(f" - {post.get('title', 'No Title')}")
                        break
                    else:
                        print(f"JSON decoded but no 'results': {data.keys()}")
                except Exception as json_err:
                    print(f"Failed to decode JSON: {json_err}")
                    print(f"Response Preview: {response.text[:200]}")
            else:
                print(f"Request failed with status {response.status_code}")
                # print(f"Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"Connection Error: {e}")

if __name__ == "__main__":
    test_cryptopanic()
