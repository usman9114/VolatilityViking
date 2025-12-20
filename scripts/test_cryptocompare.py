import requests
import pandas as pd

def test_cryptocompare():
    url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
    print(f"Requesting: {url}...")
    
    try:
        response = requests.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if 'Data' in data:
                print(f"Success! Found {len(data['Data'])} articles.")
                for item in data['Data'][:3]:
                    print(f" - {item.get('published_on')}: {item.get('title')}")
            else:
                print("No 'Data' field in response.")
        else:
            print(f"Failed. Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_cryptocompare()
