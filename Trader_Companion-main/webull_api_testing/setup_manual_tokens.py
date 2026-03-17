import json
import os

SESSION_FILE = 'webull_session.json'

def main():
    print("--- Webull Manual Token Setup ---")
    print("Please follow the instructions in get_tokens.js to extract these from your browser.")
    
    access_token = input("Enter accessToken: ").strip()
    refresh_token = input("Enter refreshToken: ").strip()
    uuid = input("Enter uuid: ").strip()
    did = input("Enter did: ").strip()
    
    data = {
        "accessToken": access_token,
        "refreshToken": refresh_token,
        "uuid": uuid,
        "did": did
    }
    
    with open(SESSION_FILE, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"\n[SUCCESS] tokens saved to {SESSION_FILE}")
    print("You can now run test_02_price_quotes.py to verify.")

if __name__ == "__main__":
    main()
