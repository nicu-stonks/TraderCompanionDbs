from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import os

SESSION_FILE = 'webull_session.json'

def main():
    print("="*60)
    print("WEBULL AUTOMATED LOGIN HELPER")
    print("="*60)
    print("This script launches a Chrome window for you to log in.")
    print("Once logged in (Dashboard visible), it grabs your tokens automatically.")
    
    try:
        # Prevent window closing immediately if script crashes
        options = webdriver.ChromeOptions()
        # options.add_argument("--headless") # Headless might trigger stricter checks
        
        print("\n[INIT] Launching Chrome...")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        print("[NAV] Going to https://app.webull.com/trade")
        driver.get("https://app.webull.com/trade")
        
        print("\n" + "*"*50)
        print("ACTION REQUIRED: Log in to your Webull account in the browser.")
        print("Complete 2FA if prompted.")
        print("*"*50 + "\n")
        
        print("[WAIT] Polling for successful login (tokens in LocalStorage)...")
        
        # Poll for accessToken
        tokens_found = False
        attempts = 0
        while not tokens_found:
            time.sleep(1)
            try:
                # 1. Attempt LocalStorage / SessionStorage
                token = driver.execute_script("return localStorage.getItem('accessToken') || sessionStorage.getItem('accessToken');")
                did = driver.execute_script("return localStorage.getItem('did') || sessionStorage.getItem('did');")
                
                # 2. Attempt Cookies (Selenium can see HTTP-only)
                cookies = driver.get_cookies()
                cookie_dict = {c['name']: c['value'] for c in cookies}
                
                if not token:
                    # Look for likely cookie names
                    # 'web_lt' (Long Term token) is often the Access Token in newer Webull Web
                    token = cookie_dict.get('web_lt') or cookie_dict.get('accessToken') or cookie_dict.get('wb_access_token')
                
                if not did:
                    did = cookie_dict.get('did') or cookie_dict.get('web_did')

                if token and did:
                    print(f"\n[SUCCESS] Login detected! Token: {token[:10]}... DID: {did}")
                    
                    # Try to get refresh/uuid
                    ref = driver.execute_script("return localStorage.getItem('refreshToken') || sessionStorage.getItem('refreshToken');")
                    if not ref: ref = cookie_dict.get('refreshToken') or cookie_dict.get('refresh_token')
                    
                    uuid = driver.execute_script("return localStorage.getItem('uuid') || sessionStorage.getItem('uuid');")
                    if not uuid: uuid = cookie_dict.get('web_uid') # Use web_uid cookie
                    
                    data = {
                        "accessToken": token,
                        "refreshToken": ref, # Might be empty, that's okay for manual session usage usually
                        "uuid": uuid if uuid else "generated_uuid",
                        "did": did,
                        "extracted_at": time.time()
                    }
                    
                    with open(SESSION_FILE, 'w') as f:
                        json.dump(data, f, indent=4)
                        
                    print(f"[SAVE] Tokens saved to {SESSION_FILE}")
                    tokens_found = True
                
                # Debug Output every 5 seconds
                if attempts % 5 == 0 and attempts > 0:
                     ls_keys = driver.execute_script("return Object.keys(localStorage);")
                     ss_keys = driver.execute_script("return Object.keys(sessionStorage);")
                     
                     print(f"\n[DEBUG] Waiting... LS: {len(ls_keys)} | SS: {len(ss_keys)} | Cookies: {len(cookies)}")
                     print(f"[DEBUG] LS Keys: {ls_keys}")
                     print(f"[DEBUG] SS Keys: {ss_keys}")
                     print(f"[DEBUG] Cookie Names: {list(cookie_dict.keys())}")
                    
            except Exception as e:
                print(f"Browser error: {e}")
                break
                
            attempts += 1

        print("\n[DONE] Closing browser in 3 seconds...")
        time.sleep(3)
        driver.quit()
        
    except Exception as e:
        print(f"\n[ERROR] Automation failed: {e}")
        print("Make sure you piped install: selenium webdriver-manager")

if __name__ == "__main__":
    main()
