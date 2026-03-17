import json
import os
from webull import webull

SESSION_FILE = 'webull_session.json'

def load_and_login():
    wb = webull()
    
    if os.path.exists(SESSION_FILE):
        print(f"[INFO] Found session file {SESSION_FILE}")
        try:
            with open(SESSION_FILE, 'r') as f:
                session = json.load(f)
            
            token = session.get('accessToken') or session.get('access_token')
            if token:
                wb._access_token = token
                wb._refresh_token = session.get('refreshToken') or session.get('refresh_token')
                wb._uuid = session.get('uuid')
                wb._did = session.get('did')
                
                # Hydrate session headers for the internal requests object
                wb._session.headers.update({
                    'access_token': wb._access_token,
                    'did': wb._did,
                    'regionId': '1'
                })
                
                print(f"[OK] Session tokens loaded (Token: {token[:10]}...)")
                
                # Check if actually logged in by fetching account ID (lightweight call)
                try:
                    wb.get_account_id()
                    return wb
                except Exception as e:
                    print(f"[WARNING] Session might be expired or invalid: {e}")
                    return wb # Still return wb, maybe some endpoints work
        except Exception as e:
            print(f"[ERROR] Failed to load session file: {e}")
            
    print("[ERROR] No valid session found. Please run setup_manual_tokens.py first.")
    return None
