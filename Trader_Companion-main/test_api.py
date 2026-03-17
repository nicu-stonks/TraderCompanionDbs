import requests

try:
    url = "http://localhost:8000/api/nutrients_tracker/chat/2026-02-23/"
    payload = {"prompt": "test"}
    resp = requests.post(url, json=payload)
    print("Status code:", resp.status_code)
    try:
        print("Response JSON:", resp.json())
    except:
        print("Response Text:", resp.text)
except Exception as e:
    print("Request failed:", e)
