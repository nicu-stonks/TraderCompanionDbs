"""
Standalone Webull API latency benchmark.
Measures get_bars() response time at different bar counts.
Run from repo root: python test_webull_speed.py
"""
import json
import time
import os

try:
    from webull import webull
except ImportError:
    print("webull package not installed - run: pip install webull")
    raise

CREDS_FILE = os.path.join(os.path.dirname(__file__), 'dbs', 'webull_credentials.json')
SYMBOL = 'SPY'
TEST_CASES = [
    ('d1', 10,   'daily 10'),
    ('m5', 5,    '5m 5'),
    ('m5', 50,   '5m 50'),
    ('m5', 100,  '5m 100'),
    ('m5', 200,  '5m 200'),
    ('m5', 500,  '5m 500'),
    ('m5', 1200, '5m 1200'),
]
REPS = 3  # repetitions per case to get avg

def load_session(wb):
    with open(CREDS_FILE) as f:
        creds = json.load(f)
    did = creds.get('did') or creds.get('device_id', '')
    wb._device_id = creds.get('device_id', '') or did
    wb._did = did
    wb._access_token  = creds.get('access_token', '')
    wb._refresh_token = creds.get('refresh_token', '')
    wb._uuid          = creds.get('uuid', '')
    wb._trade_token   = creds.get('trade_token', '')

def measure(wb, interval, count, label, reps):
    times = []
    rows  = 0
    for i in range(reps):
        t0 = time.perf_counter()
        try:
            df = wb.get_bars(stock=SYMBOL, interval=interval, count=count)
            elapsed = (time.perf_counter() - t0) * 1000
            rows = len(df) if df is not None and hasattr(df, '__len__') else 0
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  [{label} rep{i+1}] ERROR: {e}")
            rows = 0
        times.append(elapsed)
        time.sleep(0.3)  # small gap between calls
    avg = sum(times) / len(times)
    mn  = min(times)
    mx  = max(times)
    return avg, mn, mx, rows

def main():
    if not os.path.exists(CREDS_FILE):
        print(f"No credentials file at {CREDS_FILE} - login via the app first.")
        return

    wb = webull()
    load_session(wb)

    print(f"\nWebull API latency benchmark — symbol={SYMBOL}, reps={REPS}")
    print(f"{'Label':<14} {'Count':>6}  {'Avg ms':>8}  {'Min ms':>8}  {'Max ms':>8}  {'Rows':>6}")
    print("-" * 58)

    for interval, count, label in TEST_CASES:
        avg, mn, mx, rows = measure(wb, interval, count, label, REPS)
        print(f"{label:<14} {count:>6}  {avg:>8.0f}  {mn:>8.0f}  {mx:>8.0f}  {rows:>6}")

    print("\nDone.")
    print("\nConclusion:")
    print("  If 5m 200 is ~300-600ms but 5m 1200 is ~2500ms, the bar count is the bottleneck.")
    print("  At 300ms/fetch + limit=26 => ~33 tickers/10s (well beyond the limit).")
    print("  At 2600ms/fetch => max ~4 tickers/10s regardless of limit setting.")

if __name__ == '__main__':
    main()
