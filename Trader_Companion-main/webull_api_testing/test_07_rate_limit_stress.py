import time
import concurrent.futures
import sys
import os
from webull_utils import load_and_login

SYMBOLS = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC', 
           'SPY', 'QQQ', 'IWM', 'DIA', 'V', 'JPM', 'BAC', 'WMT', 'TGT', 'DIS']

def fetch_data(wb, symbol):
    start = time.time()
    try:
        quote = wb.get_quote(stock=symbol)
        price = float(quote.get('close', quote.get('price', 0))) if quote else 0.0
        vol = quote.get('volume', 0)
        return symbol, price, vol, time.time() - start, True
    except Exception:
        return symbol, 0, 0, time.time() - start, False

def loop(wb):
    print("="*60)
    print(f"WEBULL RATE LIMIT STRESS TEST")
    print("Press Ctrl+C to stop.")
    print("="*60)
    
    start_time = time.time()
    TARGET_INTERVAL = 2.0
    
    try:
        while True:
            iter_start = time.time()
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(fetch_data, wb, sym): sym for sym in SYMBOLS}
                for f in concurrent.futures.as_completed(futures):
                    results.append(f.result())
            
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"STRESS TEST | Time: {int(time.time() - start_time)}s")
            print("-" * 55)
            for sym, price, vol, lat, success in sorted(results, key=lambda x: x[0]):
                status = f"${price:.2f}" if success else "ERROR"
                print(f"{sym:<8} | {status:<12} | {vol:<15} | {lat:.2f}s")
            
            elapsed = time.time() - iter_start
            time.sleep(max(0, TARGET_INTERVAL - elapsed))
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    wb = load_and_login()
    if wb: loop(wb)
