import time
import threading
import queue
import sys
import os
from webull_utils import load_and_login

# 20 Tickers for demo
SYMBOLS = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC', 
           'SPY', 'QQQ', 'IWM', 'DIA', 'V', 'JPM', 'BAC', 'WMT', 'TGT', 'DIS']

class RateLimiter:
    def __init__(self, rate_per_sec):
        self.rate = rate_per_sec
        self.tokens = rate_per_sec
        self.max_tokens = rate_per_sec
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_update
                new_tokens = elapsed * self.rate
                if new_tokens > 0:
                    self.tokens = min(self.max_tokens, self.tokens + new_tokens)
                    self.last_update = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True
            time.sleep(0.01)

class SmartPoller:
    def __init__(self, wb, symbols, target_rps=2.0):
        self.wb = wb
        self.symbols = symbols
        self.queue = queue.Queue()
        self.limiter = RateLimiter(target_rps)
        self.running = True
        self.total_reqs = 0
        self.latency_sum = 0
        self.data_snapshot = {}
        self.stats_lock = threading.Lock()
        for s in symbols:
            self.queue.put(s)

    def worker_loop(self, worker_id):
        while self.running:
            try:
                sym = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue
            self.limiter.acquire()
            start_t = time.time()
            try:
                quote = self.wb.get_quote(stock=sym)
                price = float(quote.get('close', quote.get('price', 0))) if quote else 0.0
                lat = time.time() - start_t
                with self.stats_lock:
                    self.total_reqs += 1
                    self.latency_sum += lat
                    self.data_snapshot[sym] = (price, lat)
            except Exception as e:
                print(f"[Worker {worker_id}] Error: {e}")
                time.sleep(5) # Adaptive Backoff
            if self.running:
                self.queue.put(sym)

    def start(self, num_threads=3):
        self.threads = []
        for i in range(num_threads):
            t = threading.Thread(target=self.worker_loop, args=(i,))
            t.daemon = True
            t.start()
            self.threads.append(t)

    def stop(self):
        self.running = False
        for t in self.threads: t.join(timeout=1.0)

def main():
    wb = load_and_login()
    if not wb: return
    TARGET_RPS = 2.0 # Updated to safer limit
    poller = SmartPoller(wb, SYMBOLS, target_rps=TARGET_RPS)
    poller.start(num_threads=3)
    start_time = time.time()
    try:
        while True:
            time.sleep(1.0)
            os.system('cls' if os.name == 'nt' else 'clear')
            duration = time.time() - start_time
            with poller.stats_lock:
                reqs = poller.total_reqs
                lat_avg = (poller.latency_sum / reqs) if reqs > 0 else 0
                snapshot = dict(poller.data_snapshot)
            rps_real = reqs / duration if duration > 0 else 0
            print(f"WEBULL SMART POLLER | Time: {int(duration)}s | Avg Latency: {lat_avg*1000:.1f}ms")
            print(f"Target RPS: {TARGET_RPS} | Actual RPS: {rps_real:.1f}")
            print("-" * 50)
            print(f"{'Ticker':<8} | {'Price':<10} | {'Latency':<10}")
            print("-" * 50)
            for sym in sorted(SYMBOLS):
                data = snapshot.get(sym)
                if data:
                    price, lat = data
                    print(f"{sym:<8} | ${price:<9.2f} | {lat*1000:.0f}ms")
            print("-" * 50)
    except KeyboardInterrupt:
        poller.stop()

if __name__ == "__main__":
    main()
