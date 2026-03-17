from webull_utils import load_and_login
import concurrent.futures
import time

SYMBOLS = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC', 
           'SPY', 'QQQ', 'IWM', 'DIA', 'V', 'JPM', 'BAC', 'WMT', 'TGT', 'DIS']

def fetch_price(wb, symbol):
    start = time.time()
    try:
        quote = wb.get_quote(stock=symbol)
        price = quote.get('close', quote.get('price', 0)) if quote else None
        return symbol, price, time.time() - start
    except Exception as e:
        return symbol, None, time.time() - start

def run_batch_test(wb):
    print(f"\n--- Batch Fetching {len(SYMBOLS)} Tickers ---")
    start_time = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_price, wb, sym): sym for sym in SYMBOLS}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    total_time = time.time() - start_time
    print("-" * 50)
    print(f"{'Ticker':<8} | {'Price':<10} | {'Latency (s)':<12}")
    print("-" * 50)
    for sym, price, dur in sorted(results, key=lambda x: x[0]):
        price_str = "FAIL"
        if price is not None:
            try:
                val = float(price)
                price_str = f"${val:.2f}"
            except:
                price_str = str(price)
        print(f"{sym:<8} | {price_str:<10} | {dur:.3f}s")
    print("-" * 50)
    print(f"Total Batch Time: {total_time:.3f}s")
    print(f"Throughput:       {len(SYMBOLS)/total_time:.1f} tickers/sec")

if __name__ == "__main__":
    wb = load_and_login()
    if wb:
        run_batch_test(wb)
