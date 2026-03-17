"""
Quick test: compare yfinance fetch times for different period strings.
Run from the repo root: python test_yf_periods.py
"""
import time
import yfinance as yf

SYMBOLS = ['SPY', 'AAPL']
PERIODS = ['2y', '5y', '10y', 'max']
SLEEP_BETWEEN = 0.5

results = []

for symbol in SYMBOLS:
    ticker = yf.Ticker(symbol)
    for period in PERIODS:
        time.sleep(SLEEP_BETWEEN)
        t0 = time.perf_counter()
        df = ticker.history(period=period, interval='1d')
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        bars = len(df)
        first = str(df.index[0].date()) if not df.empty else 'N/A'
        last  = str(df.index[-1].date()) if not df.empty else 'N/A'
        results.append((symbol, period, elapsed, bars, first, last))
        print(f"{symbol:6s}  {period:5s}  {elapsed:7.1f}ms  {bars:5d} bars  {first} → {last}")

print()
print("Summary (ms):")
print(f"{'Symbol':<8} {'Period':<6} {'ms':>8} {'Bars':>6}")
for symbol, period, elapsed, bars, first, last in results:
    print(f"{symbol:<8} {period:<6} {elapsed:>8.1f} {bars:>6}")
