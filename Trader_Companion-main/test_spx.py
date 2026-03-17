import sys
import os

sys.path.append(os.path.abspath('Buy_Seller/ticker_data_fetcher'))
sys.path.append(os.path.abspath('.'))

import server

def test_spx():
    print("Testing '^GSPC' with Webull")
    
    server.stock_server.data_provider = 'webull'
    quote = server.stock_server.webull_manager.get_quote('^GSPC')
    print("Quote:")
    print(quote)
    
    daily = server.stock_server._webull_history_daily('^GSPC')
    print(f"Daily points: {len(daily)}")
    
    hist_5m = server.stock_server._webull_history_5m('^GSPC')
    print(f"5m points: {len(hist_5m)}")

if __name__ == "__main__":
    test_spx()
