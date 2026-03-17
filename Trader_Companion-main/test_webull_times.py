import sys
import os
import pprint

sys.path.append(os.path.abspath('Buy_Seller/ticker_data_fetcher'))
sys.path.append(os.path.abspath('.'))

import server

def test_webull_times():
    print("Testing Webull 5m times for ECO")
    server.stock_server.data_provider = 'webull'
    
    # get 5m
    hist_5m = server.stock_server._webull_history_5m('ECO')
    
    if hist_5m.empty:
        print("No data for ECO.")
        return
        
    print(f"Total bars: {len(hist_5m)}")
    print("First 20 bars:")
    print(hist_5m.head(20))
    print("\nLast 20 bars:")
    print(hist_5m.tail(20))

if __name__ == "__main__":
    test_webull_times()
