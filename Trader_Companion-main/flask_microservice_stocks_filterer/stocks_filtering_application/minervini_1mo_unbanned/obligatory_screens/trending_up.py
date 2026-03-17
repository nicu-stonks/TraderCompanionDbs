import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_ma(data, window):
    return data['Close'].rolling(window=window).mean()

def check_conditions(group):
    # Check if we have enough data
    if len(group) < 200:
        return False

    # Get the last row of data
    last_row = group.iloc[-1]

    # Check all conditions
    conditions = [
        last_row['150MA'] > last_row['200MA'],
        last_row['50MA'] > last_row['150MA'],
        last_row['Close'] > last_row['50MA'],
        last_row['Close'] > last_row['150MA'],
        last_row['Close'] > last_row['200MA'],
        is_200ma_trending_up(group)
    ]

    return all(conditions)

def is_200ma_trending_up(group):
    # Reduce to last 30 trading days
    last_four_months = group.iloc[-30:]
    
    if len(last_four_months) < 30:
        return False  # Not enough data
    
    # Assuming approximately 20 trading days per month
    m1_ago = last_four_months['200MA'].iloc[-22]     # 1 month ago
    current = last_four_months['200MA'].iloc[-1]    # Current
    
    # Check if each month's 200MA is higher than the previous month
    return  current > m1_ago


def main():
    import os

    # Get the absolute path of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Find the absolute path of the "flask_microservice_stocks_filterer" directory
    while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
        script_dir = os.path.dirname(script_dir)

    # Define the input and output file paths
    input_file = os.path.join(script_dir, "stocks_filtering_application", "price_data", "all_tickers_historical.csv")
    output_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_1mo_unbanned", "obligatory_screens", "results", "trending_up_stocks.csv")

    print(f"Resolved input file path: {input_file}")
    print(f"Resolved output file path: {output_file}")
    # Read the CSV file
    df = pd.read_csv(input_file, parse_dates=['Date'])
    
    # Group the data by stock symbol
    grouped = df.groupby('Symbol')
    
    qualifying_symbols = []
    
    for symbol, group in grouped:
        # Sort by date
        group = group.sort_values('Date')
        
        # Calculate moving averages
        group['200MA'] = calculate_ma(group, 200)
        group['150MA'] = calculate_ma(group, 150)
        group['50MA'] = calculate_ma(group, 50)
        
        # Check all conditions
        if check_conditions(group):
            qualifying_symbols.append(symbol)
    
    # Save the symbols to a new CSV file
    pd.DataFrame({'Symbol': qualifying_symbols}).to_csv(output_file, index=False)
    print(f"Trending up analysis complete. Saved {len(qualifying_symbols)} symbols to {output_file}")

if __name__ == "__main__":
    main()