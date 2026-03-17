import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def calculate_ma(data, window):
    return data['Close'].rolling(window=window).mean()

def main():
    # Get the absolute path of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find the absolute path of the "flask_microservice_stocks_filterer" directory
    while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
        script_dir = os.path.dirname(script_dir)
    
    # Append the correct relative path to the input file
    input_file = os.path.join(script_dir, "stocks_filtering_application", "price_data", "all_tickers_historical.csv")
    
    # Define the output file
    output_file = os.path.join(script_dir, "stocks_filtering_application", "sentiment_graphs", "results", "above_200ma.csv")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read the CSV file
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Get the start and end dates from the data
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    
    print(f"Data range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # For each symbol, count number of days with data
    symbol_days = df.groupby('Symbol').size()
    
    # Filter to only use symbols with enough data for 200MA
    valid_symbols = symbol_days[symbol_days >= 200].index.tolist()
    
    print(f"Found {len(valid_symbols)} out of {len(symbol_days)} stocks with at least 200 days of data")
    
    # Filter dataframe to only include valid symbols
    df = df[df['Symbol'].isin(valid_symbols)]
    
    # Get unique dates after filtering
    # all_dates = sorted(df['Date'].unique())
    
    # Creating a dictionary to store results for each date
    results_dict = {}
    
    # Pre-calculate 200MA for each stock
    print("Pre-calculating 200-day moving averages...")
    
    # Track progress
    total_symbols = len(valid_symbols)
    for i, symbol in enumerate(valid_symbols):
        # Get data for this symbol
        stock_data = df[df['Symbol'] == symbol].sort_values('Date')
        
        # Calculate 200-day moving average
        stock_data['MA_200'] = calculate_ma(stock_data, 200)
        
        # For each date that has MA_200 (not NaN), check if price > MA
        for _, row in stock_data.iterrows():
            if pd.notna(row['MA_200']):
                date_str = row['Date'].strftime('%Y-%m-%d')
                
                # Initialize counters for this date if not already present
                if date_str not in results_dict:
                    results_dict[date_str] = {'total': 0, 'above_ma': 0}
                
                # Increment total counter
                results_dict[date_str]['total'] += 1
                
                # Check if price is above MA
                if row['Close'] > row['MA_200']:
                    results_dict[date_str]['above_ma'] += 1
        
        # Print progress every 100 stocks or when complete
        if (i + 1) % 100 == 0 or (i + 1) == total_symbols:
            print(f"Pre-calculating MA: {i + 1}/{total_symbols} stocks processed ({((i + 1)/total_symbols*100):.1f}%)")
    
    # Create results DataFrame
    results = []
    for date_str in sorted(results_dict.keys()):
        total = results_dict[date_str]['total']
        above_ma = results_dict[date_str]['above_ma']
        
        # Calculate percentage
        percentage = (above_ma / total * 100) if total > 0 else 0
        percentage = round(percentage, 2)  # Round to 2 decimal places
        
        # Print progress for each day
        print(f"Date: {date_str} - Stocks above 200MA: {above_ma}/{total} ({percentage}%)")
        
        results.append({
            'Date': date_str,
            'Percentage_Above_200MA': percentage
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV with updated column name
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file} with percentage values")
    print(f"Total trading days processed: {len(results_df)}")

if __name__ == "__main__":
    main()