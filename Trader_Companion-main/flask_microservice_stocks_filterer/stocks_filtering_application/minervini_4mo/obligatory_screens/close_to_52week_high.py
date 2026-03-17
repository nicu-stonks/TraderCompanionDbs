import csv
from datetime import datetime

def process_stocks(input_file, output_file):
    stocks = {}
    skipped_rows = 0
    
    # Read the CSV file and process the data
    with open(input_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row_num, row in enumerate(csv_reader, start=2):  # Start at 2 to account for header row
            symbol = row['Symbol']
            date_str = row['Date'].split()[0] if row['Date'] else None
            close_price_str = row['Close']
            
            # Skip rows with missing data
            if not symbol or not date_str or not close_price_str:
                skipped_rows += 1
                continue
            
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
                close_price = float(close_price_str)
            except ValueError as e:
                print(f"Error processing row {row_num}: {e}")
                skipped_rows += 1
                continue
            
            if symbol not in stocks:
                stocks[symbol] = {'prices': [], 'dates': []}
            
            stocks[symbol]['prices'].append(close_price)
            stocks[symbol]['dates'].append(date)

    # Find stocks that are at least 25% close to their 52-week high
    qualified_stocks = []
    for symbol, data in stocks.items():
        prices = data['prices']
        dates = data['dates']
        
        if not prices or not dates:
            continue
        
        # Find the 52-week period
        latest_date = max(dates)
        one_year_ago = latest_date.replace(year=latest_date.year - 1)
        
        # Filter prices within the last 52 weeks
        prices_52_weeks = [price for price, date in zip(prices, dates) if date >= one_year_ago]
        
        if prices_52_weeks:
            high_52_week = max(prices_52_weeks)
            current_price = prices[-1]
            
            # Check if the current price is within 25% of the 52-week high
            if current_price >= high_52_week * 0.75:
                qualified_stocks.append(symbol)

    # Write the qualified stocks to a new CSV file
    with open(output_file, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Symbol'])
        for symbol in qualified_stocks:
            csv_writer.writerow([symbol])

    print(f"Close to 52-week high analysis complete. {len(qualified_stocks)} stocks meeting the criteria have been saved to {output_file}.")
    print(f"Skipped {skipped_rows} rows due to missing or invalid data.")

# Usage
import os

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Find the absolute path of the "flask_microservice_stocks_filterer" directory
while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
    script_dir = os.path.dirname(script_dir)

# Append the correct relative path to the input and output files
input_file = os.path.join(script_dir, "stocks_filtering_application", "price_data", "all_tickers_historical.csv")
output_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_4mo", "obligatory_screens", "results", "close_to_52week_high.csv")

process_stocks(input_file, output_file)