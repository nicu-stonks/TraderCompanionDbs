import csv
from datetime import datetime, timedelta

def process_recent_stocks(input_file, output_file, max_months=12):
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

    # Filter stocks trading for at most `max_months` months
    qualified_stocks = []
    for symbol, data in stocks.items():
        dates = data['dates']
        
        if not dates:
            continue
        
        first_date = min(dates)
        latest_date = max(dates)
        
        # Check if the stock has been trading for at most `max_months` months
        if (latest_date - first_date) <= timedelta(days=max_months * 30):
            qualified_stocks.append(symbol)

    # Write the qualified stocks to a new CSV file
    with open(output_file, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Symbol'])
        for symbol in qualified_stocks:
            csv_writer.writerow([symbol])

    print(f"Stocks trading for at most {max_months} months analysis complete. {len(qualified_stocks)} stocks meeting the criteria have been saved to {output_file}.")
    print(f"Skipped {skipped_rows} rows due to missing or invalid data.")

# Usage
import os

# Get the absolute path of the current script

script_dir = os.path.dirname(os.path.abspath(__file__))
# Find the absolute path of the "flask_microservice_stocks_filterer" directory
while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
    script_dir = os.path.dirname(script_dir)

# Append the correct relative path to the input file
input_file = os.path.join(script_dir, "stocks_filtering_application", "price_data", "all_tickers_historical.csv")

# Define the output file
output_file = os.path.join(script_dir, "stocks_filtering_application", "ipos", "obligatory_screens", "results", "trading_for_at_most_3mo.csv")

print(f"Resolved input file path: {input_file}")
print(f"Resolved output file path: {output_file}")
process_recent_stocks(input_file, output_file, max_months=12)