import csv
import os
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

    # Find stocks with the last closing price at least $10
    qualified_stocks = []
    for symbol, data in stocks.items():
        if data['prices'] and data['dates']:
            last_price = data['prices'][-1]
            if last_price >= 10:
                qualified_stocks.append(symbol)

    # Write the qualified stocks to a new CSV file
    with open(output_file, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Symbol'])
        for symbol in qualified_stocks:
            csv_writer.writerow([symbol])

    print(f"Last price check complete. {len(qualified_stocks)} stocks meeting the criteria have been saved to {output_file}.")
    print(f"Skipped {skipped_rows} rows due to missing or invalid data.")

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Find the absolute path of the "flask_microservice_stocks_filterer" directory
while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
    script_dir = os.path.dirname(script_dir)

# Append the correct relative path to the input file
input_file = os.path.join(script_dir, "stocks_filtering_application", "price_data", "all_tickers_historical.csv")

# Define the output file
output_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_4mo", "obligatory_screens", "results", "last_price_above_10.csv")

print(f"Resolved input file path: {input_file}")
print(f"Resolved output file path: {output_file}")
process_stocks(input_file, output_file)