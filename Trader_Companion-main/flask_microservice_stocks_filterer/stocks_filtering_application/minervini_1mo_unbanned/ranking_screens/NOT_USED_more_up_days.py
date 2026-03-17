import csv
from collections import defaultdict
from datetime import datetime

def process_csv(input_file, output_file):
    stocks = defaultdict(list)
    
    # Read the input CSV file
    with open(input_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            date = datetime.strptime(row['Date'].split()[0], '%Y-%m-%d')
            close = float(row['Close'])
            open_price = float(row['Open'])
            symbol = row['Symbol']
            
            stocks[symbol].append((date, close > open_price))
    
    # Process each stock
    qualifying_stocks = []
    for symbol, data in stocks.items():
        # Sort data by date in descending order
        sorted_data = sorted(data, key=lambda x: x[0], reverse=True)
        
        # Check the last 15 days
        if len(sorted_data) >= 15:
            up_days = sum(1 for _, is_up in sorted_data[:15] if is_up)
            if up_days >= 10:
                qualifying_stocks.append((symbol, up_days))
    
    # Write qualifying stocks to output CSV
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Symbol', 'UpDays'])
        for symbol, up_days in qualifying_stocks:
            writer.writerow([symbol, up_days])

    print(f"More up days Analysis complete. {len(qualifying_stocks)} qualifying stocks saved to {output_file}")

# Usage
input_file = './ranking_screens/passed_stocks_input_data/filtered_price_data.csv'
output_file = './ranking_screens/results/more_up_days_stocks.csv'
process_csv(input_file, output_file)
