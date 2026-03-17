import csv
from datetime import datetime, timedelta

def process_stocks(input_file, output_file):
    stocks = {}
    
    # Read the CSV file and process the data
    with open(input_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            symbol = row['Symbol']
            date = datetime.strptime(row['Date'].split()[0], '%Y-%m-%d')
            close_price = float(row['Close'])
            
            if symbol not in stocks:
                stocks[symbol] = {'prices': [], 'dates': []}
            
            stocks[symbol]['prices'].append(close_price)
            stocks[symbol]['dates'].append(date)

    # Find stocks with low volatility in the last 10 days
    qualified_stocks = []
    for symbol, data in stocks.items():
        prices = data['prices']
        dates = data['dates']
        
        # Find the last 10 days period
        latest_date = max(dates)
        ten_days_ago = latest_date - timedelta(days=10)
        
        # Filter prices within the last 10 days
        prices_2_weeks = [price for price, date in zip(prices, dates) if date >= ten_days_ago]
        
        if prices_2_weeks:
            low_2_weeks = min(prices_2_weeks)
            high_2_weeks = max(prices_2_weeks)
            
            # Check if the difference between high and low is not more than 10%
            if high_2_weeks <= low_2_weeks * 1.15:
                qualified_stocks.append(symbol)

    # Write the qualified stocks to a new CSV file
    with open(output_file, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Symbol'])
        for symbol in qualified_stocks:
            csv_writer.writerow([symbol])

    print(f"Low volatility analysis complete. {len(qualified_stocks)} stocks meeting the criteria have been saved to {output_file}.")

# Usage
input_file = '../../stock_api_data/amex_arca_bats_nasdaq_nyse_otc_stocks_1_year_price_data.csv'
output_file = './obligatory_screens/results/low_volatility_2weeks.csv'
process_stocks(input_file, output_file)