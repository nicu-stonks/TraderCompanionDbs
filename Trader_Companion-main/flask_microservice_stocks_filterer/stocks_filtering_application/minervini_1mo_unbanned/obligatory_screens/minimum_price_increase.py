import csv
from datetime import datetime, timedelta
import sys
from collections import defaultdict

def calculate_price_increase(highs, lows):
    if not highs or not lows:  # Check if lists are empty
        return None
    year_high = max(highs)
    year_low = min(lows)
    
    # Avoid division by zero
    if year_low <= 0:
        return None
        
    return (year_high - year_low) / year_low * 100

def process_stocks(input_file, output_file, minimum_percentage, N):
    # Dictionary to store high and low prices for each symbol
    stocks = defaultdict(lambda: {'highs': [], 'lows': []})
    skipped_rows = 0

    # Define the cutoff date for filtering (last year)
    cutoff_date = datetime.today() - timedelta(days=365)

    # Read the CSV file and process the data
    with open(input_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row_num, row in enumerate(csv_reader, start=2):
            symbol = row['Symbol']
            date_str = row['Date'].split()[0] if row['Date'] else None
            
            try:
                high_price = float(row['High'])
                low_price = float(row['Low'])
                date = datetime.strptime(date_str, '%Y-%m-%d')

                # Skip data older than the last year
                if date < cutoff_date:
                    continue
                
                # Skip if prices are invalid
                if high_price <= 0 or low_price <= 0:
                    skipped_rows += 1
                    continue
                    
            except (ValueError, KeyError) as e:
                print(f"Error processing row {row_num}: {e}")
                skipped_rows += 1
                continue
            
            # Store high and low prices within the last year
            stocks[symbol]['highs'].append(high_price)
            stocks[symbol]['lows'].append(low_price)

    # Calculate price increases for each stock
    qualified_stocks = []
    for symbol, data in stocks.items():
        price_increase = calculate_price_increase(data['highs'], data['lows'])
        
        if price_increase is not None and price_increase >= minimum_percentage:
            qualified_stocks.append((symbol, price_increase))

    # Sort stocks by price increase (descending) and take top N
    qualified_stocks.sort(key=lambda x: x[1], reverse=True)
    qualified_stocks = qualified_stocks[:N] if N else qualified_stocks

    # Write the qualified stocks to a new CSV file
    with open(output_file, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Symbol', 'Price_Increase_Percentage'])
        for symbol, increase in qualified_stocks:
            csv_writer.writerow([symbol, f"{increase:.2f}"])

    print(f"Analysis complete. {len(qualified_stocks)} stocks with minimum {minimum_percentage}% "
          f"increase have been saved to {output_file}.")
    print(f"Skipped {skipped_rows} rows due to missing, invalid, or outdated data.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <minimum_percentage>")
        sys.exit(1)
    
    try:
        minimum_percentage = float(sys.argv[1])
        N = int(sys.argv[2]) if len(sys.argv) > 2 else None
    except ValueError:
        print("Error: minimum_percentage must be a number")
        sys.exit(1)

    import os

    # Get the absolute path of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Find the absolute path of the "flask_microservice_stocks_filterer" directory
    while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
        script_dir = os.path.dirname(script_dir)

    # Append the correct relative path to the input file
    input_file = os.path.join(script_dir, "stocks_filtering_application", "price_data", "all_tickers_historical.csv")

    # Define the output file
    output_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_1mo_unbanned", "obligatory_screens", "results", "minimum_price_increase.csv")

    print(f"Resolved input file path: {input_file}")
    print(f"Resolved output file path: {output_file}")
    process_stocks(input_file, output_file, minimum_percentage, N)
