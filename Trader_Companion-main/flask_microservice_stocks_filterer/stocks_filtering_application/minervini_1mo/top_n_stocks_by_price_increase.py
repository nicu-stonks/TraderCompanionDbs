import csv
import os
import time
import argparse
from collections import defaultdict

def wait_for_file(file_path, max_attempts=5, wait_time=60):
    """Waits for a file to exist, retrying for a set number of attempts."""
    attempts = 0
    while attempts < max_attempts:
        if os.path.exists(file_path):
            return True  # File found
        print(f"Waiting for '{file_path}' to be created... ({attempts+1}/{max_attempts})")
        time.sleep(wait_time)
        attempts += 1
    return False  # File still not found after max attempts

def process_csv_files(directory, top_n, output_file, exclusion_file):
    # Ensure the exclusion file exists (wait for it)
    if not wait_for_file(exclusion_file):
        print(f"Error: '{exclusion_file}' not found after multiple attempts. Exiting.")
        return

    # Dictionary to store all data
    data = defaultdict(dict)
    characteristics = set()
    price_increases = {}

    # Read the price increase data
    price_increase_file = os.path.join(directory, 'top_price_increase_1y.csv')
    try:
        with open(price_increase_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    symbol, increase = row[:2]
                    try:
                        increase = float(increase)
                        if increase >= 100:  # Only consider stocks with at least 100% increase
                            price_increases[symbol] = increase
                    except ValueError:
                        print(f"Warning: Invalid price increase value for {symbol}. Skipping.")
    except Exception as e:
        print(f"Error processing top_price_increase_1y.csv: {str(e)}")
        return

    # Read exclusion list (symbols that should be removed)
    excluded_symbols = set()
    try:
        with open(exclusion_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                if row:
                    excluded_symbols.add(row[0].strip())  # Store only the symbol column
    except Exception as e:
        print(f"Error processing exclusion file '{exclusion_file}': {str(e)}")
        return

    # Process all other CSV files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and filename != 'top_price_increase_1y.csv':
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r') as file:
                    reader = csv.reader(file)
                    header = next(reader, None)
                    
                    if not header or len(header) < 2:
                        print(f"Warning: File '{filename}' has an invalid header. Skipping.")
                        continue
                    
                    characteristic = header[1]
                    characteristics.add(characteristic)  # Ensure characteristic is stored
                    
                    has_data = False
                    for row in reader:
                        if len(row) < 2:
                            print(f"Warning: Invalid row in '{filename}'. Skipping row.")
                            continue
                        symbol, value = row[:2]
                        if symbol in price_increases:  # Only keep stocks with >= 80% increase
                            data[symbol][characteristic] = value
                            has_data = True

                    # If no valid data rows, ensure empty column appears
                    if not has_data:
                        for symbol in data.keys():  # Ensure existing symbols have an empty column
                            data[symbol][characteristic] = ''
            except Exception as e:
                print(f"Error processing file '{filename}': {str(e)}")

    # Ensure all symbols appear even if they have no data
    all_symbols = set(data.keys()) | set(price_increases.keys())

    # Prepare sorted data (excluding symbols in the exclusion list)
    sorted_data = sorted(
        [(symbol, data.get(symbol, {})) for symbol in all_symbols if symbol not in excluded_symbols],
        key=lambda item: price_increases.get(item[0], 0),
        reverse=True
    )
    
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        # Write header ensuring all characteristics appear
        header = ['Symbol', 'Price_Increase_Percentage', 'Screeners'] + list(characteristics)
        writer.writerow(header)
        
        # Write data for the top N symbols
        for symbol, char_dict in sorted_data[:top_n]:
            row = [symbol, price_increases.get(symbol, ''), len(char_dict)]
            for char in characteristics:
                row.append(char_dict.get(char, ''))  # Ensure empty columns exist
            writer.writerow(row)

    print(f"'{output_file}' has been created with filtered symbols.")

def main():
    parser = argparse.ArgumentParser(description='Process top N stocks by price increase, excluding specific symbols')
    parser.add_argument('top_n', type=int, help='Number of top stocks to select')
    args = parser.parse_args()
    
    # Get the absolute path of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Find the absolute path of the "flask_microservice_stocks_filterer" directory
    while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
        script_dir = os.path.dirname(script_dir)

    # Define paths
    directory = os.path.join(script_dir, "stocks_filtering_application", "minervini_1mo", "ranking_screens", "results")
    output_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_1mo", "stocks_ranking_by_price.csv")
    exclusion_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_4mo", "obligatory_screens", "results", "obligatory_passed_stocks.csv")

    process_csv_files(directory, args.top_n, output_file, exclusion_file)

if __name__ == "__main__":
    main()