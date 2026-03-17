import csv
import os
from collections import defaultdict
import argparse

def process_csv_files(directory, top_n, output_file):
    # Dictionary to store all data
    data = defaultdict(dict)
    characteristics = set()
    price_increases = {}

    # First, read the price increase data
    price_increase_file = os.path.join(directory, 'top_price_increase_1y.csv')
    try:
        with open(price_increase_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    symbol, increase = row[:2]
                    try:
                        price_increases[symbol] = float(increase)
                    except ValueError:
                        print(f"Warning: Invalid price increase value for {symbol}. Skipping.")
    except Exception as e:
        print(f"Error processing top_price_increase_1y.csv: {str(e)}")
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

    # Prepare sorted data
    sorted_data = sorted(
        [(symbol, data.get(symbol, {})) for symbol in all_symbols],
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

    print(f"'{output_file}' has been created.")

def main():
    parser = argparse.ArgumentParser(description='Process top N stocks by price increase')
    parser.add_argument('top_n', type=int, help='Number of top stocks to select')
    args = parser.parse_args()
    
        # Get the absolute path of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Find the absolute path of the "flask_microservice_stocks_filterer" directory
    while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
        script_dir = os.path.dirname(script_dir)

    # Append the correct relative path to the directory
    directory = os.path.join(script_dir, "stocks_filtering_application", "ipos", "ranking_screens", "results")
    output_file = os.path.join(script_dir, "stocks_filtering_application", "ipos", "stocks_ranking_by_price.csv")

    
    process_csv_files(directory, args.top_n, output_file)

if __name__ == "__main__":
    main()
