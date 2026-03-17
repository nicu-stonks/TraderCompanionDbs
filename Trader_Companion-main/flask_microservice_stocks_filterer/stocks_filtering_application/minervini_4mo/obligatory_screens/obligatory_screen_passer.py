import os
import csv

def get_existing_csv_files(directory):
    """Find all existing CSV files in the specified directory."""
    csv_files = []
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            full_path = os.path.join(directory, file)
            csv_files.append(full_path)
    return csv_files

def read_csv(file_name):
    """Read symbols from a CSV file."""
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        try:
            next(reader)  # Skip header
        except StopIteration:
            return set()  # Return empty set if file is empty
        return set(row[0] for row in reader)

def find_common_stocks(file_names):
    """Find stocks common to all input files."""
    if not file_names:
        print("No CSV files found in the directory!")
        return set()
        
    stock_sets = [read_csv(file_name) for file_name in file_names]
    print(f"Processing {len(file_names)} CSV files:")
    for file_name in file_names:
        print(f"- {os.path.basename(file_name)}")
    
    return set.intersection(*stock_sets)

def save_common_stocks(stocks, output_file):
    """Save the common stocks to an output CSV file."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Symbol"])
        for stock in sorted(stocks):
            writer.writerow([stock])
    print(f"\nFound {len(stocks)} stocks that passed all available screens")
    print(f"Results saved to {os.path.basename(output_file)}")

def main():
    import os

    # Get the absolute path of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Find the absolute path of the "flask_microservice_stocks_filterer" directory
    while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
        script_dir = os.path.dirname(script_dir)

    # Define the input directory and output file path
    input_directory = os.path.join(script_dir, "stocks_filtering_application", "minervini_4mo", "obligatory_screens", "results")
    output_file = os.path.join(input_directory, "obligatory_passed_stocks.csv")

    
    # Get all existing CSV files except the output file
    input_files = [f for f in get_existing_csv_files(input_directory)
                  if os.path.basename(f) != "obligatory_passed_stocks.csv"]
    
    common_stocks = find_common_stocks(input_files)
    save_common_stocks(common_stocks, output_file)

if __name__ == "__main__":
    main()