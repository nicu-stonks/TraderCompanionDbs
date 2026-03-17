import os
import csv
import subprocess
import sys
import logging
from datetime import date, timedelta
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

# Find the absolute path of the "flask_microservice_stocks_filterer" directory
script_dir = os.path.dirname(os.path.abspath(__file__))
while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
    script_dir = os.path.dirname(script_dir)

# Paths to scripts and CSV output files
scripts_dir = os.path.join(script_dir, "stocks_filtering_application", "market_sentiment_screens")
results_dir = os.path.join(scripts_dir, "results")
os.makedirs(results_dir, exist_ok=True)  # Ensure the results directory exists

# Define script-to-CSV mapping
script_mapping = {
    "52week_high_1d.py": "52week_high_1_days.csv",
    "52week_high_2w.py": "52week_high_2_weeks.csv",
    "52week_low_1d.py": "52week_low_1_days.csv",
    "52week_low_2w.py": "52week_low_2_weeks.csv",
    "rs_over_70.py": "rsi_over_70.csv",
    "rsi_under_30.py": "rsi_under_30.csv",
    "rsi_trending_down.py": "rsi_trending_down_stocks.csv",
    "rsi_trending_up.py": "rsi_trending_up_stocks.csv",
    "above_200MA.py": "above_200ma.csv",
}

# Get the full paths for scripts and corresponding CSVs
scripts = {os.path.join(scripts_dir, script): os.path.join(results_dir, csv_file) for script, csv_file in script_mapping.items()}

# Define the final output file
output_file = os.path.join(script_dir, "stocks_filtering_application", "sentiment_history", "sentiment_history.csv")
os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure directory exists

def run_script(script_path):
    """Runs a Python script and logs its output."""
    command = [sys.executable, '-u', script_path]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    for line in process.stdout:
        logging.info(f"[{os.path.basename(script_path)}] {line.strip()}")
    for line in process.stderr:
        logging.error(f"[{os.path.basename(script_path)} ERROR] {line.strip()}")

    process.wait()
    return process.returncode

def execute_scripts_in_parallel(scripts):
    """Runs all scripts in parallel using ThreadPoolExecutor."""
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(run_script, script): script for script in scripts.keys()}
        for future in as_completed(futures):
            future.result()  # Wait for completion

def read_percentage(filename):
    """Reads the percentage value from a CSV file and rounds it to 2 decimal places."""
    if not os.path.exists(filename):
        return "0.00"  # Return zero if the file does not exist

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip header
        for row in reader:
            try:
                return f"{float(row[0]):.2f}"  # Format as string with 2 decimal places
            except ValueError:
                return "0.00"
    return "0.00"

def get_index_change(ticker):
    """Fetches the percentage change for a given index (e.g., NASDAQ, SPY) and rounds it to 2 decimal places."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5d")
    if len(hist) >= 2:
        yesterday_close = hist['Close'].iloc[-2]
        today_close = hist['Close'].iloc[-1]
        return f"{((today_close - yesterday_close) / yesterday_close * 100):.2f}"
    return "0.00"

def generate_csv():
    """Generates the sentiment history CSV after running all scripts."""
    today = (date.today() - timedelta(days=2)).isoformat()
    
    # Run scripts before reading percentages
    execute_scripts_in_parallel(scripts)

    # Read percentages from each file
    percentages = {csv: read_percentage(csv) for csv in scripts.values()}

    # Remove path and extension from file names
    file_names_short = [os.path.splitext(os.path.basename(csv))[0] for csv in scripts.values()]
    
    # Get NASDAQ and SPY price changes
    nasdaq_change = get_index_change("^IXIC")
    spy_change = get_index_change("SPY")
    
    # Check if the output file exists
    if os.path.exists(output_file):
        with open(output_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            existing_data = list(reader)
            
        if any(row[0] == today for row in existing_data):
            print(f"Data for {today} already exists in the history file. No changes made.")
            return
            
        with open(output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([today, spy_change, nasdaq_change] + [percentages[csv] for csv in scripts.values()])
            
        print(f"New row for {today} has been appended to '{output_file}'.")
    else:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Date', 'SPY', 'NASDAQ'] + file_names_short)
            writer.writerow([today, spy_change, nasdaq_change] + [percentages[csv] for csv in scripts.values()])
            
        print(f"New file '{output_file}' has been created with data for {today}.")

# Run the full history handler process
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    generate_csv()