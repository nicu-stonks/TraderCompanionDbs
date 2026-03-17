import os
import csv
import subprocess
import sys
import logging
from datetime import date, timedelta, datetime
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
    script_dir = os.path.dirname(script_dir)

scripts_dir = os.path.join(script_dir, "stocks_filtering_application", "market_sentiment_screens")
results_dir = os.path.join(scripts_dir, "results")
os.makedirs(results_dir, exist_ok=True)

amex_arca_bats_nasdaq_nyse_otc_stocks_file = os.path.join(script_dir, "stocks_filtering_application", "price_data", "all_tickers_historical.csv")

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

scripts = {os.path.join(scripts_dir, script): os.path.join(results_dir, csv_file) for script, csv_file in script_mapping.items()}

output_file = os.path.join(script_dir, "stocks_filtering_application", "sentiment_history", "sentiment_history.csv")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

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
            future.result()

def get_most_recent_date(csv_file):
    """Finds the most recent date in the CSV file."""
    if not os.path.exists(csv_file):
        return None
    df = pd.read_csv(csv_file, parse_dates=['Date'])
    if df.empty:
        return None
    return df['Date'].max()

def remove_most_recent_rows(csv_file, date_to_remove):
    """Deletes rows containing the given date from the CSV file."""
    if not os.path.exists(csv_file):
        return
    df = pd.read_csv(csv_file, parse_dates=['Date'])
    if df.empty:
        return
    df = df[df['Date'] != date_to_remove]
    df.to_csv(csv_file, index=False)
    print(f"Deleted rows with date {date_to_remove} from {csv_file}")

def read_percentage(filename):
    """Reads the percentage value from a CSV file and rounds it to 2 decimal places."""
    if not os.path.exists(filename):
        return "0.00"
    
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            try:
                return f"{float(row[0]):.2f}"
            except ValueError:
                return "0.00"
    return "0.00"

def get_index_change(ticker):
    """Fetches the percentage change for a given index and rounds it."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5d")
    if len(hist) >= 2:
        yesterday_close = hist['Close'].iloc[-2]
        today_close = hist['Close'].iloc[-1]
        return f"{((today_close - yesterday_close) / yesterday_close * 100):.2f}"
    return "0.00"

def generate_csv():
    """Iterates back from today to 250 days ago, deleting recent data before each step."""
    today = date.today() - timedelta(days=2)
    for days in range(250):
        print(f"Processing day {days + 1} / 250")
        most_recent_date = get_most_recent_date(amex_arca_bats_nasdaq_nyse_otc_stocks_file)
        if most_recent_date:
            remove_most_recent_rows(amex_arca_bats_nasdaq_nyse_otc_stocks_file, most_recent_date)
        
        execute_scripts_in_parallel(scripts)
        percentages = {csv: read_percentage(csv) for csv in scripts.values()}
        file_names_short = [os.path.splitext(os.path.basename(csv))[0] for csv in scripts.values()]
        nasdaq_change = get_index_change("^IXIC")
        spy_change = get_index_change("SPY")
        
        if os.path.exists(output_file):
            with open(output_file, 'r') as csvfile:
                reader = csv.reader(csvfile)
                existing_data = list(reader)
            
            if any(row[0] == today.isoformat() for row in existing_data):
                print(f"Data for {today} already exists. Skipping.")
                today -= timedelta(days=1)
                continue
            
            with open(output_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([today, spy_change, nasdaq_change] + [percentages[csv] for csv in scripts.values()])
        else:
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Date', 'SPY', 'NASDAQ'] + file_names_short)
                writer.writerow([today, spy_change, nasdaq_change] + [percentages[csv] for csv in scripts.values()])
        
        today -= timedelta(days=1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    generate_csv()
