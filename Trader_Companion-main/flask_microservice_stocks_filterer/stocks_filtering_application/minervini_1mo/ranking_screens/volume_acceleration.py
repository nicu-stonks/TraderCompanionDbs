import csv
from datetime import datetime, timedelta
import statistics

def calculate_volume_acceleration(file_path, output_path):
    stocks = {}

    # Read the CSV file
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            symbol = row['Symbol']
            date = datetime.strptime(row['Date'].split()[0], '%Y-%m-%d')
            volume = int(row['Volume'])

            if symbol not in stocks:
                stocks[symbol] = []
            stocks[symbol].append((date, volume))

    # Calculate volume acceleration for each stock
    accelerated_stocks = []
    two_months_ago = datetime.now() - timedelta(days=60)

    for symbol, data in stocks.items():
        # Sort data by date
        data.sort(key=lambda x: x[0])

        # Filter data for the last 2 months
        recent_data = [d for d in data if d[0] >= two_months_ago]

        if len(recent_data) < 2:
            continue

        # Calculate average volume for the entire period and the last 2 months
        all_volumes = [d[1] for d in data]
        recent_volumes = [d[1] for d in recent_data]

        avg_volume_all = statistics.mean(all_volumes)
        avg_volume_recent = statistics.mean(recent_volumes)

        # Calculate volume acceleration, handling the case where avg_volume_all is zero
        if avg_volume_all == 0:
            if avg_volume_recent > 0:
                volume_acceleration = float('inf')  # Infinite acceleration
            else:
                volume_acceleration = 0  # No acceleration
        else:
            volume_acceleration = (avg_volume_recent - avg_volume_all) / avg_volume_all * 100

        if volume_acceleration > 50:
            accelerated_stocks.append((symbol, volume_acceleration))

    # Write results to a new CSV file with symbol and volume acceleration
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Symbol', 'Volume-Acceleration(%)'])
        for symbol, acceleration in accelerated_stocks:
            writer.writerow([symbol, f"{acceleration:.2f}" if acceleration != float('inf') else "inf"])

    print(f"Volume Acceleration Analysis complete. {len(accelerated_stocks)} Results saved to {output_path}")

import os

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Find the absolute path of the "flask_microservice_stocks_filterer" directory
while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
    script_dir = os.path.dirname(script_dir)

# Append the correct relative path to the input and output files
input_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_1mo", "ranking_screens", "passed_stocks_input_data", "filtered_price_data.csv")
output_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_1mo", "ranking_screens", "results", "volume_acceleration_stocks.csv")
calculate_volume_acceleration(input_file, output_file)