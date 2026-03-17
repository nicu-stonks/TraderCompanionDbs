import csv
from datetime import datetime
from collections import defaultdict

def check_52_week_high(input_file, output_file):
    stocks = defaultdict(list)
    skipped_rows = 0
    total_stocks = set()

    with open(input_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row_num, row in enumerate(reader, start=2):
            try:
                date = datetime.strptime(row['Date'].split()[0], '%Y-%m-%d')
                high_str = row['High'].strip()
                symbol = row['Symbol']

                if not high_str or not symbol:
                    raise ValueError("Missing 'High' or 'Symbol' value")

                high = float(high_str)
                stocks[symbol].append((date, high))
                total_stocks.add(symbol)
            except (ValueError, KeyError) as e:
                skipped_rows += 1
                continue

    results = []
    for symbol, data in stocks.items():
        data.sort(key=lambda x: x[0])
        last_year_data = data[-252:]

        if last_year_data:
            last_day = last_year_data[-1]
            year_high = max(last_year_data, key=lambda x: x[1])[1]

            if last_day[1] == year_high:
                results.append(symbol)

    percentage = (len(results) / len(total_stocks)) * 100 if total_stocks else 0

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Percentage'])
        writer.writerow([percentage])

    print(f"New 52-week-high analysis complete. {percentage:.2f}% of stocks({len(results)}) hit a 52-week high in the last trading day.")
    print(f"Skipped {skipped_rows} rows due to missing or invalid data.")

import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Find the absolute path of the "flask_microservice_stocks_filterer" directory
while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
    script_dir = os.path.dirname(script_dir)

# Append the correct relative paths to the input and output files
input_file = os.path.join(script_dir, "stocks_filtering_application", "price_data", "all_tickers_historical.csv")
output_file = os.path.join(script_dir, "stocks_filtering_application", "market_sentiment_screens", "results", "52week_high_1_days.csv")

check_52_week_high(input_file, output_file)