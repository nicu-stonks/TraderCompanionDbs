import pandas as pd
from datetime import datetime, timedelta

def get_stocks_with_52week_low(file_path, output_file):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df.sort_values('Date')

    last_date = df['Date'].max()
    two_weeks_ago = last_date - timedelta(days=14)

    total_stocks = df['Symbol'].nunique()
    grouped = df.groupby('Symbol')

    symbols_with_52week_low = []

    for symbol, group in grouped:
        last_year_data = group[group['Date'] >= last_date - timedelta(days=365)]
        last_two_weeks_data = group[group['Date'] >= two_weeks_ago]

        if last_two_weeks_data['Low'].min() <= last_year_data['Low'].min():
            symbols_with_52week_low.append(symbol)

    percentage = (len(symbols_with_52week_low) / total_stocks) * 100 if total_stocks else 0

    pd.DataFrame([[percentage]], columns=['Percentage']).to_csv(output_file, index=False)

    print(f"Saved {percentage:.2f}% ({len(symbols_with_52week_low)}) symbols to {output_file}. Total stocks: {total_stocks}.")


# Usage
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Find the absolute path of the "flask_microservice_stocks_filterer" directory
while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
    script_dir = os.path.dirname(script_dir)

# Append the correct relative paths to the input and output files
input_file = os.path.join(script_dir, "stocks_filtering_application", "price_data", "all_tickers_historical.csv")
output_file = os.path.join(script_dir, "stocks_filtering_application", "market_sentiment_screens", "results", "52week_low_2_weeks.csv")

get_stocks_with_52week_low(input_file, output_file)
