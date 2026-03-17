import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Find the absolute path of the "flask_microservice_stocks_filterer" directory
while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
    script_dir = os.path.dirname(script_dir)

# Append the correct relative paths to the input and output files
input_file = os.path.join(script_dir, "stocks_filtering_application", "price_data", "all_tickers_historical.csv")
output_file = os.path.join(script_dir, "stocks_filtering_application", "market_sentiment_screens", "results", "above_200ma.csv")

# Read the CSV file
df = pd.read_csv(input_file, parse_dates=['Date'])

def calculate_200ma(group):
    group = group.sort_values('Date')
    group['200MA'] = group['Close'].rolling(window=200, min_periods=200).mean()
    return group

# Apply the moving average calculation
df = df.groupby('Symbol', group_keys=False).apply(calculate_200ma)

# Filter stocks where the latest close price is above the 200-day moving average
latest_data = df.groupby('Symbol').last().reset_index()
stocks_above_200ma = latest_data[latest_data['Close'] > latest_data['200MA']]

# Calculate percentage of stocks above the 200-day moving average
total_stocks = latest_data['Symbol'].nunique()
percentage = (len(stocks_above_200ma) / total_stocks) * 100 if total_stocks else 0

# Save only the percentage to a new CSV file
pd.DataFrame([[percentage]], columns=['Percentage']).to_csv(output_file, index=False)

print(f"Saved {percentage:.2f}% ({len(stocks_above_200ma)}) stocks with Close price above 200-day MA to '{output_file}'. Total stocks: {total_stocks}.")
