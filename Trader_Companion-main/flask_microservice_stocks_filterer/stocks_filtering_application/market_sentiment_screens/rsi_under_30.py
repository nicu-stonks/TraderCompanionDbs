import pandas as pd
import numpy as np
import os

def calculate_rsi(data, periods=14):
    close_delta = data['Close'].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    # Calculate the EWMA
    ma_up = up.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
    
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

script_dir = os.path.dirname(os.path.abspath(__file__))

# Find the absolute path of the "flask_microservice_stocks_filterer" directory
while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
    script_dir = os.path.dirname(script_dir)

# Append the correct relative paths to the input and output files
input_file = os.path.join(script_dir, "stocks_filtering_application", "price_data", "all_tickers_historical.csv")
output_file = os.path.join(script_dir, "stocks_filtering_application", "market_sentiment_screens", "results", "rsi_under_30.csv")

# Read the CSV file
df = pd.read_csv(input_file, parse_dates=['Date'])

# Group by Symbol and calculate RSI
def calculate_group_rsi(group):
    return pd.Series({'RSI': calculate_rsi(group).iloc[-1]})

rsi_df = df.groupby('Symbol').apply(calculate_group_rsi).reset_index()

# Total number of stocks
total_stocks = rsi_df['Symbol'].nunique()

# Filter stocks with RSI under 30
low_rsi_stocks = rsi_df[rsi_df['RSI'] < 30]

# Calculate percentage
percentage = (len(low_rsi_stocks) / total_stocks) * 100 if total_stocks else 0

# Save only the percentage to a new CSV file
pd.DataFrame([[percentage]], columns=['Percentage']).to_csv(output_file, index=False)

print(f"Saved {percentage:.2f}% ({len(low_rsi_stocks)}) stocks with 14-day RSI < 30 to {output_file}. Total stocks: {total_stocks}.")
