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
    rsi = 100 - (100 / (1 + rsi))
    return rsi

def rsi_trending_up(rsi_series, window, trend_threshold=0.7):
    rsi_ma = rsi_series.rolling(window=window).mean()
    rsi_ma_clean = rsi_ma.dropna()

    if len(rsi_ma_clean) == 0:
        return False

    upward_trend = (rsi_ma_clean.diff() > 0).sum()  
    upward_proportion = upward_trend / len(rsi_ma_clean) if len(rsi_ma_clean) > 0 else 0
    
    return upward_proportion >= trend_threshold and rsi_ma_clean.iloc[-1] > rsi_ma_clean.iloc[0]

script_dir = os.path.dirname(os.path.abspath(__file__))

# Find the absolute path of the "flask_microservice_stocks_filterer" directory
while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
    script_dir = os.path.dirname(script_dir)

# Append the correct relative paths to the input and output files
input_file = os.path.join(script_dir, "stocks_filtering_application", "price_data", "all_tickers_historical.csv")
output_file = os.path.join(script_dir, "stocks_filtering_application", "market_sentiment_screens", "results", "rsi_trending_up_stocks.csv")

# Read the CSV file
df = pd.read_csv(input_file, parse_dates=['Date'])

# Initialize an empty list to store the results
result = []

# Define number of trading days in a month (assuming 21 days per month)
days_in_month = 21
window = 30

# Group by Symbol and calculate RSI
for symbol, group in df.groupby('Symbol'):
    group = group.sort_values('Date')
    group['RSI'] = calculate_rsi(group)

    months_trending_up = 0
    for i in range(1, int((len(group) - window) / days_in_month)):
        start_idx = -i * days_in_month
        end_idx = -(i - 1) * days_in_month if i > 1 else None

        rsi_last_month = group['RSI'].iloc[start_idx - window:end_idx]

        if rsi_trending_up(rsi_last_month, window=window):
            months_trending_up += 1
        else:
            break

    if months_trending_up >= 1:
        result.append({'Symbol': symbol, 'MonthsTrendingUp': months_trending_up})

# Calculate total stocks
total_stocks = df['Symbol'].nunique()

# Convert results to DataFrame
result_df = pd.DataFrame(result)

# Calculate and save percentage of stocks with RSI trending up
percentage = (len(result_df) / total_stocks) * 100 if total_stocks else 0
pd.DataFrame([[percentage]], columns=['Percentage']).to_csv(output_file, index=False)

print(f"Saved {percentage:.2f}% ({len(result_df)}) stocks whose RSI has been trending up for at least 1 month to '{output_file}'. Total stocks: {total_stocks}.")
