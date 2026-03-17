import pandas as pd
import numpy as np
import os

def calculate_price_increase(group):
    year_high = group['High'].max()
    year_low = group['Low'].min()
    return (year_high - year_low) / year_low * 100

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Find the absolute path of the "flask_microservice_stocks_filterer" directory
while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
    script_dir = os.path.dirname(script_dir)

# Append the correct relative path to the input and output files
input_file = os.path.join(script_dir, "stocks_filtering_application", "ipos", "ranking_screens", "passed_stocks_input_data", "filtered_price_data.csv")
output_file = os.path.join(script_dir, "stocks_filtering_application", "ipos", "ranking_screens", "results", "top_price_increase_1y.csv")

df = pd.read_csv(input_file, parse_dates=['Date'])

# Group by symbol and calculate price increase
# Add include_groups=False to avoid the deprecation warning
price_increases = df.groupby('Symbol').apply(calculate_price_increase, include_groups=False)

# Ensure price_increases is a Series with symbol as index
if isinstance(price_increases, pd.DataFrame):
    # If it's a DataFrame, convert to Series
    price_increases = price_increases.iloc[:, 0]

# Sort price increases in descending order
top_100 = price_increases.sort_values(ascending=False)

# Create a DataFrame with the results
result_df = pd.DataFrame({
    'Symbol': top_100.index,
    'Price_Increase_Percentage': top_100.values
})

# Write the results to the output CSV file
result_df.to_csv(output_file, index=False)

print(f"Top 100 stocks by price increase have been saved to {output_file}")
