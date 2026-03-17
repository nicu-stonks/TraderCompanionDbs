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
input_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_1mo", "ranking_screens", "passed_stocks_input_data", "filtered_price_data.csv")
output_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_1mo", "ranking_screens", "results", "top_price_increase_1y.csv")

# Read CSV with date parsing
df = pd.read_csv(input_file, parse_dates=['Date'])

# Determine the latest date in the dataset
latest_date = df['Date'].max()

# Filter data to only include the last 1 year
one_year_ago = latest_date - pd.DateOffset(years=1)
df_filtered = df[df['Date'] >= one_year_ago]

# Group by symbol and calculate price increase
price_increases = df_filtered.groupby('Symbol').apply(calculate_price_increase)

top = price_increases.sort_values(ascending=False)

# Create a DataFrame with the results
result_df = pd.DataFrame({
    'Symbol': top.index,
    'Price_Increase_Percentage': top.values
})

# Write the results to the output CSV file
result_df.to_csv(output_file, index=False)

print(f"Top stocks by price increase (last 1 year) have been saved to {output_file}")
