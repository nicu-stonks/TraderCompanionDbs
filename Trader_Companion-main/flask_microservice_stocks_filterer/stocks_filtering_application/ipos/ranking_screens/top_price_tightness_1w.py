import pandas as pd
import os

def calculate_price_tightness(group):
    week_high = group['High'].max()
    week_low = group['Low'].min()
    return (week_high - week_low) / week_high * 100  # Tightness as a percentage of the high

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Find the absolute path of the "flask_microservice_stocks_filterer" directory
while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
    script_dir = os.path.dirname(script_dir)

# Append the correct relative path to the input and output files
input_file = os.path.join(script_dir, "stocks_filtering_application", "ipos", "ranking_screens", "passed_stocks_input_data", "filtered_price_data.csv")
output_file = os.path.join(script_dir, "stocks_filtering_application", "ipos", "ranking_screens", "results", "top_price_tightness_1w.csv")

# Read CSV with date parsing
df = pd.read_csv(input_file, parse_dates=['Date'])

# Determine the latest date in the dataset
latest_date = df['Date'].max()

# Filter data to only include the last 1 week
one_week_ago = latest_date - pd.DateOffset(weeks=1)
df_filtered = df[df['Date'] >= one_week_ago]

# Group by symbol and calculate price tightness
price_tightness = df_filtered.groupby('Symbol').apply(calculate_price_tightness)

top = price_tightness.sort_values(ascending=True)  # Lower tightness indicates less volatility

# Create a DataFrame with the results
result_df = pd.DataFrame({
    'Symbol': top.index,
    'Price_Tightness_1W': top.values
})

# Write the results to the output CSV file
result_df.to_csv(output_file, index=False)

print(f"Top stocks by price tightness (last 1 week) have been saved to {output_file}")
