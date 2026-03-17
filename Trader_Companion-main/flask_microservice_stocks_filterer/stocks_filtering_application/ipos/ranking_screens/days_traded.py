import pandas as pd
import os

def count_trading_days(group):
    # Count the number of unique trading days for this ticker
    return len(group['Date'].unique())

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Find the absolute path of the "flask_microservice_stocks_filterer" directory
while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
    script_dir = os.path.dirname(script_dir)

# Append the correct relative path to the input and output files
input_file = os.path.join(script_dir, "stocks_filtering_application", "ipos", "ranking_screens", "passed_stocks_input_data", "filtered_price_data.csv")
output_file = os.path.join(script_dir, "stocks_filtering_application", "ipos", "ranking_screens", "results", "Days_Traded.csv")

# Read CSV with date parsing
df = pd.read_csv(input_file, parse_dates=['Date'])

# Determine the latest date in the dataset
latest_date = df['Date'].max()

# Count trading days for each symbol
trading_days = df.groupby('Symbol').apply(count_trading_days)

# Sort by number of trading days (descending)
sorted_trading_days = trading_days.sort_values(ascending=False)

# Create a DataFrame with the results
result_df = pd.DataFrame({
    'Symbol': sorted_trading_days.index,
    'Days_Traded': sorted_trading_days.values
})

# Write the results to the output CSV file
result_df.to_csv(output_file, index=False)

print(f"Stocks ranked by number of trading days have been saved to {output_file}")