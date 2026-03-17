import pandas as pd
import os
import numpy as np

def calculate_rsi(data, window=14):
    """
    Calculate the Relative Strength Index (RSI) for a given stock data.
    
    Parameters:
    data (DataFrame): DataFrame containing 'Close' prices
    window (int): Look-back period for RSI calculation, default is 14 days
    
    Returns:
    Series: RSI values
    """
    # Calculate price changes
    delta = data['Close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and average loss
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Find the absolute path of the "flask_microservice_stocks_filterer" directory
while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
    script_dir = os.path.dirname(script_dir)

# Append the correct relative path to the input and output files
input_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_1mo", "ranking_screens", "passed_stocks_input_data", "filtered_price_data.csv")
output_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_1mo", "ranking_screens", "results", "max_rsi_12m.csv")

# Read CSV with date parsing
df = pd.read_csv(input_file, parse_dates=['Date'])

# Determine the latest date in the dataset
latest_date = df['Date'].max()

# Filter data to only include the last 12 months
three_months_ago = latest_date - pd.DateOffset(months=12)
df_filtered = df[df['Date'] >= three_months_ago]

# Initialize a dictionary to store maximum RSI values
max_rsi_values = {}

# Loop through each unique symbol
for symbol in df_filtered['Symbol'].unique():
    # Get data for this symbol
    symbol_data = df_filtered[df_filtered['Symbol'] == symbol].sort_values('Date')
    
    # Calculate RSI if there's enough data (at least 15 trading days)
    if len(symbol_data) >= 15:  # Ensure we have at least 15 days of data
        symbol_data['RSI'] = calculate_rsi(symbol_data)
        
        # Check if RSI calculation was successful (not all NaN)
        if not symbol_data['RSI'].isna().all():
            # Find the maximum RSI value for this symbol (ignoring NaN values)
            max_rsi = symbol_data['RSI'].max()
            
            # Store the result
            max_rsi_values[symbol] = max_rsi
    else:
        # Skip symbols with insufficient data
        print(f"Skipping {symbol}: Insufficient trading data (only {len(symbol_data)} days)")


# Create a DataFrame with the results
result_df = pd.DataFrame({
    'Symbol': list(max_rsi_values.keys()),
    'RSI_12M': list(max_rsi_values.values())
})

# Sort by maximum RSI in descending order
result_df = result_df.sort_values('RSI_12M', ascending=False)

# Write the results to the output CSV file
result_df.to_csv(output_file, index=False)

print(f"Top stocks by maximum RSI (last 12 months) have been saved to {output_file}")
print(f"Total stocks analyzed: {len(result_df)}")
print(f"Stocks excluded due to insufficient trading data: {len(df_filtered['Symbol'].unique()) - len(result_df)}")