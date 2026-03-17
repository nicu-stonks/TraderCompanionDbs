import pandas as pd
import numpy as np
import os
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import socket

# Suppress pandas FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

def calculate_rsi(price_series, period=14):
    """
    Calculate RSI (Relative Strength Index) for a given price series
    """
    # Calculate price changes
    delta = price_series.diff()
    
    # Create gains (positive) and losses (negative) series
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss over the specified period
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS (Relative Strength), handling division by zero
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    # Convert back to series
    rsi_series = pd.Series(rsi, index=price_series.index)
    
    return rsi_series

def calculate_relative_strength_vs_market(stock_data, market_data, period=14):
    """
    Calculate relative strength of a stock compared to the market
    Using RSI methodology to measure relative performance
    """
    try:
        # Debug prints
        print(f"Stock data type: {type(stock_data)}")
        print(f"Stock data columns: {stock_data.columns.tolist() if hasattr(stock_data, 'columns') else 'No columns attribute'}")
        print(f"Market data type: {type(market_data)}")
        print(f"Market data columns: {market_data.columns.tolist() if hasattr(market_data, 'columns') else 'No columns attribute'}")
        
        # Extract Close prices (ensure we're working with Series, not DataFrames)
        if 'Close' not in stock_data:
            print(f"'Close' not in stock_data. Available keys/columns: {dir(stock_data)}")
            # Try alternative column names
            close_col = [col for col in stock_data.columns if 'close' in col.lower()] if hasattr(stock_data, 'columns') else []
            if close_col:
                print(f"Found alternative close column: {close_col[0]}")
                stock_close = stock_data[close_col[0]]
            else:
                raise KeyError("No 'Close' or alternative close column found in stock_data")
        else:
            stock_close = stock_data['Close']
            
        market_close = market_data['Close']
        
        # Calculate daily returns for both stock and market
        stock_returns = stock_close.pct_change()
        market_returns = market_close.pct_change()
        
        # Ensure stock_returns and market_returns have the same index
        common_index = stock_returns.index.intersection(market_returns.index)
        stock_returns = stock_returns.loc[common_index]
        market_returns = market_returns.loc[common_index]
        
        # Calculate relative performance (stock returns - market returns)
        relative_performance = stock_returns - market_returns
        
        # Apply RSI formula to the relative performance data
        relative_rsi = calculate_rsi(relative_performance, period)
        
        return relative_rsi
    except Exception as e:
        print(f"Error in calculate_relative_strength_vs_market: {e}")
        return pd.Series()

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Find the absolute path of the "flask_microservice_stocks_filterer" directory
while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
    script_dir = os.path.dirname(script_dir)

# Append the correct relative path to the input and output files
input_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_1mo", "ranking_screens", "passed_stocks_input_data", "filtered_price_data.csv")
output_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_1mo", "ranking_screens", "results", "max_rsi_vs_market_3m.csv")

print("Reading CSV file...")
# Read CSV file
df = pd.read_csv(input_file)

print("DataFrame info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# Check if 'Date' column exists
if 'Date' not in df.columns:
    print(f"Error: 'Date' column not found. Available columns: {df.columns.tolist()}")
    raise ValueError("Date column not found in input file")

# Print sample dates for debugging
print("\nSample Date values:")
print(df['Date'].head())

# Manual approach to handle date conversion
try:
    print("Converting dates manually...")
    # Extract date part from the string (remove time and timezone)
    date_strings = []
    for date_str in df['Date']:
        # Split by space and take first part (date only)
        date_part = date_str.split()[0]
        date_strings.append(date_part)
    
    # Create new datetime column
    df['DateClean'] = pd.to_datetime(date_strings)
    print(f"Date conversion successful. Data type: {df['DateClean'].dtype}")
    
    # Verify conversion
    print("Sample converted dates:")
    print(df['DateClean'].head())
    
    # Use the clean date column for all further operations
    date_col = 'DateClean'
except Exception as e:
    print(f"Error in manual date conversion: {e}")
    # Fallback - try standard conversion without timezone handling
    try:
        print("Trying fallback date conversion...")
        df['DateClean'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
        if df['DateClean'].isna().all():
            raise ValueError("All dates converted to NaT")
        date_col = 'DateClean'
    except Exception as e2:
        print(f"Fallback conversion also failed: {e2}")
        # Last resort - use string dates and parse them later
        date_col = 'Date'
        print("Will use original date strings for filtering")

# Determine the date range for analysis
if date_col == 'DateClean':
    latest_date = df[date_col].max()
    # Calculate 3 months ago using pandas
    three_months_ago = latest_date - pd.DateOffset(months=3)
    print(f"Analysis period: {three_months_ago} to {latest_date}")
    
    # Filter data to only include the last 3 months
    df_filtered = df[df[date_col] >= three_months_ago].copy()
else:
    # If we're using string dates, we need a different approach
    print("Using string-based date filtering")
    # Sort the dates in descending order
    unique_dates = sorted(df['Date'].unique(), reverse=True)
    
    # Take the latest date
    latest_date_str = unique_dates[0]
    print(f"Latest date: {latest_date_str}")
    
    # Keep all data (can't easily filter by 3 months)
    df_filtered = df.copy()
    
    # Try to parse the latest date for yfinance
    try:
        latest_date = pd.to_datetime(latest_date_str.split()[0])
        three_months_ago = latest_date - pd.DateOffset(months=3)
    except:
        # Fallback to current date minus 3 months
        latest_date = datetime.now()
        three_months_ago = latest_date - pd.DateOffset(months=3)

print(f"Filtered data contains {len(df_filtered)} rows for {df_filtered['Symbol'].nunique()} symbols")

# Get the SPX (S&P 500) data for the same period using yfinance
try:
    # Format dates for yfinance
    end_date = latest_date.strftime('%Y-%m-%d')
    start_date = three_months_ago.strftime('%Y-%m-%d')
    
    print(f"Downloading SPX data from {start_date} to {end_date}")
    # Set timeout for network requests
    socket.setdefaulttimeout(15)  # 15 second timeout
    spx_data = yf.download('^GSPC', start=start_date, end=end_date)
    
    print(f"Downloaded {len(spx_data)} days of SPX data")
    if len(spx_data) < 5:
        print("Warning: Very little SPX data downloaded!")
        print(spx_data)
except Exception as e:
    print(f"Error downloading SPX data: {e}")
    raise

# Process each stock and calculate its max RSI against the market
symbols = df_filtered['Symbol'].unique()
print(f"Processing {len(symbols)} symbols...")

# Count successful and failed calculations
success_count = 0
error_count = 0
results = []

for symbol in symbols:
    try:
        print(f"Processing {symbol}...")
        # Get data for this specific stock
        stock_data = df_filtered[df_filtered['Symbol'] == symbol].copy()
        
        print(f"Stock data for {symbol} has columns: {stock_data.columns.tolist()}")
        print(f"First few rows of stock data:")
        print(stock_data.head())
        
        # Skip if not enough data points
        if len(stock_data) < 15:  # Need at least 15 days for a meaningful 14-day RSI
            print(f"Skipping {symbol}: Not enough data points ({len(stock_data)})")
            continue
        
        # Make sure stock data has a proper datetime index
        if date_col == 'DateClean':
            stock_data = stock_data.set_index(date_col)
        else:
            # If we're using string dates, create a new index
            try:
                # Extract date parts only
                date_strings = [d.split()[0] for d in stock_data['Date']]
                temp_dates = pd.to_datetime(date_strings)
                stock_data['temp_date'] = temp_dates
                stock_data = stock_data.set_index('temp_date')
            except Exception as de:
                print(f"Error creating date index for {symbol}: {de}")
                continue
        
        # Ensure stock_data has required columns
        required_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
        missing_cols = [col for col in required_cols if col not in stock_data.columns]
        if missing_cols:
            print(f"Skipping {symbol}: Missing columns {missing_cols}")
            continue
        
        # Find common dates between stock and market data
        common_dates = stock_data.index.intersection(spx_data.index)
        
        # Skip if not enough data points
        if len(common_dates) < 15:  # Need at least 15 days for a meaningful 14-day RSI
            print(f"Skipping {symbol}: Not enough common dates with SPX data ({len(common_dates)})")
            continue
            
        # Extract just the Close prices Series for the stock
        stock_data_subset = stock_data.loc[common_dates, ['Close']]
        spx_data_subset = spx_data.loc[common_dates, ['Close']]

        # Calculate relative strength RSI against the market
        rs_vs_market = calculate_relative_strength_vs_market(stock_data_subset, spx_data_subset)


        
        # Skip if RSI calculation failed
        if rs_vs_market.empty:
            print(f"Skipping {symbol}: RSI calculation failed")
            continue
            
        # Get the maximum RSI value in the period
        max_rsi = rs_vs_market.max()
        
        # Skip if max_rsi is NaN
        if pd.isna(max_rsi):
            print(f"Skipping {symbol}: Maximum RSI is NaN")
            continue
        
        # Store the result
        results.append({
            'Symbol': symbol,
            'Max_RSI_vs_Market_3M': max_rsi
        })
        success_count += 1
        
        # Print progress after every 10 successful calculations
        if success_count % 10 == 0:
            print(f"Progress: {success_count}/{len(symbols)} stocks processed successfully")
            
    except Exception as e:
        error_count += 1
        print(f"Error processing {symbol}: {e}")
        continue

print(f"\nProcessing complete: {success_count} stocks successful, {error_count} failed")

# Create results DataFrame
if results:
    result_df = pd.DataFrame(results)
    
    # Ensure data types are consistent
    result_df['Max_RSI_vs_Market_3M'] = pd.to_numeric(result_df['Max_RSI_vs_Market_3M'], errors='coerce')
    
    # Handle NaN values
    result_df = result_df.dropna()
    
    # Sort by Max RSI in descending order
    if not result_df.empty:
        result_df = result_df.sort_values('Max_RSI_vs_Market_3M', ascending=False)
else:
    # Create an empty DataFrame with the right columns if no results
    result_df = pd.DataFrame(columns=['Symbol', 'Max_RSI_vs_Market_3M'])

# Write the results to the output CSV file
result_df.to_csv(output_file, index=False)

# Print summary of results
print(f"\nResults summary:")
print(f"- Total stocks processed: {len(symbols)}")
print(f"- Stocks with valid RSI results: {len(result_df)}")
print(f"- Top 5 stocks by RSI against market:")
if not result_df.empty and len(result_df) >= 5:
    for i, (symbol, max_rsi) in enumerate(zip(result_df['Symbol'].head(5), result_df['Max_RSI_vs_Market_3M'].head(5))):
        print(f"  {i+1}. {symbol}: {max_rsi:.2f}")
elif not result_df.empty:
    for i, (symbol, max_rsi) in enumerate(zip(result_df['Symbol'], result_df['Max_RSI_vs_Market_3M'])):
        print(f"  {i+1}. {symbol}: {max_rsi:.2f}")
else:
    print("  No valid results found")

print(f"\nResults have been saved to {output_file}")