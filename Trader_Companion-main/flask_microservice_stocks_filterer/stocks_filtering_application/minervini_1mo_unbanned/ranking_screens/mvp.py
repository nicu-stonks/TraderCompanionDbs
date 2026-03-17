import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta

def calculate_50day_volume_ma(group, end_date, window_days=50):
    """Calculate 50-day volume moving average ending at a specific date"""
    # Filter data up to and including the end_date
    data_up_to_date = group[group['Date'] <= end_date].sort_values('Date')
    
    # Take the last 50 days (or all available if less than 50)
    volume_data = data_up_to_date['Volume'].tail(window_days)
    
    # Return the average, or NaN if no data
    if len(volume_data) == 0:
        return np.nan
    
    return volume_data.mean()

def calculate_mvp_score_for_window(window, group_data):
    """Calculate MVP score for a 15-day window"""
    if len(window) < 15:
        return 0
    
    # Sort by date to ensure correct order
    window = window.sort_values('Date').reset_index(drop=True)
    
    # Calculate daily price changes
    window['Daily_Change'] = window['Close'].diff()
    
    # 1. Momentum: Check if the stock is up 12 out of 15 days
    # Skip the first row since it will have NaN for Daily_Change
    positive_days = sum(window['Daily_Change'].iloc[1:] > 0)
    momentum_check = positive_days >= 12
    
    # 2. Volume: Check if volume increased by 25% during the 15-day period
    # Calculate 50-day volume MA as of the start of the window
    window_start_date = window['Date'].iloc[0]
    reference_50day_volume_ma = calculate_50day_volume_ma(group_data, window_start_date, 50)
    
    avg_volume_window = window['Volume'].mean()
    
    if np.isnan(reference_50day_volume_ma) or reference_50day_volume_ma == 0:
        volume_check = False  # Can't calculate if reference volume is NaN or 0
    else:
        volume_increase = (avg_volume_window / reference_50day_volume_ma) - 1
        volume_check = volume_increase >= 0.25
    
    # 3. Price: Check if the stock price is up 20% or more during the 15-day period
    first_close = window['Close'].iloc[0]
    last_close = window['Close'].iloc[-1]
    
    if first_close == 0 or np.isnan(first_close) or np.isnan(last_close):
        price_check = False  # Can't calculate if price data is invalid
    else:
        price_percent_change = ((last_close / first_close) - 1) * 100
        price_check = price_percent_change >= 20
    
    # Return 1 if all conditions are met, 0 otherwise
    if momentum_check and volume_check and price_check:
        return 1
    else:
        return 0

def check_mvp_criteria_in_period(group, lookback_days=180):
    """Check if the stock met MVP criteria at least once in the lookback period"""
    # Handle empty or very small groups
    if len(group) < 15:
        return 0
    
    # Sort by date to ensure correct order
    group = group.sort_values('Date').reset_index(drop=True)
    
    # Calculate the date cutoff (6 months ago)
    latest_date = group['Date'].max()
    cutoff_date = latest_date - pd.Timedelta(days=lookback_days)
    
    # Filter for the lookback period
    period_data = group[group['Date'] >= cutoff_date].reset_index(drop=True)
    
    if len(period_data) < 15:  # Not enough data in the period
        return 0
    
    # Sliding window approach: check each possible 15-day window in the lookback period
    met_criteria = False
    
    # Make sure we don't go out of bounds
    max_start_idx = len(period_data) - 15
    
    for start_idx in range(max_start_idx + 1):
        window = period_data.iloc[start_idx:start_idx + 15].copy()
        
        # Double-check window has exactly 15 days
        if len(window) != 15:
            continue
            
        # Calculate MVP score for this window
        try:
            score = calculate_mvp_score_for_window(window, group)
            
            if score == 1:
                met_criteria = True
                break
        except Exception as e:
            # Skip this window if there's any calculation error
            print(f"Error calculating MVP score for window: {e}")
            continue
    
    return 1 if met_criteria else 0

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Find the absolute path of the "flask_microservice_stocks_filterer" directory
while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
    script_dir = os.path.dirname(script_dir)

# Append the correct relative path to the input and output files
input_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_1mo_unbanned", "ranking_screens", "passed_stocks_input_data", "filtered_price_data.csv")
output_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_1mo_unbanned", "ranking_screens", "results", "mvp_stocks_6mo.csv")

try:
    # Read CSV with date parsing
    df = pd.read_csv(input_file, parse_dates=['Date'])
    print(f"Loaded {len(df)} rows of data")
    
    # Group by symbol and check for MVP criteria in the lookback period
    results = []
    
    symbols_processed = 0
    for symbol, group in df.groupby('Symbol'):
        try:
            mvp_score = check_mvp_criteria_in_period(group, lookback_days=180)  # 180 days = ~6 months
            if mvp_score == 1:
                results.append({'Symbol': symbol, 'MVP_Last_6Mo': 1})
            
            symbols_processed += 1
            if symbols_processed % 50 == 0:  # Progress indicator
                print(f"Processed {symbols_processed} symbols...")
                
        except Exception as e:
            print(f"Error processing symbol {symbol}: {e}")
            continue
    
    print(f"Processed {symbols_processed} symbols total")
    print(f"Found {len(results)} stocks meeting MVP criteria")
    
    # Create a DataFrame with the results
    if results:
        result_df = pd.DataFrame(results)
        # Write the results to the output CSV file
        result_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        # Create empty DataFrame with proper columns if no results
        result_df = pd.DataFrame(columns=['Symbol', 'MVP_Last_6Mo'])
        result_df.to_csv(output_file, index=False)
        print(f"No stocks met MVP criteria. Empty results file saved to {output_file}")

except Exception as e:
    print(f"Error reading input file or processing data: {e}")
    print(f"Make sure the input file exists at: {input_file}")

print(f"Stocks meeting MVP criteria at least once in the last 6 months have been saved to {output_file}")