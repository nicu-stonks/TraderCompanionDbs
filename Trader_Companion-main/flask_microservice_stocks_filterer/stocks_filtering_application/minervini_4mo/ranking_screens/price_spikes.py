import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_stocks(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file, parse_dates=['Date'])
    
    # Group by Symbol
    grouped = df.groupby('Symbol')
    
    results = []
    
    for symbol, group in grouped:
        # Sort by date in descending order
        group = group.sort_values('Date', ascending=False)
        
        # Get data for the last 2 months
        two_months_ago = group['Date'].max() - timedelta(days=60)
        recent_data = group[group['Date'] > two_months_ago]
        
        if len(recent_data) < 2:  # Skip if not enough data
            continue
        
        # Calculate average volume
        avg_volume = recent_data['Volume'].mean()
        
        # Calculate average price increase on up days
        up_days = recent_data[recent_data['Close'] > recent_data['Open']]
        avg_price_increase = (up_days['Close'] - up_days['Open']).mean()
        
        # Identify price spikes
        price_spikes = recent_data[
            (recent_data['Close'] > recent_data['Open']) &  # Up day
            (recent_data['Close'] - recent_data['Open'] > 3 * avg_price_increase) &  # Price spike
            (recent_data['Volume'] > 3 * avg_volume)  # High volume
        ]
        
        num_spikes = len(price_spikes)
        
        if num_spikes > 0:
            results.append({'Symbol': symbol, 'Nr_Of_Spikes': num_spikes})
    
    # Create and save the output DataFrame
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_file, index=False)
    print(f"Analysis complete. Results saved to {output_file}")

# Usage
import os

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Find the absolute path of the "flask_microservice_stocks_filterer" directory
while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
    script_dir = os.path.dirname(script_dir)

# Append the correct relative path to the input and output files
input_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_4mo", "ranking_screens", "passed_stocks_input_data", "filtered_price_data.csv")
output_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_4mo", "ranking_screens", "results", "price_spikes.csv")

analyze_stocks(input_file, output_file)