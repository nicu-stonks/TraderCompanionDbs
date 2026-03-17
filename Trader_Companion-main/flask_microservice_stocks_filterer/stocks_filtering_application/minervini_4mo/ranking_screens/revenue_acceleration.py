import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def analyze_revenue_acceleration(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file, parse_dates=['Date'])
    
    # Group by Symbol
    grouped = df.groupby('Symbol')
    
    results = []
    
    for symbol, group in grouped:
        # Sort by date in descending order (most recent first)
        group = group.sort_values('Date', ascending=False)
        
        # Initialize counter for consecutive quarters with revenue acceleration
        consecutive_acceleration = 0
        
        # Need at least 2 quarters to calculate acceleration
        if len(group) < 2:
            continue
        
        # Iterate through quarters to check for acceleration
        for i in range(len(group) - 1):
            current_revenue = group.iloc[i]['Revenue']
            previous_revenue = group.iloc[i + 1]['Revenue']
            
            # Skip if either Revenue value is NaN or negative
            if pd.isna(current_revenue) or pd.isna(previous_revenue) or previous_revenue <= 0:
                break
            
            # Calculate percentage increase
            percent_increase = (current_revenue - previous_revenue) / previous_revenue * 100
            
            # Check if there's at least 10% increase
            if percent_increase >= 10:
                consecutive_acceleration += 1
            else:
                # Break the loop if acceleration stops
                break
        
        # Only add to results if there's at least 1 quarter with acceleration
        if consecutive_acceleration > 0:
            results.append({'Symbol': symbol, 'Revenue_Quarters': consecutive_acceleration})
    
    # Create and save the output DataFrame
    output_df = pd.DataFrame(results)
    output_df = output_df.sort_values('Revenue_Quarters', ascending=False)
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
input_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_4mo", "ranking_screens", "passed_stocks_input_data", "filtered_quarterly_fundamental_data.csv")
output_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_4mo", "ranking_screens", "results", "revenue_acceleration.csv")

analyze_revenue_acceleration(input_file, output_file)