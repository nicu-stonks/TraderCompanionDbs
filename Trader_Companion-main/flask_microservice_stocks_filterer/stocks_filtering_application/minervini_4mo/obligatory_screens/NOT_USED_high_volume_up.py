import pandas as pd
from datetime import timedelta
import csv

def analyze_stock_volume(file_path, output_file):
    
    # Read the CSV file, explicitly parsing the Date column
    df = pd.read_csv(file_path, parse_dates=['Date'], date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S%z', utc=True))
    
    # Convert Date to local time and remove time zone information
    df['Date'] = df['Date'].dt.tz_convert(None)
    
    # Calculate the date 3 months ago from the most recent date
    latest_date = df['Date'].max()
    three_months_ago = latest_date - timedelta(days=100)
    
    # Filter for the last 3 months of data
    df = df[df['Date'] > three_months_ago]
    
    # Group the data by stock symbol
    grouped = df.groupby('Symbol')
    
    results = []
    
    for symbol, group in grouped:
        # Sort the group by date
        group = group.sort_values('Date')
        
        # Calculate daily price change
        group['PriceChange'] = group['Close'] - group['Close'].shift(1)
        
        # Calculate average volume
        avg_volume = group['Volume'].mean()
        
        # Separate up days and down days
        up_days = group[group['PriceChange'] > 0]
        down_days = group[group['PriceChange'] < 0]
        
        # Calculate average volume for up and down days
        avg_volume_up = up_days['Volume'].mean()
        avg_volume_down = down_days['Volume'].mean()
        
        # Count up and down days on above average volume
        up_days_high_volume = up_days[up_days['Volume'] > avg_volume]
        down_days_high_volume = down_days[down_days['Volume'] > avg_volume]
        
        # Weekly analysis
        group['Week'] = group['Date'].dt.to_period('W')
        weekly_data = group.groupby('Week').agg({
            'Close': lambda x: x.iloc[-1] - x.iloc[0],
            'Volume': 'mean'
        }).reset_index()
        
        # Separate up weeks and down weeks
        up_weeks = weekly_data[weekly_data['Close'] > 0]
        down_weeks = weekly_data[weekly_data['Close'] < 0]
        
        # Count up and down weeks on above average volume
        avg_weekly_volume = weekly_data['Volume'].mean()
        up_weeks_high_volume = up_weeks[up_weeks['Volume'] > avg_weekly_volume]
        down_weeks_high_volume = down_weeks[down_weeks['Volume'] > avg_weekly_volume]
        
        # Check all criteria
        if (avg_volume_up > avg_volume_down and
            len(up_days_high_volume) > len(down_days_high_volume) and
            len(up_weeks_high_volume) > len(down_weeks_high_volume)):
            results.append(symbol)
    
    # Save results to a new CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Symbol'])
        for symbol in results:
            writer.writerow([symbol])
            
    print(f"Higher volume up analysis complete. {len(results)} Results saved to {output_file}")

# Usage
input_file = '../stock_api_data/amex_arca_bats_nasdaq_nyse_otc_stocks_1_year_price_data.csv'
output_file = './obligatory_screens/results/high_volume_up_stocks.csv'
analyze_stock_volume(input_file, output_file)
