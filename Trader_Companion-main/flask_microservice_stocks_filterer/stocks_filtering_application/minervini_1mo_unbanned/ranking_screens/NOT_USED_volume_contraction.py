import pandas as pd
import csv
from datetime import datetime, timedelta

def calculate_volume_trend(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, parse_dates=['Date'])

    # Sort by Date and Symbol
    df = df.sort_values(['Symbol', 'Date'])

    # Calculate 4-day moving average of volume
    df['Volume_MA4'] = df.groupby('Symbol')['Volume'].rolling(window=4).mean().reset_index(0, drop=True)

    # Get the date 2 weeks ago from the latest date
    latest_date = df['Date'].max()
    two_weeks_ago = latest_date - timedelta(days=14)

    # Filter for the last 2 weeks of data
    df_last_2_weeks = df[df['Date'] >= two_weeks_ago]

    # Function to check if volume is trending down and calculate contraction
    def volume_trend_and_contraction(group):
        if len(group) < 2:  # Need at least 2 points to determine a trend
            return pd.Series({'trending_down': False, 'contraction': 0})
        
        start_ma = group.iloc[0]['Volume_MA4']
        end_ma = group.iloc[-1]['Volume_MA4']
        
        trending_down = end_ma < start_ma
        if trending_down:
            contraction = (start_ma - end_ma) / start_ma * 100
        else:
            contraction = 0
        
        return pd.Series({'trending_down': trending_down, 'contraction': contraction})

    # Group by Symbol and check for downward trend and calculate contraction
    volume_trend = df_last_2_weeks.groupby('Symbol').apply(volume_trend_and_contraction)

    # Filter for stocks with volume contraction
    stocks_with_contraction = volume_trend[volume_trend['trending_down']]

    return stocks_with_contraction

def save_tickers_to_csv(stocks_data, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Symbol', 'Volume-Contraction(%)'])  # Header
        for symbol, data in stocks_data.iterrows():
            writer.writerow([symbol, f"{data['contraction']:.2f}"])

# Main execution
input_file = './ranking_screens/passed_stocks_input_data/filtered_price_data.csv'
output_file = './ranking_screens/results/stocks_with_volume_contraction.csv'

stocks_with_contraction = calculate_volume_trend(input_file)
save_tickers_to_csv(stocks_with_contraction, output_file)

print(f"{len(stocks_with_contraction)} stocks with volume contraction in the last 2 weeks saved to {output_file}.")