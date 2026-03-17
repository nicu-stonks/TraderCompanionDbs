import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def calculate_quarterly_averages(group):
    """Calculate average prices for each quarter (3-month periods)"""
    # Get the most recent date
    latest_date = group['Date'].max()
    
    # Define quarter boundaries (in days, approximately)
    quarters = {
        'Q0_current': 0,      # Current price (most recent)
        'Q1_avg': 63,         # 0-3 months ago average
        'Q2_avg': 126,        # 3-6 months ago average  
        'Q3_avg': 189,        # 6-9 months ago average
        'Q4_avg': 252         # 9-12 months ago average
    }
    
    quarterly_prices = {}
    
    # Current price (most recent)
    quarterly_prices['current'] = group['Close'].iloc[-1]
    
    # Calculate average price for each quarter
    for quarter, days_back in quarters.items():
        if quarter == 'Q0_current':
            continue
            
        # Define the date range for this quarter
        if quarter == 'Q1_avg':  # 0-3 months ago
            start_days = 0
            end_days = 63
        elif quarter == 'Q2_avg':  # 3-6 months ago
            start_days = 63
            end_days = 126
        elif quarter == 'Q3_avg':  # 6-9 months ago
            start_days = 126
            end_days = 189
        elif quarter == 'Q4_avg':  # 9-12 months ago
            start_days = 189
            end_days = 252
        
        # Get data for this quarter
        quarter_data = group.iloc[-(end_days):-(start_days)] if start_days > 0 else group.iloc[-(end_days):]
        
        if len(quarter_data) > 0:
            quarterly_prices[quarter] = quarter_data['Close'].mean()
        else:
            quarterly_prices[quarter] = np.nan
    
    return quarterly_prices

def calculate_roc_from_averages(current_price, avg_price):
    """Calculate Rate of Change from current price to quarterly average"""
    if pd.isna(avg_price) or avg_price == 0:
        return np.nan
    return (current_price / avg_price - 1) * 100

def calculate_rs_rating(group):
    """Calculate IBD-style Relative Strength rating using quarterly averages"""
    # Ensure we have enough data for the longest period (252 days)
    if len(group) <= 252:
        return np.nan
    
    # Sort by date to ensure proper order
    group = group.sort_values('Date')
    
    # Calculate quarterly averages
    quarterly_prices = calculate_quarterly_averages(group)
    current_price = quarterly_prices['current']
    
    # Calculate Rate of Change for different periods using quarterly averages
    roc_q1 = calculate_roc_from_averages(current_price, quarterly_prices.get('Q1_avg'))  # vs 0-3 months avg
    roc_q2 = calculate_roc_from_averages(current_price, quarterly_prices.get('Q2_avg'))  # vs 3-6 months avg
    roc_q3 = calculate_roc_from_averages(current_price, quarterly_prices.get('Q3_avg'))  # vs 6-9 months avg
    roc_q4 = calculate_roc_from_averages(current_price, quarterly_prices.get('Q4_avg'))  # vs 9-12 months avg
    
    # Check if we have all required ROC values
    if pd.isna(roc_q1) or pd.isna(roc_q2) or pd.isna(roc_q3) or pd.isna(roc_q4):
        return np.nan
    
    # Calculate Strength Factor using the IBD formula with quarterly averages
    # StrengthFactor = 0.4 * ROC(Q1) + 0.2 * ROC(Q2) + 0.2 * ROC(Q3) + 0.2 * ROC(Q4)
    strength_factor = (0.4 * roc_q1 + 0.2 * roc_q2 + 0.2 * roc_q3 + 0.2 * roc_q4)
    
    return strength_factor

def main():
    # Get the absolute path of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find the absolute path of the "flask_microservice_stocks_filterer" directory
    while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
        script_dir = os.path.dirname(script_dir)
    
    # Define the input and output file paths
    input_file = os.path.join(script_dir, "stocks_filtering_application", "price_data", "all_tickers_historical.csv")
    output_file = os.path.join(script_dir, "stocks_filtering_application", "minervini_1mo", "obligatory_screens", "results", "raw_rs_file.csv")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Resolved input file path: {input_file}")
    print(f"Resolved output file path: {output_file}")
    
    # Read the CSV file
    print("Reading historical stock data...")
    df = pd.read_csv(input_file, parse_dates=['Date'])
    
    # Group the data by stock symbol
    grouped = df.groupby('Symbol')
    
    # Calculate RS rating for each stock
    print("Calculating RS ratings using quarterly averages for all stocks...")
    rs_values = {}
    processed_count = 0
    total_stocks = len(grouped)
    
    for symbol, group in grouped:
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{total_stocks} stocks...")
        
        # Sort by date
        group = group.sort_values('Date')
        
        # Calculate RS rating using quarterly averages
        rs_value = calculate_rs_rating(group)
        if not np.isnan(rs_value):
            rs_values[symbol] = rs_value
    
    # Convert to DataFrame
    rs_df = pd.DataFrame(list(rs_values.items()), columns=['Symbol', 'StrengthFactor'])
    
    # Rank stocks and calculate percentile (0-99)
    print("Ranking stocks based on RS rating...")
    rs_df['IBD_RSI'] = rs_df['StrengthFactor'].rank(pct=True) * 100
    rs_df['IBD_RSI'] = rs_df['IBD_RSI'].apply(lambda x: int(min(99, max(0, x))))  # Ensure between 0-99
    
    # Sort by RS rating in descending order
    rs_df = rs_df.sort_values('IBD_RSI', ascending=False)
    
    # Keep only Symbol and IBD_RSI columns
    rs_df = rs_df[['Symbol', 'IBD_RSI']]
    
    # Save to CSV
    print(f"Saving RS ratings for {len(rs_df)} stocks to {output_file}")
    rs_df.to_csv(output_file, index=False)
    print("RS rating calculation complete!")
    print(f"Successfully calculated RS ratings using quarterly averages for {len(rs_df)} stocks")

if __name__ == "__main__":
    main()