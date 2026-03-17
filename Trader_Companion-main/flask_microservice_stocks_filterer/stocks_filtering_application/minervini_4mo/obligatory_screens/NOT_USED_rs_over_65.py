import pandas as pd
import numpy as np

def calculate_rsi(data, periods=14):
    close_delta = data['Close'].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    # Calculate the EWMA
    ma_up = up.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
    

    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

# Read the CSV file
df = pd.read_csv('../../stock_api_data/amex_arca_bats_nasdaq_nyse_otc_stocks_1_year_price_data.csv', parse_dates=['Date'])

# Group by Symbol and calculate RSI
def calculate_group_rsi(group):
    return pd.Series({'RSI': calculate_rsi(group).iloc[-1]})

rsi_df = df.groupby('Symbol').apply(calculate_group_rsi).reset_index()

# Filter stocks with RSI above 65
high_rsi_stocks = rsi_df[rsi_df['RSI'] > 40]

# Save the filtered ticker symbols and RSI to a new CSV file
high_rsi_stocks[['Symbol', 'RSI']].to_csv('./obligatory_screens/results/rs_over_65.csv', index=False)

print(f"Saved {len(high_rsi_stocks)} stocks with 14-day RSI > 65 to './obligatory_screens/results/rs_over_65.csv'")