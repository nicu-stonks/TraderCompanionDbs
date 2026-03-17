import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def main():
    # --- SETUP & PATHS ---
    
    # Get the absolute path of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Find the absolute path of the "flask_microservice_stocks_filterer" directory
    while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
        script_dir = os.path.dirname(script_dir)

    # Input file (same format as your existing code)
    input_file = os.path.join(
        script_dir,
        "stocks_filtering_application",
        "stock_api_data",
        "amex_arca_bats_nasdaq_nyse_otc_stocks_1_year_price_data.csv"
    )

    # Output file (only the 4-day rolling percentage)
    output_file = os.path.join(
        script_dir,
        "stocks_filtering_application",
        "sentiment_graphs",
        "results",
        "rolling_4day_percentage_new_32week_highs.csv"
    )

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by symbol then by date
    df.sort_values(['Symbol', 'Date'], inplace=True)

    # Print date range
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    print(f"Data range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Group by symbol
    grouped = df.groupby('Symbol')

    # Dictionary to store daily results:
    #   results_dict[date_str] = {"total": X, "new_high": Y}
    #      'total' = # of symbols with 224 prior days of data (i.e. RollingMax224 not NaN)
    #      'new_high' = # of those that actually set a new 32-week high
    results_dict = {}

    # -- PROCESS SYMBOL BY SYMBOL --
    total_symbols = len(grouped)
    print(f"Total symbols found: {total_symbols}")

    for i, (symbol, stock_data) in enumerate(grouped, start=1):
        # If there's not enough rows for 224 prior days, skip (can never form a 32wk high)
        if len(stock_data) < 224:
            # Progress print
            if (i % 100 == 0) or (i == total_symbols):
                print(f"Processed {i}/{total_symbols} symbols ({i/total_symbols*100:.1f}%)")
            continue

        # Sort by date just in case (should already be sorted)
        stock_data = stock_data.sort_values('Date').copy()

        # Compute the rolling max of the prior 224 days' High (so shift by 1 day, then rolling)
        # This ensures for row i, we look at the 224 days behind it (excluding current day).
        stock_data['RollingMax224'] = (
            stock_data['High'].shift(1).rolling(window=224, min_periods=224).max()
        )

        # Identify new 32-week highs (current day's High is strictly > max of previous 224 days)
        stock_data['New_32wk_High'] = stock_data['High'] > stock_data['RollingMax224']

        # For each row, if RollingMax224 is not NaN, that means the symbol is "eligible"
        # If 'New_32wk_High' is True, that means it set a new 32-week high that day
        for idx_row, row in stock_data.iterrows():
            date_obj = row['Date']
            date_str = date_obj.strftime('%Y-%m-%d')

            # If RollingMax224 is NaN, skip
            if pd.isna(row['RollingMax224']):
                continue

            if date_str not in results_dict:
                results_dict[date_str] = {'total': 0, 'new_high': 0}

            # Increase count of symbols with enough data on this date
            results_dict[date_str]['total'] += 1

            # If it's a new 32-week high
            if row['New_32wk_High']:
                results_dict[date_str]['new_high'] += 1

        # Print progress every 100 symbols or when done
        if (i % 100 == 0) or (i == total_symbols):
            print(f"Processed {i}/{total_symbols} symbols ({(i/total_symbols)*100:.1f}%)")

    # -- BUILD A DAILY PERCENTAGE DATAFRAME --
    # Sort dates in ascending order
    sorted_dates = sorted(results_dict.keys())

    daily_records = []
    for date_str in sorted_dates:
        day_info = results_dict[date_str]
        total = day_info['total']
        new_high = day_info['new_high']

        # Calculate the daily percentage
        daily_pct = 0.0
        if total > 0:
            daily_pct = (new_high / total) * 100

        daily_records.append({
            'Date': date_str,
            'Daily_Percentage_New_32Week_Highs': daily_pct
        })

    daily_df = pd.DataFrame(daily_records)
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    daily_df.sort_values('Date', inplace=True)
    daily_df.reset_index(drop=True, inplace=True)

    # -- 4-DAY ROLLING AVERAGE OF DAILY PERCENTAGE --
    daily_df['Rolling_4Day_Percentage_32Week_Highs'] = (
        daily_df['Daily_Percentage_New_32Week_Highs']
        .rolling(window=4)
        .mean()
    )

    # -- FINAL OUTPUT --
    # The user asked to "leave only the 4 day rolling percentage in the output file."
    # We'll keep the Date for reference, plus the 4-day rolling column.
    # (If you truly want only the rolling column, remove "Date" from the final columns.)
    final_df = daily_df[['Date', 'Rolling_4Day_Percentage_32Week_Highs']].copy()

    # Save to CSV
    final_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print(f"Total days with eligible data: {len(final_df)}")


if __name__ == "__main__":
    main()
