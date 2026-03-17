import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import concurrent.futures
import time
import random


def get_stock_data(ticker):
    try:
        # Add random delay between requests to avoid rate limiting
        time.sleep(random.uniform(0.5, 1.5))

        stock = yf.Ticker(ticker)

        # Get price data for the last year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2 * 365)  # 2 years of data

        try:
            price_data = stock.history(start=start_date, end=end_date, interval='1d')
        except Exception as e:
            print(f"Error fetching price data for {ticker}: {e}")
            return pd.DataFrame(), [], []

        if price_data.empty:
            print(f"No price data found for {ticker}")
            return pd.DataFrame(), [], []

        price_data['Symbol'] = ticker
        price_data.reset_index(inplace=True)
        price_data = price_data.round(2)

        # Get financial statements with error handling
        try:
            yearly_balance_sheet = stock.balance_sheet
            quarterly_balance_sheet = stock.quarterly_balance_sheet
            yearly_income_stmt = stock.income_stmt
            quarterly_income_stmt = stock.quarterly_income_stmt
        except Exception as e:
            print(f"Error fetching financial statements for {ticker}: {e}")
            return price_data, [], []  # Return price data even if financials fail

        quarterly_data = []
        annual_data = []

        def extract_fundamental_data(date, is_annual=False):
            try:
                if date in quarterly_balance_sheet.columns:
                    bs = quarterly_balance_sheet.loc[:, date]
                else:
                    return None
                if date in quarterly_income_stmt.columns:
                    is_ = quarterly_income_stmt.loc[:, date] if not is_annual else yearly_income_stmt.loc[:, date]
                else:
                    return None

                total_revenue = is_.get('Total Revenue', 0)
                net_income = is_.get('Net Income', 0)

                return {
                    'Symbol': ticker,
                    'Date': date.strftime('%Y-%m-%d'),
                    'Total Assets': bs.get('Total Assets', ''),
                    'Total Liabilities': bs.get('Total Liabilities Net Minority Interest', ''),
                    'Total Equity': bs.get('Total Equity Gross Minority Interest', ''),
                    'Total Revenue': total_revenue,
                    'Gross Profit': is_.get('Gross Profit', ''),
                    'Operating Income': is_.get('Operating Income', ''),
                    'Net Income': net_income,
                    'Net profit margin': (net_income / total_revenue if total_revenue != 0 else ''),
                    'EPS': is_.get('Diluted EPS', ''),
                }
            except Exception as e:
                print(f"Error extracting fundamental data for {ticker} at {date}: {e}")
                return None

        # Get quarterly data for last 3 years
        two_years_ago = datetime.now() - timedelta(days=3 * 365)
        for date in quarterly_balance_sheet.columns:
            if date >= two_years_ago:
                data = extract_fundamental_data(date)
                if data:
                    quarterly_data.append(data)

        # Get annual data for last 3 years
        for date in yearly_balance_sheet.columns.year:
            if date < two_years_ago.year:
                continue
            annual_date = yearly_balance_sheet.columns[yearly_balance_sheet.columns.year == date][-1]
            data = extract_fundamental_data(annual_date, is_annual=True)
            if data:
                annual_data.append(data)

        print(f"Successfully fetched data for {ticker}")
        return price_data, quarterly_data, annual_data

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame(), [], []


def main():
    # Read NASDAQ stock list
    amex_arca_bats_nasdaq_nyse_otc_stocks = pd.read_csv('./stock_tickers/amex_arca_bats_nasdaq_nyse_otc_stocks.csv')

    # Create lists to store all data
    all_price_data = []
    all_quarterly_fundamental_data = []
    all_annual_fundamental_data = []

    # Reduce max_workers to avoid overwhelming the API
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Process stocks in smaller batches
        batch_size = 50
        for i in range(0, len(amex_arca_bats_nasdaq_nyse_otc_stocks), batch_size):
            batch = amex_arca_bats_nasdaq_nyse_otc_stocks['Symbol'][i:i + batch_size]

            # Submit batch of tasks
            futures = [executor.submit(get_stock_data, ticker) for ticker in batch]

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    price_data, quarterly_data, annual_data = future.result()
                    if not price_data.empty:
                        all_price_data.append(price_data)
                    all_quarterly_fundamental_data.extend(quarterly_data)
                    all_annual_fundamental_data.extend(annual_data)
                except Exception as e:
                    print(f"Error processing future: {e}")

            print(f"Completed batch {i // batch_size + 1} of {(len(amex_arca_bats_nasdaq_nyse_otc_stocks) + batch_size - 1) // batch_size}")
            time.sleep(45)  # Add delay between batches

    # Save the data
    try:
        if all_price_data:
            combined_price_data = pd.concat(all_price_data, ignore_index=True)
            combined_price_data.to_csv('./stock_api_data/amex_arca_bats_nasdaq_nyse_otc_stocks_1_year_price_data.csv', index=False)
            print(f"Saved price data for {len(all_price_data)} stocks")

        quarterly_fundamental_data = [item for item in all_quarterly_fundamental_data if item is not None]
        if quarterly_fundamental_data:
            quarterly_fundamental_df = pd.DataFrame(quarterly_fundamental_data)
            quarterly_fundamental_df.to_csv('./stock_api_data/quarterly_fundamental_data_2years.csv', index=False)
            print(f"Saved quarterly data for {len(quarterly_fundamental_data)} records")

        annual_fundamental_data = [item for item in all_annual_fundamental_data if item is not None]
        if annual_fundamental_data:
            annual_fundamental_df = pd.DataFrame(annual_fundamental_data)
            annual_fundamental_df.to_csv('./stock_api_data/annual_fundamental_data_2years.csv', index=False)
            print(f"Saved annual data for {len(annual_fundamental_data)} records")

    except Exception as e:
        print(f"Error saving data: {e}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

