import pandas as pd
import numpy as np
import yfinance as yf
import os

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Computes the RSI (Relative Strength Index) for a given price series.
    Returns a pandas Series of RSI values aligned with the input index.
    Handles edge cases where there are no losses (RSI=100) or no gains (RSI=0).
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Handle edge cases
    # When loss is zero, RSI should be 100 (complete bullish)
    # When gain is zero, RSI should be 0 (complete bearish)
    rs = np.where(
        avg_loss == 0, 
        float('inf'),  # When no losses, RS is infinity, RSI will be 100
        np.where(
            avg_gain == 0,
            0,          # When no gains, RS is 0, RSI will be 0
            avg_gain / avg_loss
        )
    )
    
    # Convert back to Series with original index
    rs = pd.Series(rs, index=avg_gain.index)
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    # Handle infinity case explicitly (when rs is infinity, RSI is 100)
    rsi = rsi.replace([np.inf, -np.inf], [100, 0])
    
    return rsi

def print_rsi_calculation_steps(symbol: str, prices: pd.Series, period: int = 14):
    """
    Prints detailed steps of RSI calculation for a specific symbol.
    Handles edge cases for RSI calculation.
    """
    # print(f"\n----- DETAILED RSI CALCULATION FOR {symbol} -----")
    # print(f"Period used for calculation: {period} days")
    
    # Calculate deltas
    delta = prices.diff()
    # print("\nStep 1: Calculate price deltas (day-to-day change)")
    # print(delta.head(period+5).to_string())
    
    # Calculate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # print("\nStep 2: Separate gains and losses")
    # print("Gains:")
    # print(gain.head(period+5).to_string())
    # print("\nLosses:")
    # print(loss.head(period+5).to_string())
    
    # Calculate average gains and losses
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    # print(f"\nStep 3: Calculate average gains and losses over {period} days")
    # print("Average Gains:")
    # print(avg_gain.head(period+5).to_string())
    # print("\nAverage Losses:")
    # print(avg_loss.head(period+5).to_string())
    
    # Calculate RS and RSI with proper handling of edge cases
    rs_values = []
    for i in range(len(avg_gain)):
        if pd.isna(avg_gain.iloc[i]) or pd.isna(avg_loss.iloc[i]):
            rs_values.append(np.nan)
        elif avg_loss.iloc[i] == 0:
            rs_values.append(float('inf'))  # When no losses, RS is infinity, RSI will be 100
        elif avg_gain.iloc[i] == 0:
            rs_values.append(0)  # When no gains, RS is 0, RSI will be 0
        else:
            rs_values.append(avg_gain.iloc[i] / avg_loss.iloc[i])
    
    rs = pd.Series(rs_values, index=avg_gain.index)
    
    # Calculate RSI with proper handling of infinity values
    rsi_values = []
    for rs_val in rs:
        if pd.isna(rs_val):
            rsi_values.append(np.nan)
        elif rs_val == float('inf'):
            rsi_values.append(100.0)  # When RS is infinity, RSI is 100
        else:
            rsi_values.append(100 - (100 / (1 + rs_val)))
    
    rsi = pd.Series(rsi_values, index=rs.index)
    
    # print("\nStep 4: Calculate Relative Strength (RS = Avg Gain / Avg Loss)")
    # print("Note: Infinity means all gains and no losses (RSI will be 100)")
    rs_display = rs.copy()
    rs_display = rs_display.replace([np.inf, -np.inf], ['Infinity', 'NegInfinity'])
    # print(rs_display.head(period+5).to_string())
    
    # print("\nStep 5: Calculate RSI = 100 - (100 / (1 + RS))")
    # print("Note: When RS is Infinity, RSI is set to 100")
    # print(rsi.head(period+5).to_string())
    
    # After period+5 rows, just show a summary
    # if len(rsi) > period+5:
    #     print("\nRSI values for remaining days (summary):")
    #     print(rsi.iloc[period+5:].describe().to_string())
    
    # print("----- END OF RSI CALCULATION -----\n")
    
    return rsi

def main():
    # ------------------------------------------------------------------------------
    # 1) Locate the input_file under "flask_microservice_stocks_filterer" directory
    # ------------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    while (not script_dir.endswith("flask_microservice_stocks_filterer")
           and os.path.dirname(script_dir) != script_dir):
        script_dir = os.path.dirname(script_dir)

    input_file = os.path.join(
        script_dir,
        "stocks_filtering_application",
        "minervini_1mo_unbanned",
        "ranking_screens",
        "passed_stocks_input_data",
        "filtered_price_data.csv"
    )

    # ------------------------------------------------------------------------------
    # 2) READ CSV & FORCE PARSE THE "Date" COLUMN AS A NAIVE DATETIME
    # ------------------------------------------------------------------------------
    # print(f"Reading CSV from: {input_file}")
    df = pd.read_csv(input_file)

    # print("\nDEBUG: Checking initial columns in df:", df.columns.tolist())
    # print("DEBUG: Head of df:\n", df.head(), "\n")

    # Force-parse date strings (including time zones) to naive datetimes
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df["Date"] = df["Date"].dt.tz_convert(None)

    # Drop any rows with no valid Date
    initial_count = len(df)
    df = df.dropna(subset=["Date"])
    # print(f"DEBUG: Dropped {initial_count - len(df)} rows due to invalid 'Date'.")
    # print("DEBUG: df shape after date parsing:", df.shape)

    # ------------------------------------------------------------------------------
    # 3) FILTER TO THE LAST ~3 MONTHS
    # ------------------------------------------------------------------------------
    last_date_in_stocks = df["Date"].max()
    three_months_ago = last_date_in_stocks - pd.DateOffset(months=3)
    df = df[df["Date"] >= three_months_ago].copy()
    # print(f"DEBUG: After filtering to last 3 months (>= {three_months_ago.date()}), df shape: {df.shape}")

    # ------------------------------------------------------------------------------
    # 4) SORT & CALCULATE STOCK RSI USING GROUPBY TRANSFORM
    # ------------------------------------------------------------------------------
    df.sort_values(by=["Symbol", "Date"], inplace=True)
    
    # Check if ROOT exists in the dataset
    if "ROOT" in df["Symbol"].unique():
        # print("\n*** ROOT found in dataset! Will track detailed RSI calculation ***")
        # Create a separate variable for ROOT data
        lth_data = df[df["Symbol"] == "ROOT"].copy()
        lth_data = lth_data.sort_values(by="Date")
        
        # Print detailed steps for ROOT RSI calculation
        lth_data["StockRSI"] = print_rsi_calculation_steps("ROOT", lth_data["Close"])
        
        # Apply normal calculation for all symbols
        df["StockRSI"] = df.groupby("Symbol")["Close"].transform(compute_rsi)
    else:
        # print("\n*** ROOT not found in dataset! Will proceed with normal calculations ***")
        df["StockRSI"] = df.groupby("Symbol")["Close"].transform(compute_rsi)

    # print("DEBUG: Checking if StockRSI is filled or mostly NaN:")
    # print(df["StockRSI"].describe())  # quick stats

    # ------------------------------------------------------------------------------
    # 5) FETCH MARKET DATA FROM YFINANCE & CALCULATE MARKET RSI
    # ------------------------------------------------------------------------------
    market_symbol = "^GSPC"  # S&P 500
    market_ticker = yf.Ticker(market_symbol)

    # print(f"\nFetching market data for {market_symbol} from {three_months_ago.date()} to {last_date_in_stocks.date()}")
    
    # Make sure to fetch daily data with no gaps
    market_data = market_ticker.history(
        start=three_months_ago.strftime("%Y-%m-%d"),
        end=(last_date_in_stocks + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),  # Add one day to ensure we get the last date
        interval="1d"
    )
    
    # Print the last 30 days of raw market data from yfinance
    # print("\n----- LAST 30 DAYS OF RAW MARKET DATA FROM YFINANCE -----")
    # Sort by date in descending order and take last 30 days
    last_30_days = market_data.sort_index(ascending=False).head(30)
    # Reset index to make Date a regular column for better display
    display_data = last_30_days.reset_index()
    # Print with full details
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 1000)  # Wide display
    # print(display_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].to_string(index=False))
    # print("-----------------------------------------------------\n")
    # Reset display options
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    
    # Check if we have enough market data
    # print(f"DEBUG: Fetched {len(market_data)} days of market data")
    if len(market_data) < 5:
        # print("WARNING: Very little market data returned. Trying alternative method...")
        # Try an alternative approach - fetch a longer period and then filter
        extended_start = three_months_ago - pd.DateOffset(days=30)  # Go back an extra month
        market_data = market_ticker.history(
            start=extended_start.strftime("%Y-%m-%d"),
            end=(last_date_in_stocks + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
            interval="1d"
        )
        # print(f"DEBUG: Fetched {len(market_data)} days of market data with extended range")
        market_data = market_data[market_data.index >= pd.Timestamp(three_months_ago)]
        # print(f"DEBUG: After filtering to our date range: {len(market_data)} days")

    market_data.reset_index(inplace=True)
    market_data.rename(columns={"Date": "MarketDate"}, inplace=True)

    # Debug the fetched dates
    # print("\nDEBUG: Range of fetched market dates:")
    # print(f"First date: {market_data['MarketDate'].min()}")
    # print(f"Last date: {market_data['MarketDate'].max()}")
    # print(f"Total dates: {market_data['MarketDate'].nunique()}")
    
    # Check for missing dates
    all_dates = pd.date_range(start=market_data['MarketDate'].min(), end=market_data['MarketDate'].max())
    missing_dates = set(all_dates) - set(market_data['MarketDate'])
    if missing_dates:
        # print(f"WARNING: Found {len(missing_dates)} missing dates in market data")
        # print(f"Missing dates (first 5): {sorted(list(missing_dates))[:5]}")
        pass  # If missing dates are weekends/holidays, this is normal

    # Ensure naive datetimes
    market_data["MarketDate"] = pd.to_datetime(market_data["MarketDate"], errors="coerce", utc=True)
    market_data["MarketDate"] = market_data["MarketDate"].dt.tz_convert(None)

    # Drop rows with invalid MarketDate
    market_data = market_data.dropna(subset=["MarketDate"])
    market_data.sort_values("MarketDate", inplace=True)

    # Compute MarketRSI with detailed calculation for reference
    # print("\n----- MARKET RSI CALCULATION (S&P 500) -----")
    market_data["MarketRSI"] = print_rsi_calculation_steps(market_symbol, market_data["Close"])
    # print("DEBUG: market_data head:\n", market_data.head(), "\n")
    # print("DEBUG: market_data MarketRSI describe:\n", market_data["MarketRSI"].describe(), "\n")

    # ------------------------------------------------------------------------------
    # 6) MERGE STOCK DATA & MARKET DATA ON DATE - FIXED APPROACH
    # ------------------------------------------------------------------------------
    
    # Ensure market data dates are properly formatted and sorted
    market_data.rename(columns={"MarketDate": "Date"}, inplace=True)
    market_data.sort_values("Date", inplace=True)
    
    # IMPORTANT: DO NOT FILL NaN VALUES IN MARKET RSI
    # REMOVING THIS LINE:
    # market_data["MarketRSI"] = market_data["MarketRSI"].ffill().bfill()

    # Print to debug
    # print("\nDEBUG: Market data with preserved NaN values:")
    # print(market_data[["Date", "Close", "MarketRSI"]].head().to_string())
    # print(market_data[["Date", "Close", "MarketRSI"]].tail().to_string())

    # This is a critical step: Create a date-based lookup function that handles weekends/holidays
    # For any given date, find the most recent valid market date (for weekends/holidays)
    def get_closest_market_date(date, market_dates):
        """Find the most recent valid market date for any given date."""
        # Convert date to datetime if it's not already
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)
        
        # Find all market dates that are on or before the given date
        valid_dates = [d for d in market_dates if d <= date]
        
        if not valid_dates:
            # If no valid date found (rare case), use the earliest market date
            return min(market_dates)
        
        # Return the most recent valid market date
        return max(valid_dates)

    # Create a dictionary mapping all possible dates to their corresponding market data
    # First, get all unique dates from stock data
    all_stock_dates = df["Date"].unique()

    # Get all market dates as a sorted list (for lookup)
    market_dates = sorted(market_data["Date"].unique())

    # Create a mapping from each stock date to the closest market date
    date_mapping = {}
    for stock_date in all_stock_dates:
        closest_market_date = get_closest_market_date(stock_date, market_dates)
        date_mapping[stock_date] = closest_market_date

    # Create a lookup dictionary for market data based on date
    # IMPORTANT: Do not modify the MarketRSI values - let NaN values remain NaN
    market_lookup = {
        date: {
            "Close": row["Close"], 
            "MarketRSI": row["MarketRSI"]  # Do not modify NaN values
        } 
        for date, row in market_data.set_index("Date").iterrows()
    }

    # Now apply the mapping to create new columns in df
    # Initialize new columns
    df["Close_Market"] = None
    df["MarketRSI"] = None

    # Fill market data based on the date mapping
    for idx, row in df.iterrows():
        stock_date = row["Date"]
        if stock_date in date_mapping:
            market_date = date_mapping[stock_date]
            if market_date in market_lookup:
                df.at[idx, "Close_Market"] = market_lookup[market_date]["Close"]
                df.at[idx, "MarketRSI"] = market_lookup[market_date]["MarketRSI"]
                # Do not modify NaN values - leave them as is

    # Create merged dataframe for further processing
    merged = df.copy()

    # Verify the data is properly populated
    # print("\nDEBUG: Sample of stock data with mapped market data:")
    sample_symbols = df["Symbol"].unique()[:3]  # Take first 3 symbols
    for symbol in sample_symbols:
        symbol_data = merged[merged["Symbol"] == symbol].head(3)
        # print(f"\nSample data for {symbol}:")
        # print(symbol_data[["Date", "Close", "StockRSI", "Close_Market", "MarketRSI"]].to_string())
        pass

    # print("\nDEBUG: merged shape:", merged.shape)
    # print("DEBUG: merged columns:", merged.columns.tolist())

    # ------------------------------------------------------------------------------
    # 7) CALCULATE RSI_vs_Market (only when both RSIs are available)
    # ------------------------------------------------------------------------------
    # Only calculate the difference when both values are available
    merged["RSI_vs_Market"] = np.where(
        pd.isna(merged["StockRSI"]) | pd.isna(merged["MarketRSI"]),
        np.nan,  # If either RSI is NaN, the difference is also NaN
        merged["StockRSI"] - merged["MarketRSI"]
    )
    
    # Print ROOT RSI vs Market if present
    if "ROOT" in df["Symbol"].unique():
        lth_merged = merged[merged["Symbol"] == "ROOT"].copy()
        lth_merged = lth_merged.sort_values("Date")
        
        # print("\n----- ROOT RSI VS MARKET DETAILS -----")
        # print("Date | ROOT Price | ROOT RSI | Market Price | Market RSI | RSI_vs_Market")
        # print("--------------------------------------------------------------------------")
        for _, row in lth_merged.iterrows():
            # Handle any remaining NaN values for display
            stock_rsi = row['StockRSI'] if not pd.isna(row['StockRSI']) else 'N/A'
            market_rsi = row['MarketRSI'] if not pd.isna(row['MarketRSI']) else 'N/A'
            rsi_diff = row['RSI_vs_Market'] if not pd.isna(row['RSI_vs_Market']) else 'N/A'
            market_price = row['Close_Market'] if not pd.isna(row['Close_Market']) else 'N/A'
            
            # Format as numbers only when they're actually numbers
            stock_rsi_fmt = f"{stock_rsi:.2f}" if isinstance(stock_rsi, (int, float)) else stock_rsi
            market_rsi_fmt = f"{market_rsi:.2f}" if isinstance(market_rsi, (int, float)) else market_rsi
            rsi_diff_fmt = f"{rsi_diff:.2f}" if isinstance(rsi_diff, (int, float)) else rsi_diff
            market_price_fmt = f"${market_price:.2f}" if isinstance(market_price, (int, float)) else market_price
            
            # print(f"{row['Date'].strftime('%Y-%m-%d')} | ${row['Close']:.2f} | {stock_rsi_fmt} | {market_price_fmt} | {market_rsi_fmt} | {rsi_diff_fmt}")
            pass
        
        # Calculate summary statistics only on valid numeric values
        valid_diff = lth_merged["RSI_vs_Market"].dropna()
        if not valid_diff.empty:
            # print("\nSummary statistics for ROOT RSI vs Market:")
            # print(valid_diff.describe().to_string())
            pass
        else:
            # print("\nNo valid RSI_vs_Market values to compute statistics.")
            pass
        # print("--------------------------------------------------------------------------\n")

    # Group by Symbol to find indices of the max / min rows for RSI_vs_Market
    groupobj = merged.groupby("Symbol")["RSI_vs_Market"]

    def safe_idxmax(series):
        if series.notna().any():
            return series.idxmax()
        return np.nan

    def safe_idxmin(series):
        if series.notna().any():
            return series.idxmin()
        return np.nan

    idx_series_max = groupobj.apply(safe_idxmax).dropna()
    idx_series_min = groupobj.apply(safe_idxmin).dropna()

    # If no valid rows for either max or min, we stop
    if len(idx_series_max) == 0 and len(idx_series_min) == 0:
        # print("WARNING: No valid rows found for any symbol (all RSI_vs_Market are NaN). Exiting early.")
        return

    # ------------------------------------------------------------------------------
    # 8) EXTRACT AND SAVE MAX & MIN RSI_vs_Market
    # ------------------------------------------------------------------------------
    # Get the rows for each symbol's max RSI_vs_Market
    best_days = merged.loc[idx_series_max, ["Symbol", "RSI_vs_Market"]].copy()
    best_days.rename(columns={"RSI_vs_Market": "Max_Market_RSI_Diff_3M"}, inplace=True)
    # Sort descending by the max difference
    best_days.sort_values(by="Max_Market_RSI_Diff_3M", ascending=False, inplace=True)
    
    # Print ROOT best day if it exists
    if "ROOT" in best_days["Symbol"].values:
        lth_best = best_days[best_days["Symbol"] == "ROOT"]
        # print(f"ROOT MAX RSI vs Market: {lth_best['Max_Market_RSI_Diff_3M'].values[0]:.2f}")
        pass
    
    max_output_file = os.path.join(
        script_dir,
        "stocks_filtering_application",
        "minervini_1mo_unbanned",
        "ranking_screens",
        "results",
        "max_rsi_vs_market_3mo.csv"
    )
    best_days.to_csv(max_output_file, index=False)
    # print(f"Saved MAX RSI difference to: {max_output_file}")

    # Get the rows for each symbol's min RSI_vs_Market
    worst_days = merged.loc[idx_series_min, ["Symbol", "RSI_vs_Market"]].copy()
    worst_days.rename(columns={"RSI_vs_Market": "Min_Market_RSI_Diff_3M"}, inplace=True)
    # Sort ascending by the min difference
    worst_days.sort_values(by="Min_Market_RSI_Diff_3M", ascending=False, inplace=True)
    
    # Print ROOT worst day if it exists
    if "ROOT" in worst_days["Symbol"].values:
        lth_worst = worst_days[worst_days["Symbol"] == "ROOT"]
        # print(f"ROOT MIN RSI vs Market: {lth_worst['Min_Market_RSI_Diff_3M'].values[0]:.2f}")
        pass
    
    min_output_file = os.path.join(
        script_dir,
        "stocks_filtering_application",
        "minervini_1mo_unbanned",
        "ranking_screens",
        "results",
        "min_rsi_vs_market_3mo.csv"
    )
    worst_days.to_csv(min_output_file, index=False)
    # print(f"Saved MIN RSI difference to: {min_output_file}")

    # print("\nDone!")

if __name__ == "__main__":
    main()