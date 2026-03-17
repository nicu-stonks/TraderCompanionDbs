from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData
import threading
import time
import datetime
import queue
import csv
import os
import pandas as pd

class HistoricalDataApp(EClient, EWrapper):
    def __init__(self, tickers_to_process, output_dir):
        EClient.__init__(self, self)
        
        self.tickers_to_process = tickers_to_process
        self.data_queue = queue.Queue()
        self.orderId = None
        self.processed_tickers = set()
        self.processing_tickers = set()  # Keep track of tickers currently being processed
        
        # Output directory
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Rate-limit tracking
        self.request_count = 0
        self.minute_start = time.time()
        self.request_lock = threading.Lock()
        
        # Timeout tracking
        self.last_response_time = time.time()
        self.processing_complete = False
        
        # Store historical data
        self.ticker_data = {}
    
    def nextValidId(self, orderId):
        self.orderId = orderId
        print(f"Next Valid Id: {orderId}")
        # Start requesting once we have a valid ID
        self.process_next_ticker()
    
    def nextId(self):
        self.orderId += 1
        return self.orderId
    
    def error(self, reqId, errorCode, errorString, advancedOrderReject=""):
        print(f"Error {errorCode}: {errorString}")
        if 0 <= reqId < len(self.tickers_to_process):
            ticker = self.tickers_to_process[reqId]
            print(f"Error for ticker {ticker}: {errorString}")
            
            # If it's a duplicate ticker ID error, but we already have data for this ticker,
            # let's just save what we have and move on
            if errorCode == 322 and "Duplicate ticker ID" in errorString and ticker in self.ticker_data and self.ticker_data[ticker]:
                print(f"Saving partial data for {ticker} despite duplicate ID error")
                self.save_ticker_data_to_csv(ticker)
            
            # Clean up processing status and mark as processed
            if ticker in self.processing_tickers:
                self.processing_tickers.remove(ticker)
            self.processed_tickers.add(ticker)
            
            # Try next ticker
            self.process_next_ticker()
            
            # Update last response time on error
            self.last_response_time = time.time()
    
    def historicalData(self, reqId, bar):
        """Called when historical data bar is received."""
        if reqId < len(self.tickers_to_process):
            ticker = self.tickers_to_process[reqId]
            
            # Initialize data dictionary for this ticker if not exists
            if ticker not in self.ticker_data:
                self.ticker_data[ticker] = []
            
            # Store the bar data
            self.ticker_data[ticker].append({
                'Date': bar.date,
                'Open': bar.open,
                'High': bar.high,
                'Low': bar.low,
                'Close': bar.close,
                'Volume': bar.volume,
                'Symbol': ticker
            })
            
            # Update last response time
            self.last_response_time = time.time()
    
    def historicalDataEnd(self, reqId, start, end):
        """Called when all historical data has been received."""
        if reqId < len(self.tickers_to_process):
            ticker = self.tickers_to_process[reqId]
            print(f"Received all historical data for {ticker}")
            
            # If we received data, save it to CSV
            if ticker in self.ticker_data and self.ticker_data[ticker]:
                self.save_ticker_data_to_csv(ticker)
            
            # Clean up processing status
            if ticker in self.processing_tickers:
                self.processing_tickers.remove(ticker)
            
            # Mark as processed and move to next ticker
            if ticker not in self.processed_tickers:
                self.processed_tickers.add(ticker)
                self.process_next_ticker()
                
            # Update last response time
            self.last_response_time = time.time()
    
    def save_ticker_data_to_csv(self, ticker):
        """Save the historical data for a ticker to a CSV file."""
        if ticker not in self.ticker_data or not self.ticker_data[ticker]:
            print(f"No data to save for {ticker}")
            return
        
        csv_filename = os.path.join(self.output_dir, f"{ticker}_historical.csv")
        
        # Create a formatted version of the data with all required columns
        formatted_data = []
        for bar in self.ticker_data[ticker]:
            # Format the date from 'YYYYMMDD' to 'YYYY-MM-DD 00:00:00-04:00'
            date_str = bar['Date']
            if len(date_str) == 8:  # If it's in the format YYYYMMDD
                year = date_str[0:4]
                month = date_str[4:6]
                day = date_str[6:8]
                date_str = f"{year}-{month}-{day} 00:00:00-04:00"
            
            # Convert volume to integer
            volume = int(bar['Volume']) if bar['Volume'] is not None else 0
            
            row = {
                'Date': date_str,
                'Open': bar['Open'],
                'High': bar['High'],
                'Low': bar['Low'],
                'Close': bar['Close'],
                'Volume': volume,  # Now an integer
                'Dividends': 0.0,
                'Stock Splits': 0.0,
                'Symbol': ticker,
                'Capital Gains': ''  # Empty string for Capital Gains
            }
            formatted_data.append(row)
        
        # Write to CSV with all required columns
        with open(csv_filename, 'w', newline='') as f:
            fieldnames = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                         'Dividends', 'Stock Splits', 'Symbol', 'Capital Gains']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(formatted_data)
        
        print(f"Saved historical data for {ticker} to {csv_filename}")
        
        # Clear data after saving to free memory
        self.ticker_data[ticker] = []
    
    def process_next_ticker(self):
        """Rate-limited processing of the next ticker in the list."""
        with self.request_lock:
            # Find tickers that are neither processed nor currently being processed
            pending_tickers = [t for t in self.tickers_to_process 
                              if t not in self.processed_tickers and t not in self.processing_tickers]
            
            # If none left, we're done for this chunk
            if not pending_tickers:
                if len(self.processing_tickers) == 0:  # Only complete if no tickers are still processing
                    print("All historical data requests in this chunk completed.")
                    self.processing_complete = True
                return
            
            # Rate limit check (100 requests per minute)
            current_time = time.time()
            if current_time - self.minute_start >= 60:
                # Reset every minute
                self.request_count = 0
                self.minute_start = current_time
            
            if self.request_count >= 100:
                # Wait if we hit 100/min
                wait_time = 60 - (current_time - self.minute_start)
                print(f"Rate limit (100/min) reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                self.request_count = 0
                self.minute_start = time.time()
            
            # Pop the next ticker, increment request count
            ticker = pending_tickers[0]
            self.processing_tickers.add(ticker)  # Mark as being processed
            self.request_count += 1
            self.request_historical_data(ticker)
    
    def request_historical_data(self, ticker):
        """Request historical data for a specific ticker."""
        if ticker in self.processed_tickers:
            print(f"Ticker {ticker} already processed, skipping")
            self.process_next_ticker()
            return
            
        contract = Contract()
        contract.symbol = ticker
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        # Use a unique reqId for each request to avoid duplicates
        # Instead of reusing the same index position for the reqId
        reqId = self.tickers_to_process.index(ticker)
        
        # Calculate date range for the last 2 years
        end_date = datetime.datetime.now().strftime("%Y%m%d 23:59:59")
        
        # 2 years ago from today
        two_years_ago = datetime.datetime.now() - datetime.timedelta(days=365*2)
        start_date = two_years_ago.strftime("%Y%m%d 00:00:00")
        
        print(f"Requesting historical data for {ticker} with reqId={reqId} ({self.request_count}/100 this minute)")
        print(f"Date range: {start_date} to {end_date}")
        
        # Clear any existing data for this ticker
        if ticker in self.ticker_data:
            self.ticker_data[ticker] = []
        
        try:
            # Request daily bars for the last 2 years
            self.reqHistoricalData(
                reqId,
                contract,
                end_date,  # End DateTime
                "2 Y",     # Duration String
                "1 day",   # Bar Size Setting
                "TRADES",  # What to Show
                1,         # Use Regular Trading Hours (1=Yes)
                2,         # Date Format (1=epoch seconds, 2=string format)
                False,     # Keep Updates
                []         # Chart Options
            )
        except Exception as e:
            print(f"Exception requesting data for {ticker}: {e}")
            self.processed_tickers.add(ticker)
            self.process_next_ticker()

def merge_csv_files(directory, output_file):
    """Merge all individual CSV files into one master CSV file."""
    all_data = []
    
    # Delete the existing output file if it exists
    if os.path.exists(output_file):
        try:
            os.remove(output_file)
            print(f"Deleted existing file: {output_file}")
        except Exception as e:
            print(f"Error deleting existing file {output_file}: {e}")
    
    for filename in os.listdir(directory):
        if filename.endswith("_historical.csv"):
            file_path = os.path.join(directory, filename)
            try:
                df = pd.read_csv(file_path)
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    if all_data:
        # Concatenate all dataframes
        master_df = pd.concat(all_data, ignore_index=True)
        
        # Save to master CSV file
        master_df.to_csv(output_file, index=False)
        print(f"Merged all historical data into {output_file}")
    else:
        print("No data files found to merge.")
        
def cleanup_ticker_files(directory):
    """Delete all individual ticker CSV files after merging."""
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith("_historical.csv") and filename != "all_tickers_historical.csv":
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                count += 1
            except Exception as e:
                print(f"Error removing {filename}: {e}")
    print(f"Deleted {count} individual ticker files")

def chunked_main():
    # Ensure paths are properly resolved
    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.join(script_dir, "stock_tickers", "amex_arca_bats_nasdaq_nyse_otc_stocks.csv")  # File with ticker symbols
    output_dir = os.path.join(script_dir, "price_data")  # Directory to store individual ticker CSVs
    master_output_file = os.path.join(output_dir, "all_tickers_historical.csv")  # Final merged output file

    # Read tickers from input
    with open(input_file, 'r') as f:
        lines = f.readlines()
        tickers = [line.strip() for line in lines if line.strip() and line.strip().upper() != "SYMBOL"]
    
    # Decide how many tickers per chunk (adjust to taste)
    CHUNK_SIZE = 200
    MAX_WAIT_TIME = 120  # Maximum wait time in seconds if no new responses
    
    # We'll process the tickers in slices
    total_tickers = len(tickers)
    print(f"Total tickers: {total_tickers}")
    
    # Track progress across disconnects
    current_chunk = 0
    
    # Try to resume from previous run
    try:
        with open("price_progress.txt", "r") as f:
            current_chunk = int(f.read().strip())
            print(f"Resuming from chunk {current_chunk}")
    except (FileNotFoundError, ValueError):
        current_chunk = 0
        print("Starting from the beginning")
    
    # Process all chunks
    for chunk_index in range(current_chunk, (total_tickers + CHUNK_SIZE - 1) // CHUNK_SIZE):
        chunk_start = chunk_index * CHUNK_SIZE
        chunk_end = min(chunk_start + CHUNK_SIZE, total_tickers)
        chunk = tickers[chunk_start:chunk_end]
        
        print("\n" + "="*50)
        print(f"Processing chunk {chunk_index}: {chunk_start} to {chunk_end-1} (size={len(chunk)})")
        print("="*50 + "\n")
        
        # Save current progress
        with open("price_progress.txt", "w") as f:
            f.write(str(chunk_index))
        
        # Keep track of which tickers we've successfully processed in this chunk
        processed_tickers_in_chunk = set()
        remaining_tickers = chunk.copy()
        
        # Continue until all tickers in the chunk are processed
        while remaining_tickers:
            print(f"Processing {len(remaining_tickers)} remaining tickers in chunk {chunk_index}")
            
            # Create and start the app with only the remaining tickers
            app = HistoricalDataApp(
                tickers_to_process=remaining_tickers,
                output_dir=output_dir
            )
            
            # Connect
            app.connect("127.0.0.1", 7497, clientId=0)
            
            # Start the networking thread
            api_thread = threading.Thread(target=app.run)
            api_thread.start()
            
            # Give IB a moment
            time.sleep(2)
            
            # Wait until the remaining tickers are fully processed or timeout
            wait_start = time.time()
            timeout_occurred = False
            
            while len(app.processed_tickers) < len(remaining_tickers):
                time.sleep(1)
                
                # Check for processing complete flag
                if app.processing_complete:
                    print("App signaled processing complete")
                    break
                    
                # Check for timeout - no response within MAX_WAIT_TIME
                current_time = time.time()
                if current_time - app.last_response_time > MAX_WAIT_TIME:
                    print(f"No response for {MAX_WAIT_TIME} seconds. Will disconnect and retry remaining tickers.")
                    timeout_occurred = True
                    break
                    
                # Also add a failsafe for entire attempt timeout
                if current_time - wait_start > MAX_WAIT_TIME * 2:
                    print(f"Processing timeout after {MAX_WAIT_TIME * 2} seconds. Will disconnect and retry remaining tickers.")
                    timeout_occurred = True
                    break
            
            # Update our tracking of which tickers we've processed
            processed_tickers_in_chunk.update(app.processed_tickers)
            
            # Calculate which tickers still need to be processed
            remaining_tickers = [t for t in remaining_tickers if t not in app.processed_tickers]
            
            # Disconnect the current API session
            app.disconnect()
            api_thread.join(timeout=5)
            
            # Log progress
            print(f"Progress: {len(processed_tickers_in_chunk)}/{len(chunk)} tickers processed in chunk {chunk_index}")
            
            if remaining_tickers:
                print(f"Still need to process: {len(remaining_tickers)} tickers in this chunk")
                # If a timeout didn't occur but we still have unprocessed tickers, that's strange
                # Could be errors or other issues, we'll wait a bit and retry
                if not timeout_occurred:
                    print("No timeout occurred but some tickers were not processed. Possibly due to errors.")
                
                # Output the first few remaining tickers for debugging
                sample = remaining_tickers[:min(5, len(remaining_tickers))]
                print(f"Sample of remaining tickers: {', '.join(sample)}")
                
                # Wait before retry to give IB time to "cool down"
                print("Waiting 10 seconds before retry...")
                time.sleep(10)
            else:
                print(f"All tickers in chunk {chunk_index} have been processed!")
        
        print(f"Chunk {chunk_index} ({chunk_start}-{chunk_end-1}) completed. Processed {len(processed_tickers_in_chunk)}/{len(chunk)} tickers.")
        
        # Wait a bit before the next chunk, to give IB time to "cool down"
        time.sleep(5)
        
        # Update progress after successful completion
        with open("price_progress.txt", "w") as f:
            f.write(str(chunk_index + 1))
    
    # After all chunks are processed, merge the CSV files
    merge_csv_files(output_dir, master_output_file)
    
    # Clean up individual files
    cleanup_ticker_files(output_dir)
    
    print("\nAll chunks completed. Exiting.")
    # Clear progress file when done
    with open("price_progress.txt", "w") as f:
        f.write("0")

if __name__ == "__main__":
    chunked_main()