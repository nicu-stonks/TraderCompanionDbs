import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import sys
import ctypes
import threading
import argparse
import uuid
import platform
from pipeline_status import PipelineStatus
from datetime import datetime
import logging

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create logs directory if it doesn't exist
logs_dir = os.path.join(script_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Define file paths
price_fundamental_script = os.path.join(script_dir, "price_1y_fundamental_2y.py")
obligatory_passed_stocks = os.path.join(script_dir, "obligatory_screens", "obligatory_screen_passer.py")
obligatory_data_filter = os.path.join(script_dir, "ranking_screens", "passed_stocks_input_data",
                                      "obligatory_screen_data_filter.py")
banned_filter = os.path.join(script_dir, "banned_stocks", "banned_filter.py")
top_n_stocks_price_increase = os.path.join(script_dir, "top_n_stocks_by_price_increase.py")
top_n_stocks_nr_screeners = os.path.join(script_dir, "top_n_stocks_by_nr_screeners.py")
history_handler = os.path.join(script_dir, "market_sentiment_screens", "history_handler.py")

# Default screens if none provided
DEFAULT_OBLIGATORY_SCREENS = [
    "above_52week_low",
    "trending_up",
    "close_to_52week_high",
    "minimum_volume_100k",
    "minimum_price_increase"
]

DEFAULT_RANKING_SCREENS = [
    "annual_EPS_acceleration",
    "annual_margin_acceleration",
    "annual_sales_acceleration",
    "quarterly_EPS_acceleration",
    "quarterly_eps_breakout",
    "quarterly_margin_acceleration",
    "quarterly_sales_acceleration",
    "rs_over_70",
    "rsi_trending_up",
    "volume_acceleration",
    "price_spikes",
    "top_price_increases_1y"
]

# Market sentiment screens (always run all of these)
MARKET_SENTIMENT_SCREENS = [
    "52week_high_2w",
    "52week_low_2w",
    "rs_over_70",
    "rsi_under_30",
    "rsi_trending_up",
    "rsi_trending_down",
    "52week_high_1d",
    "52week_low_1d"
]

def put_computer_to_sleep():
    """Put the computer to sleep based on the operating system."""
    system = platform.system().lower()

    try:
        if system == 'windows':
            ctypes.windll.PowrProf.SetSuspendState(0, 1, 0)
        elif system == 'darwin':  # macOS
            os.system('pmset sleepnow')
        elif system == 'linux':
            os.system('systemctl suspend')
        else:
            logging.error(f"Sleep not supported on {system}")
            return False
        return True
    except Exception as e:
        logging.error(f"Failed to put computer to sleep: {e}")
        return False

# Set up logging
def setup_logging():
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Define log file path
    log_file = os.path.join(logs_dir, "last_run.log")

    # Remove old log file if it exists
    if os.path.exists(log_file):
        os.remove(log_file)

    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return log_file

def get_dirs_to_cleanup(run_obligatory, run_sentiment):
    """Get directories to clean up based on which screens are being run."""
    dirs = []

    # Always include ranking screens results
    dirs.append(os.path.join(script_dir, "ranking_screens", "results"))

    # Add obligatory screens results if running obligatory screens
    if run_obligatory:
        dirs.append(os.path.join(script_dir, "obligatory_screens", "results"))

    # Add sentiment screens results if running sentiment screens
    if run_sentiment:
        dirs.append(os.path.join(script_dir, "market_sentiment_screens", "results"))

    return dirs


def find_csv_files(directories):
    """Find all CSV files in the specified directories."""
    csv_files = []
    for directory in directories:
        if os.path.exists(directory):
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
    return csv_files


def parse_args():
    parser = argparse.ArgumentParser(description='Stock screening pipeline')
    parser.add_argument('price_increase', type=float,
                        help='Minimum price increase percentage')
    parser.add_argument('--ranking-method', type=str,
                        choices=['price', 'screeners'],
                        default='price',
                        help='Ranking method: price (by price increase) or screeners (by number of screeners)')
    parser.add_argument('--fetch-data', action='store_true',
                        help='Run price fundamental script to fetch new data')
    parser.add_argument('--top-n', type=int, default=100,
                        help='Number of top stocks to select in the ranking')
    parser.add_argument('--sleep-after', action='store_true',
                        help='Put the computer to sleep after completion')

    # Add arguments for screens
    parser.add_argument('--obligatory-screens', nargs='+',
                        default=DEFAULT_OBLIGATORY_SCREENS,
                        help='List of obligatory screens to run (without .py extension)')
    parser.add_argument('--ranking-screens', nargs='+',
                        default=DEFAULT_RANKING_SCREENS,
                        help='List of ranking screens to run (without .py extension)')

    # Add flags to control which screen types to run
    parser.add_argument('--skip-obligatory', action='store_true',
                        help='Skip running obligatory screens')
    parser.add_argument('--skip-sentiment', action='store_true',
                        help='Skip running market sentiment screens')

    return parser.parse_args()


def get_full_paths(screen_names, screen_type):
    """Convert screen names to full paths based on screen type."""
    base_path = {
        'obligatory': 'obligatory_screens',
        'ranking': 'ranking_screens',
        'sentiment': 'market_sentiment_screens'
    }[screen_type]

    return [os.path.join(script_dir, base_path, f"{name}.py") for name in screen_names]


def delete_file(file_path):
    """Delete a file if it exists."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logging.error(f"Error deleting {file_path}: {e}")

def run_script(script_path, args=None, status_tracker=None):
    """Run a Python script with optional arguments and capture its output in real-time."""
    script_name = os.path.basename(script_path)

    command = [sys.executable, '-u', script_path]
    if args:
        command.extend(args)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        universal_newlines=True
    )

    def handle_output(pipe, prefix, is_error=False):
        """Handle output from a pipe with a prefix."""
        try:
            for line in pipe:
                log_level = logging.ERROR if is_error else logging.INFO
                logging.log(log_level, f"{prefix}: {line.strip()}")
                if status_tracker:
                    status_tracker.handle_script_output(line, script_name)
        except Exception as e:
            logging.error(f"Error reading output: {e}")

    stdout_thread = threading.Thread(
        target=handle_output,
        args=(process.stdout, f"[{script_name}]"),
        daemon=True
    )
    stderr_thread = threading.Thread(
        target=handle_output,
        args=(process.stderr, f"[{script_name} ERROR]", True),
        daemon=True
    )

    stdout_thread.start()
    stderr_thread.start()

    process.wait()

    process.stdout.close()
    process.stderr.close()

    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)

    return process.returncode

def run_scripts_in_parallel(scripts, description, price_increase=None):
    """Run multiple scripts in parallel and show their output."""
    logging.info(f"\nRunning {description}...")
    with ThreadPoolExecutor() as executor:
        futures = []
        for script in scripts:
            # Check if this is the minimum_price_increase script and we have a price_increase value
            if os.path.basename(script) == "minimum_price_increase.py" and price_increase is not None:
                args = [str(price_increase)]
                futures.append(executor.submit(run_script, script, args))
            else:
                futures.append(executor.submit(run_script, script))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error running script: {e}")


def main():
    status_tracker = None
    try:
        # Set up logging at the start
        log_file = setup_logging()
        logging.info(f"Starting pipeline run. Logs will be saved to: {log_file}")

        args = parse_args()
        logging.info(f"Pipeline arguments: {args}")

        # Start the pipeline with this process PID
        status_tracker = PipelineStatus(os.getpid())

        # Add top_price_increases_1y to ranking screens if ranking method is price
        ranking_screen_list = args.ranking_screens
        if args.ranking_method == 'price' and 'top_price_increases_1y' not in ranking_screen_list:
            ranking_screen_list.append('top_price_increases_1y')
            logging.info("Added top_price_increases_1y to ranking screens due to price ranking method")

        # Convert screen names to full paths
        obligatory_screens = get_full_paths(args.obligatory_screens, 'obligatory') if not args.skip_obligatory else []
        ranking_screens = get_full_paths(args.ranking_screens, 'ranking')
        market_sentiment_screens = get_full_paths(MARKET_SENTIMENT_SCREENS, 'sentiment') if not args.skip_sentiment else []

        # Clean up old CSV files based on which screens we're running
        status_tracker.update_step("cleaning_old_files")
        logging.info("Finding and deleting old CSV files...")
        dirs_to_cleanup = get_dirs_to_cleanup(not args.skip_obligatory, not args.skip_sentiment)
        csv_files = find_csv_files(dirs_to_cleanup)
        logging.info(f"Found {len(csv_files)} CSV files to delete")
        for file_path in csv_files:
            logging.info(f"Deleting: {os.path.basename(file_path)}")
            delete_file(file_path)

        # Fetch stock data if requested
        if args.fetch_data:
            status_tracker.update_step("fetching_stock_data")
            logging.info("\nFetching stock data from the API...")
            run_script(price_fundamental_script, status_tracker=status_tracker)
        else:
            logging.info("\nSkipping data fetch, using existing data...")

        # Run obligatory screens if not skipped
        if not args.skip_obligatory:
            status_tracker.update_step("running_obligatory_screens")
            logging.info("\nRunning obligatory screen scripts...")
            run_scripts_in_parallel(obligatory_screens, "obligatory screens", args.price_increase)

            status_tracker.update_step("checking_obligatory_screens")
            logging.info("\nChecking which stocks passed the obligatory screens...")
            run_script(obligatory_passed_stocks)

            status_tracker.update_step("checking_banned_stocks")
            logging.info("\nChecking which files are banned, creating unbanned stocks list...")
            run_script(banned_filter)

            status_tracker.update_step("filtering_passed_stocks")
            logging.info("\nRunning the filter for passed and unbanned stocks...")
            run_script(obligatory_data_filter)

        status_tracker.update_step("running_ranking_screens")
        run_scripts_in_parallel(ranking_screens, "ranking screen scripts")

        status_tracker.update_step("finding_top_stocks")
        logging.info(f"\nSearching for the top {args.top_n} stocks...")
        if args.ranking_method == 'price':
            run_script(top_n_stocks_price_increase, [str(args.top_n)])
        else:  # ranking_method == 'screeners'
            run_script(top_n_stocks_nr_screeners, [str(args.top_n)])

        # Run sentiment screens if not skipped
        if not args.skip_sentiment:
            status_tracker.update_step("running_sentiment_screens")
            run_scripts_in_parallel(market_sentiment_screens, "market sentiment screen scripts")

            status_tracker.update_step("running_history_handler")
            logging.info("\nRunning history handler script...")
            run_script(history_handler)

        logging.info("\nAll scripts completed.")
        status_tracker.complete_pipeline()
    except Exception as e:
        logging.error(f"Pipeline failed with error: {str(e)}")
        if status_tracker:
            status_tracker.fail_pipeline(str(e))
        raise e
    finally:
        # Ensure pipeline status is marked as complete even if an error occurs
        if status_tracker:
            status_tracker.complete_pipeline()

    # Put computer to sleep after all cleanup is done
    logging.info("Checking if computer should be put to sleep...")
    logging.info("Args: " + str(args))
    if args.sleep_after:
        logging.info("Putting computer to sleep in 5 seconds...")
        time.sleep(5)  # Give time for logs to be written
        if put_computer_to_sleep():
            logging.info("Sleep command sent successfully")
        else:
            logging.error("Failed to put computer to sleep")


if __name__ == "__main__":
    main()