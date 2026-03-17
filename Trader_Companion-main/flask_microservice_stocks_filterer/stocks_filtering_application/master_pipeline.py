import os
import subprocess
import sys
import logging
import argparse
import time
import platform
import ctypes
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pipeline_status import PipelineStatus
import psutil

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define pipelines to run
PIPELINE_PATHS = [
    os.path.join(script_dir, "minervini_1mo", "stock_screening_pipeline.py"),
    os.path.join(script_dir, "minervini_4mo", "stock_screening_pipeline.py"),
    os.path.join(script_dir, "minervini_1mo_unbanned", "stock_screening_pipeline.py"),
    os.path.join(script_dir, "ipos", "stock_screening_pipeline.py"),
]

# Define fetch data script
fetch_data_script = os.path.join(script_dir, "extract_price_multiple.py")
# fetch_fundamentals_script = os.path.join(script_dir, "extract_fundamental_multiple.py")


# Set up logging
def setup_logging():
    logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, "last_run.log")

    if os.path.exists(log_file):
        os.remove(log_file)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file


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


def parse_args():
    parser = argparse.ArgumentParser(description="Master stock screening pipeline")
    parser.add_argument("price_increase", type=float, help="Minimum price increase percentage")
    parser.add_argument("--top-n", type=int, default=100, help="Number of top stocks to select")
    parser.add_argument("--fetch-data", action="store_true", help="Fetch stock data before running pipelines")
    parser.add_argument("--sleep-after", action="store_true", help="Put computer to sleep after completion")
    parser.add_argument("--skip-sentiment", action="store_true", help="Skip sentiment analysis steps")
    return parser.parse_args()


def run_sentiment_scripts():
    sentiment_screens_script = os.path.join(script_dir, "market_sentiment_screens", "sentiment_screens_handler.py")
    sentiment_graphs_script = os.path.join(script_dir, "sentiment_graphs", "sentiment_graphs_runner.py")

    logging.info("Running sentiment analysis scripts...")
    run_script(sentiment_screens_script)
    run_script(sentiment_graphs_script)
    logging.info("Sentiment analysis scripts completed.")


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


def run_all_pipelines_parallel(price_increase, top_n, status_tracker):
    """Run all pipelines in parallel using threading."""
    logging.info("Running all pipelines in parallel...")

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(run_script, pipeline, [str(price_increase), '--top-n', str(top_n)],
                            status_tracker): pipeline
            for pipeline in PIPELINE_PATHS
        }

        for future in as_completed(futures):
            pipeline_name = os.path.basename(futures[future])
            try:
                result = future.result()
                if result == 0:
                    logging.info(f"{pipeline_name} completed successfully.")
                else:
                    logging.error(f"{pipeline_name} failed with exit code {result}.")
            except Exception as e:
                logging.error(f"Error in {pipeline_name}: {e}")

def kill_ib_processes():
    """Kill all Python processes related to IB API."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Look for python processes with IB-related arguments
            if proc.info['name'] in ['python', 'python.exe']:
                cmdline = ' '.join(proc.info['cmdline'] if proc.info['cmdline'] else [])
                if any(term in cmdline for term in ['ibapi', 'extract_price', 'extract_fundamental']):
                    print(f"Killing process {proc.pid}: {cmdline}")
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

def main():
    kill_ib_processes()
    print("IB process cleanup complete")
    
    log_file = setup_logging()
    logging.info(f"Starting pipeline run. Logs will be saved to: {log_file}")

    args = parse_args()
    logging.info(f"Pipeline arguments: {args}")

    status_tracker = PipelineStatus(os.getpid())

    try:
        # Fetch stock data if requested
        if args.fetch_data:
            logging.info("Fetching stock price data from the API...")
            status_tracker.update_step("Fetching price data")
            run_script(fetch_data_script, status_tracker=status_tracker)
            logging.info("Price data fetch completed.")
        else:
            logging.info("Skipping data fetch, using existing data...")

        # Run pipelines in parallel
        status_tracker.update_step("Running pipelines")
        run_all_pipelines_parallel(args.price_increase, args.top_n, status_tracker)
        
        # if not args.skip_sentiment:
        #     logging.info("Running sentiment analysis...")
        #     status_tracker.update_step("Running sentiment analysis")
        #     run_sentiment_scripts()
        # else:
        #     logging.info("Skipping sentiment analysis as per user request.")


        logging.info("All pipelines completed.")
        status_tracker.complete_pipeline()
    except Exception as e:
        logging.error(f"Pipeline failed with error: {str(e)}")
        status_tracker.fail_pipeline(str(e))
        raise e

    # Put computer to sleep if requested
    if args.sleep_after:
        logging.info("Putting computer to sleep in 5 seconds...")
        time.sleep(5)
        if put_computer_to_sleep():
            logging.info("Sleep command sent successfully")
        else:
            logging.error("Failed to put computer to sleep")


if __name__ == "__main__":
    main()
