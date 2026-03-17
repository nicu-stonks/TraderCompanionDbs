import os
import subprocess
import sys
import psutil
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Get the absolute path of the "flask_microservice_stocks_filterer" directory
script_dir = os.path.dirname(os.path.abspath(__file__))
while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
    script_dir = os.path.dirname(script_dir)

# Define paths based on the correct absolute directory
logs_dir = os.path.join(script_dir, "stocks_filtering_application", "ipos", "logs")
os.makedirs(logs_dir, exist_ok=True)


def setup_logging():
    log_file = os.path.join(logs_dir, "last_run.log")

    # Function to find and terminate processes using the log file
    def kill_processes_using_file(file_path):
        for proc in psutil.process_iter(['pid', 'name', 'open_files']):
            try:
                if proc.info['open_files']:
                    for file in proc.info['open_files']:
                        if file.path == file_path:
                            print(f"Closing process {proc.info['name']} (PID: {proc.info['pid']}) using {file_path}")
                            proc.terminate()  # Try a graceful shutdown
                            proc.wait(timeout=5)  # Wait for process to exit
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

    # If the log file exists, find and kill any process holding it
    if os.path.exists(log_file):
        kill_processes_using_file(log_file)

        try:
            os.remove(log_file)
            print(f"Successfully deleted {log_file}")
        except Exception as e:
            print(f"Error deleting {log_file}: {e}")

    # Set up logging after ensuring the file is deleted
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file


def parse_args():
    parser = argparse.ArgumentParser(description='Stock screening pipeline')
    parser.add_argument('price_increase', type=float, help='Minimum price increase percentage')
    parser.add_argument('--top-n', type=int, default=100, help='Number of top stocks to select')
    return parser.parse_args()


def run_script(script_path, args=None):
    command = [sys.executable, '-u', script_path]
    if args:
        command.extend(args)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    for line in process.stdout:
        logging.info(f"[{os.path.basename(script_path)}] {line.strip()}")
    for line in process.stderr:
        logging.error(f"[{os.path.basename(script_path)} ERROR] {line.strip()}")

    process.wait()
    return process.returncode


def run_scripts_in_parallel(scripts, price_increase=None):
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(run_script, script,
                            [str(price_increase)] if 'minimum_price_increase.py' in script else None)
            for script in scripts
        ]
        for future in as_completed(futures):
            future.result()


def get_dirs_to_cleanup():
    return [
        os.path.join(script_dir, "stocks_filtering_application", "ipos", "ranking_screens", "results"),
        os.path.join(script_dir, "stocks_filtering_application", "ipos", "obligatory_screens", "results")
    ]


def find_csv_files(directories):
    csv_files = []
    for directory in directories:
        if os.path.exists(directory):
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
    return csv_files


def delete_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logging.error(f"Error deleting {file_path}: {e}")


def main():
    setup_logging()
    args = parse_args()
    logging.info("Starting stock screening pipeline")

    obligatory_screens = [
        os.path.join(script_dir, "stocks_filtering_application", "ipos", "obligatory_screens", f"{name}.py")
        for name in
        ["close_to_52week_high", "trading_for_at_most_3mo", "minimum_5_dollar"]
    ]
    ranking_screens = [
        os.path.join(script_dir, "stocks_filtering_application", "ipos", "ranking_screens", f"{name}.py")
        for name in ["top_price_increases_1y", "price_spikes", "volume_acceleration", "top_price_tightness_1w", "top_rsi", "mvp", "days_traded", "top_rsi_6m", "top_rsi_12m"]
    ]

    logging.info("Finding and deleting old CSV files...")
    dirs_to_cleanup = get_dirs_to_cleanup()
    csv_files = find_csv_files(dirs_to_cleanup)
    logging.info(f"Found {len(csv_files)} CSV files to delete")
    for file_path in csv_files:
        logging.info(f"Deleting: {os.path.basename(file_path)}")
        delete_file(file_path)

    run_scripts_in_parallel(obligatory_screens, args.price_increase)

    run_script(os.path.join(script_dir, "stocks_filtering_application", "ipos", "obligatory_screens", "obligatory_screen_passer.py"))

    run_script(os.path.join(script_dir, "stocks_filtering_application", "ipos", "banned_stocks", "banned_filter.py"))

    run_script(os.path.join(script_dir, "stocks_filtering_application", "ipos", "ranking_screens", "passed_stocks_input_data",
                            "obligatory_screen_data_filter.py"))

    run_scripts_in_parallel(ranking_screens)

    run_script(os.path.join(script_dir, "stocks_filtering_application", "ipos", "top_n_stocks_by_price_increase.py"), [str(args.top_n)])

    logging.info("Stock screening pipeline completed successfully.")


if __name__ == "__main__":
    main()
