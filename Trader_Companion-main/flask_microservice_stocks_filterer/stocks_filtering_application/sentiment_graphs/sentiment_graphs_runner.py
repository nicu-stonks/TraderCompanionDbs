import os
import subprocess
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Find the absolute path of the "flask_microservice_stocks_filterer" directory
script_dir = os.path.dirname(os.path.abspath(__file__))
while not script_dir.endswith("flask_microservice_stocks_filterer") and os.path.dirname(script_dir) != script_dir:
    script_dir = os.path.dirname(script_dir)

# Define the directory containing the scripts
scripts_dir = os.path.join(script_dir, "stocks_filtering_application", "sentiment_graphs")

# List of scripts to run
scripts = [
    os.path.join(scripts_dir, "32week_high.py"),
    os.path.join(scripts_dir, "32week_low.py"),
    os.path.join(scripts_dir, "above_200ma.py")
]

def run_script(script_path):
    """Runs a Python script and logs its output."""
    command = [sys.executable, '-u', script_path]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    for line in process.stdout:
        logging.info(f"[{os.path.basename(script_path)}] {line.strip()}")
    for line in process.stderr:
        logging.error(f"[{os.path.basename(script_path)} ERROR] {line.strip()}")

    process.wait()
    return process.returncode

def execute_scripts_in_parallel(scripts):
    """Runs the selected scripts in parallel."""
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(run_script, script): script for script in scripts}
        for future in as_completed(futures):
            future.result()  # Wait for completion

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    execute_scripts_in_parallel(scripts)
