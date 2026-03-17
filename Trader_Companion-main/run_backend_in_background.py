import os
import subprocess
import sys

def start_hidden_servers():
    # Get current directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    api_dir = os.path.join(base_dir, 'flask_microservice_stocks_filterer')

    # Start Django server without window
    subprocess.Popen(
        [sys.executable, 'manage.py', 'runserver'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,  # This prevents terminal window creation
        close_fds=True  # This ensures file descriptors aren't inherited
    )

    # Start API server without window
    api_script = os.path.join(api_dir, 'api_endpoints.py')
    subprocess.Popen(
        [sys.executable, api_script],
        cwd=api_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
        close_fds=True
    )

if __name__ == "__main__":
    start_hidden_servers()