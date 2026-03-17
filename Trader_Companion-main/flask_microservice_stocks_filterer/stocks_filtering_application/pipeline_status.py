import json
import os
import time
import re
from typing import Optional, Dict, Any

class PipelineStatus:
    """Handles saving and loading pipeline status information"""

    STATUS_FILE = "status.json"
    STATUS_DIR = "pipeline_status"

    def __init__(self, pid: int) -> None:
        # Find or initialize status file path
        status_path = self._find_status_file()
        if not status_path:
            # Create status directory if it doesn't exist
            status_path = os.path.join(".", self.STATUS_DIR, self.STATUS_FILE)
            os.makedirs(os.path.dirname(status_path), exist_ok=True)

        self.status_path = status_path

        # Initialize status file
        self._save_status({
            "start_time": time.time(),
            "current_step": "initializing",
            "steps_completed": [],
            "current_batch": None,
            "total_batches": None,
            "status": "running",
            "last_updated": time.time(),
            "process_pid": pid
        })

    @classmethod
    def start_pipeline(cls, pid: int) -> None:
        """
        Initialize the pipeline status as running with a new start time.
        """
        status_path = cls._find_status_file()
        if not status_path:
            # Create status directory if it doesn't exist
            status_path = os.path.join(".", cls.STATUS_DIR, cls.STATUS_FILE)
            os.makedirs(os.path.dirname(status_path), exist_ok=True)

        initial_status = {
            "start_time": time.time(),
            "current_step": "initializing",
            "steps_completed": [],
            "current_batch": None,
            "total_batches": None,
            "status": "running",
            "last_updated": time.time(),
            "process_pid": pid
        }

        with open(status_path, 'w') as f:
            json.dump(initial_status, f, indent=2)

    @classmethod
    def set_process_pid(cls, pid: int) -> None:
        """Set the process ID for the running pipeline"""
        status_path = cls._find_status_file()
        if not status_path:
            return None

        with open(status_path, 'r') as f:
            current_status = json.load(f)

        current_status["process_pid"] = pid
        current_status["last_updated"] = time.time()

        with open(status_path, 'w') as f:
            json.dump(current_status, f, indent=2)

    def update_step(self, step: str) -> None:
        """Update the current step of the pipeline"""
        current_status = self._load_status()
        current_status["current_step"] = step
        current_status["steps_completed"].append(step)
        current_status["last_updated"] = time.time()
        self._save_status(current_status)

    def handle_script_output(self, line: str, script_name: str) -> None:
        """
        Handle a line of output from a script and update status accordingly

        Args:
            line: The line of output from the script
            script_name: Name of the script producing the output
        """
        if not line:
            return

        # Remove ANSI color codes and strip whitespace
        line = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', line).strip()

        # Handle batch progress from price_fundamental_script
        if "price_1y_fundamental_2y.py" in script_name:
            self._handle_batch_progress(line)

        # Handle other script-specific output patterns here
        if "Error:" in line:
            self._handle_error(line, script_name)

    def _handle_batch_progress(self, line: str) -> None:
        """Parse and handle batch progress information"""
        batch_match = re.search(r"Completed batch (\d+) of (\d+)", line)
        if batch_match:
            current_batch = int(batch_match.group(1))
            total_batches = int(batch_match.group(2))
            current_status = self._load_status()
            current_status["current_batch"] = current_batch
            current_status["total_batches"] = total_batches
            current_status["last_updated"] = time.time()
            self._save_status(current_status)

    def _handle_error(self, line: str, script_name: str) -> None:
        """Handle error messages in the output"""
        current_status = self._load_status()
        if "errors" not in current_status:
            current_status["errors"] = []
        current_status["errors"].append({
            "script": script_name,
            "error": line,
            "timestamp": time.time()
        })
        self._save_status(current_status)

    @classmethod
    def complete_pipeline(cls) -> None:
        """Mark the pipeline as completed"""
        status_path = cls._find_status_file()
        if not status_path:
            return None

        with open(status_path, 'r') as f:
            current_status = json.load(f)

        current_status["status"] = "completed"
        current_status["end_time"] = time.time()
        current_status["last_updated"] = time.time()
        current_status["process_pid"] = None  # Clear PID on completion

        with open(status_path, 'w') as f:
            json.dump(current_status, f, indent=2)

    def fail_pipeline(self, error: str) -> None:
        """Mark the pipeline as failed"""
        current_status = self._load_status()
        current_status["status"] = "failed"
        current_status["error"] = error
        current_status["end_time"] = time.time()
        current_status["last_updated"] = time.time()
        self._save_status(current_status)

    def _save_status(self, status: Dict[str, Any]) -> None:
        """Save the current status to file"""
        with open(self.status_path, 'w') as f:
            json.dump(status, f, indent=2)

    def _load_status(self) -> Dict[str, Any]:
        """Load the current status from file"""
        if not os.path.exists(self.status_path):
            return {
                "start_time": time.time(),
                "current_step": "initializing",
                "steps_completed": [],
                "current_batch": None,
                "total_batches": None,
                "status": "running",
                "last_updated": time.time()
            }

        with open(self.status_path, 'r') as f:
            return json.load(f)

    @classmethod
    def _find_status_file(cls) -> Optional[str]:
        """Find status.json by walking down through subdirectories"""
        for root, _, files in os.walk('.'):
            if cls.STATUS_FILE in files and cls.STATUS_DIR in root.split(os.sep):
                return os.path.join(root, cls.STATUS_FILE)
        return None

    @classmethod
    def get_status(cls) -> Optional[Dict[str, Any]]:
        """Get the current pipeline status"""
        status_path = cls._find_status_file()
        if not status_path:
            return None

        with open(status_path, 'r') as f:
            return json.load(f)