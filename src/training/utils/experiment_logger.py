"""
Experiment logging utilities for tracking configuration and runs
"""
import os
import sys
import csv
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def flatten_config(config: Any, parent_key: str = '') -> Dict[str, Any]:
    """
    Flatten nested configuration object into a dictionary with hierarchical keys.

    Args:
        config: Configuration object (nested attributes)
        parent_key: Parent key for nested attributes

    Returns:
        Flattened dictionary with keys like 'afno2d.hidden_size'
    """
    items = {}

    # Handle config objects with __dict__
    try:
        if isinstance(config, dict):
            config_dict = config
        elif hasattr(config, '__dict__'):
            config_dict = config.__dict__
        else:
            return {parent_key: config}
    except Exception:
        # If we can't introspect the config, return it as-is
        return {parent_key: str(config)}

    for key, value in config_dict.items():
        # Skip private attributes
        if key.startswith('_'):
            continue

        new_key = f"{parent_key}.{key}" if parent_key else key

        # Recursively flatten nested objects
        try:
            if isinstance(value, (str, int, float, bool, type(None))):
                items[new_key] = value
            elif isinstance(value, list):
                # Convert lists to string representation for CSV
                items[new_key] = str(value)
            elif isinstance(value, dict):
                items.update(flatten_config(value, new_key))
            elif hasattr(value, '__dict__'):
                items.update(flatten_config(value, new_key))
            else:
                # Fallback: convert to string
                items[new_key] = str(value)
        except Exception:
            # If we can't process this value, skip it
            continue

    return items


def create_multicolumn_headers(flat_config: Dict[str, Any]) -> tuple:
    """
    Create multi-column headers from flattened config.
    Preserves the order from the configuration file.

    Returns:
        Tuple of (top_headers, sub_headers) where:
        - top_headers: List of top-level categories repeated for each column
        - sub_headers: List of actual column names (e.g., 'name', 'hidden_size')
    """
    # Group keys by their top-level category, preserving order
    from collections import OrderedDict
    categories = OrderedDict()

    for key in flat_config.keys():
        if '.' in key:
            category, subkey = key.split('.', 1)
        else:
            category = 'general'
            subkey = key

        if category not in categories:
            categories[category] = []
        categories[category].append((key, subkey))

    # Build headers with special ordering: 'name' first, then rest in original order
    top_headers = []
    sub_headers = []
    column_keys = []

    # First, add 'name' if it exists
    if 'name' in flat_config:
        top_headers.append('general')
        sub_headers.append('name')
        column_keys.append('name')

    # Then add all other fields in their original order
    for category, items in categories.items():
        for full_key, subkey in items:
            # Skip 'name' as we already added it first
            if full_key == 'name':
                continue

            # Add category header for each column (no blanks)
            top_headers.append(category)
            sub_headers.append(subkey)
            column_keys.append(full_key)

    return top_headers, sub_headers, column_keys


def log_experiment(config: Any, experiments_dir: str = "results/experiments", csv_filename: str = "runtable.csv"):
    """
    Log experiment configuration to runtable CSV file with multi-column headers.

    Args:
        config: Configuration object
        experiments_dir: Directory to store the runtable CSV file
        csv_filename: Name of the CSV file (default: "runtable.csv")
                     For model-specific tables, use "runtable_FNO.csv", "runtable_TFNO.csv", etc.
    """
    # Create experiments directory if it doesn't exist
    os.makedirs(experiments_dir, exist_ok=True)

    csv_path = os.path.join(experiments_dir, csv_filename)
    file_exists = os.path.exists(csv_path)

    # Flatten configuration
    flat_config = flatten_config(config)

    # Create headers
    top_headers, sub_headers, column_keys = create_multicolumn_headers(flat_config)

    # Open file in append mode
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write headers if file is new
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writerow(top_headers)
            writer.writerow(sub_headers)

        # Write configuration values
        row_values = [flat_config.get(key, '') for key in column_keys]
        writer.writerow(row_values)

    print(f"Experiment logged to: {csv_path}")
    return csv_path


def get_model_filename(config: Any) -> str:
    """
    Generate model checkpoint filename based on experiment name.

    Args:
        config: Configuration object with 'name' attribute

    Returns:
        Model filename: {name}.pth
    """
    experiment_name = getattr(config, 'name', 'model')
    return f"{experiment_name}.pth"


class TeeLogger:
    """
    A logger that writes to both console and file simultaneously.
    Filters out TQDM progress bar updates to keep log files clean.
    """
    def __init__(self, log_path: str):
        """
        Initialize TeeLogger.

        Args:
            log_path: Path to the log file
        """
        self.log_path = log_path
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'a', buffering=1)  # Line buffered
        self.last_line_was_progress = False
        self.progress_buffer = ""

        # Write header
        self.log_file.write(f"\n{'='*80}\n")
        self.log_file.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write(f"{'='*80}\n\n")
        self.log_file.flush()

    def write(self, message):
        """Write message to both terminal and log file, filtering TQDM updates."""
        # Always write to terminal
        self.terminal.write(message)

        # Filter TQDM progress bars from log file
        # TQDM uses carriage return (\r) to overwrite the same line
        if '\r' in message and '\n' not in message:
            # This is a TQDM update (carriage return without newline)
            # Store it but don't write to file yet
            self.progress_buffer = message
            self.last_line_was_progress = True
        elif self.last_line_was_progress and '\n' in message:
            # This is the final update from TQDM (contains newline)
            # Write only this final state to the log file
            self.log_file.write(self.progress_buffer.replace('\r', '') + message)
            self.progress_buffer = ""
            self.last_line_was_progress = False
        elif '\r' in message and '\n' in message:
            # Mixed carriage return and newline - write only after last \r
            parts = message.split('\r')
            final_part = parts[-1]
            self.log_file.write(final_part)
            self.last_line_was_progress = False
        else:
            # Normal message without carriage return
            if self.last_line_was_progress:
                # Clear the progress buffer
                self.progress_buffer = ""
                self.last_line_was_progress = False
            self.log_file.write(message)

    def flush(self):
        """Flush both terminal and log file."""
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        """Close the log file."""
        self.log_file.write(f"\n{'='*80}\n")
        self.log_file.write(f"Training ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write(f"{'='*80}\n")
        self.log_file.close()


def setup_logging(config: Any, experiments_dir: str = "results/experiments") -> TeeLogger:
    """
    Set up logging to capture all console output to a log file.

    Args:
        config: Configuration object with 'name' attribute
        experiments_dir: Directory to store logs

    Returns:
        TeeLogger instance
    """
    # Create logs directory
    logs_dir = os.path.join(experiments_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Get experiment name
    experiment_name = getattr(config, 'name', 'experiment')
    log_path = os.path.join(logs_dir, f"{experiment_name}.log")

    # Create and return TeeLogger
    tee_logger = TeeLogger(log_path)
    sys.stdout = tee_logger
    sys.stderr = tee_logger

    print(f"Logging to: {log_path}")
    return tee_logger


def cleanup_logging(tee_logger: TeeLogger):
    """
    Restore original stdout/stderr and close log file.

    Args:
        tee_logger: TeeLogger instance to clean up
    """
    sys.stdout = tee_logger.terminal
    sys.stderr = tee_logger.terminal
    tee_logger.close()
