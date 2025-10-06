#!/usr/bin/env python3
"""
Main script to run the Coffee Forecasting pipeline.
This script runs database operations and model training/forecasting as subprocesses.
"""

import subprocess
import sys
import os
from datetime import datetime


def run_subprocess(script_name, args=None):
    """Run a Python script as a subprocess and handle errors."""
    if args is None:
        args = []

    cmd = [sys.executable, script_name] + args
    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Successfully completed {script_name}")
        if result.stdout:
            print(f"Output:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print(f"Script {script_name} not found")
        return False


def main():
    print("Coffee Forecasting Pipeline - Main Execution")
    print("=" * 50)
    print(f"Started at: {datetime.now()}")
    print()

    # Run database operations (loading data)
    print("Step 1: Running database operations...")
    db_success = run_subprocess("database.py")

    if not db_success:
        print("Database operations failed. Exiting.")
        sys.exit(1)

    print()

    # Run model training and forecasting
    print("Step 2: Running model training and forecasting...")
    model_success = run_subprocess("model.py")

    if not model_success:
        print("Model operations failed. Exiting.")
        sys.exit(1)

    print()
    print("=" * 50)
    print("All pipeline steps completed successfully!")
    print(f"Finished at: {datetime.now()}")


if __name__ == "__main__":
    main()