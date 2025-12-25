#!/usr/bin/env python3
"""
Batch Crowding Analysis Runner

This script reads experiment configurations from a JSON file and runs crowding analysis
for multiple experiments in batch mode. Useful for processing ablation studies.

Usage:
    python batch_crowding_analysis.py --config experiments_config.json
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def load_config(config_file):
    """Load experiment configuration from JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)


def run_crowding_analysis(experiment_config):
    """Run crowding analysis for a single experiment configuration."""

    # Build command
    cmd = [
        sys.executable, "evaluate_by_crowding_and_knee.py",
        "--input_file", experiment_config["input_file"],
        "--output_dir", experiment_config["output_dir"],
        "--experiment_suffix", experiment_config["experiment_suffix"],
        "--few_shot", str(experiment_config["few_shot"]),
        "--provider", experiment_config.get("provider", "gemini"),
        "--max_capacity", str(experiment_config.get("max_capacity", 1000)),
        "--filter_threshold", str(experiment_config.get("filter_threshold", 0.6)),
        "--crowding_iterations", str(
            experiment_config.get("crowding_iterations", 3)),
        "--group_size", str(experiment_config.get("group_size", 30))
    ]

    print(f"Running: {' '.join(cmd)}")

    # Run the command
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True)
        print(f"✅ Success: {experiment_config['name']}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {experiment_config['name']}")
        print(f"Error: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch Crowding Analysis Runner")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to JSON configuration file")
    parser.add_argument("--parallel", action="store_true",
                        help="Run experiments in parallel (not implemented yet)")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    experiments = config["experiments"]

    print(f"Loaded {len(experiments)} experiment configurations")
    print("=" * 50)

    # Run experiments
    successful = 0
    failed = 0

    for i, experiment in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Processing: {experiment['name']}")

        # Check if input file exists
        if not os.path.exists(experiment["input_file"]):
            print(f"⚠️  Input file not found: {experiment['input_file']}")
            failed += 1
            continue

        # Create output directory if it doesn't exist
        os.makedirs(experiment["output_dir"], exist_ok=True)

        # Run the analysis
        if run_crowding_analysis(experiment):
            successful += 1
        else:
            failed += 1

    # Summary
    print("\n" + "=" * 50)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 50)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(experiments)}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
