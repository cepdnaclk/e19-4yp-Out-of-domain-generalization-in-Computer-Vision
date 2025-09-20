#!/usr/bin/env python3
"""
Learning Curve Plotter

This script extracts iteration scores from log files and plots learning curves
for multiple experiments on the same graph.

Usage:
    python draw_learning_plots.py --input_files file1.log file2.log file3.log --labels "Experiment 1" "Experiment 2" "Experiment 3"
    python draw_learning_plots.py --input_files logs/*.log --labels "Setup A" "Setup B" "Setup C" --output learning_curves.png
"""

import argparse
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import os

# Set seaborn style for nicer plots
sns.set_style("whitegrid")
sns.set_palette("husl")


def extract_iteration_scores(file_path: str) -> Tuple[List[int], List[float]]:
    """
    Extract iteration numbers and scores from a log file.

    Args:
        file_path: Path to the log file

    Returns:
        Tuple of (iterations, scores) lists
    """
    iterations = []
    scores = []

    # Pattern to match lines like "Iteration 20: mean accuracy of top 10: 0.3131" or "0.3131."
    pattern = r'Iteration\s+(\d+):\s+mean\s+\w+\s+of\s+top\s+\d+:\s+([\d.]+)\.?'

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                match = re.search(pattern, line)
                if match:
                    iteration = int(match.group(1))
                    # Clean the score string by removing any trailing periods
                    score_str = match.group(2).rstrip('.')
                    try:
                        score = float(score_str)
                        iterations.append(iteration)
                        scores.append(score)
                    except ValueError as ve:
                        print(
                            f"Warning: Could not parse score '{match.group(2)}' on line {line_num} in {file_path}: {ve}")
                        continue
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return [], []

    print(f"Extracted {len(iterations)} data points from {file_path}")
    if iterations:
        print(f"  Iteration range: {min(iterations)} - {max(iterations)}")
        print(f"  Score range: {min(scores):.4f} - {max(scores):.4f}")

    return iterations, scores


def plot_learning_curves(data: Dict[str, Tuple[List[int], List[float]]],
                         output_file: str = None,
                         title: str = "Learning Curves",
                         figsize: Tuple[int, int] = (6, 6)):
    """
    Plot learning curves for multiple experiments using seaborn styling.

    Args:
        data: Dictionary mapping labels to (iterations, scores) tuples
        output_file: Optional output file path
        title: Plot title
        figsize: Figure size tuple (default: square 6x6 for single column)
    """
    # Create figure with square aspect ratio suitable for single column
    fig, ax = plt.subplots(figsize=figsize)

    # Use seaborn color palette
    colors = sns.color_palette("husl", len(data))
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

    # Plot each experiment
    for idx, (label, (iterations, scores)) in enumerate(data.items()):
        if not iterations:
            print(f"Warning: No data found for {label}")
            continue

        color = colors[idx % len(colors)]
        line_style = line_styles[idx % len(line_styles)]

        # Use seaborn lineplot style with matplotlib
        ax.plot(iterations, scores,
                label=label,
                color=color,
                linestyle=line_style,
                linewidth=2.5,
                marker='o' if len(iterations) <= 50 else None,
                markersize=5 if len(iterations) <= 50 else 0,
                markerfacecolor=color,
                markeredgecolor='white',
                markeredgewidth=0.5,
                alpha=0.9)

    # Customize the plot with seaborn-style formatting and 12pt fonts
    ax.set_xlabel('Iteration', fontsize=12, fontweight='medium')
    ax.set_ylabel('Mean Score of Top 10', fontsize=12, fontweight='medium')
    ax.set_title(title, fontsize=12, pad=20)

    # Set tick label font sizes to 12pt
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Customize grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Legend positioning for square layout with 12pt font
    ax.legend(frameon=True, fancybox=True, shadow=True,
              loc='best', fontsize=12)

    # Remove top and right spines for cleaner look
    sns.despine(ax=ax)

    # Set axis limits with some padding
    all_iterations = []
    all_scores = []
    max_iterations_per_experiment = []

    for iterations, scores in data.values():
        if iterations:  # Only process non-empty data
            all_iterations.extend(iterations)
            all_scores.extend(scores)
            max_iterations_per_experiment.append(max(iterations))

    if all_iterations and max_iterations_per_experiment:
        # Use the minimum of the maximum iterations across all experiments
        min_max_iterations = min(max_iterations_per_experiment)

        # Filter data to only include iterations up to this limit
        filtered_iterations = [
            it for it in all_iterations if it <= min_max_iterations]

        # Set x-axis limit based on the minimum common range
        ax.set_xlim(min(all_iterations) - 5, min_max_iterations + 5)

        print(
            f"Setting x-axis limit to {min_max_iterations} (minimum of max iterations across experiments)")

        # Set y-axis limits based on all scores
        score_range = max(all_scores) - min(all_scores)
        ax.set_ylim(min(all_scores) - score_range * 0.05,
                    max(all_scores) + score_range * 0.05)

    # Tight layout for better spacing
    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

    plt.close()


def create_summary_statistics(data: Dict[str, Tuple[List[int], List[float]]]) -> str:
    """
    Create summary statistics for all experiments.

    Args:
        data: Dictionary mapping labels to (iterations, scores) tuples

    Returns:
        Summary statistics as a formatted string
    """
    summary = "=== LEARNING CURVE SUMMARY ===\n\n"

    for label, (iterations, scores) in data.items():
        if not scores:
            summary += f"{label}: No data found\n\n"
            continue

        summary += f"{label}:\n"
        summary += f"  Total iterations: {len(iterations)}\n"
        summary += f"  Iteration range: {min(iterations)} - {max(iterations)}\n"
        summary += f"  Initial score: {scores[0]:.4f}\n"
        summary += f"  Final score: {scores[-1]:.4f}\n"
        summary += f"  Best score: {max(scores):.4f} (iteration {iterations[scores.index(max(scores))]})\n"
        summary += f"  Worst score: {min(scores):.4f} (iteration {iterations[scores.index(min(scores))]})\n"
        summary += f"  Score improvement: {scores[-1] - scores[0]:.4f}\n"
        summary += f"  Average score: {np.mean(scores):.4f}\n"
        summary += f"  Score std dev: {np.std(scores):.4f}\n\n"

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Plot learning curves from log files")
    parser.add_argument("--input_files", nargs='+', required=True,
                        help="List of input log files")
    parser.add_argument("--labels", nargs='+', required=True,
                        help="Labels for each input file (must match number of files)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (if not specified, shows plot)")
    parser.add_argument("--title", type=str, default="Learning Curves",
                        help="Plot title")
    parser.add_argument("--figsize", nargs=2, type=int, default=[6, 6],
                        help="Figure size as width height")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary statistics")
    parser.add_argument("--summary_file", type=str, default=None,
                        help="Save summary statistics to file")

    args = parser.parse_args()

    # Validate arguments
    if len(args.input_files) != len(args.labels):
        print(
            f"Error: Number of input files ({len(args.input_files)}) must match number of labels ({len(args.labels)})")
        return 1

    # Check if input files exist
    for file_path in args.input_files:
        if not os.path.exists(file_path):
            print(f"Error: Input file does not exist: {file_path}")
            return 1

    print(f"Processing {len(args.input_files)} log files...")

    # Extract data from all files
    data = {}
    for file_path, label in zip(args.input_files, args.labels):
        print(f"\nProcessing: {file_path} -> {label}")
        iterations, scores = extract_iteration_scores(file_path)
        data[label] = (iterations, scores)

    # Check if we have any data
    total_points = sum(len(scores) for iterations, scores in data.values())
    if total_points == 0:
        print("Error: No iteration data found in any of the input files")
        return 1

    print(f"\nTotal data points extracted: {total_points}")

    # Create summary statistics
    if args.summary or args.summary_file:
        summary = create_summary_statistics(data)

        if args.summary:
            print("\n" + summary)

        if args.summary_file:
            with open(args.summary_file, 'w') as f:
                f.write(summary)
            print(f"Summary statistics saved to {args.summary_file}")

    # Plot the curves
    print(f"\nGenerating plot...")
    plot_learning_curves(data,
                         output_file=args.output,
                         title=args.title,
                         figsize=tuple(args.figsize))

    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
