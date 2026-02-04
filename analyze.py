#!/usr/bin/env python3
"""
Post-evolution analysis CLI.
Analyze evolutionary runs and generate reports.
"""

from pathlib import Path

import typer
from loguru import logger

from biomedxpro.analysis import EvolutionHistory, generate_report
from biomedxpro.analysis.visualizations import generate_all_plots

app = typer.Typer(help="BioMedXPro Evolution Analysis Tools")


@app.command()
def analyze(
    history_file: Path = typer.Argument(
        ...,
        help="Path to *_history.jsonl file",
        exists=True,
    ),
    output_dir: Path = typer.Option(
        Path("reports"),
        "--output-dir",
        "-o",
        help="Directory to save reports and plots",
    ),
    format: str = typer.Option(
        "all",
        "--format",
        "-f",
        help="Report format: json, markdown, html, or all",
    ),
    plateau_threshold: int = typer.Option(
        5,
        "--plateau-threshold",
        "-p",
        help="Number of generations without improvement to consider converged",
    ),
) -> None:
    """
    Analyze an evolutionary run and generate reports.

    Example:
        uv run analyze.py logs/btmri_20260204_085949_history.jsonl
        uv run analyze.py logs/exp_history.jsonl --format html --output-dir reports/exp/
    """
    logger.info(f"Loading history from {history_file}")
    history = EvolutionHistory(history_file)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Reports will be saved to {output_dir}")

    # Generate report
    report = generate_report(history, plateau_threshold=plateau_threshold)

    # Save in requested formats
    formats = ["json", "markdown", "html"] if format == "all" else [format]

    for fmt in formats:
        if fmt == "json":
            report.to_json(output_dir / "report.json")
        elif fmt == "markdown":
            report.to_markdown(output_dir / "report.md")
        elif fmt == "html":
            report.to_html(history, output_dir / "report.html", plateau_threshold)
        else:
            logger.warning(f"Unknown format: {fmt}")

    # Generate plots
    logger.info("Generating visualizations...")
    plots_dir = output_dir / "plots"
    plot_files = generate_all_plots(history, plots_dir, plateau_threshold)

    logger.success(f"✅ Analysis complete! Reports saved to {output_dir}")
    logger.info(f"   - Generated {len(formats)} report(s)")
    logger.info(f"   - Generated {len(plot_files)} plot(s)")


@app.command()
def list_runs(
    log_dir: Path = typer.Option(
        Path("logs"),
        "--log-dir",
        "-d",
        help="Directory containing history logs",
    ),
) -> None:
    """
    List all available evolutionary runs in the logs directory.

    Example:
        uv run analyze.py list-runs
        uv run analyze.py list-runs --log-dir custom_logs/
    """
    if not log_dir.exists():
        logger.error(f"Log directory not found: {log_dir}")
        raise typer.Exit(1)

    # Find all history files
    history_files = sorted(log_dir.glob("*_history.jsonl"))

    if not history_files:
        logger.warning(f"No history files found in {log_dir}")
        return

    logger.info(f"Found {len(history_files)} evolutionary run(s):\n")

    for idx, file_path in enumerate(history_files, start=1):
        # Quick peek at file size and line count
        file_size = file_path.stat().st_size / 1024  # KB
        with open(file_path) as f:
            num_lines = sum(1 for _ in f)

        logger.info(
            f"{idx}. {file_path.name}\n"
            f"   Path: {file_path}\n"
            f"   Size: {file_size:.1f} KB, Records: {num_lines}\n"
        )


if __name__ == "__main__":
    app()
